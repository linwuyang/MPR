import torch
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn as nn
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

EPS = 1E-20
def diff_in_weights(model, proxy):
    with torch.no_grad():
        diff_dict = OrderedDict()
        model_state_dict = model.state_dict()
        proxy_state_dict = proxy.state_dict()
        for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
            if len(old_w.size()) <= 1:
                continue
            if 'weight' in old_k:
                diff_w = new_w - old_w
                diff_dict[old_k] = diff_w # old_w.norm() / (diff_w.norm() + EPS) *
        return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])

def normalize(perturbations, weights):
    perturbations.mul_(weights.norm()/(perturbations.norm() + EPS))

def normalize_grad(weights, ref_weights):
    with torch.no_grad():
        for w, ref_w in zip(weights, ref_weights):
            if w.grad is None:  # 新增检查
                continue
            if w.dim() <= 1:
                w.grad.data.fill_(0)  # ignore perturbations with 1 dimension (e.g. BN, bias)
            else:
                normalize(w.grad.data, ref_w)


class Perturber():
    EPS = 1E-20
    def __init__(self, continual_model):
        self.continual_model = continual_model
        self.net = continual_model
        self.proxy = copy.deepcopy(continual_model)
        self.steps = 5
        self.lam = 0.01
        self.gamma = 0.05
        self.diff = None
        self.mask = None
    
    def _compute_param_stability(self, X, Y, seq_start_end, n_samples=1):
        original_output = self.net(X, seq_start_end).detach()
        stability_scores = {}
        psc_ade_pairs = []
        
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                if param.dim() <= 1:
                    stability_scores[name] = float('inf')
                    continue
                    
                original_data = param.data.clone()
                total_delta = 0
                total_ade = 0
                
                # 多次扰动取平均
                for _ in range(n_samples):
                    perturbation = torch.randn_like(param) * 0.1 * param.std()
                    param.data.add_(perturbation)
                    perturbed_output = self.net(X, seq_start_end)

                    total_delta += F.mse_loss(original_output, perturbed_output).item()
                    total_ade += F.mse_loss(Y, perturbed_output).item()
                    param.data.copy_(original_data)  # 恢复原始值
                    
                avg_score = total_delta / n_samples
                avg_ade = total_ade / n_samples
                stability_scores[name] = avg_score
                psc_ade_pairs.append((avg_score, avg_ade))
        # plt.figure(figsize=(12, 8), dpi=1200)
        # scores, ades = zip(*psc_ade_pairs)
        # print(scores)
        # print('\n')
        # print(ades)
        
        # # 散点图（按稳定性分数着色）
        # scatter = plt.scatter(scores, ades, 
        #                     cmap='viridis',
        #                     alpha=0.7,
        #                     s=100,
        #                     edgecolor='white')
        
        
        # # 坐标轴设置
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('Parameter Stability Criticality (log scale)', fontsize=20)
        # plt.ylabel('ADE Loss (log scale)', fontsize=20)

        # # 添加网格线（关键修改点）
        # plt.grid(True, which='both',  # 主次刻度均显示网格
        #         linestyle='--',       # 虚线样式
        #         linewidth=0.5,        # 细线
        #         alpha=0.5,           # 半透明
        #         color='gray')        # 中性灰色
        
        # # 添加统计信息
        # corr_coef = np.corrcoef(scores, ades)[0,1]
        # plt.text(0.95, 0.05, 
        #         f'Pearson r = {corr_coef:.2f}',
        #         transform=plt.gca().transAxes,
        #         ha='right', va='bottom',
        #         fontsize=20,
        #         bbox=dict(boxstyle='round', alpha=0.1))
        
        # # 保存图表
        # plot_path='results/stability_ade.png'
        # plt.tight_layout()
        # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"稳定性-ADE关系图已保存至: {os.path.abspath(plot_path)}")
        # sys.exit()
        
        return sorted(stability_scores.items(), key=lambda x: x[1])


    def _freeze_stable_params(self, stability_scores, freeze_ratio=0.2):
        """冻结最稳定的前20%参数"""
        num_to_freeze = int(len(stability_scores) * freeze_ratio)
        stable_params = [name for name, _ in stability_scores[:num_to_freeze]]
        
        for name, param in self.net.named_parameters():
            param.requires_grad = (name not in stable_params)
    
    def init_rand(self, model):
        with torch.no_grad():
            for w in model.parameters():
                if w.dim() <= 1:
                    continue
                else:
                    # z = torch.randn_like(w)
                    # z = z / torch.norm(z)
                    # delta = z * torch.norm(w) * self.gamma
                    #w.add_(delta)
                    w.add_(torch.randn_like(w) * torch.norm(w) * EPS)

    
    def perturb_model(self, X, y, seq_start_end_):
        out_o = self.net(X, seq_start_end_).detach()  # shape: [batch, pred_len*2]
    
        self.proxy.load_state_dict(self.net.state_dict())
        self.init_rand(self.proxy)
        #self.diff = diff_in_weights(self.net, self.proxy) # 消融实验添加这一行
        self.proxy.train()

        pertopt = torch.optim.SGD(self.proxy.parameters(), lr=self.gamma/self.steps)
    
        with torch.no_grad():
            pred_error = F.mse_loss(out_o, y, reduction='none').mean(dim=[0,2])  # [batch]
            threshold = pred_error.quantile(0.5)
            mask = (pred_error <= threshold).float()
            self.mask = mask
        
        #return out_o, torch.ones_like(mask).float() # 消融实验添加这一行
    
        if mask.sum() < 2:
            return None, mask

        min_loss = 10000000
        for idx in range(self.steps):
            pertopt.zero_grad()
        
            proxy_out = self.proxy(X, seq_start_end_)
            loss = -F.huber_loss(proxy_out, out_o, reduction='none').mean(dim=[0,2])  # 负号表示最大化差异
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)  # 只计算mask样本的损失
        
            loss.backward()
            normalize_grad(self.proxy.parameters(), self.net.parameters())
            pertopt.step()

        self.diff = diff_in_weights(self.net, self.proxy)
    
        return out_o, mask
    
    def get_loss(self, X, y, seq_start_end_):
        outs, mask = self.perturb_model(X, y, seq_start_end_)
        add_into_weights(self.net, self.diff, coeff=1.0)
        out_n = self.net(X, seq_start_end_)
        if outs is not None:
            loss1 = F.huber_loss(out_n, outs, reduction='none')  # [12,30,2]
            loss_per_sample1 = loss1.mean(dim=[0,2])
            weighted_loss1 = (loss_per_sample1 * mask).sum() / (mask.sum() + 1e-6)
            return self.lam *(weighted_loss1), outs
        else:
            return None


    def restore_model(self):
        add_into_weights(self.net, self.diff, coeff=-1.0)
        for param in self.net.parameters():
            param.requires_grad = True
    
    def __call__(self, X, y, seq_start_end_):
        X = X.cuda()
        y = y.cuda()
        stability_scores = self._compute_param_stability(X, y, seq_start_end_)
        loss_kl, outs = self.get_loss(X, y, seq_start_end_)
        if loss_kl is not None:
            self._freeze_stable_params(stability_scores)
            loss_kl.backward()
            self.restore_model()