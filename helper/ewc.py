import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from data.loader import data_loader, data_dset
import numpy as np
from torch.utils.data import DataLoader


class ElasticWeightConsolidation:

    def __init__(self, model, crit, lr=0.001, weight=1000000):
        self.model = model
        self.weight = weight
        self.crit = crit
        self.optimizer = optim.Adam(self.model.parameters(), lr)

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params(self, dl, num_batch):
        gradients = []
        for i, batch in enumerate(dl):
            if i > num_batch:
                break
            batch = [tensor.cuda() for tensor in batch]
            predicted_trajectory = self.model(batch[2], batch[6])
            # 计算MSE损失（替代分类的对数似然）
            loss = self.crit(predicted_trajectory, batch[3])
            # 计算梯度
            grad = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
            gradients.append([g.detach() for g in grad])
        
        # 计算Fisher信息（梯度平方的均值）
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            fisher = torch.mean(torch.stack([g[idx]**2 for g in gradients]), dim=0)
            self.model.register_buffer(f"{name.replace('.', '__')}_estimated_fisher", fisher)
        #     output = F.log_softmax(self.model(batch[2]), dim=1)
        #     log_liklihoods.append(output[:, batch[3]])
        # log_likelihood = torch.cat(log_liklihoods).mean()
        # grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        # _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        # for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
        #     self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, dataset, num_batches):
        self._update_fisher_params(dataset, num_batches)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0

    def forward_backward_update(self, input, target, seq_start_end):
        output = self.model(input, seq_start_end)
        loss = self._compute_consolidation_loss(self.weight) + self.crit(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)