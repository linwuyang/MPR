import argparse
import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.trajectories import TrajectoryDataset as RawTrajectoryDataset

def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def analyze_missing_probs(data_dir, obs_len=8, p_base=0.1, p_high=0.7, a_thresh=0.5, k=5.0, delim="tab"):
    try:
        dset = RawTrajectoryDataset(data_dir, obs_len=obs_len, pred_len=12, skip=1, delim=delim, device="cpu")
        obs_traj = dset.obs_traj  # [N, 2, obs_len]
        
        N = obs_traj.size(0)
        T = obs_len
        p_missing = torch.full((N, T), float(p_base), dtype=torch.float32)
        
        # 计算速度
        v = obs_traj[:, :, 1:] - obs_traj[:, :, :-1]  # [N,2,T-1]
        # 计算加速度
        a = torch.norm(v[:, :, 1:] - v[:, :, :-1], dim=1)  # [N,T-2]
        
        # 计算动态概率
        p_dyn = float(p_base) + (float(p_high) - float(p_base)) * _sigmoid(float(k) * (a - float(a_thresh)))
        p_missing[:, 2:] = p_dyn
        
        # 统计每个位置的缺失概率
        pos_probs = p_missing.mean(dim=0).numpy()  # [T]
        pos_stds = p_missing.std(dim=0).numpy()
        
        # 统计加速度分布
        a_mean = a.mean().item()
        a_std = a.std().item()
        a_min = a.min().item()
        a_max = a.max().item()
        
        return {
            'name': os.path.basename(data_dir),
            'N': N,
            'pos_probs': pos_probs,
            'pos_stds': pos_stds,
            'a_mean': a_mean,
            'a_std': a_std,
            'a_min': a_min,
            'a_max': a_max
        }
    except Exception as e:
        return {'name': os.path.basename(data_dir), 'error': str(e)}

def main():
    datasets = [
        ('datasets/ETH/train', 'tab'),
        ('datasets/UCY/train', 'tab'),
        ('datasets/inD/train', 'tab'),
        ('datasets/INTERACTION/train', 'tab'),
    ]
    
    results = []
    for data_dir, delim in datasets:
        print(f"\n处理: {data_dir}")
        result = analyze_missing_probs(data_dir, delim=delim, a_thresh=0.1)
        results.append(result)
    
    print("\n" + "="*80)
    print("统计结果汇总")
    print("="*80)
    
    for r in results:
        if 'error' in r:
            print(f"\n{r['name']}: 错误 - {r['error']}")
            continue
            
        print(f"\n数据集: {r['name']}")
        print(f"轨迹数量: {r['N']}")
        print(f"\n各位置缺失概率 (均值 ± 标准差):")
        for t in range(8):
            print(f"  位置 {t}: {r['pos_probs'][t]:.4f} ± {r['pos_stds'][t]:.4f}")
        
        print(f"\n加速度统计 (位置2-7):")
        print(f"  均值: {r['a_mean']:.4f}")
        print(f"  标准差: {r['a_std']:.4f}")
        print(f"  最小值: {r['a_min']:.4f}")
        print(f"  最大值: {r['a_max']:.4f}")

if __name__ == "__main__":
    main()

