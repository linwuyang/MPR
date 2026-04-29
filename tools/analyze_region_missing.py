import argparse
import torch
import numpy as np
import sys
import os
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.trajectories import TrajectoryDataset as RawTrajectoryDataset

def analyze_region_missing(data_dir, obs_len=8, w_ratio=0.25, h_ratio=0.25, delim="tab", seed=0, region_on="obs"):
    try:
        rng = random.Random(seed)
        
        dset = RawTrajectoryDataset(data_dir, obs_len=obs_len, pred_len=12, skip=1, delim=delim, device="cpu")
        obs_traj = dset.obs_traj.clone()  # [N, 2, obs_len]
        pred_traj = dset.pred_traj.clone()
        
        if region_on == "obs+pred":
            xy = torch.cat([obs_traj, pred_traj], dim=2)  # [N,2,T]
        else:
            xy = obs_traj
        
        x_all = xy[:, 0, :].reshape(-1)
        y_all = xy[:, 1, :].reshape(-1)
        x_min = float(x_all.min().item())
        x_max = float(x_all.max().item())
        y_min = float(y_all.min().item())
        y_max = float(y_all.max().item())
        
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)
        w = float(w_ratio) * x_range
        h = float(h_ratio) * y_range
        
        cx = rng.uniform(x_min + w / 2.0, x_max - w / 2.0) if w < x_range else (x_min + x_max) / 2.0
        cy = rng.uniform(y_min + h / 2.0, y_max - h / 2.0) if h < y_range else (y_min + y_max) / 2.0
        x0, x1 = cx - w / 2.0, cx + w / 2.0
        y0, y1 = cy - h / 2.0, cy + h / 2.0
        
        # mask on obs frames only
        x = obs_traj[:, 0, :]  # [N,obs_len]
        y = obs_traj[:, 1, :]
        miss = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        obs_mask = (~miss).to(torch.float32)
        
        N = obs_traj.size(0)
        T = obs_len
        
        # 统计每个位置的缺失比例
        pos_missing_ratio = miss.float().mean(dim=0).numpy()  # [T]
        
        # 统计整体缺失情况
        total_missing_ratio = miss.float().mean().item()
        traj_with_missing = (miss.sum(dim=1) > 0).float().mean().item()
        avg_missing_per_traj = miss.sum(dim=1).float().mean().item()
        
        return {
            'name': os.path.basename(data_dir),
            'N': N,
            'pos_missing_ratio': pos_missing_ratio,
            'total_missing_ratio': total_missing_ratio,
            'traj_with_missing': traj_with_missing,
            'avg_missing_per_traj': avg_missing_per_traj,
            'region': {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1},
            'x_range': x_range,
            'y_range': y_range,
            'w': w,
            'h': h,
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
    
    w_ratio = 0.25
    h_ratio = 0.25
    
    results = []
    for data_dir, delim in datasets:
        print(f"\n处理: {data_dir}")
        result = analyze_region_missing(data_dir, w_ratio=w_ratio, h_ratio=h_ratio, delim=delim, region_on="obs")
        results.append(result)
    
    print("\n" + "="*80)
    print(f"区域缺失统计结果汇总 (w_ratio={w_ratio}, h_ratio={h_ratio})")
    print("="*80)
    
    for r in results:
        if 'error' in r:
            print(f"\n{r['name']}: 错误 - {r['error']}")
            continue
            
        print(f"\n数据集: {r['name']}")
        print(f"轨迹数量: {r['N']}")
        print(f"\n缺失区域信息:")
        print(f"  坐标范围: x=[{r['region']['x0']:.2f}, {r['region']['x1']:.2f}], y=[{r['region']['y0']:.2f}, {r['region']['y1']:.2f}]")
        print(f"  区域大小: w={r['w']:.2f}, h={r['h']:.2f}")
        print(f"  数据范围: x_range={r['x_range']:.2f}, y_range={r['y_range']:.2f}")
        print(f"\n各位置缺失比例:")
        for t in range(8):
            print(f"  位置 {t}: {r['pos_missing_ratio'][t]:.4f} ({r['pos_missing_ratio'][t]*100:.2f}%)")
        
        print(f"\n整体统计:")
        print(f"  总缺失比例: {r['total_missing_ratio']:.4f} ({r['total_missing_ratio']*100:.2f}%)")
        print(f"  有缺失的轨迹比例: {r['traj_with_missing']:.4f} ({r['traj_with_missing']*100:.2f}%)")
        print(f"  平均每条轨迹缺失帧数: {r['avg_missing_per_traj']:.2f}")

if __name__ == "__main__":
    main()

