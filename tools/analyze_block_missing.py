import sys
import os
import random

import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.trajectories import TrajectoryDataset as RawTrajectoryDataset


def _pick_block(obs_len: int, min_len: int, max_len: int, rng: random.Random):
    L = rng.randint(min_len, max_len)
    s = rng.randint(0, obs_len - L)
    return s, L


def analyze_block_missing(data_dir, obs_len=8, min_occ=1, max_occ=4, delim="tab", seed=0):
    rng = random.Random(seed)

    dset = RawTrajectoryDataset(
        data_dir,
        obs_len=obs_len,
        pred_len=12,
        skip=1,
        delim=delim,
        device="cpu",
    )

    N = dset.obs_traj.size(0)
    T = obs_len

    obs_mask = torch.ones((N, T), dtype=torch.float32)

    for i in range(N):
        s, L = _pick_block(T, min_occ, max_occ, rng)
        obs_mask[i, s : s + L] = 0.0

    miss = obs_mask == 0.0

    pos_missing_ratio = miss.float().mean(dim=0).numpy()
    total_missing_ratio = miss.float().mean().item()
    traj_with_missing = (miss.sum(dim=1) > 0).float().mean().item()
    avg_missing_per_traj = miss.sum(dim=1).float().mean().item()

    return {
        "name": os.path.basename(data_dir),
        "N": N,
        "pos_missing_ratio": pos_missing_ratio,
        "total_missing_ratio": total_missing_ratio,
        "traj_with_missing": traj_with_missing,
        "avg_missing_per_traj": avg_missing_per_traj,
    }


def main():
    datasets = [
        ("datasets/ETH/train", "tab"),
        ("datasets/UCY/train", "tab"),
        ("datasets/inD/train", "tab"),
        ("datasets/INTERACTION/train", "tab"),
    ]

    results = []
    for data_dir, delim in datasets:
        print(f"\n处理: {data_dir}")
        r = analyze_block_missing(data_dir, delim=delim)
        results.append(r)

    print("\n" + "=" * 80)
    print("块缺失统计结果汇总 (min_occ=1, max_occ=4)" )
    print("=" * 80)

    for r in results:
        print(f"\n数据集: {r['name']}")
        print(f"轨迹数量: {r['N']}")
        print("\n各位置缺失比例:")
        for t in range(8):
            v = r["pos_missing_ratio"][t]
            print(f"  位置 {t}: {v:.4f} ({v*100:.2f}%)")
        print("\n整体统计:")
        print(f"  总缺失比例: {r['total_missing_ratio']:.4f} ({r['total_missing_ratio']*100:.2f}%)")
        print(f"  有缺失的轨迹比例: {r['traj_with_missing']:.4f} ({r['traj_with_missing']*100:.2f}%)")
        print(f"  平均每条轨迹缺失帧数: {r['avg_missing_per_traj']:.2f}")


if __name__ == "__main__":
    main()
