import argparse
import os
import random

import torch

from data.trajectories import TrajectoryDataset as RawTrajectoryDataset


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="训练集目录，里面是多个txt轨迹文件")
    ap.add_argument("--out", type=str, required=True, help="输出.pt路径")
    ap.add_argument("--obs_len", type=int, default=8)
    ap.add_argument("--pred_len", type=int, default=12)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--delim", type=str, default="tab", help="tab/space 或者直接传 \\t/ ' '")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fill", type=str, default="zero", choices=["zero", "nan"])

    ap.add_argument("--p_base", type=float, default=0.1)
    ap.add_argument("--p_high", type=float, default=0.7)
    ap.add_argument("--a_thresh", type=float, default=0.5)
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--ensure_missing", action="store_true", help="保证每条轨迹至少缺失1帧(否则重采样)")
    ap.add_argument("--max_resample", type=int, default=50)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)

    dset = RawTrajectoryDataset(
        args.data_dir,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        device="cpu",
    )

    obs_traj = dset.obs_traj.clone()  # [N, 2, obs_len]
    pred_traj = dset.pred_traj.clone()
    obs_traj_rel = dset.obs_traj_rel.clone()
    pred_traj_rel = dset.pred_traj_rel.clone()
    loss_mask = dset.loss_mask.clone()
    non_linear_ped = dset.non_linear_ped.clone()
    seq_start_end = torch.as_tensor(dset.seq_start_end, dtype=torch.long)  # [num_seq, 2]

    N = obs_traj.size(0)
    T = args.obs_len
    obs_mask = torch.ones((N, T), dtype=torch.float32)
    p_missing = torch.full((N, T), float(args.p_base), dtype=torch.float32)

    if args.fill == "nan":
        fill_value = float("nan")
    else:
        fill_value = 0.0

    # v[t] = pos[t] - pos[t-1], t>=1  => shape [N,2,T-1]
    v = obs_traj[:, :, 1:] - obs_traj[:, :, :-1]
    # a[t] = ||v[t] - v[t-1]||, t>=2  => shape [N,T-2]
    a = torch.norm(v[:, :, 1:] - v[:, :, :-1], dim=1)
    # map to prob for frames t=2..T-1
    p_dyn = float(args.p_base) + (float(args.p_high) - float(args.p_base)) * _sigmoid(
        float(args.k) * (a - float(args.a_thresh))
    )
    p_missing[:, 2:] = p_dyn
    p_missing.clamp_(0.0, 1.0)

    # sample missing per timepoint
    for i in range(N):
        for _ in range(args.max_resample if args.ensure_missing else 1):
            u = torch.rand((T,), generator=g)
            miss = (u < p_missing[i]).to(torch.float32)
            # 可选：避免全缺失（至少留1帧）
            if miss.sum().item() >= T:
                keep_idx = rng.randrange(T)
                miss[keep_idx] = 0.0
            if (not args.ensure_missing) or (miss.sum().item() >= 1):
                obs_mask[i] = 1.0 - miss
                break

    miss_idx = obs_mask == 0.0
    # apply missing
    obs_traj[miss_idx.unsqueeze(1).expand(-1, 2, -1)] = fill_value
    obs_traj_rel[miss_idx.unsqueeze(1).expand(-1, 2, -1)] = 0.0

    payload = {
        "obs_traj": obs_traj,
        "pred_traj": pred_traj,
        "obs_traj_rel": obs_traj_rel,
        "pred_traj_rel": pred_traj_rel,
        "loss_mask": loss_mask,
        "non_linear_ped": non_linear_ped,
        "seq_start_end": seq_start_end,
        "obs_mask": obs_mask,      # [N,T] 1=可见 0=缺失
        "p_missing": p_missing,    # [N,T] 每帧缺失概率
        "meta": {
            "data_dir": os.path.abspath(args.data_dir),
            "obs_len": args.obs_len,
            "pred_len": args.pred_len,
            "skip": args.skip,
            "delim": args.delim,
            "seed": args.seed,
            "fill": args.fill,
            "p_base": args.p_base,
            "p_high": args.p_high,
            "a_thresh": args.a_thresh,
            "k": args.k,
            "ensure_missing": bool(args.ensure_missing),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)


if __name__ == "__main__":
    main()


