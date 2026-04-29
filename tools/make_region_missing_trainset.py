import argparse
import os
import random

import torch

from data.trajectories import TrajectoryDataset as RawTrajectoryDataset


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

    ap.add_argument("--w_ratio", type=float, default=0.2, help="缺失矩形宽度=数据x范围*w_ratio")
    ap.add_argument("--h_ratio", type=float, default=0.2, help="缺失矩形高度=数据y范围*h_ratio")
    ap.add_argument("--region_on", type=str, default="obs", choices=["obs", "obs+pred"])
    args = ap.parse_args()

    rng = random.Random(args.seed)

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

    if args.region_on == "obs+pred":
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
    w = float(args.w_ratio) * x_range
    h = float(args.h_ratio) * y_range

    cx = rng.uniform(x_min + w / 2.0, x_max - w / 2.0) if w < x_range else (x_min + x_max) / 2.0
    cy = rng.uniform(y_min + h / 2.0, y_max - h / 2.0) if h < y_range else (y_min + y_max) / 2.0
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0

    if args.fill == "nan":
        fill_value = float("nan")
    else:
        fill_value = 0.0

    # mask on obs frames only
    x = obs_traj[:, 0, :]  # [N,obs_len]
    y = obs_traj[:, 1, :]
    miss = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    obs_mask = (~miss).to(torch.float32)

    # apply missing
    obs_traj[miss.unsqueeze(1).expand(-1, 2, -1)] = fill_value
    obs_traj_rel[miss.unsqueeze(1).expand(-1, 2, -1)] = 0.0

    payload = {
        "obs_traj": obs_traj,
        "pred_traj": pred_traj,
        "obs_traj_rel": obs_traj_rel,
        "pred_traj_rel": pred_traj_rel,
        "loss_mask": loss_mask,
        "non_linear_ped": non_linear_ped,
        "seq_start_end": seq_start_end,
        "obs_mask": obs_mask,  # [N,obs_len]
        "region": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "meta": {
            "data_dir": os.path.abspath(args.data_dir),
            "obs_len": args.obs_len,
            "pred_len": args.pred_len,
            "skip": args.skip,
            "delim": args.delim,
            "seed": args.seed,
            "fill": args.fill,
            "w_ratio": args.w_ratio,
            "h_ratio": args.h_ratio,
            "region_on": args.region_on,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)


if __name__ == "__main__":
    main()


