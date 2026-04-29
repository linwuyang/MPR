import argparse
import os
import random

import torch

from data.trajectories import TrajectoryDataset as RawTrajectoryDataset


def _pick_block(obs_len: int, min_len: int, max_len: int, rng: random.Random):
    L = rng.randint(min_len, max_len)
    s = rng.randint(0, obs_len - L)
    return s, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="训练集目录，里面是多个txt轨迹文件")
    ap.add_argument("--out", type=str, required=True, help="输出.pt路径")
    ap.add_argument("--obs_len", type=int, default=8)
    ap.add_argument("--pred_len", type=int, default=12)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--delim", type=str, default="tab", help="tab/space 或者直接传 \\t/ ' '")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_occ", type=int, default=1)
    ap.add_argument("--max_occ", type=int, default=4)
    ap.add_argument("--fill", type=str, default="zero", choices=["zero", "nan"])
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

    N = obs_traj.size(0)
    obs_mask = torch.ones((N, args.obs_len), dtype=torch.float32)
    occ_start = torch.empty((N,), dtype=torch.long)
    occ_len = torch.empty((N,), dtype=torch.long)

    if args.fill == "nan":
        fill_value = float("nan")
    else:
        fill_value = 0.0

    for i in range(N):
        s, L = _pick_block(args.obs_len, args.min_occ, args.max_occ, rng)
        occ_start[i] = s
        occ_len[i] = L
        obs_mask[i, s : s + L] = 0.0
        obs_traj[i, :, s : s + L] = fill_value
        obs_traj_rel[i, :, s : s + L] = 0.0

    payload = {
        "obs_traj": obs_traj,  # [N,2,obs_len]
        "pred_traj": pred_traj,  # [N,2,pred_len]
        "obs_traj_rel": obs_traj_rel,
        "pred_traj_rel": pred_traj_rel,
        "loss_mask": loss_mask,
        "non_linear_ped": non_linear_ped,
        "seq_start_end": seq_start_end,  # [num_seq,2]
        "obs_mask": obs_mask,  # [N,obs_len] 1=可见 0=遮挡
        "occ_start": occ_start,
        "occ_len": occ_len,
        "meta": {
            "data_dir": os.path.abspath(args.data_dir),
            "obs_len": args.obs_len,
            "pred_len": args.pred_len,
            "skip": args.skip,
            "delim": args.delim,
            "seed": args.seed,
            "min_occ": args.min_occ,
            "max_occ": args.max_occ,
            "fill": args.fill,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)


if __name__ == "__main__":
    main()


