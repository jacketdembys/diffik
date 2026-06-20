"""Re-evaluate saved checkpoints for single-sample vs best-of-K accuracy.

Appends to <out_dir>/bestofk.csv (skip-if-done). best-of-K = draw K candidates
per pose, keep the lowest-position-error one (FK-selected, refinement-free) -- the
realistic generative-IK deployment metric. Reports BOTH position (mm) and
orientation (deg) for single and best-of-K.

    python scripts/eval_bestofk.py --runs runs_es/lbe_traj_n50 runs_es/lbe_traj_n100 --K 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import pandas as pd
import torch

from diffik.checkpoint import load_checkpoint
from diffik.data import build_datasets, build_datasets_lbe, load_dataset
from diffik.diffusion import GaussianDiffusion, LBEDiffusion, NoiseSchedule
from diffik.eval import evaluate
from diffik.models import LBEDenoiser, MLPDenoiser

FIELDS = ["name", "n_train", "K", "pos_single", "ori_single", "pos_bestk", "ori_bestk",
          "pct_pos_le_1mm_bestk", "pct_ori_le_1deg_bestk"]


def eval_run(run_dir, K, n_poses, device):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
    ds = load_dataset(os.path.join(run_dir, "dataset.npz"))
    seed = cfg["seed"]
    dc, mc, dfc = cfg["data"], cfg["model"], cfg["diffusion"]
    pose_dim, dof = ds.pose.shape[1], ds.q.shape[1]

    if dc["lbe"]:
        _, _, test, q_norm, _ = build_datasets_lbe(ds, v_deg=dc["v_deg"], v_mm=dc["v_mm"], seed=seed)
    else:
        _, _, test, q_norm, _ = build_datasets(ds, seed=seed)
    test = test.head(n_poses) if len(test) > n_poses else test

    if mc["type"] == "lbe":
        model = LBEDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc["hidden_dim"], n_layers=mc["n_layers"])
        diffusion = LBEDiffusion(model, NoiseSchedule(T=dfc["T"]), dof=dof)
    else:
        model = MLPDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc["hidden_dim"], n_layers=mc["n_layers"])
        diffusion = GaussianDiffusion(model, NoiseSchedule(T=dfc["T"]), dof=dof)
    diffusion, q_norm, _, _ = load_checkpoint(os.path.join(run_dir, "checkpoint.pth"), diffusion, map_location=device)

    kw = {}
    if mc["type"] == "lbe" and cfg["eval"].get("seeded", True):
        kw["example"] = test.example.to(device)

    g = torch.Generator().manual_seed(0)
    single = evaluate(diffusion, test, q_norm, robot=dc["robot"], n_per_pose=1, device=device, generator=g, **kw)
    g = torch.Generator().manual_seed(0)
    bestk = evaluate(diffusion, test, q_norm, robot=dc["robot"], n_per_pose=K, device=device, generator=g, **kw)
    return {
        "name": cfg["name"], "n_train": metrics["n_train"], "K": K,
        "pos_single": round(single.best_of_n.pos_mm_avg, 4), "ori_single": round(single.best_of_n.ori_deg_avg, 4),
        "pos_bestk": round(bestk.best_of_n.pos_mm_avg, 4), "ori_bestk": round(bestk.best_of_n.ori_deg_avg, 4),
        "pct_pos_le_1mm_bestk": round(bestk.best_of_n.pct_pos_le_1mm, 4),
        "pct_ori_le_1deg_bestk": round(bestk.best_of_n.pct_ori_le_1deg, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--n_poses", type=int, default=256)
    ap.add_argument("--out_dir", default="runs_es")
    ap.add_argument("--device", default="cpu", help="cpu avoids contending with MPS training")
    args = ap.parse_args()

    out_csv = os.path.join(args.out_dir, "bestofk.csv")
    done = set(pd.read_csv(out_csv)["name"].astype(str)) if os.path.exists(out_csv) else set()

    for run_dir in args.runs:
        name = json.load(open(os.path.join(run_dir, "config.json")))["name"]
        if name in done:
            print(f"[skip] {name}")
            continue
        row = eval_run(run_dir, args.K, args.n_poses, args.device)
        exists = os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if not exists:
                w.writeheader()
            w.writerow(row)
        print(f"{row['name']:>18}  n={row['n_train']:>7}  single {row['pos_single']:.2f}mm/{row['ori_single']:.2f}deg"
              f"  -> best-of-{args.K} {row['pos_bestk']:.2f}mm/{row['ori_bestk']:.2f}deg"
              f"  (<=1mm {row['pct_pos_le_1mm_bestk']:.1f}%)")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
