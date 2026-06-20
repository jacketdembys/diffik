"""Evaluate multimodality of a trained run, seedless vs seeded (the knob).

    python scripts/eval_multimodality.py --run_dir runs_es/lbe_traj_n3200 --K 50

Loads the run's config/checkpoint/dataset, draws K candidates per test pose, and
reports best-of-K accuracy + diversity + valid-solution coverage for the seedless
regime and (for LBE models) the seeded regime. Saves multimodality.json.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from diffik.checkpoint import load_checkpoint
from diffik.data import build_datasets, build_datasets_lbe, load_dataset
from diffik.diffusion import GaussianDiffusion, LBEDiffusion, NoiseSchedule
from diffik.eval import evaluate_multimodality
from diffik.models import LBEDenoiser, MLPDenoiser
from diffik.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--n_poses", type=int, default=256, help="cap test poses (P x K samples)")
    ap.add_argument("--tol_mm", type=float, default=10.0)
    ap.add_argument("--tol_deg", type=float, default=5.0)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    ds = load_dataset(os.path.join(args.run_dir, "dataset.npz"))
    seed = cfg["seed"]
    dc, mc, dfc = cfg["data"], cfg["model"], cfg["diffusion"]
    pose_dim, dof = ds.pose.shape[1], ds.q.shape[1]
    device = get_device()

    if dc["lbe"]:
        _, _, test, q_norm, _ = build_datasets_lbe(ds, v_deg=dc["v_deg"], v_mm=dc["v_mm"], seed=seed)
    else:
        _, _, test, q_norm, _ = build_datasets(ds, seed=seed)
    test = test.head(args.n_poses) if len(test) > args.n_poses else test

    if mc["type"] == "lbe":
        model = LBEDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc["hidden_dim"], n_layers=mc["n_layers"])
        diffusion = LBEDiffusion(model, NoiseSchedule(T=dfc["T"]), dof=dof)
    else:
        model = MLPDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc["hidden_dim"], n_layers=mc["n_layers"])
        diffusion = GaussianDiffusion(model, NoiseSchedule(T=dfc["T"]), dof=dof)
    diffusion, q_norm, _, _ = load_checkpoint(os.path.join(args.run_dir, "checkpoint.pth"), diffusion, map_location=device)

    regimes = {"seedless": {}}
    if mc["type"] == "lbe":
        regimes["seeded"] = {"example": test.example.to(device)}

    out = {"run": args.run_dir, "K": args.K, "n_poses": len(test), "n_params": sum(p.numel() for p in model.parameters())}
    for name, kw in regimes.items():
        g = torch.Generator().manual_seed(0)
        res = evaluate_multimodality(diffusion, test, q_norm, robot=dc["robot"], K=args.K,
                                     device=device, generator=g, tol_mm=args.tol_mm, tol_deg=args.tol_deg, **kw)
        print(f"\n=== {name} ===\n{res}")
        out[name] = {
            "pos_mm_avg_bestK": res.best_of_k.pos_mm_avg, "ori_deg_avg_bestK": res.best_of_k.ori_deg_avg,
            "diversity_all": res.diversity_all, "mean_valid_per_pose": res.mean_valid_per_pose,
            "frac_poses_multi": res.frac_poses_multi, "valid_diversity": res.valid_diversity,
        }
    with open(os.path.join(args.run_dir, "multimodality.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {os.path.join(args.run_dir, 'multimodality.json')}")


if __name__ == "__main__":
    main()
