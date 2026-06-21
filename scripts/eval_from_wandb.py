"""Download finished-run checkpoints from wandb and re-evaluate CANONICALLY:
the run's ACTUAL held-out test split (regenerated deterministically by seed), the
FULL test set (chunked), fixed K=50. Reports per-sample and best-of-K with the
IROS-style range buckets. Big runs are skipped locally (--max_n_traj) -> cluster.

    WANDB_API_KEY=<key> python scripts/eval_from_wandb.py --filter lbe_n --K 50
"""
from __future__ import annotations

import argparse
import os

import torch
import wandb

from diffik.checkpoint import load_checkpoint
from diffik.data import build_datasets_lbe, build_datasets, generate_trajectory, generate_random
from diffik.diffusion import GaussianDiffusion, LBEDiffusion, NoiseSchedule
from diffik.eval import evaluate
from diffik.models import LBEDenoiser, MLPDenoiser
from diffik.utils import get_device


def build_diffusion(cfg, pose_dim, dof):
    mc, dfc = cfg["model"], cfg["diffusion"]
    if mc["type"] == "lbe":
        m = LBEDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc["hidden_dim"], n_layers=mc["n_layers"])
        return LBEDiffusion(m, NoiseSchedule(T=dfc["T"]), dof=dof)
    m = MLPDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc["hidden_dim"], n_layers=mc["n_layers"])
    return GaussianDiffusion(m, NoiseSchedule(T=dfc["T"]), dof=dof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="jacketdembys")
    ap.add_argument("--project", default="diffik")
    ap.add_argument("--filter", default="lbe_n")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--max_n_traj", type=int, default=1600,
                    help="skip runs with n_trajectories above this locally (run those on the cluster)")
    args = ap.parse_args()

    device = get_device()
    api = wandb.Api()
    runs = [r for r in api.runs(f"{args.entity}/{args.project}")
            if args.filter in r.name and r.state == "finished"]
    runs.sort(key=lambda r: r.summary.get("n_train", 0))
    print(f"canonical eval: actual test split, FULL set, K={args.K} (device={device})\n")

    for r in runs:
        try:
            art = api.artifact(f"{args.entity}/{args.project}/{r.name}-ckpt:latest")
            ckpt = os.path.join(art.download(), "checkpoint.pth")
        except Exception as e:
            print(f"{r.name}: no checkpoint artifact ({e})"); continue

        blob = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = blob["config"]
        dc = cfg["data"]
        if dc["kind"] == "trajectory" and dc["n_trajectories"] > args.max_n_traj:
            print(f"=== {r.name} (n_train={r.summary.get('n_train','?')}) SKIPPED locally "
                  f"(n_traj={dc['n_trajectories']} > {args.max_n_traj}); run on cluster")
            continue

        # regenerate the EXACT dataset + leakage-safe test split (deterministic by seed)
        if dc["kind"] == "trajectory":
            ds = generate_trajectory(robot=dc["robot"], n_trajectories=dc["n_trajectories"],
                                     steps_per_traj=dc["steps_per_traj"], v_deg=dc["v_deg"], seed=cfg["seed"])
            _, _, test, q_norm, _ = build_datasets_lbe(ds, v_deg=dc["v_deg"], v_mm=dc["v_mm"], seed=cfg["seed"])
        else:
            ds = generate_random(robot=dc["robot"], n_samples=dc["n_samples"], seed=cfg["seed"])
            _, _, test, q_norm, _ = build_datasets(ds, seed=cfg["seed"])

        pose_dim, dof = ds.pose.shape[1], ds.q.shape[1]
        diff = build_diffusion(cfg, pose_dim, dof)
        diff, q_norm, _, _ = load_checkpoint(ckpt, diff, map_location=device)

        kw = {"example": test.example.to(device)} if (cfg["model"]["type"] == "lbe" and test.example is not None) else {}
        g = torch.Generator().manual_seed(0)
        res = evaluate(diff, test, q_norm, robot=dc["robot"], n_per_pose=args.K, device=device, generator=g, **kw)
        print(f"=== {r.name}  (n_train={r.summary.get('n_train','?')}, test={len(test)})")
        print(f"  best-of-{args.K} (min): {res.best_of_n}")
        print(f"  mean-of-{args.K}      : {res.mean}")
        print(f"  worst-of-{args.K}(max): {res.worst_of_n}")


if __name__ == "__main__":
    main()
