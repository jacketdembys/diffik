"""Scaling sweep: train DiffIK on increasingly larger datasets, saving after EACH
run so partial progress survives interruption (results live in runs/experiments.csv
and runs/<name>/). Re-running skips runs whose registry row already exists.

Examples:
    python scripts/sweep.py --regime lbe --sizes 25,50,100,200,400 --epochs 300
    python scripts/sweep.py --regime seedless --sizes 2000,8000,32000 --epochs 300
Run in background for long sweeps; each completed size is persisted immediately.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from diffik.config import Config, DataConfig, DiffusionConfig, EvalConfig, ModelConfig, TrainConfig
from train import run  # scripts/train.py


def already_done(out_dir, name):
    reg = os.path.join(out_dir, "experiments.csv")
    if not os.path.exists(reg):
        return False
    return name in set(pd.read_csv(reg)["name"].astype(str))


def make_cfg(regime, size, epochs, out_dir, hidden, n_layers, T, patience, monitor_every, suffix=""):
    """size = n_trajectories (lbe) or n_samples (seedless). epochs = MAX epochs."""
    if regime == "lbe":
        name = f"lbe_traj_n{size}{suffix}"
        data = DataConfig(kind="trajectory", lbe=True, n_trajectories=size, steps_per_traj=40, v_deg=1.0)
        model = ModelConfig(type="lbe", hidden_dim=hidden, n_layers=n_layers)
        evalc = EvalConfig(n_per_pose=1, sampler="ddpm", seeded=True)
    else:
        name = f"seedless_rand_n{size}{suffix}"
        data = DataConfig(kind="random", lbe=False, n_samples=size)
        model = ModelConfig(type="mlp", hidden_dim=hidden, n_layers=n_layers)
        evalc = EvalConfig(n_per_pose=10, sampler="ddpm")
    return Config(
        name=name, out_dir=out_dir,
        data=data, model=model,
        diffusion=DiffusionConfig(T=T, fk_loss_weight=10.0, rot_weight=0.1, p_example_dropout=0.3),
        train=TrainConfig(epochs=epochs, batch_size=256, lr=1e-3, checkpoint_every=max(epochs // 4, 1),
                          patience=patience, early_stop_metric="val_pose", monitor_every=monitor_every),
        eval=evalc,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["lbe", "seedless"], default="lbe")
    ap.add_argument("--sizes", default="25,50,100,200,400",
                    help="comma list: n_trajectories (lbe) or n_samples (seedless)")
    ap.add_argument("--epochs", type=int, default=1000, help="MAX epochs (early stopping may stop sooner)")
    ap.add_argument("--patience", type=int, default=8, help="early-stop patience in CHECKS; 0 disables")
    ap.add_argument("--monitor_every", type=int, default=10, help="epochs between val pose-error checks")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--out_dir", default="runs_es")
    ap.add_argument("--suffix", default="", help="appended to run names (for reruns w/o collision)")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"sweep regime={args.regime} sizes={sizes} max_epochs={args.epochs} patience={args.patience}")
    for size in sizes:
        cfg = make_cfg(args.regime, size, args.epochs, args.out_dir, args.hidden, args.n_layers,
                       args.T, args.patience, args.monitor_every, args.suffix)
        if already_done(args.out_dir, cfg.name):
            print(f"[skip] {cfg.name} already in registry")
            continue
        print(f"\n========== {cfg.name} ==========")
        run(cfg)
    print(f"\nsweep complete. Registry: {os.path.join(args.out_dir, 'experiments.csv')}")


if __name__ == "__main__":
    main()
