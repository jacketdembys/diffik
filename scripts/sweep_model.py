"""Model-size sweep at a FIXED dataset size (the best from the data sweep).

Varies denoiser capacity (hidden x layers -> #params); trains each with the same
early-stopping protocol, saving incrementally. Pair with eval_multimodality.py to
also get the diversity/coverage of each size.

    python scripts/sweep_model.py --regime lbe --n 6400 \
        --configs 256x3,512x4,768x6,1024x4 --epochs 1000 --patience 8 --out_dir runs_model
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from diffik.config import Config, DataConfig, DiffusionConfig, EvalConfig, ModelConfig, TrainConfig
from train import run


def already_done(out_dir, name):
    reg = os.path.join(out_dir, "experiments.csv")
    return os.path.exists(reg) and name in set(pd.read_csv(reg)["name"].astype(str))


def make_cfg(regime, n, hidden, n_layers, epochs, patience, monitor_every, T, out_dir):
    name = f"{regime}_n{n}_h{hidden}_l{n_layers}"
    if regime == "lbe":
        data = DataConfig(kind="trajectory", lbe=True, n_trajectories=n, steps_per_traj=40, v_deg=1.0)
        model = ModelConfig(type="lbe", hidden_dim=hidden, n_layers=n_layers)
        evalc = EvalConfig(n_per_pose=1, sampler="ddpm", seeded=True)
    else:
        data = DataConfig(kind="random", lbe=False, n_samples=n)
        model = ModelConfig(type="mlp", hidden_dim=hidden, n_layers=n_layers)
        evalc = EvalConfig(n_per_pose=10, sampler="ddpm")
    return Config(
        name=name, out_dir=out_dir, data=data, model=model,
        diffusion=DiffusionConfig(T=T, fk_loss_weight=10.0, rot_weight=0.1, p_example_dropout=0.3),
        train=TrainConfig(epochs=epochs, batch_size=256, lr=1e-3, checkpoint_every=max(epochs // 4, 1),
                          patience=patience, early_stop_metric="val_pose", monitor_every=monitor_every),
        eval=evalc,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["lbe", "seedless"], default="lbe")
    ap.add_argument("--n", type=int, required=True, help="dataset size: n_trajectories (lbe) or n_samples (seedless)")
    ap.add_argument("--configs", default="128x2,256x3,512x4,768x6,1024x4,1280x6", help="comma list of hiddenXlayers")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--monitor_every", type=int, default=10)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--out_dir", default="runs_model")
    args = ap.parse_args()

    specs = [(int(h), int(l)) for h, l in (c.split("x") for c in args.configs.split(","))]
    print(f"model-size sweep regime={args.regime} n={args.n} specs={specs}")
    for hidden, n_layers in specs:
        cfg = make_cfg(args.regime, args.n, hidden, n_layers, args.epochs, args.patience,
                       args.monitor_every, args.T, args.out_dir)
        if already_done(args.out_dir, cfg.name):
            print(f"[skip] {cfg.name}")
            continue
        print(f"\n========== {cfg.name} ==========")
        run(cfg)
    print(f"\nmodel-size sweep complete. Registry: {os.path.join(args.out_dir, 'experiments.csv')}")


if __name__ == "__main__":
    main()
