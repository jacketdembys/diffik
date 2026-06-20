"""Vanilla-DDPM baseline on the two datasets (random vs trajectory), held-out.

Same model/config for both so the comparison is apples-to-apples. This is a
small CPU-scale baseline (not the full ~1M-sample training of Phase 9); its
purpose is to establish the pre-FK-loss reference on each dataset separately.
"""
from __future__ import annotations

import torch

from diffik.data import build_datasets, generate_random, generate_trajectory
from diffik.diffusion import GaussianDiffusion, NoiseSchedule
from diffik.eval import evaluate
from diffik.models import MLPDenoiser
from diffik.training import train_diffusion
from diffik.utils import set_seed

CONFIG = dict(n_samples=4000, T=100, hidden=512, n_layers=4, epochs=150, batch=256, lr=1e-3)


def run(ds, name):
    set_seed(0)
    train, val, test, q_norm, _ = build_datasets(ds, seed=0)
    diffusion = GaussianDiffusion(
        MLPDenoiser(hidden_dim=CONFIG["hidden"], n_layers=CONFIG["n_layers"]),
        NoiseSchedule(T=CONFIG["T"]),
        dof=7,
    )
    train_diffusion(diffusion, train, epochs=CONFIG["epochs"], batch_size=CONFIG["batch"],
                    lr=CONFIG["lr"], device="cpu")
    g = torch.Generator().manual_seed(0)
    res = evaluate(diffusion, test, q_norm, n_per_pose=1, device="cpu", generator=g)
    print(f"\n=== {name} dataset (held-out test, {res.mean.n} poses) ===")
    print(f"  {res.mean}")
    return res


if __name__ == "__main__":
    print(f"config: {CONFIG}")
    rnd = generate_random(n_samples=CONFIG["n_samples"], seed=0)
    # ~4000 samples as 100 trajectories x 40 steps, v=1 deg
    traj = generate_trajectory(n_trajectories=100, steps_per_traj=40, v_deg=1.0, seed=0)

    r_rnd = run(rnd, "RANDOM")
    r_traj = run(traj, "TRAJECTORY")

    print("\n--- summary (vanilla DDPM, pre-FK-loss) ---")
    print(f"  RANDOM     : pos {r_rnd.mean.pos_mm_avg:7.1f} mm | ori {r_rnd.mean.ori_deg_avg:6.1f} deg | <=1mm {r_rnd.mean.pct_pos_le_1mm:.1f}%")
    print(f"  TRAJECTORY : pos {r_traj.mean.pos_mm_avg:7.1f} mm | ori {r_traj.mean.ori_deg_avg:6.1f} deg | <=1mm {r_traj.mean.pct_pos_le_1mm:.1f}%")
