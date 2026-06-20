"""Phase 5 ablation: no-FK vs +FK loss, on both datasets, held-out.

Same model/config throughout; the only change is the FK loss. Small CPU scale
(not Phase 9's full training) -- purpose is to show the FK-loss effect per dataset.
"""
from __future__ import annotations

import torch

from diffik.data import build_datasets, generate_random, generate_trajectory
from diffik.diffusion import GaussianDiffusion, NoiseSchedule
from diffik.eval import evaluate
from diffik.kinematics import get_robot
from diffik.models import MLPDenoiser
from diffik.training import train_diffusion
from diffik.utils import set_seed

CFG = dict(T=100, hidden=512, n_layers=4, epochs=150, batch=256, lr=1e-3, fk_w=10.0, rot_w=0.1)


def run(ds, fk_on):
    set_seed(0)
    train, _, test, q_norm, _ = build_datasets(ds, seed=0)
    diff = GaussianDiffusion(
        MLPDenoiser(hidden_dim=CFG["hidden"], n_layers=CFG["n_layers"]),
        NoiseSchedule(T=CFG["T"]), dof=7,
        chain=get_robot("panda_7r"), q_norm=q_norm,
        fk_loss_weight=CFG["fk_w"] if fk_on else 0.0, rot_weight=CFG["rot_w"],
    )
    train_diffusion(diff, train, epochs=CFG["epochs"], batch_size=CFG["batch"], lr=CFG["lr"], device="cpu")
    g = torch.Generator().manual_seed(0)
    return evaluate(diff, test, q_norm, n_per_pose=1, device="cpu", generator=g).mean


if __name__ == "__main__":
    print(f"config: {CFG}")
    datasets = {
        "RANDOM": generate_random(n_samples=4000, seed=0),
        "TRAJECTORY": generate_trajectory(n_trajectories=100, steps_per_traj=40, v_deg=1.0, seed=0),
    }
    print(f"\n{'dataset':<12} {'variant':<8} {'pos(mm)':>9} {'ori(deg)':>9} {'<=1mm':>7}")
    for name, ds in datasets.items():
        for fk_on in (False, True):
            s = run(ds, fk_on)
            print(f"{name:<12} {'+FK' if fk_on else 'no-FK':<8} "
                  f"{s.pos_mm_avg:>9.1f} {s.ori_deg_avg:>9.1f} {s.pct_pos_le_1mm:>6.1f}%")
