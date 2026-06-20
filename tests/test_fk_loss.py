"""Phase 5 verification: differentiable-FK loss.

Unit checks (FK loss is zero at a perfect prediction, produces gradients, and is
wired into the loss dict) plus a quick single-dataset ablation showing the FK
loss reduces held-out position error vs the no-FK baseline. The full both-dataset
ablation lives in scripts/ablation_fk.py.
"""
from __future__ import annotations

import torch

from diffik.data import build_datasets, generate_random
from diffik.diffusion import GaussianDiffusion, NoiseSchedule
from diffik.eval import evaluate
from diffik.kinematics import get_robot
from diffik.models import MLPDenoiser
from diffik.training import train_diffusion
from diffik.utils import set_seed


def _make(fk_w, q_norm, rot_w=0.1, T=100):
    return GaussianDiffusion(
        MLPDenoiser(hidden_dim=512, n_layers=4),
        NoiseSchedule(T=T),
        dof=7,
        chain=get_robot("panda_7r"),
        q_norm=q_norm,
        fk_loss_weight=fk_w,
        rot_weight=rot_w,
    )


def test_fk_loss_zero_at_perfect_prediction():
    set_seed(0)
    ds = generate_random(n_samples=64, seed=0)
    _, _, _, q_norm, _ = build_datasets(ds, fractions=(1.0, 0.0, 0.0), seed=0)
    diff = _make(10.0, q_norm)
    x0 = torch.randn(32, 7)
    t = torch.randint(0, diff.T, (32,))
    # perfect estimate: x0_hat == x0  ->  FK(x0_hat) == FK(x0)  ->  loss 0
    fk = diff._fk_loss(x0, x0, t)
    assert float(fk) < 1e-10


def test_fk_loss_positive_and_differentiable():
    set_seed(0)
    ds = generate_random(n_samples=64, seed=0)
    _, _, _, q_norm, _ = build_datasets(ds, fractions=(1.0, 0.0, 0.0), seed=0)
    diff = _make(10.0, q_norm)
    x0 = torch.randn(16, 7)
    pose = torch.randn(16, 6)
    total, info = diff.loss(x0, pose)
    assert info["fk"] > 0.0 and info["denoise"] > 0.0
    total.backward()
    grads = [p.grad for p in diff.model.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads)


def test_fk_loss_disabled_by_default():
    set_seed(0)
    diff = GaussianDiffusion(MLPDenoiser(), NoiseSchedule(T=50), dof=7)
    assert not diff.use_fk_loss
    _, info = diff.loss(torch.randn(8, 7), torch.randn(8, 6))
    assert info["fk"] == 0.0


def test_fk_loss_improves_heldout_position(capsys):
    """Quick ablation: FK loss lowers held-out position error vs no-FK baseline."""
    ds = generate_random(n_samples=3000, seed=0)

    set_seed(0)
    train, _, test, q_norm, _ = build_datasets(ds, seed=0)
    base = _make(0.0, q_norm)
    train_diffusion(base, train, epochs=120, batch_size=256, lr=1e-3, device="cpu")
    g = torch.Generator().manual_seed(0)
    r_base = evaluate(base, test, q_norm, n_per_pose=1, device="cpu", generator=g)

    set_seed(0)
    train, _, test, q_norm, _ = build_datasets(ds, seed=0)
    fk = _make(10.0, q_norm, rot_w=0.1)
    train_diffusion(fk, train, epochs=120, batch_size=256, lr=1e-3, device="cpu")
    g = torch.Generator().manual_seed(0)
    r_fk = evaluate(fk, test, q_norm, n_per_pose=1, device="cpu", generator=g)

    with capsys.disabled():
        print(
            f"\n  [FK ablation, held-out]"
            f"\n    no-FK : pos {r_base.mean.pos_mm_avg:.1f} mm, ori {r_base.mean.ori_deg_avg:.1f} deg"
            f"\n    +FK   : pos {r_fk.mean.pos_mm_avg:.1f} mm, ori {r_fk.mean.ori_deg_avg:.1f} deg"
        )
    # FK loss clearly improves position; should not worsen orientation materially
    assert r_fk.mean.pos_mm_avg < 0.85 * r_base.mean.pos_mm_avg
    assert r_fk.mean.ori_deg_avg < 1.1 * r_base.mean.ori_deg_avg
