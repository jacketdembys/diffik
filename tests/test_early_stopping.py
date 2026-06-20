"""Verify validation-based early stopping + best-checkpoint restore."""
from __future__ import annotations

import torch

from diffik.data import build_datasets, generate_random
from diffik.diffusion import GaussianDiffusion, NoiseSchedule
from diffik.models import MLPDenoiser
from diffik.training import compute_val_loss, train_diffusion
from diffik.utils import set_seed


def _make(T=50):
    return GaussianDiffusion(MLPDenoiser(hidden_dim=128, n_layers=2), NoiseSchedule(T=T), dof=7)


def test_val_loss_is_deterministic():
    """Fixed-seed val loss must be identical across calls (stable ES signal)."""
    set_seed(0)
    ds = generate_random(n_samples=200, seed=0)
    _, val, _, _, _ = build_datasets(ds, seed=0)
    from torch.utils.data import DataLoader
    diff = _make()
    loader = DataLoader(val, batch_size=64, shuffle=False)
    a = compute_val_loss(diff, loader, "cpu")
    b = compute_val_loss(diff, loader, "cpu")
    assert abs(a - b) < 1e-9


def test_history_has_val_and_records_epochs():
    set_seed(0)
    ds = generate_random(n_samples=400, seed=0)
    train, val, _, _, _ = build_datasets(ds, seed=0)
    diff = _make()
    hist = train_diffusion(diff, train, val_dataset=val, epochs=6, batch_size=128, device="cpu", patience=0)
    assert len(hist) == 6
    assert all("val" in h for h in hist) and all("total" in h for h in hist)


def test_early_stopping_triggers():
    """Frozen model (lr=0) -> constant val loss -> no improvement -> stop at patience."""
    set_seed(0)
    ds = generate_random(n_samples=600, seed=0)
    train, val, _, _, _ = build_datasets(ds, seed=0)
    diff = _make()
    hist = train_diffusion(diff, train, val_dataset=val, epochs=50, batch_size=128,
                           lr=0.0, device="cpu", patience=3)
    assert len(hist) < 50, "early stopping should halt before max epochs"
    assert len(hist) <= 6, "with a frozen model it should stop ~patience epochs in"


def test_best_checkpoint_restored():
    """Overfit a tiny train set; the restored weights must be the best-val ones
    (not the last epoch's), even with early stopping disabled."""
    set_seed(0)
    ds = generate_random(n_samples=64, seed=0)
    train, val, _, _, _ = build_datasets(ds, seed=0)
    diff = _make()
    hist = train_diffusion(diff, train, val_dataset=val, epochs=120, batch_size=64,
                           lr=2e-3, device="cpu", patience=0)
    from torch.utils.data import DataLoader
    loader = DataLoader(val, batch_size=256, shuffle=False)
    restored_val = compute_val_loss(diff, loader, "cpu")
    best_val = min(h["val"] for h in hist if "val" in h)
    last_val = hist[-1]["val"]
    assert abs(restored_val - best_val) < 1e-6, "best-val weights must be restored"
    assert restored_val <= last_val + 1e-6, "restored (best) should be <= last epoch's val"
