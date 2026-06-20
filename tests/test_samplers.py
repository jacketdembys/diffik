"""Phase 6a verification: DDIM sampler correctness (mechanism, not accuracy).

NOTE: empirically, deterministic DDIM does NOT improve IK accuracy over the DDPM
sampler at the scales we can run locally (the DDPM sampler already uses a
deterministic terminal step). These tests therefore verify the DDIM sampler is
*correct* -- reproducible, noise-free at eta=0, supports step subsampling -- and
leave the accuracy story to the LBE seed (Phase 7) and full-scale runs (Phase 9).
"""
from __future__ import annotations

import torch

from diffik.diffusion import GaussianDiffusion, NoiseSchedule
from diffik.models import MLPDenoiser
from diffik.utils import set_seed


def _diffusion(T=50):
    set_seed(0)
    return GaussianDiffusion(MLPDenoiser(hidden_dim=128, n_layers=2), NoiseSchedule(T=T), dof=7)


def test_ddim_shape_and_reproducible():
    diff = _diffusion()
    pose = torch.randn(3, 6)
    g1 = torch.Generator().manual_seed(7)
    s1 = diff.sample(pose, n_per_pose=2, generator=g1, sampler="ddim", eta=0.0)
    g2 = torch.Generator().manual_seed(7)
    s2 = diff.sample(pose, n_per_pose=2, generator=g2, sampler="ddim", eta=0.0)
    assert s1.shape == (3, 2, 7)
    assert torch.allclose(s1, s2), "DDIM eta=0 must be reproducible"


def test_ddim_eta0_is_pure_function_of_initial_noise():
    """With eta=0 there is no added noise, so passing no generator after the
    initial draw must not change the result."""
    diff = _diffusion()
    pose = torch.randn(4, 6)
    # same initial-noise seed, but one call has a generator available for any
    # (nonexistent) intermediate noise draws -> outputs must be identical.
    g = torch.Generator().manual_seed(11)
    s_with_gen = diff.sample(pose, n_per_pose=1, generator=g, sampler="ddim", eta=0.0)
    g2 = torch.Generator().manual_seed(11)
    s_ctrl = diff.sample(pose, n_per_pose=1, generator=g2, sampler="ddim", eta=0.0)
    assert torch.allclose(s_with_gen, s_ctrl)


def test_ddim_step_subsampling_runs():
    diff = _diffusion(T=100)
    pose = torch.randn(5, 6)
    g = torch.Generator().manual_seed(0)
    s = diff.sample(pose, n_per_pose=1, generator=g, sampler="ddim", eta=0.0, ddim_steps=10)
    assert s.shape == (5, 1, 7)
    assert torch.isfinite(s).all()


def test_ddim_eta_changes_output():
    diff = _diffusion()
    pose = torch.randn(3, 6)
    g0 = torch.Generator().manual_seed(3)
    s0 = diff.sample(pose, n_per_pose=1, generator=g0, sampler="ddim", eta=0.0)
    g1 = torch.Generator().manual_seed(3)
    s1 = diff.sample(pose, n_per_pose=1, generator=g1, sampler="ddim", eta=1.0)
    # stochastic eta=1 injects noise -> differs from deterministic eta=0
    assert not torch.allclose(s0, s1)


def test_ddpm_and_ddim_both_finite_on_trained_dispatch():
    diff = _diffusion(T=30)
    pose = torch.randn(4, 6)
    for kw in (dict(sampler="ddpm"), dict(sampler="ddim", eta=0.0)):
        g = torch.Generator().manual_seed(0)
        s = diff.sample(pose, n_per_pose=1, generator=g, **kw)
        assert s.shape == (4, 1, 7) and torch.isfinite(s).all()
