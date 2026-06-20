"""Phase 4 verification: evaluation harness & metrics.

Includes a generalization smoke test that trains on a few thousand samples and
evaluates on a *held-out* split (honest, representative measurement -- still
vanilla-DDPM-weak, but on unseen poses), unlike the Phase 3 overfit sanity test.
"""
from __future__ import annotations

import torch

from diffik.data import build_datasets, generate_random
from diffik.diffusion import GaussianDiffusion, NoiseSchedule
from diffik.eval import diversity, evaluate, summarize_errors
from diffik.models import MLPDenoiser
from diffik.training import train_diffusion
from diffik.utils import set_seed


def test_summarize_errors_handcomputed():
    pos = torch.tensor([0.5, 1.0, 2.0, 4.0])  # mm
    ori = torch.tensor([0.2, 0.8, 1.0, 5.0])  # deg
    s = summarize_errors(pos, ori)
    assert s.n == 4
    assert abs(s.pos_mm_avg - 1.875) < 1e-6
    assert s.pos_mm_min == 0.5 and s.pos_mm_max == 4.0
    # <=1mm: {0.5, 1.0} -> 50% ; <=1deg: {0.2,0.8,1.0} -> 75%
    assert abs(s.pct_pos_le_1mm - 50.0) < 1e-6
    assert abs(s.pct_ori_le_1deg - 75.0) < 1e-6


def test_diversity_zero_when_identical_else_positive():
    base = torch.randn(5, 1, 7)
    identical = base.repeat(1, 4, 1)
    assert diversity(identical) < 1e-7
    varied = torch.randn(5, 4, 7)
    assert diversity(varied) > 0.1
    assert diversity(torch.randn(5, 1, 7)) == 0.0  # N=1


def test_perfect_solver_gives_zero_error():
    """A 'diffusion' that returns the ground-truth joints must score ~0 error."""
    set_seed(0)
    ds = generate_random(n_samples=64, seed=0)
    train, _, test, q_norm, pose_norm = build_datasets(ds, seed=0)

    class PerfectDiffusion:
        dof = 7

        def __init__(self, dataset):
            self._q = dataset.q  # normalized ground-truth joints

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def sample(self, pose, n_per_pose=1, generator=None):
            P = pose.shape[0]
            return self._q.unsqueeze(1).repeat(1, n_per_pose, 1)

    res = evaluate(PerfectDiffusion(test), test, q_norm, n_per_pose=1, device="cpu")
    assert res.mean.pos_mm_avg < 1e-3
    assert res.mean.ori_deg_avg < 1e-3
    assert res.mean.pct_pos_le_1mm == 100.0
    assert res.mean.pct_ori_le_1deg == 100.0


def test_best_of_n_not_worse_than_mean():
    set_seed(0)
    ds = generate_random(n_samples=64, seed=1)
    _, _, test, q_norm, _ = build_datasets(ds, seed=1)
    diffusion = GaussianDiffusion(MLPDenoiser(hidden_dim=128, n_layers=2), NoiseSchedule(T=30), dof=7)
    g = torch.Generator().manual_seed(0)
    res = evaluate(diffusion, test, q_norm, n_per_pose=8, device="cpu", generator=g)
    # best-of-N selects the lowest position error per pose -> <= mean
    assert res.best_of_n.pos_mm_avg <= res.mean.pos_mm_avg + 1e-6
    assert res.diversity > 0.0  # untrained model still produces varied samples


def test_generalization_smoke_heldout(capsys):
    """Train on a few thousand samples; evaluate on a held-out test split."""
    set_seed(0)
    ds = generate_random(n_samples=4000, seed=0)
    train, val, test, q_norm, pose_norm = build_datasets(ds, seed=0)

    diffusion = GaussianDiffusion(MLPDenoiser(hidden_dim=512, n_layers=4), NoiseSchedule(T=100), dof=7)
    train_diffusion(diffusion, train, epochs=150, batch_size=256, lr=1e-3, device="cpu")

    g = torch.Generator().manual_seed(0)
    res = evaluate(diffusion, test, q_norm, n_per_pose=1, device="cpu", generator=g)

    # random-joint baseline on the same held-out targets
    class RandomSolver:
        dof = 7

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def sample(self, pose, n_per_pose=1, generator=None):
            return torch.randn(pose.shape[0], n_per_pose, 7, generator=generator)

    gr = torch.Generator().manual_seed(0)
    rnd = evaluate(RandomSolver(), test, q_norm, n_per_pose=1, device="cpu", generator=gr)

    with capsys.disabled():
        print(f"\n  [held-out test, {res.mean.n} poses]")
        print(f"    DDPM baseline : {res.mean}")
        print(f"    random joints : {rnd.mean}")

    # honest expectation: vanilla DDPM generalizes clearly better than random,
    # but is NOT yet sub-mm (that's Phases 5-6).
    assert res.mean.pos_mm_avg < 0.6 * rnd.mean.pos_mm_avg
    assert res.mean.pos_mm_avg > 1.0  # still well above sub-mm at this stage
    assert res.mean.pct_pos_le_1mm < 5.0  # essentially no sub-mm hits yet
