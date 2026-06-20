"""Phase: multimodality evaluation metric checks."""
from __future__ import annotations

import torch

from diffik.data import build_datasets, generate_random
from diffik.diffusion import GaussianDiffusion, NoiseSchedule
from diffik.eval import evaluate_multimodality
from diffik.eval.multimodality import _valid_diversity
from diffik.models import MLPDenoiser
from diffik.utils import set_seed


def test_valid_diversity_helper():
    # pose 0: two valid, distinct -> positive diversity; pose 1: one valid -> ignored
    q = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [9.0, 9.0]],
                      [[0.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    valid = torch.tensor([[True, True, False], [True, False, False]])
    vdiv, n = _valid_diversity(q, valid)
    assert n == 1 and abs(vdiv - 1.0) < 1e-6  # only pose 0 counts; dist([0,0],[1,0])=1


def test_multimodality_runs_and_reports_structure():
    set_seed(0)
    ds = generate_random(n_samples=64, seed=0)
    _, _, test, q_norm, _ = build_datasets(ds, seed=0)
    diff = GaussianDiffusion(MLPDenoiser(hidden_dim=128, n_layers=2), NoiseSchedule(T=20), dof=7)
    g = torch.Generator().manual_seed(0)
    res = evaluate_multimodality(diff, test.head(16), q_norm, K=8, device="cpu",
                                 generator=g, tol_mm=1e9, tol_deg=1e9)  # all valid
    # untrained stochastic model -> diverse samples; with infinite tol all are valid
    assert res.diversity_all > 0.0
    assert res.mean_valid_per_pose == 8.0          # all K valid under huge tol
    assert res.frac_poses_multi == 1.0
    assert res.valid_diversity > 0.0
    assert res.best_of_k.pos_mm_avg <= res.best_of_k.pos_mm_max + 1e-6
