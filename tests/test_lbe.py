"""Phase 7 verification: Learning-by-Example diffusion with classifier-free guidance.

Covers example-pair correctness, the LBE denoiser / CFG-dropout mechanism, and the
decisive result: the seed (example) substantially improves held-out accuracy over
the seedless regime.
"""
from __future__ import annotations

import numpy as np
import torch

from diffik.data import (
    add_examples,
    build_datasets_lbe,
    generate_random,
    generate_trajectory,
)
from diffik.diffusion import LBEDiffusion, NoiseSchedule
from diffik.eval import evaluate
from diffik.kinematics import forward_kinematics, get_robot, pose_error
from diffik.models import LBEDenoiser
from diffik.training import train_diffusion
from diffik.utils import set_seed

DT = torch.float64


def _fk_pose_consistency(ex_q, ex_pose, robot):
    chain = get_robot(robot, dtype=DT)
    from tests.test_data import _matrix_from_pose

    T_fk = forward_kinematics(torch.as_tensor(ex_q, dtype=DT), chain)
    T_st = _matrix_from_pose(torch.as_tensor(ex_pose, dtype=DT), "xyzrpy")
    return pose_error(T_fk, T_st)


def test_trajectory_example_is_previous_waypoint():
    ds = generate_trajectory(n_trajectories=5, steps_per_traj=10, seed=0)
    ex_q, ex_pose = add_examples(ds, v_deg=1.0, seed=0)
    # for step>0, example == previous sample; for step 0, example == itself
    for i in range(len(ds)):
        j = i - 1 if ds.step[i] > 0 else i
        assert np.allclose(ex_q[i], ds.q[j]) and np.allclose(ex_pose[i], ds.pose[j])


def test_example_fk_consistency_both_datasets():
    rnd = generate_random(n_samples=1000, seed=0)
    traj = generate_trajectory(n_trajectories=30, steps_per_traj=30, seed=0)
    for ds in (rnd, traj):
        ex_q, ex_pose = add_examples(ds, v_deg=1.0, seed=0)
        pos_mm, ori_deg = _fk_pose_consistency(ex_q, ex_pose, ds.robot)
        assert float(pos_mm.max()) < 1e-6 and float(ori_deg.max()) < 1e-3, ds.kind


def test_random_example_within_v():
    ds = generate_random(n_samples=2000, seed=0)
    ex_q, _ = add_examples(ds, v_deg=1.0, seed=0)
    # synthesized example perturbs each joint by at most 1 deg (revolute)
    delta = np.abs(ex_q - ds.q)
    assert delta.max() <= np.deg2rad(1.0) + 1e-6


def test_lbe_denoiser_runs_seeded_and_seedless():
    set_seed(0)
    model = LBEDenoiser(dof=7, pose_dim=6, hidden_dim=128, n_layers=2)
    x_t = torch.randn(8, 7)
    pose = torch.randn(8, 6)
    t = torch.randint(0, 100, (8,))
    example = torch.randn(8, 6 + 7)
    out_seeded = model(x_t, pose, t, example=example)
    out_seedless = model(x_t, pose, t, example=None)
    assert out_seeded.shape == (8, 7) and out_seedless.shape == (8, 7)
    # dropping all examples must equal the seedless (null) path
    drop_all = torch.ones(8, dtype=torch.bool)
    out_dropped = model(x_t, pose, t, example=example, drop_mask=drop_all)
    assert torch.allclose(out_dropped, out_seedless, atol=1e-6)


def test_lbe_loss_with_example_and_fk_backprops():
    set_seed(0)
    ds = generate_trajectory(n_trajectories=10, steps_per_traj=10, seed=0)
    train, _, _, qn, _ = build_datasets_lbe(ds, seed=0)
    diff = LBEDiffusion(
        LBEDenoiser(hidden_dim=256, n_layers=3), NoiseSchedule(T=50), dof=7,
        chain=get_robot("panda_7r"), q_norm=qn, fk_loss_weight=10.0, p_example_dropout=0.3,
    )
    b = train[0]
    x0 = b["q"].unsqueeze(0)
    pose = b["pose"].unsqueeze(0)
    example = b["example"].unsqueeze(0)
    total, info = diff.loss(x0, pose, example)
    assert info["fk"] > 0 and info["denoise"] > 0
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in diff.model.parameters())


def test_seed_beats_seedless_decisive(capsys):
    """The core Phase 7 claim: the example seed substantially lowers held-out error."""
    set_seed(0)
    ds = generate_trajectory(n_trajectories=100, steps_per_traj=40, v_deg=1.0, seed=0)
    train, _, test, qn, _ = build_datasets_lbe(ds, v_deg=1.0, seed=0)
    diff = LBEDiffusion(
        LBEDenoiser(hidden_dim=512, n_layers=4), NoiseSchedule(T=100), dof=7,
        chain=get_robot("panda_7r"), q_norm=qn, fk_loss_weight=10.0, rot_weight=0.1,
        p_example_dropout=0.3,
    )
    train_diffusion(diff, train, epochs=150, batch_size=256, lr=1e-3, device="cpu")

    g = torch.Generator().manual_seed(0)
    seedless = evaluate(diff, test, qn, n_per_pose=1, device="cpu", generator=g)
    g = torch.Generator().manual_seed(0)
    seeded = evaluate(diff, test, qn, n_per_pose=1, device="cpu", generator=g, example=test.example)

    with capsys.disabled():
        print(
            f"\n  [LBE decisive, held-out]"
            f"\n    seedless: pos {seedless.mean.pos_mm_avg:.1f} mm, ori {seedless.mean.ori_deg_avg:.1f} deg"
            f"\n    seeded  : pos {seeded.mean.pos_mm_avg:.1f} mm, ori {seeded.mean.ori_deg_avg:.1f} deg"
        )
    # the seed should clearly beat seedless on both position and orientation
    assert seeded.mean.pos_mm_avg < 0.6 * seedless.mean.pos_mm_avg
    assert seeded.mean.ori_deg_avg < 0.6 * seedless.mean.ori_deg_avg
