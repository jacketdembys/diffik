"""Phase 3 verification: minimal conditional diffusion baseline.

Checks: forward-noising sanity (x_t ~ x0 at t=0; x_t ~ N(0,1) at t=T-1),
overfitting a small set drives the denoising loss down and yields sampled joints
whose FK is far better than random joints, and sampling is shape-correct and
seed-deterministic.
"""
from __future__ import annotations

import torch

from diffik.data import build_datasets, generate_random
from diffik.diffusion import GaussianDiffusion, NoiseSchedule
from diffik.kinematics import forward_kinematics, get_robot, pose_error
from diffik.models import MLPDenoiser
from diffik.training import train_diffusion
from diffik.utils import set_seed


def _make_diffusion(T=200, hidden=256, n_layers=3):
    schedule = NoiseSchedule(T=T)
    model = MLPDenoiser(dof=7, pose_dim=6, hidden_dim=hidden, n_layers=n_layers)
    return GaussianDiffusion(model, schedule, dof=7)


def test_forward_noising_sanity():
    set_seed(0)
    schedule = NoiseSchedule(T=1000)
    x0 = torch.randn(4096, 7)  # ~unit-scale normalized joints

    # t = 0: barely any noise -> x_t ~ x0
    t0 = torch.zeros(4096, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t0 = schedule.q_sample(x0, t0, noise)
    resid_std = (x_t0 - x0).std().item()
    assert resid_std < 0.05, f"residual std at t=0 too large: {resid_std}"

    # t = T-1: almost pure noise -> x_t ~ N(0, 1)
    tT = torch.full((4096,), schedule.T - 1, dtype=torch.long)
    x_tT = schedule.q_sample(x0, tT, torch.randn_like(x0))
    assert abs(x_tT.mean().item()) < 0.05
    assert abs(x_tT.std().item() - 1.0) < 0.05


def test_predict_x0_inverts_q_sample():
    schedule = NoiseSchedule(T=500)
    x0 = torch.randn(256, 7)
    t = torch.randint(0, 500, (256,))
    noise = torch.randn_like(x0)
    x_t = schedule.q_sample(x0, t, noise)
    x0_rec = schedule.predict_x0_from_eps(x_t, t, noise)
    assert torch.allclose(x0, x0_rec, atol=1e-4)


def test_overfit_small_set_drives_loss_down_and_fk_better_than_random(capsys):
    set_seed(0)
    ds = generate_random(n_samples=32, seed=0)
    train, _, _, q_norm, pose_norm = build_datasets(ds, fractions=(1.0, 0.0, 0.0), seed=0)

    diffusion = _make_diffusion(T=100, hidden=256, n_layers=3)
    history = train_diffusion(diffusion, train, epochs=800, batch_size=32, lr=1e-3, device="cpu")

    losses = [h["total"] for h in history]
    init_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    assert final_loss < 0.4 * init_loss, f"loss did not drop enough: {init_loss:.3f}->{final_loss:.3f}"

    # sample for the training poses; FK error should beat random joints
    set_seed(1)
    poses_n = train.pose
    g = torch.Generator().manual_seed(0)
    samples_n = diffusion.sample(poses_n, n_per_pose=1, generator=g)[:, 0, :]
    q_pred = q_norm.inverse_transform(samples_n)
    q_true = q_norm.inverse_transform(train.q)

    chain = get_robot(ds.robot, dtype=torch.float32)
    T_pred = forward_kinematics(q_pred, chain)
    T_true = forward_kinematics(q_true, chain)
    pos_pred, ori_pred = pose_error(T_pred, T_true)

    # random-joint baseline
    q_rand = q_norm.inverse_transform(torch.randn_like(samples_n))
    pos_rand, ori_rand = pose_error(forward_kinematics(q_rand, chain), T_true)

    n_samples = len(train)
    with capsys.disabled():
        print(
            f"\n  [overfit {n_samples} poses, 1 sample/pose] loss {init_loss:.3f}->{final_loss:.3f}"
            f"\n    sampled: pos {pos_pred.mean():.1f} mm,  ori {ori_pred.mean():.2f} deg"
            f"\n    random : pos {pos_rand.mean():.1f} mm,  ori {ori_rand.mean():.2f} deg"
        )
    # both position and orientation should clearly beat random-joint baseline
    assert pos_pred.mean() < 0.5 * pos_rand.mean()
    assert ori_pred.mean() < 0.7 * ori_rand.mean()


def test_sampling_shape_and_determinism():
    set_seed(0)
    diffusion = _make_diffusion(T=50)
    pose = torch.randn(3, 6)

    g1 = torch.Generator().manual_seed(123)
    s1 = diffusion.sample(pose, n_per_pose=4, generator=g1)
    g2 = torch.Generator().manual_seed(123)
    s2 = diffusion.sample(pose, n_per_pose=4, generator=g2)

    assert s1.shape == (3, 4, 7)
    assert torch.allclose(s1, s2), "sampling is not seed-deterministic"

    g3 = torch.Generator().manual_seed(999)
    s3 = diffusion.sample(pose, n_per_pose=4, generator=g3)
    assert not torch.allclose(s1, s3), "different seeds should give different samples"
