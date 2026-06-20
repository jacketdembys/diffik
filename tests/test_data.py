"""Phase 2 verification: dataset generation, loading, normalization.

The critical check is FK(q) == pose for every generated sample (catches any
generation bug), plus joint-limit compliance, leakage-safe splits, normalizer
round-trip, and that the two generators stay structurally distinct.
"""
from __future__ import annotations

import numpy as np
import torch

from diffik.data import (
    Normalizer,
    build_datasets,
    generate_random,
    generate_trajectory,
    get_joint_limits,
    load_dataset,
    pose_from_matrix,
    save_dataset,
    split_indices,
)
from diffik.kinematics import forward_kinematics, get_robot, pose_error

DTYPE = torch.float64


def _fk_consistency_mm_deg(ds):
    """Return (max position err mm, max orientation err deg) of FK(q) vs stored pose."""
    chain = get_robot(ds.robot, dtype=DTYPE)
    q = torch.as_tensor(ds.q, dtype=DTYPE)
    T_fk = forward_kinematics(q, chain)
    T_stored = _matrix_from_pose(torch.as_tensor(ds.pose, dtype=DTYPE), ds.pose_repr)
    pos_mm, ori_deg = pose_error(T_fk, T_stored)
    return float(pos_mm.max()), float(ori_deg.max())


def _matrix_from_pose(pose: torch.Tensor, repr: str) -> torch.Tensor:
    """Rebuild SE(3) from an xyzrpy pose vector (inverse of pose_from_matrix)."""
    assert repr == "xyzrpy"
    B = pose.shape[0]
    x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
    roll, pitch, yaw = pose[:, 3], pose[:, 4], pose[:, 5]
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    R = torch.zeros(B, 3, 3, dtype=pose.dtype)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    T = torch.eye(4, dtype=pose.dtype).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, 0, 3], T[:, 1, 3], T[:, 2, 3] = x, y, z
    return T


def test_random_fk_consistency(capsys):
    ds = generate_random(n_samples=2000, seed=0)
    pos_mm, ori_deg = _fk_consistency_mm_deg(ds)
    with capsys.disabled():
        print(f"\n  [random] FK(q) vs stored pose: pos<={pos_mm:.2e} mm, ori<={ori_deg:.2e} deg")
    assert pos_mm < 1e-6 and ori_deg < 1e-3


def test_trajectory_fk_consistency(capsys):
    ds = generate_trajectory(n_trajectories=40, steps_per_traj=50, v_deg=1.0, seed=0)
    pos_mm, ori_deg = _fk_consistency_mm_deg(ds)
    with capsys.disabled():
        print(f"\n  [trajectory] FK(q) vs stored pose: pos<={pos_mm:.2e} mm, ori<={ori_deg:.2e} deg")
    assert pos_mm < 1e-6 and ori_deg < 1e-3


def test_joint_limits_respected():
    limits = get_joint_limits("panda_7r")
    lo, hi = limits[:, 0].numpy(), limits[:, 1].numpy()
    for ds in (generate_random(n_samples=2000, seed=1),
               generate_trajectory(n_trajectories=40, steps_per_traj=50, seed=1)):
        assert (ds.q >= lo - 1e-9).all() and (ds.q <= hi + 1e-9).all(), ds.kind


def test_generators_are_distinct():
    rnd = generate_random(n_samples=100, seed=2)
    traj = generate_trajectory(n_trajectories=10, steps_per_traj=10, seed=2)
    assert rnd.kind == "random" and rnd.traj_id is None
    assert traj.kind == "trajectory" and traj.traj_id is not None
    # consecutive trajectory steps are close; random pairs are not
    traj_step_gap = np.linalg.norm(np.diff(traj.q[:10], axis=0), axis=1).mean()
    rnd_gap = np.linalg.norm(np.diff(rnd.q[:10], axis=0), axis=1).mean()
    assert traj_step_gap < rnd_gap


def test_normalizer_roundtrip():
    ds = generate_random(n_samples=500, seed=3)
    x = torch.as_tensor(ds.q, dtype=torch.float32)
    norm = Normalizer.fit(x)
    x_rt = norm.inverse_transform(norm.transform(x))
    assert torch.allclose(x, x_rt, atol=1e-5)
    # normalized data is ~zero-mean, ~unit-std
    z = norm.transform(x)
    assert z.mean().abs() < 1e-5 and abs(z.std().item() - 1.0) < 1e-2


def test_split_no_trajectory_leakage():
    ds = generate_trajectory(n_trajectories=50, steps_per_traj=20, seed=4)
    tr, va, te = split_indices(ds, seed=4)
    # disjoint indices
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0
    # no trajectory id appears in more than one split
    tids = lambda idx: set(ds.traj_id[idx].tolist())
    assert tids(tr).isdisjoint(tids(va))
    assert tids(tr).isdisjoint(tids(te))
    assert tids(va).isdisjoint(tids(te))


def test_build_datasets_and_normalizer_fit_on_train():
    ds = generate_random(n_samples=1000, seed=5)
    train, val, test, q_norm, pose_norm = build_datasets(ds, seed=5)
    assert len(train) + len(val) + len(test) == len(ds)
    sample = train[0]
    assert sample["q"].shape == (7,) and sample["pose"].shape == (6,)
    assert sample["q"].dtype == torch.float32
    # train split is standardized to ~unit scale
    all_q = torch.stack([train[i]["q"] for i in range(len(train))])
    assert all_q.mean().abs() < 1e-2 and abs(all_q.std().item() - 1.0) < 5e-2


def test_save_load_roundtrip(tmp_path):
    ds = generate_trajectory(n_trajectories=10, steps_per_traj=10, seed=6)
    p = tmp_path / "traj.npz"
    save_dataset(ds, str(p))
    ds2 = load_dataset(str(p))
    assert np.allclose(ds.q, ds2.q) and np.allclose(ds.pose, ds2.pose)
    assert ds2.kind == "trajectory" and ds2.traj_id is not None
    assert ds2.meta["v_deg"] == ds.meta["v_deg"]
