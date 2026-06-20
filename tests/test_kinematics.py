"""Phase 1 verification: differentiable forward kinematics.

The authoritative correctness check builds a roboticstoolbox DHRobot from the
*identical* standard-DH parameters and compares end-effector poses. That makes
this a pure math check of our FK implementation, independent of any particular
robot model that rtb ships.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from diffik.kinematics import (
    DHChain,
    forward_kinematics,
    matrix_to_rpy,
    panda_7r,
    pose_error,
    position,
    rotation,
)

rtb = pytest.importorskip("roboticstoolbox")

DTYPE = torch.float64


def _rtb_robot_from_chain(chain: DHChain):
    """Build an rtb standard-DH robot from the same parameters (revolute only)."""
    import roboticstoolbox as rtb_

    links = []
    for i in range(chain.n):
        links.append(
            rtb_.RevoluteDH(
                d=float(chain.d[i]),
                a=float(chain.a[i]),
                alpha=float(chain.alpha[i]),
                offset=float(chain.theta_offset[i]),
            )
        )
    return rtb_.DHRobot(links)


def _rtb_fkine_batch(robot, q_np: np.ndarray) -> np.ndarray:
    return np.stack([robot.fkine(q_np[b]).A for b in range(q_np.shape[0])], axis=0)


@pytest.fixture(scope="module")
def chain() -> DHChain:
    return panda_7r(dtype=DTYPE)


@pytest.fixture(scope="module")
def rtb_robot(chain):
    return _rtb_robot_from_chain(chain)


def test_fk_zero_config_matches_rtb(chain, rtb_robot):
    q = torch.zeros(1, chain.n, dtype=DTYPE)
    T_ours = forward_kinematics(q, chain)[0].numpy()
    T_rtb = rtb_robot.fkine(np.zeros(chain.n)).A
    assert np.allclose(T_ours, T_rtb, atol=1e-9), f"\nours=\n{T_ours}\nrtb=\n{T_rtb}"


def test_fk_random_configs_match_rtb(chain, rtb_robot, capsys):
    torch.manual_seed(0)
    B = 256
    q = (torch.rand(B, chain.n, dtype=DTYPE) * 2 - 1) * math.pi
    T_ours = forward_kinematics(q, chain)
    T_rtb = torch.from_numpy(_rtb_fkine_batch(rtb_robot, q.numpy()))

    pos_err_mm, ori_err_deg = pose_error(T_ours, T_rtb)

    with capsys.disabled():
        print(
            f"\n  [FK vs rtb over {B} random configs]"
            f"\n    position error (mm):  max={pos_err_mm.max():.3e}  mean={pos_err_mm.mean():.3e}"
            f"\n    orientation err (deg): max={ori_err_deg.max():.3e}  mean={ori_err_deg.mean():.3e}"
        )

    # Tight tolerances: errors are at float64 round-off level. Note the geodesic
    # angle uses arccos, which is ill-conditioned near 0deg (a ~1e-13 matrix
    # round-off amplifies to ~1e-6 deg), so its floor is looser than position's.
    assert pos_err_mm.max() < 1e-6, f"max position err {pos_err_mm.max():.2e} mm"
    assert ori_err_deg.max() < 1e-3, f"max orientation err {ori_err_deg.max():.2e} deg"


def test_batched_equals_loop(chain):
    torch.manual_seed(1)
    q = torch.randn(17, chain.n, dtype=DTYPE)
    T_batched = forward_kinematics(q, chain)
    T_loop = torch.stack([forward_kinematics(q[i : i + 1], chain)[0] for i in range(q.shape[0])])
    assert torch.allclose(T_batched, T_loop, atol=1e-12)


def test_valid_se3(chain):
    torch.manual_seed(2)
    q = torch.randn(64, chain.n, dtype=DTYPE)
    T = forward_kinematics(q, chain)
    R = rotation(T)
    # rotation is orthonormal with det +1
    eye = torch.eye(3, dtype=DTYPE).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-9)
    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-9)
    # bottom row is [0, 0, 0, 1]
    bottom = T[:, 3, :]
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DTYPE)
    assert torch.allclose(bottom, expected.expand_as(bottom), atol=1e-12)


def test_fk_is_differentiable_gradcheck(chain):
    q = torch.randn(2, chain.n, dtype=DTYPE, requires_grad=True)

    def f(qq):
        return forward_kinematics(qq, chain)[:, :3, 3]

    assert torch.autograd.gradcheck(f, (q,), atol=1e-6, rtol=1e-4)


def test_orientation_rpy_matches_rtb(chain, rtb_robot):
    torch.manual_seed(3)
    B = 128
    q = (torch.rand(B, chain.n, dtype=DTYPE) * 2 - 1) * (0.9 * math.pi)
    T_ours = forward_kinematics(q, chain)
    rpy_ours = matrix_to_rpy(rotation(T_ours)).numpy()

    T_rtb = _rtb_fkine_batch(rtb_robot, q.numpy())
    rpy_rtb = np.stack(
        [_rpy_from_matrix(T_rtb[b, :3, :3]) for b in range(B)], axis=0
    )
    # compare on the sin/cos of each angle to avoid +-pi wrap artifacts
    assert np.allclose(np.sin(rpy_ours), np.sin(rpy_rtb), atol=1e-7)
    assert np.allclose(np.cos(rpy_ours), np.cos(rpy_rtb), atol=1e-7)


def _rpy_from_matrix(R: np.ndarray) -> np.ndarray:
    cy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if cy < 1e-7:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], cy)
        yaw = 0.0
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], cy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])
