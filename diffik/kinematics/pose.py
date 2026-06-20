"""Pose extraction / conversion utilities for SE(3) transforms.

Conventions:
- Position is the translation column of the homogeneous matrix (metres).
- RPY is the X-Y-Z fixed-axis (roll, pitch, yaw) representation matching the
  reconstruction-error protocol used in the IROS 2024/2025 papers.
"""
from __future__ import annotations

import torch


def position(T: torch.Tensor) -> torch.Tensor:
    """Translation column, shape ``[B, 3]`` (metres)."""
    return T[..., :3, 3]


def rotation(T: torch.Tensor) -> torch.Tensor:
    """Rotation block, shape ``[B, 3, 3]``."""
    return T[..., :3, :3]


def matrix_to_rpy(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> roll/pitch/yaw (X-Y-Z), shape ``[B, 3]`` (rad).

    R = Rz(yaw) . Ry(pitch) . Rx(roll). Uses the robust ``cy`` formulation that
    falls back to a degenerate branch at the gimbal-lock pitch = +/- pi/2.
    """
    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    cy = torch.sqrt(r00 * r00 + r10 * r10)
    locked = cy < 1e-7

    pitch = torch.atan2(-r20, cy)
    roll = torch.where(locked, torch.atan2(-r12, r11), torch.atan2(r21, r22))
    yaw = torch.where(locked, torch.zeros_like(pitch), torch.atan2(r10, r00))
    return torch.stack([roll, pitch, yaw], dim=-1)


def rotation_angle_deg(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two rotations, in degrees, shape ``[B]``.

    angle = arccos((trace(R_a^T R_b) - 1) / 2).
    """
    R_rel = R_a.transpose(-1, -2) @ R_b
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def pose_error(T_pred: torch.Tensor, T_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Position error (mm) and orientation error (deg) between two SE(3) poses.

    Returns ``(pos_err_mm [B], ori_err_deg [B])``. Position error is the
    Euclidean norm of the translation difference (m -> mm); orientation error is
    the geodesic angle between the rotation blocks.
    """
    pos_err_mm = torch.linalg.norm(position(T_pred) - position(T_target), dim=-1) * 1000.0
    ori_err_deg = rotation_angle_deg(rotation(T_pred), rotation(T_target))
    return pos_err_mm, ori_err_deg


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> unit quaternion ``[B, 4]`` ordered (w, x, y, z)."""
    m = R
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.zeros((*R.shape[:-2], 4), dtype=R.dtype, device=R.device)

    s = torch.sqrt(torch.clamp(trace + 1.0, min=0.0)) * 2.0  # 4w
    safe = s > 1e-8
    w = torch.where(safe, 0.25 * s, torch.zeros_like(s))
    x = torch.where(safe, (m[..., 2, 1] - m[..., 1, 2]) / torch.clamp(s, min=1e-8), torch.zeros_like(s))
    y = torch.where(safe, (m[..., 0, 2] - m[..., 2, 0]) / torch.clamp(s, min=1e-8), torch.zeros_like(s))
    z = torch.where(safe, (m[..., 1, 0] - m[..., 0, 1]) / torch.clamp(s, min=1e-8), torch.zeros_like(s))
    q[..., 0], q[..., 1], q[..., 2], q[..., 3] = w, x, y, z
    # Fallback for the rare trace<=-1 case: use a simple per-sample recompute.
    if (~safe).any():
        idx = (~safe).nonzero(as_tuple=True)
        for b in zip(*idx):
            q[b] = _quat_from_matrix_fallback(R[b])
    return q


def _quat_from_matrix_fallback(m: torch.Tensor) -> torch.Tensor:
    # Largest-diagonal method for numerical robustness when trace <= -1.
    i = int(torch.argmax(torch.stack([m[0, 0], m[1, 1], m[2, 2]])))
    if i == 0:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif i == 1:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return torch.stack([w, x, y, z])
