"""Pose-vector representations used as the diffusion conditioning signal.

The conditioning pose D is a flat vector derived from the SE(3) FK output. We
default to ``xyzrpy`` (position in metres + roll/pitch/yaw in radians, D in R^6)
to match the IROS 2024/2025 papers, but keep this pluggable because a continuous
rotation representation may help accuracy later.
"""
from __future__ import annotations

import torch

from ..kinematics.pose import matrix_to_rpy, position, rotation

POSE_DIMS = {"xyzrpy": 6}


def pose_from_matrix(T: torch.Tensor, repr: str = "xyzrpy") -> torch.Tensor:
    """SE(3) ``[B,4,4]`` -> pose vector ``[B, pose_dim]``."""
    if repr == "xyzrpy":
        return torch.cat([position(T), matrix_to_rpy(rotation(T))], dim=-1)
    raise ValueError(f"unknown pose repr '{repr}'")


def pose_dim(repr: str = "xyzrpy") -> int:
    return POSE_DIMS[repr]
