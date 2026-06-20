"""Robot definitions (standard-DH), in SI units (metres, radians).

The 7-DoF-7R Panda parameters reproduce Table I of the IROS 2024 paper
(d, a originally in mm and alpha in deg):

    i  theta   d(mm)   a(mm)   alpha(deg)
    1  th1     333     0       0
    2  th2     0       0       -90
    3  th3     316     0       90
    4  th4     0       82.5    90
    5  th5     384     -82.5   -90
    6  th6     0       0       90
    7  th7     107     88      90
"""
from __future__ import annotations

import math

import torch

from .dh import DHChain

_DEG = math.pi / 180.0


def panda_7r(dtype: torch.dtype = torch.float64) -> DHChain:
    """7-DoF-7R commensurate Panda arm (IROS 2024 Table I)."""
    d_mm = [333.0, 0.0, 316.0, 0.0, 384.0, 0.0, 107.0]
    a_mm = [0.0, 0.0, 0.0, 82.5, -82.5, 0.0, 88.0]
    alpha_deg = [0.0, -90.0, 90.0, 90.0, -90.0, 90.0, 90.0]
    n = 7
    return DHChain(
        d=torch.tensor([x / 1000.0 for x in d_mm], dtype=dtype),
        a=torch.tensor([x / 1000.0 for x in a_mm], dtype=dtype),
        alpha=torch.tensor([x * _DEG for x in alpha_deg], dtype=dtype),
        theta_offset=torch.zeros(n, dtype=dtype),
        is_revolute=torch.ones(n, dtype=torch.bool),
    )


# Panda joint limits (rad), used later for dataset generation.
PANDA_JOINT_LIMITS = torch.tensor(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=torch.float64,
)


ROBOTS = {"panda_7r": panda_7r}


def get_robot(name: str, dtype: torch.dtype = torch.float64) -> DHChain:
    if name not in ROBOTS:
        raise KeyError(f"unknown robot '{name}'; available: {list(ROBOTS)}")
    return ROBOTS[name](dtype=dtype)
