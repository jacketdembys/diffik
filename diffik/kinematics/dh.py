"""Batched, differentiable forward kinematics using standard Denavit-Hartenberg.

Standard (distal) DH convention. The homogeneous transform for link ``i`` with
parameters (theta, d, a, alpha) is::

    A_i = Rz(theta) . Tz(d) . Tx(a) . Rx(alpha)

        | cos t   -sin t cos al    sin t sin al    a cos t |
        | sin t    cos t cos al   -cos t sin al    a sin t |
    =   | 0        sin al          cos al          d       |
        | 0        0               0               1       |

All angles are in radians and all lengths in metres (SI). For a revolute joint
the joint value adds to ``theta``; for a prismatic joint it adds to ``d``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DHChain:
    """Standard-DH parameters of a serial kinematic chain.

    Each field is a 1-D tensor of length ``n`` (the number of joints), except
    ``is_revolute`` which is a boolean tensor of the same length.
    """

    d: torch.Tensor            # link offsets (m)
    a: torch.Tensor            # link lengths (m)
    alpha: torch.Tensor        # link twists (rad)
    theta_offset: torch.Tensor  # constant offset added to revolute theta (rad)
    is_revolute: torch.Tensor   # True=revolute, False=prismatic

    @property
    def n(self) -> int:
        return int(self.d.shape[0])

    def to(self, device=None, dtype=None) -> "DHChain":
        return DHChain(
            d=self.d.to(device=device, dtype=dtype),
            a=self.a.to(device=device, dtype=dtype),
            alpha=self.alpha.to(device=device, dtype=dtype),
            theta_offset=self.theta_offset.to(device=device, dtype=dtype),
            is_revolute=self.is_revolute.to(device=device),
        )


def dh_matrix(
    theta: torch.Tensor, d: torch.Tensor, a: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Single standard-DH link transform.

    Args are broadcastable tensors of shape ``[...]``; returns ``[..., 4, 4]``.
    """
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    zero = torch.zeros_like(ct)
    one = torch.ones_like(ct)
    # broadcast scalars (a, ca, sa) up to the batch shape of ct
    a = a + zero
    ca = ca + zero
    sa = sa + zero
    d = d + zero

    row0 = torch.stack([ct, -st * ca, st * sa, a * ct], dim=-1)
    row1 = torch.stack([st, ct * ca, -ct * sa, a * st], dim=-1)
    row2 = torch.stack([zero, sa, ca, d], dim=-1)
    row3 = torch.stack([zero, zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


def forward_kinematics(
    q: torch.Tensor, chain: DHChain, return_all: bool = False
) -> torch.Tensor:
    """Forward kinematics of a serial chain.

    Args:
        q: joint values, shape ``[B, n]`` (rad for revolute, m for prismatic).
        chain: the DH parameters.
        return_all: if True, also return per-joint cumulative frames.

    Returns:
        End-effector pose ``[B, 4, 4]``. If ``return_all``, returns a tuple
        ``(T_ee [B,4,4], frames [B, n+1, 4, 4])`` where ``frames[:, 0]`` is the
        base (identity) and ``frames[:, i]`` is the pose after joint ``i``.
    """
    if q.dim() != 2:
        raise ValueError(f"q must be [B, n]; got shape {tuple(q.shape)}")
    chain = chain.to(device=q.device, dtype=q.dtype)
    B, n = q.shape
    if n != chain.n:
        raise ValueError(f"q has {n} joints but chain has {chain.n}")

    is_rev = chain.is_revolute.view(1, n)
    theta = torch.where(is_rev, q + chain.theta_offset.view(1, n), chain.theta_offset.view(1, n).expand(B, n))
    d_eff = torch.where(is_rev, chain.d.view(1, n).expand(B, n), q + chain.d.view(1, n))

    T = torch.eye(4, device=q.device, dtype=q.dtype).expand(B, 4, 4).contiguous()
    frames = [T] if return_all else None
    for i in range(n):
        A_i = dh_matrix(theta[:, i], d_eff[:, i], chain.a[i], chain.alpha[i])
        T = T @ A_i
        if return_all:
            frames.append(T)

    if return_all:
        return T, torch.stack(frames, dim=1)
    return T
