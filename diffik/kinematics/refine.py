"""Differentiable damped-least-squares (Gauss-Newton) IK refinement using the
analytic geometric Jacobian. Refines candidate joint configs toward a target pose;
a few steps take diffusion samples to sub-mm (see scripts/newton_refine_validation.py).
"""
from __future__ import annotations

import torch

from .dh import forward_kinematics


def geo_jacobian(q, chain):
    """Analytic geometric Jacobian [N,6,dof] for a revolute DH chain."""
    _, frames = forward_kinematics(q, chain, return_all=True)   # frames: [N, dof+1, 4, 4]
    o_n = frames[:, -1, :3, 3]
    Jv, Jw = [], []
    for i in range(q.shape[1]):
        z = frames[:, i, :3, 2]          # joint (i+1) axis = z of frame i
        o = frames[:, i, :3, 3]
        Jv.append(torch.linalg.cross(z, o_n - o))
        Jw.append(z)
    return torch.cat([torch.stack(Jv, -1), torch.stack(Jw, -1)], dim=1)


def pose_error_vec(q, T_d, chain):
    """6D task-space error [p_d - p ; orientation error] -> [N,6]."""
    T = forward_kinematics(q, chain)
    p, R = T[:, :3, 3], T[:, :3, :3]
    p_d, R_d = T_d[:, :3, 3], T_d[:, :3, :3]
    Re = R_d @ R.transpose(1, 2)
    e_rot = 0.5 * torch.stack([Re[:, 2, 1] - Re[:, 1, 2],
                               Re[:, 0, 2] - Re[:, 2, 0],
                               Re[:, 1, 0] - Re[:, 0, 1]], dim=-1)
    return torch.cat([p_d - p, e_rot], dim=-1)


def dls_step(q, T_d, chain, lam=1e-3):
    """One damped-least-squares Newton step: dq = J^T (JJ^T + lam I)^-1 e."""
    e = pose_error_vec(q, T_d, chain)
    J = geo_jacobian(q, chain)
    JJt = J @ J.transpose(1, 2)
    I = torch.eye(6, dtype=q.dtype, device=q.device).expand_as(JJt)
    y = torch.linalg.solve(JJt + lam * I, e.unsqueeze(-1))
    return q + (J.transpose(1, 2) @ y).squeeze(-1)


def refine(q, T_d, chain, steps=1, lam=1e-3, limits=None):
    """Apply `steps` DLS steps, clamping to joint `limits` [dof,2] if given."""
    for _ in range(steps):
        q = dls_step(q, T_d, chain, lam)
        if limits is not None:
            q = torch.clamp(q, limits[:, 0], limits[:, 1])
    return q
