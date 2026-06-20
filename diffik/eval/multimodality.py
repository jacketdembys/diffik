"""Multimodality evaluation: does the model produce MULTIPLE valid IK solutions
per pose, and how diverse are they?

For each pose we draw K candidates and measure:
  - best_of_k         : accuracy of the best candidate (pos mm / ori deg)
  - diversity_all      : mean per-dim joint std over the K candidates (raw spread)
  - mean_valid_per_pose: avg number of candidates within (tol_mm, tol_deg)
  - frac_poses_multi   : fraction of poses with >= 2 VALID solutions
  - valid_diversity    : mean pairwise joint distance among VALID candidates
                         (diverse AND correct -> the real multimodality signal;
                          for a redundant arm this is null-space/self-motion coverage)

Run it seedless (no example) vs seeded (LBE example) to show the diversity<->precision
knob: seedless should be diverse, seeded should collapse toward one solution.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..kinematics import forward_kinematics, get_robot, pose_error
from .metrics import ErrorSummary, diversity, summarize_errors

FK_DTYPE = torch.float64


@dataclass
class MultimodalResult:
    K: int
    tol_mm: float
    tol_deg: float
    best_of_k: ErrorSummary
    diversity_all: float
    mean_valid_per_pose: float
    frac_poses_multi: float
    valid_diversity: float

    def __str__(self):
        return (f"[K={self.K}, valid<= {self.tol_mm}mm/{self.tol_deg}deg]\n"
                f"  best-of-K: {self.best_of_k}\n"
                f"  diversity(all)={self.diversity_all:.4f} | valid/pose={self.mean_valid_per_pose:.2f} | "
                f"poses w/ >=2 valid={self.frac_poses_multi*100:.1f}% | valid_diversity={self.valid_diversity:.4f}")


def _valid_diversity(q_pred, valid):
    """Mean pairwise joint distance among VALID candidates, averaged over poses
    that have >= 2 valid solutions."""
    vals = []
    P = q_pred.shape[0]
    for p in range(P):
        idx = valid[p].nonzero(as_tuple=True)[0]
        v = idx.numel()
        if v >= 2:
            qv = q_pred[p, idx]
            d = torch.cdist(qv, qv)
            vals.append(d.sum() / (v * (v - 1)))  # mean over off-diagonal pairs
    if not vals:
        return 0.0, 0
    return float(torch.stack(vals).mean()), len(vals)


@torch.no_grad()
def evaluate_multimodality(diffusion, dataset, q_norm, robot="panda_7r", K=50,
                           device="cpu", generator=None, tol_mm=10.0, tol_deg=5.0,
                           **sample_kwargs) -> MultimodalResult:
    device = torch.device(device)
    diffusion.to(device); diffusion.eval()
    chain = get_robot(robot, dtype=FK_DTYPE)

    pose_n = dataset.pose.to(device)
    q_true_n = dataset.q.to(device)
    P = pose_n.shape[0]

    samples_n = diffusion.sample(pose_n, n_per_pose=K, generator=generator, **sample_kwargs)  # [P,K,dof]
    q_pred = q_norm.inverse_transform(samples_n.cpu()).to(FK_DTYPE)   # [P,K,dof]
    q_true = q_norm.inverse_transform(q_true_n.cpu()).to(FK_DTYPE)    # [P,dof]
    dof = q_pred.shape[-1]

    T_pred = forward_kinematics(q_pred.reshape(P * K, dof), chain)
    T_true = forward_kinematics(q_true, chain)
    pos_mm, ori_deg = pose_error(T_pred, T_true.repeat_interleave(K, dim=0))
    pos_mm = pos_mm.reshape(P, K)
    ori_deg = ori_deg.reshape(P, K)

    # best-of-K (per pose, lowest position error)
    best_idx = pos_mm.argmin(dim=1)
    rows = torch.arange(P)
    best = summarize_errors(pos_mm[rows, best_idx], ori_deg[rows, best_idx])

    valid = (pos_mm <= tol_mm) & (ori_deg <= tol_deg)         # [P,K]
    n_valid = valid.sum(dim=1)
    vdiv, _ = _valid_diversity(q_pred, valid)

    return MultimodalResult(
        K=K, tol_mm=tol_mm, tol_deg=tol_deg,
        best_of_k=best,
        diversity_all=diversity(q_pred),
        mean_valid_per_pose=float(n_valid.double().mean()),
        frac_poses_multi=float((n_valid >= 2).double().mean()),
        valid_diversity=vdiv,
    )
