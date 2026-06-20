"""Evaluation harness: sample from a diffusion solver and score against targets.

Reports both the per-sample ("mean") summary and the practical generative-IK
"best-of-N" summary (per pose, keep the candidate with lowest position error),
plus diversity and inference time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..data.dataset import DiffIKDataset, Normalizer
from ..kinematics import forward_kinematics, get_robot, pose_error
from .metrics import ErrorSummary, diversity, summarize_errors

FK_DTYPE = torch.float64


@dataclass
class EvalResult:
    mean: ErrorSummary           # over all P*N samples
    best_of_n: ErrorSummary      # per-pose best (by position error), over P
    n_per_pose: int
    diversity: float
    sample_time_s: float
    ms_per_solution: float
    pos_mm_per_pose: np.ndarray = None   # [P] best-of-N position error (mm)
    ori_deg_per_pose: np.ndarray = None  # [P] best-of-N orientation error (deg)

    def __str__(self) -> str:
        return (
            f"[eval n_per_pose={self.n_per_pose}]\n"
            f"  mean      : {self.mean}\n"
            f"  best-of-{self.n_per_pose}: {self.best_of_n}\n"
            f"  diversity : {self.diversity:.4f} | {self.ms_per_solution:.3f} ms/solution"
        )


@torch.no_grad()
def evaluate(
    diffusion,
    dataset: DiffIKDataset,
    q_norm: Normalizer,
    robot: str = "panda_7r",
    n_per_pose: int = 1,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    **sample_kwargs,
) -> EvalResult:
    device = torch.device(device)
    diffusion.to(device)
    diffusion.eval()
    chain = get_robot(robot, dtype=FK_DTYPE)

    pose_n = dataset.pose.to(device)                 # [P, pose_dim] normalized
    q_true_n = dataset.q.to(device)                  # [P, dof] normalized
    P = pose_n.shape[0]

    t0 = time.perf_counter()
    samples_n = diffusion.sample(pose_n, n_per_pose=n_per_pose, generator=generator, **sample_kwargs)
    sample_time = time.perf_counter() - t0

    # denormalize to joint space, run FK in float64
    q_pred = q_norm.inverse_transform(samples_n.cpu()).to(FK_DTYPE)          # [P,N,dof]
    q_true = q_norm.inverse_transform(q_true_n.cpu()).to(FK_DTYPE)           # [P,dof]

    dof = q_pred.shape[-1]
    T_pred = forward_kinematics(q_pred.reshape(P * n_per_pose, dof), chain)
    T_true = forward_kinematics(q_true, chain)
    T_true_rep = T_true.repeat_interleave(n_per_pose, dim=0)

    pos_mm, ori_deg = pose_error(T_pred, T_true_rep)        # [P*N]
    pos_mm = pos_mm.reshape(P, n_per_pose)
    ori_deg = ori_deg.reshape(P, n_per_pose)

    mean_summary = summarize_errors(pos_mm, ori_deg)

    # best-of-N: per pose, pick the candidate with smallest position error
    best_idx = pos_mm.argmin(dim=1)                         # [P]
    rows = torch.arange(P)
    best_pos = pos_mm[rows, best_idx]
    best_ori = ori_deg[rows, best_idx]
    best_summary = summarize_errors(best_pos, best_ori)

    return EvalResult(
        mean=mean_summary,
        best_of_n=best_summary,
        n_per_pose=n_per_pose,
        diversity=diversity(q_pred),
        sample_time_s=sample_time,
        ms_per_solution=sample_time / (P * n_per_pose) * 1000.0,
        pos_mm_per_pose=best_pos.cpu().numpy(),
        ori_deg_per_pose=best_ori.cpu().numpy(),
    )
