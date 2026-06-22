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
    mean: ErrorSummary           # over all P*N samples (= mean-of-K per pose)
    best_of_n: ErrorSummary      # per-pose MIN (by position error), over P
    n_per_pose: int
    diversity: float
    sample_time_s: float
    ms_per_solution: float
    worst_of_n: ErrorSummary = None      # per-pose MAX (by position error), over P
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
    chunk_samples: int = 8192,
    **sample_kwargs,
) -> EvalResult:
    """Evaluate over the FULL dataset, chunked to bound the per-chunk sampling
    batch to ~chunk_samples (= poses x n_per_pose). This keeps best-of-K on large
    test sets from OOMing on bigger models. The LBE ``example`` (if present in
    sample_kwargs) is sliced per chunk to stay aligned with its poses."""
    device = torch.device(device)
    diffusion.to(device)
    diffusion.eval()
    chain = get_robot(robot, dtype=FK_DTYPE)

    pose_n = dataset.pose.to(device)                 # [P, pose_dim] normalized
    q_true_n = dataset.q.to(device)                  # [P, dof] normalized
    P, dof = pose_n.shape[0], q_true_n.shape[1]
    example = sample_kwargs.pop("example", None)
    if example is not None:
        example = example.to(device)

    all_pos, all_ori = [], []
    best_pos, best_ori, worst_pos, worst_ori = [], [], [], []
    div_sum, div_n = 0.0, 0
    step = max(1, chunk_samples // max(n_per_pose, 1))   # poses per chunk -> ~chunk_samples batch
    t0 = time.perf_counter()
    for s0 in range(0, P, step):
        sl = slice(s0, min(s0 + step, P))
        kw = dict(sample_kwargs)
        if example is not None:
            kw["example"] = example[sl]
        samples_n = diffusion.sample(pose_n[sl], n_per_pose=n_per_pose, generator=generator, **kw)  # [c,N,dof]
        c = samples_n.shape[0]
        q_pred = q_norm.inverse_transform(samples_n.cpu()).to(FK_DTYPE)
        q_true = q_norm.inverse_transform(q_true_n[sl].cpu()).to(FK_DTYPE)
        T_pred = forward_kinematics(q_pred.reshape(c * n_per_pose, dof), chain)
        T_true = forward_kinematics(q_true, chain)
        pos, ori = pose_error(T_pred, T_true.repeat_interleave(n_per_pose, dim=0))
        pos = pos.reshape(c, n_per_pose)
        ori = ori.reshape(c, n_per_pose)
        all_pos.append(pos); all_ori.append(ori)
        r = torch.arange(c)
        bi = pos.argmin(dim=1); best_pos.append(pos[r, bi]); best_ori.append(ori[r, bi])
        wi = pos.argmax(dim=1); worst_pos.append(pos[r, wi]); worst_ori.append(ori[r, wi])
        if n_per_pose >= 2:
            div_sum += float(q_pred.double().std(dim=1, unbiased=False).mean()) * c
            div_n += c
    sample_time = time.perf_counter() - t0

    all_pos = torch.cat(all_pos); all_ori = torch.cat(all_ori)        # [P, N]
    best_pos = torch.cat(best_pos); best_ori = torch.cat(best_ori)    # [P]
    worst_pos = torch.cat(worst_pos); worst_ori = torch.cat(worst_ori)  # [P]

    return EvalResult(
        mean=summarize_errors(all_pos, all_ori),
        best_of_n=summarize_errors(best_pos, best_ori),
        worst_of_n=summarize_errors(worst_pos, worst_ori),
        n_per_pose=n_per_pose,
        diversity=(div_sum / div_n) if div_n else 0.0,
        sample_time_s=sample_time,
        ms_per_solution=sample_time / (P * n_per_pose) * 1000.0,
        pos_mm_per_pose=best_pos.numpy(),
        ori_deg_per_pose=best_ori.numpy(),
    )
