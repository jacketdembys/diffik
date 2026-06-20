"""Evaluation metrics for IK solutions.

Primary scalars (used to drive the Phase 5/6 ablations):
- position error (mm): Euclidean distance of the reconstructed end-effector
- orientation error (deg): geodesic angle of the reconstructed end-effector
- %<=1mm and %<=1deg success rates (matching the IROS tables)
- diversity: spread of multiple samples per pose (multimodality signal)

Per-axis (X/Y/Z, Ro/Pi/Ya) breakdowns for the final paper tables are added in
Phase 9; here we keep the scalar summary that the ablations need.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ErrorSummary:
    n: int
    pos_mm_avg: float
    pos_mm_min: float
    pos_mm_max: float
    pos_mm_std: float
    ori_deg_avg: float
    ori_deg_min: float
    ori_deg_max: float
    ori_deg_std: float
    pct_pos_le_1mm: float
    pct_ori_le_1deg: float
    # error-distribution buckets (matching the IROS papers)
    pct_pos_1_5mm: float = 0.0     # (1, 5] mm
    pct_pos_5_10mm: float = 0.0    # (5, 10] mm
    pct_pos_gt_10mm: float = 0.0   # > 10 mm
    pct_ori_1_3deg: float = 0.0    # (1, 3] deg
    pct_ori_gt_3deg: float = 0.0   # > 3 deg

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def __str__(self) -> str:
        return (
            f"n={self.n} | "
            f"pos(mm) avg={self.pos_mm_avg:.3f} [min {self.pos_mm_min:.3f}, max {self.pos_mm_max:.3f}] "
            f"| ranges %: <=1 {self.pct_pos_le_1mm:.1f}, (1,5] {self.pct_pos_1_5mm:.1f}, "
            f"(5,10] {self.pct_pos_5_10mm:.1f}, >10 {self.pct_pos_gt_10mm:.1f} | "
            f"ori(deg) avg={self.ori_deg_avg:.3f} ranges %: <=1 {self.pct_ori_le_1deg:.1f}, "
            f"(1,3] {self.pct_ori_1_3deg:.1f}, >3 {self.pct_ori_gt_3deg:.1f}"
        )


def _pct(mask) -> float:
    return float(mask.double().mean() * 100.0)


def summarize_errors(
    pos_mm: torch.Tensor,
    ori_deg: torch.Tensor,
    pos_thresh_mm: float = 1.0,
    ori_thresh_deg: float = 1.0,
) -> ErrorSummary:
    """Summarize per-sample position (mm) and orientation (deg) error vectors,
    including the error-distribution buckets used in the IROS papers."""
    pos_mm = pos_mm.flatten().double()
    ori_deg = ori_deg.flatten().double()
    return ErrorSummary(
        n=pos_mm.numel(),
        pos_mm_avg=float(pos_mm.mean()),
        pos_mm_min=float(pos_mm.min()),
        pos_mm_max=float(pos_mm.max()),
        pos_mm_std=float(pos_mm.std(unbiased=False)),
        ori_deg_avg=float(ori_deg.mean()),
        ori_deg_min=float(ori_deg.min()),
        ori_deg_max=float(ori_deg.max()),
        ori_deg_std=float(ori_deg.std(unbiased=False)),
        pct_pos_le_1mm=_pct(pos_mm <= pos_thresh_mm),
        pct_ori_le_1deg=_pct(ori_deg <= ori_thresh_deg),
        pct_pos_1_5mm=_pct((pos_mm > 1.0) & (pos_mm <= 5.0)),
        pct_pos_5_10mm=_pct((pos_mm > 5.0) & (pos_mm <= 10.0)),
        pct_pos_gt_10mm=_pct(pos_mm > 10.0),
        pct_ori_1_3deg=_pct((ori_deg > 1.0) & (ori_deg <= 3.0)),
        pct_ori_gt_3deg=_pct(ori_deg > 3.0),
    )


def diversity(samples: torch.Tensor) -> float:
    """Mean per-dimension std across the N samples drawn per pose.

    Args:
        samples: ``[P, N, dof]`` (joint space). Returns 0 for N==1.
    """
    if samples.shape[1] < 2:
        return 0.0
    return float(samples.double().std(dim=1, unbiased=False).mean())
