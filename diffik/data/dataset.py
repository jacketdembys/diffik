"""Torch dataset, normalization, and leakage-safe splitting for DiffIK."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from .generate import Dataset


class Normalizer:
    """Per-dimension standardization: ``(x - mean) / std``."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, x: torch.Tensor, eps: float = 1e-8) -> "Normalizer":
        return cls(mean=x.mean(dim=0, keepdim=True), std=x.std(dim=0, keepdim=True) + eps)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x)) / self.std.to(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x) + self.mean.to(x)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def load_state_dict(cls, sd: dict) -> "Normalizer":
        return cls(mean=sd["mean"], std=sd["std"])


class DiffIKDataset(TorchDataset):
    """Yields ``{"q": [n], "pose": [pose_dim]}`` as normalized float32 tensors.

    If example pairs are provided, also yields ``"example": [pose_dim+dof]`` =
    concat(normalized example pose, normalized example joints), and exposes the
    full normalized example tensor as ``self.example`` (for evaluation).
    """

    def __init__(
        self,
        q: np.ndarray | torch.Tensor,
        pose: np.ndarray | torch.Tensor,
        q_norm: Normalizer,
        pose_norm: Normalizer,
        example_q: np.ndarray | torch.Tensor | None = None,
        example_pose: np.ndarray | torch.Tensor | None = None,
    ):
        q_t = torch.as_tensor(q, dtype=torch.float32)
        pose_t = torch.as_tensor(pose, dtype=torch.float32)
        self.q = q_norm.transform(q_t)
        self.pose = pose_norm.transform(pose_t)
        self.q_norm = q_norm
        self.pose_norm = pose_norm

        self.example = None
        if example_q is not None and example_pose is not None:
            eq = q_norm.transform(torch.as_tensor(example_q, dtype=torch.float32))
            ep = pose_norm.transform(torch.as_tensor(example_pose, dtype=torch.float32))
            self.example = torch.cat([ep, eq], dim=-1)  # [N, pose_dim+dof]

    def __len__(self) -> int:
        return self.q.shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {"q": self.q[idx], "pose": self.pose[idx]}
        if self.example is not None:
            item["example"] = self.example[idx]
        return item

    def head(self, n: int) -> "DiffIKDataset":
        """A cheap subset view of the first n samples (for fast val monitoring)."""
        import copy
        d = copy.copy(self)
        d.q = self.q[:n]
        d.pose = self.pose[:n]
        d.example = self.example[:n] if self.example is not None else None
        return d


def split_indices(
    ds: Dataset,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/val/test index split.

    For trajectory datasets the split is by *trajectory* (so near-duplicate
    consecutive frames never leak across splits); for random datasets it is a
    plain per-sample shuffle.
    """
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1"
    rng = np.random.default_rng(seed)

    if ds.kind == "trajectory" and ds.traj_id is not None:
        traj_ids = np.unique(ds.traj_id)
        rng.shuffle(traj_ids)
        n = len(traj_ids)
        n_tr = int(fractions[0] * n)
        n_va = int(fractions[1] * n)
        groups = {
            "train": set(traj_ids[:n_tr].tolist()),
            "val": set(traj_ids[n_tr : n_tr + n_va].tolist()),
            "test": set(traj_ids[n_tr + n_va :].tolist()),
        }
        member = {k: np.array([i for i, t in enumerate(ds.traj_id) if t in v], dtype=np.int64)
                  for k, v in groups.items()}
        return member["train"], member["val"], member["test"]

    idx = np.arange(len(ds))
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(fractions[0] * n)
    n_va = int(fractions[1] * n)
    return idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]


def build_datasets(
    ds: Dataset,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[DiffIKDataset, DiffIKDataset, DiffIKDataset, Normalizer, Normalizer]:
    """Split, fit normalizers on the *train* split only, and wrap all splits."""
    tr, va, te = split_indices(ds, fractions, seed)
    q = torch.as_tensor(ds.q, dtype=torch.float32)
    pose = torch.as_tensor(ds.pose, dtype=torch.float32)

    q_norm = Normalizer.fit(q[tr])
    pose_norm = Normalizer.fit(pose[tr])

    make = lambda idx: DiffIKDataset(q[idx], pose[idx], q_norm, pose_norm)
    return make(tr), make(va), make(te), q_norm, pose_norm


def build_datasets_lbe(
    ds: Dataset,
    v_deg: float = 1.0,
    v_mm: float = 1.0,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
):
    """Like ``build_datasets`` but also attaches LBE example pairs to each split.

    Examples are computed globally (raw space) then subset per split, so the
    leakage-safe split is preserved. Normalizers are fit on the train split only.
    """
    from .generate import add_examples

    ex_q_all, ex_pose_all = add_examples(ds, v_deg=v_deg, v_mm=v_mm, seed=seed)
    tr, va, te = split_indices(ds, fractions, seed)
    q = torch.as_tensor(ds.q, dtype=torch.float32)
    pose = torch.as_tensor(ds.pose, dtype=torch.float32)
    ex_q = torch.as_tensor(ex_q_all, dtype=torch.float32)
    ex_pose = torch.as_tensor(ex_pose_all, dtype=torch.float32)

    q_norm = Normalizer.fit(q[tr])
    pose_norm = Normalizer.fit(pose[tr])

    make = lambda idx: DiffIKDataset(
        q[idx], pose[idx], q_norm, pose_norm, ex_q[idx], ex_pose[idx]
    )
    return make(tr), make(va), make(te), q_norm, pose_norm
