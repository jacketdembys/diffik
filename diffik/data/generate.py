"""Dataset generation for DiffIK.

Two *separate* generators (never mixed into one dataset):

- ``generate_random``: joint configurations sampled i.i.d. uniformly within the
  joint limits. Cleanest setting for the seedless multimodal story.
- ``generate_trajectory``: trajectories built by perturbing the current joint
  state by +/- v at each step (the IROS 2024/2025 scheme); the LBE example is
  then the previous waypoint in the same trajectory.

Every generated dataset stores joints ``q`` and the corresponding pose vector
``pose`` = FK(q), so the consistency check FK(q) == pose holds by construction.
Generation runs in float64 for accuracy; cast to float32 at training time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..kinematics import forward_kinematics, get_robot
from ..kinematics.robots import PANDA_JOINT_LIMITS
from .poses import pose_dim, pose_from_matrix

_JOINT_LIMITS = {"panda_7r": PANDA_JOINT_LIMITS}

DTYPE = torch.float64


@dataclass
class Dataset:
    """In-memory dataset container."""

    q: np.ndarray                    # [N, n] joint configs
    pose: np.ndarray                 # [N, pose_dim] pose vectors = FK(q)
    robot: str
    pose_repr: str
    kind: str                        # "random" or "trajectory"
    traj_id: np.ndarray | None = None   # [N] trajectory index (trajectory only)
    step: np.ndarray | None = None      # [N] step within trajectory (trajectory only)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.q.shape[0]


def get_joint_limits(robot: str) -> torch.Tensor:
    if robot not in _JOINT_LIMITS:
        raise KeyError(f"no joint limits registered for '{robot}'")
    return _JOINT_LIMITS[robot].to(DTYPE)


def _fk_poses(q: torch.Tensor, robot: str, pose_repr: str) -> torch.Tensor:
    chain = get_robot(robot, dtype=DTYPE)
    T = forward_kinematics(q, chain)
    return pose_from_matrix(T, pose_repr)


def sample_uniform_joints(n: int, limits: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Uniform samples in the joint box, shape ``[n, dof]``."""
    dof = limits.shape[0]
    lo, hi = limits[:, 0], limits[:, 1]
    u = torch.rand(n, dof, generator=generator, dtype=DTYPE)
    return lo + u * (hi - lo)


def generate_random(
    robot: str = "panda_7r",
    n_samples: int = 1000,
    seed: int = 0,
    pose_repr: str = "xyzrpy",
) -> Dataset:
    """i.i.d. uniform joint samples and their FK poses."""
    g = torch.Generator().manual_seed(seed)
    limits = get_joint_limits(robot)
    q = sample_uniform_joints(n_samples, limits, g)
    pose = _fk_poses(q, robot, pose_repr)
    return Dataset(
        q=q.numpy(),
        pose=pose.numpy(),
        robot=robot,
        pose_repr=pose_repr,
        kind="random",
        meta={"n_samples": n_samples, "seed": seed},
    )


def generate_trajectory(
    robot: str = "panda_7r",
    n_trajectories: int = 100,
    steps_per_traj: int = 100,
    v_deg: float = 1.0,
    v_mm: float = 1.0,
    seed: int = 0,
    pose_repr: str = "xyzrpy",
) -> Dataset:
    """Trajectory-based generation (IROS scheme).

    Each trajectory starts from a uniform sample; each subsequent step perturbs
    every joint by U(-v, +v) and clamps to the joint limits. v is ``v_deg`` for
    revolute joints (converted to rad) and ``v_mm`` for prismatic joints.
    """
    g = torch.Generator().manual_seed(seed)
    limits = get_joint_limits(robot)
    chain = get_robot(robot, dtype=DTYPE)
    dof = limits.shape[0]
    lo, hi = limits[:, 0], limits[:, 1]

    v = torch.where(
        chain.is_revolute,
        torch.full((dof,), v_deg * math.pi / 180.0, dtype=DTYPE),
        torch.full((dof,), v_mm / 1000.0, dtype=DTYPE),
    )

    q_all, tid_all, step_all = [], [], []
    for t in range(n_trajectories):
        q_cur = sample_uniform_joints(1, limits, g)[0]
        for s in range(steps_per_traj):
            q_all.append(q_cur.clone())
            tid_all.append(t)
            step_all.append(s)
            delta = (torch.rand(dof, generator=g, dtype=DTYPE) * 2 - 1) * v
            q_cur = torch.clamp(q_cur + delta, lo, hi)

    q = torch.stack(q_all, dim=0)
    pose = _fk_poses(q, robot, pose_repr)
    return Dataset(
        q=q.numpy(),
        pose=pose.numpy(),
        robot=robot,
        pose_repr=pose_repr,
        kind="trajectory",
        traj_id=np.asarray(tid_all, dtype=np.int64),
        step=np.asarray(step_all, dtype=np.int64),
        meta={
            "n_trajectories": n_trajectories,
            "steps_per_traj": steps_per_traj,
            "v_deg": v_deg,
            "v_mm": v_mm,
            "seed": seed,
        },
    )


def add_examples(
    ds: Dataset, v_deg: float = 1.0, v_mm: float = 1.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-sample LBE example pairs (Qe, De), aligned with ds (raw space).

    - trajectory: example = previous waypoint in the same trajectory (step 0 uses
      itself). This is the natural, free seed.
    - random: synthesize an example by perturbing the query joints by U(-v, v)
      and clamping to the limits, then De = FK(Qe).

    Returns ``(example_q [N, dof], example_pose [N, pose_dim])``.
    """
    if ds.kind == "trajectory":
        assert ds.step is not None
        n = len(ds)
        ex_idx = np.arange(n)
        prev_ok = ds.step > 0
        ex_idx[prev_ok] = ex_idx[prev_ok] - 1  # previous waypoint (same trajectory)
        return ds.q[ex_idx].copy(), ds.pose[ex_idx].copy()

    # random: synthesize a within-v example
    g = torch.Generator().manual_seed(seed)
    limits = get_joint_limits(ds.robot)
    chain = get_robot(ds.robot, dtype=DTYPE)
    lo, hi = limits[:, 0], limits[:, 1]
    v = torch.where(
        chain.is_revolute,
        torch.full((limits.shape[0],), v_deg * math.pi / 180.0, dtype=DTYPE),
        torch.full((limits.shape[0],), v_mm / 1000.0, dtype=DTYPE),
    )
    q = torch.as_tensor(ds.q, dtype=DTYPE)
    delta = (torch.rand(q.shape, generator=g, dtype=DTYPE) * 2 - 1) * v
    ex_q = torch.clamp(q + delta, lo, hi)
    ex_pose = _fk_poses(ex_q, ds.robot, ds.pose_repr)
    return ex_q.numpy(), ex_pose.numpy()


def save_dataset(ds: Dataset, path: str) -> None:
    arrays = {
        "q": ds.q,
        "pose": ds.pose,
        "robot": np.array(ds.robot),
        "pose_repr": np.array(ds.pose_repr),
        "kind": np.array(ds.kind),
    }
    if ds.traj_id is not None:
        arrays["traj_id"] = ds.traj_id
    if ds.step is not None:
        arrays["step"] = ds.step
    arrays["meta"] = np.array(repr(ds.meta))
    np.savez(path, **arrays)


def load_dataset(path: str) -> Dataset:
    z = np.load(path, allow_pickle=False)
    import ast

    return Dataset(
        q=z["q"],
        pose=z["pose"],
        robot=str(z["robot"]),
        pose_repr=str(z["pose_repr"]),
        kind=str(z["kind"]),
        traj_id=z["traj_id"] if "traj_id" in z else None,
        step=z["step"] if "step" in z else None,
        meta=ast.literal_eval(str(z["meta"])),
    )
