"""Config schema for training/eval runs (YAML-driven, cluster-ready)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import yaml


@dataclass
class DataConfig:
    kind: str = "trajectory"           # "trajectory" or "random"
    lbe: bool = True                   # build example pairs (LBE)
    robot: str = "panda_7r"
    # random
    n_samples: int = 1_000_000
    # trajectory
    n_trajectories: int = 10_000
    steps_per_traj: int = 100
    v_deg: float = 1.0
    v_mm: float = 1.0
    seed: int = 0


@dataclass
class ModelConfig:
    type: str = "lbe"                  # "lbe" or "mlp"
    backbone: str = "plain"           # "plain" | "rmlp" | "dmlp" (LBE denoiser)
    hidden_dim: int = 1024
    n_layers: int = 4
    time_embed_dim: int = 128
    pose_embed_dim: int = 128
    example_embed_dim: int = 128
    dropout: float = 0.0
    self_cond: bool = False           # self-conditioning (feed prev x0_hat back in)


@dataclass
class DiffusionConfig:
    T: int = 1000
    prediction_type: str = "eps"
    fk_loss_weight: float = 10.0
    rot_weight: float = 0.1
    fk_weighting: str = "alpha_bar"   # none|alpha_bar|alpha_bar_pow|snr|low_t_window
    fk_weight_gamma: float = 1.0      # exponent (alpha_bar_pow) or min-SNR clamp (snr)
    fk_t_window: int = 0              # low_t_window: supervise FK only for t < this (0 -> T//10)
    p_example_dropout: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 1000                 # max epochs (upper cap; early stopping may stop sooner)
    batch_size: int = 128
    lr: float = 3e-4
    device: str | None = None          # None -> auto (cuda/mps/cpu)
    checkpoint_every: int = 50
    patience: int = 0                  # 0 = no early stopping; >0 = patience in CHECKS
    early_stop_metric: str = "val_pose"  # "val_pose" (metric-aligned) or "val_loss"
    monitor_every: int = 10            # epochs between monitor checks
    monitor_cap: int = 512             # cap val poses used for the monitor (speed)
    min_delta: float = 0.0


@dataclass
class EvalConfig:
    n_per_pose: int = 1
    sampler: str = "ddim"             # "ddpm" or "ddim"; DDIM eta=0 adopted (sharper, multimodality kept)
    ddim_steps: int | None = None
    eta: float = 0.0
    guidance_scale: float = 1.0
    seeded: bool = True                # use the example at eval (LBE only)


@dataclass
class Config:
    name: str = "diffik_run"
    out_dir: str = "runs"
    seed: int = 0
    wandb: bool = False
    wandb_project: str = "diffik"
    wandb_entity: str = "jacketdembys"
    wandb_mode: str = "online"   # force online (override any WANDB_MODE=offline in the image)
    wandb_group: str = ""        # groups related runs in the UI (e.g. a sweep); "" = no group
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, d: dict):
    if d is None:
        return cls()
    valid = {f.name for f in fields(cls)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}")
    return cls(**d)


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    sub = {
        "data": _build(DataConfig, raw.pop("data", None)),
        "model": _build(ModelConfig, raw.pop("model", None)),
        "diffusion": _build(DiffusionConfig, raw.pop("diffusion", None)),
        "train": _build(TrainConfig, raw.pop("train", None)),
        "eval": _build(EvalConfig, raw.pop("eval", None)),
    }
    top_valid = {f.name for f in fields(Config)} - set(sub)
    unknown = set(raw) - top_valid
    if unknown:
        raise ValueError(f"unknown top-level config keys: {unknown}")
    return Config(**raw, **sub)
