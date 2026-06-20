"""Checkpoint save/load: model weights + normalizers + config."""
from __future__ import annotations

import torch

from .data.dataset import Normalizer


def save_checkpoint(path, diffusion, q_norm: Normalizer, pose_norm: Normalizer, config_dict: dict):
    torch.save(
        {
            "model_state": diffusion.model.state_dict(),
            "q_norm": q_norm.state_dict(),
            "pose_norm": pose_norm.state_dict(),
            "config": config_dict,
        },
        path,
    )


def load_checkpoint(path, diffusion, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    diffusion.model.load_state_dict(ckpt["model_state"])
    q_norm = Normalizer.load_state_dict(ckpt["q_norm"])
    pose_norm = Normalizer.load_state_dict(ckpt["pose_norm"])
    return diffusion, q_norm, pose_norm, ckpt.get("config", {})
