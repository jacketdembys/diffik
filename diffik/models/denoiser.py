"""Pose-conditioned MLP denoiser (epsilon prediction).

Baseline architecture for Phase 3: embeds the timestep (sinusoidal) and the
conditioning pose, concatenates them with the noisy joint vector, and predicts
the noise. Deliberately simple; residual/precision variants come later.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .embeddings import sinusoidal_embedding


class MLPDenoiser(nn.Module):
    def __init__(
        self,
        dof: int = 7,
        pose_dim: int = 6,
        hidden_dim: int = 512,
        time_embed_dim: int = 128,
        pose_embed_dim: int = 128,
        n_layers: int = 4,
    ):
        super().__init__()
        self.dof = dof
        self.pose_dim = pose_dim
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, pose_embed_dim),
            nn.SiLU(),
            nn.Linear(pose_embed_dim, pose_embed_dim),
        )

        in_dim = dof + pose_embed_dim + time_embed_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, dof)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, pose: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t: [B, dof], pose: [B, pose_dim], t: [B] (long) -> eps_pred [B, dof]."""
        te = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))
        pe = self.pose_mlp(pose)
        h = torch.cat([x_t, pe, te], dim=-1)
        return self.net(h)
