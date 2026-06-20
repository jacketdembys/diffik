"""LBE-aware denoiser: conditions on the query pose AND an example (De, Qe).

The example is the Learning-by-Example signal -- a (pose, joints) tuple, e.g. the
previous waypoint in a trajectory. It is embedded separately and can be *dropped*
(replaced by a learned null embedding) for classifier-free guidance: training
with random example-dropout yields one model that runs both seedless (no example)
and seeded (with example), and supports guidance scaling at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .embeddings import sinusoidal_embedding


class LBEDenoiser(nn.Module):
    def __init__(
        self,
        dof: int = 7,
        pose_dim: int = 6,
        hidden_dim: int = 512,
        time_embed_dim: int = 128,
        pose_embed_dim: int = 128,
        example_embed_dim: int = 128,
        n_layers: int = 4,
    ):
        super().__init__()
        self.dof = dof
        self.pose_dim = pose_dim
        self.time_embed_dim = time_embed_dim
        self.example_dim = pose_dim + dof  # (De, Qe)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, pose_embed_dim), nn.SiLU(),
            nn.Linear(pose_embed_dim, pose_embed_dim),
        )
        self.example_mlp = nn.Sequential(
            nn.Linear(self.example_dim, example_embed_dim), nn.SiLU(),
            nn.Linear(example_embed_dim, example_embed_dim),
        )
        # learned embedding used when the example is absent / dropped
        self.null_example = nn.Parameter(torch.randn(example_embed_dim) * 0.02)

        in_dim = dof + pose_embed_dim + time_embed_dim + example_embed_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, dof)]
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        pose: torch.Tensor,
        t: torch.Tensor,
        example: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x_t:[B,dof], pose:[B,pose_dim], t:[B], example:[B,pose_dim+dof] or None.

        drop_mask:[B] bool -> True rows use the null example (CFG dropout).
        """
        B = x_t.shape[0]
        te = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))
        pe = self.pose_mlp(pose)

        if example is None:
            ee = self.null_example.unsqueeze(0).expand(B, -1)
        else:
            ee = self.example_mlp(example)
            if drop_mask is not None:
                null = self.null_example.unsqueeze(0).expand(B, -1)
                ee = torch.where(drop_mask.unsqueeze(-1), null, ee)

        h = torch.cat([x_t, pe, te, ee], dim=-1)
        return self.net(h)
