"""LBE-aware denoiser with a selectable MLP backbone (plain / rmlp / dmlp).

Conditions on the query pose (always) and an example (De, Qe) with a learned
null embedding for classifier-free dropout. The backbone that maps the conditioned
hidden vector to the noise prediction can be:
  - plain : stacked FC + SiLU (the Phase-3 baseline)
  - rmlp  : residual blocks (ResNet-style summation skips) -- best in the IROS papers
  - dmlp  : dense block (DenseNet-style concatenation skips + transition)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .embeddings import sinusoidal_embedding


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.act(x + self.drop(self.fc2(self.act(self.fc1(x)))))


class DenseBlock(nn.Module):
    """n FC layers, each fed the concatenation of all previous outputs; a
    transition layer reduces the concatenation back to ``dim``."""

    def __init__(self, dim: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        in_dim = dim
        for _ in range(n_layers):
            self.layers.append(nn.Linear(in_dim, dim))
            in_dim += dim
        self.transition = nn.Linear(in_dim, dim)

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(self.drop(self.act(layer(torch.cat(feats, dim=-1)))))
        return self.act(self.transition(torch.cat(feats, dim=-1)))


def make_backbone(kind: str, hidden_dim: int, n_layers: int, dropout: float = 0.0) -> nn.Module:
    if kind == "plain":
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        return nn.Sequential(*layers)
    if kind == "rmlp":
        return nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)])
    if kind == "dmlp":
        return DenseBlock(hidden_dim, n_layers, dropout)
    raise ValueError(f"unknown backbone '{kind}'")


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
        backbone: str = "plain",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dof = dof
        self.pose_dim = pose_dim
        self.time_embed_dim = time_embed_dim
        self.example_dim = pose_dim + dof  # (De, Qe)
        self.backbone_kind = backbone

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
        self.null_example = nn.Parameter(torch.randn(example_embed_dim) * 0.02)

        in_dim = dof + pose_embed_dim + time_embed_dim + example_embed_dim
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        self.backbone = make_backbone(backbone, hidden_dim, n_layers, dropout)
        self.output_proj = nn.Linear(hidden_dim, dof)

    def forward(self, x_t, pose, t, example=None, drop_mask=None):
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
        h = self.input_proj(torch.cat([x_t, pe, te, ee], dim=-1))
        return self.output_proj(self.backbone(h))
