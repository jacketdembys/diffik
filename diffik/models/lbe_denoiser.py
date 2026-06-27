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
    """Faithful to ResMLPSum's ResidualBlockSum (identity shortcut at hidden==hidden):
    fc1 -> ReLU -> fc2, + x, -> ReLU."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.drop(self.fc2(self.relu(self.fc1(x))))
        return self.relu(out + x)


class DenseBackbone(nn.Module):
    """Faithful to DenseMLP: each block is fc -> ReLU -> fc; its output is
    concatenated with all previous block outputs, then a transition Linear maps
    back to ``dim`` (per-block FC layers, unlike the original's shared self.fc)."""

    def __init__(self, dim: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.fc2 = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.trans = nn.ModuleList(nn.Linear((i + 2) * dim, dim) for i in range(n_layers))

    def forward(self, x):
        feats = [x]
        prev = x
        for i in range(len(self.fc1)):
            h = self.drop(self.fc2[i](self.relu(self.fc1[i](prev))))
            feats.append(h)
            prev = self.trans[i](torch.cat(feats, dim=-1))
        return prev


def make_backbone(kind: str, hidden_dim: int, n_layers: int, dropout: float = 0.0) -> nn.Module:
    if kind == "plain":
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        return nn.Sequential(*layers)
    if kind == "rmlp":
        return nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)])
    if kind == "dmlp":
        return DenseBackbone(hidden_dim, n_layers, dropout)
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
        self_cond: bool = False,
    ):
        super().__init__()
        self.dof = dof
        self.pose_dim = pose_dim
        self.time_embed_dim = time_embed_dim
        self.example_dim = pose_dim + dof  # (De, Qe)
        self.backbone_kind = backbone
        self.self_cond = self_cond

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

        in_dim = dof + pose_embed_dim + time_embed_dim + example_embed_dim + (dof if self_cond else 0)
        self.input_proj = nn.Linear(in_dim, hidden_dim)   # bare Linear, like ResMLPSum/DenseMLP
        self.backbone = make_backbone(backbone, hidden_dim, n_layers, dropout)
        self.output_proj = nn.Linear(hidden_dim, dof)

    def forward(self, x_t, pose, t, example=None, drop_mask=None, x_self=None):
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
        parts = [x_t, pe, te, ee]
        if self.self_cond:                                    # previous x0_hat estimate (zeros if none)
            parts.append(x_self if x_self is not None else torch.zeros_like(x_t))
        h = self.input_proj(torch.cat(parts, dim=-1))
        return self.output_proj(self.backbone(h))
