"""DDPM noise schedule and forward (q-sample) process.

Standard DDPM with a linear beta schedule. All schedule tensors are kept as
buffers on a small ``nn.Module`` so they move with ``.to(device)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


class NoiseSchedule(nn.Module):
    """Holds betas/alphas/alpha_bar and the forward noising operation."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.T = T
        betas = make_beta_schedule(T, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward noising: x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps."""
        sab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        somab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sab * x0 + somab * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Recover x0 estimate from a noise prediction."""
        sab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        somab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return (x_t - somab * eps) / sab
