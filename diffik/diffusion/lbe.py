"""Learning-by-Example diffusion with classifier-free guidance.

Trains one model that conditions on the query pose always, and on the example
(De, Qe) with random dropout. At inference it runs seedless (no example),
seeded (example, guidance_scale=1), or guided (guidance_scale>1 amplifies the
example's effect via classifier-free guidance).
"""
from __future__ import annotations

import torch

from .gaussian import GaussianDiffusion


class LBEDiffusion(GaussianDiffusion):
    def __init__(self, *args, p_example_dropout: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_example_dropout = p_example_dropout

    def loss(self, x0: torch.Tensor, pose: torch.Tensor, example: torch.Tensor):
        """Denoising (+FK) loss with random example-dropout for CFG."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        drop_mask = torch.rand(B, device=x0.device) < self.p_example_dropout
        pred = self.model(x_t, pose, t, example=example, drop_mask=drop_mask)
        return self._losses_from_pred(x0, x_t, t, noise, pred)

    @torch.no_grad()
    def sample(
        self,
        pose: torch.Tensor,
        example: torch.Tensor | None = None,
        n_per_pose: int = 1,
        generator: torch.Generator | None = None,
        sampler: str = "ddpm",
        ddim_steps: int | None = None,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample joints. example=None -> seedless; example given -> seeded.

        guidance_scale>1 applies classifier-free guidance toward the example.
        """
        self.model.eval()
        device = pose.device
        P = pose.shape[0]
        B = P * n_per_pose
        cond_pose = pose.repeat_interleave(n_per_pose, dim=0)
        ex = example.repeat_interleave(n_per_pose, dim=0) if example is not None else None
        x = self._randn((B, self.dof), device, generator)

        def eps_fn(xx, t):
            if ex is None:
                pred = self.model(xx, cond_pose, t, example=None)
                return self._eps_from_pred(xx, None, t, pred)
            if guidance_scale == 1.0:
                pred = self.model(xx, cond_pose, t, example=ex)
                return self._eps_from_pred(xx, None, t, pred)
            pred_c = self.model(xx, cond_pose, t, example=ex)
            pred_u = self.model(xx, cond_pose, t, example=None)
            eps_c = self._eps_from_pred(xx, None, t, pred_c)
            eps_u = self._eps_from_pred(xx, None, t, pred_u)
            return eps_u + guidance_scale * (eps_c - eps_u)

        if sampler == "ddpm":
            x = self._ddpm_sample(x, eps_fn, generator)
        elif sampler == "ddim":
            x = self._ddim_sample(x, eps_fn, generator, ddim_steps, eta)
        else:
            raise ValueError(f"unknown sampler '{sampler}'")
        return x.view(P, n_per_pose, self.dof)
