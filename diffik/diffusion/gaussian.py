"""Gaussian (DDPM) diffusion: training loss and ancestral sampling.

Phase 3 baseline: epsilon-prediction denoising loss and standard ancestral DDPM
sampling, conditioned on the pose. No FK loss yet (added in Phase 5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kinematics import forward_kinematics
from .schedule import NoiseSchedule


class GaussianDiffusion(nn.Module):
    """Conditional Gaussian diffusion with an optional differentiable-FK loss.

    The base (Phase 3) behaviour is epsilon-prediction with a denoising MSE.
    Setting ``fk_loss_weight > 0`` (and providing ``chain`` + ``q_norm``) adds the
    Phase 5 FK loss: recover the clean joint estimate x0_hat, denormalize to joint
    space, run it through the differentiable FK, and penalize the pose error
    against the target pose (= FK of the ground-truth joints). The FK term is
    weighted per timestep (alpha_bar_t) so it dominates at low-noise steps where
    x0_hat is meaningful and fades at high-noise steps where it is garbage.
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: NoiseSchedule,
        dof: int = 7,
        chain=None,
        q_norm=None,
        fk_loss_weight: float = 0.0,
        fk_weighting: str = "alpha_bar",
        fk_weight_gamma: float = 1.0,
        fk_t_window: int = 0,
        rot_weight: float = 0.1,
        prediction_type: str = "eps",
        self_cond_clamp: float = 4.0,
    ):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.dof = dof
        self.chain = chain
        self.q_norm = q_norm
        self.fk_loss_weight = fk_loss_weight
        self.fk_weighting = fk_weighting
        self.fk_weight_gamma = fk_weight_gamma
        self.fk_t_window = fk_t_window
        self.rot_weight = rot_weight
        self.prediction_type = prediction_type
        self.self_cond_clamp = self_cond_clamp   # bound x0_hat fed back as x_self (norm space ~N(0,1))

    @property
    def T(self) -> int:
        return self.schedule.T

    @staticmethod
    def _randn(shape, device, generator):
        """Device-portable seeded noise (a CPU generator can't drive MPS/CUDA
        randn directly, so we draw on the generator's device then move)."""
        if generator is None:
            return torch.randn(*shape, device=device)
        return torch.randn(*shape, device=generator.device, generator=generator).to(device)

    @property
    def use_fk_loss(self) -> bool:
        return self.fk_loss_weight > 0.0 and self.chain is not None and self.q_norm is not None

    def _clamp_self(self, x0):
        """Clamp x0_hat before reusing it as the self-conditioning input: eps-pred
        x0_hat explodes at high-noise t, which destabilizes self-conditioning."""
        c = self.self_cond_clamp
        return x0.clamp(-c, c) if c and c > 0 else x0

    def _x0_hat(self, x_t, pose, t, pred):
        if self.prediction_type == "eps":
            return self.schedule.predict_x0_from_eps(x_t, t, pred)
        if self.prediction_type == "x0":
            return pred
        raise ValueError(self.prediction_type)

    def _fk_loss(self, x0_hat_n, x0_n, t):
        """Weighted pose-reconstruction loss between FK(x0_hat) and FK(x0)."""
        q_hat = self.q_norm.inverse_transform(x0_hat_n)
        q_true = self.q_norm.inverse_transform(x0_n)
        T_pred = forward_kinematics(q_hat, self.chain)
        T_tgt = forward_kinematics(q_true, self.chain)

        pos_l = ((T_pred[:, :3, 3] - T_tgt[:, :3, 3]) ** 2).sum(dim=-1)          # [B] m^2
        rot_l = ((T_pred[:, :3, :3] - T_tgt[:, :3, :3]) ** 2).sum(dim=(-1, -2))  # [B]
        fk_per = pos_l + self.rot_weight * rot_l

        w = self._fk_weight(t, fk_per)
        return (w * fk_per).sum() / (w.sum() + 1e-8)

    def _fk_weight(self, t, fk_per):
        """Per-sample weight that concentrates the FK loss on low-noise timesteps,
        where x0_hat is close enough to the manifold for FK to sharpen the endpoint:
          none          : uniform
          alpha_bar     : w = ab_t                       (gentle)
          alpha_bar_pow : w = ab_t ** gamma              (gamma>1 -> sharper low-noise focus)
          snr           : w = min(ab_t/(1-ab_t), gamma)  (min-SNR-clamped; strong low-noise focus)
          low_t_window  : w = 1 for t < fk_t_window else 0 (hard: only the lowest-noise steps)
        """
        kind = self.fk_weighting
        if kind == "none":
            return torch.ones_like(fk_per)
        ab = self.schedule.alpha_bar[t]
        if kind == "alpha_bar":
            return ab
        if kind == "alpha_bar_pow":
            return ab ** self.fk_weight_gamma
        if kind == "snr":
            snr = ab / (1.0 - ab + 1e-8)
            return torch.clamp(snr, max=self.fk_weight_gamma) if self.fk_weight_gamma > 0 else snr
        if kind == "low_t_window":
            win = self.fk_t_window if self.fk_t_window > 0 else max(1, self.T // 10)
            return (t < win).to(fk_per.dtype)
        raise ValueError(f"unknown fk_weighting '{kind}'")

    def _losses_from_pred(self, x0, x_t, t, noise, pred):
        """Denoising (+ optional FK) loss given a model prediction. Shared by the
        base diffusion and the LBE subclass."""
        if self.prediction_type == "eps":
            denoise = F.mse_loss(pred, noise)
        else:  # x0-prediction
            denoise = F.mse_loss(pred, x0)

        total = denoise
        fk = torch.zeros((), device=x0.device)
        if self.use_fk_loss:
            x0_hat = self._x0_hat(x_t, None, t, pred)
            fk = self._fk_loss(x0_hat, x0, t)
            total = denoise + self.fk_loss_weight * fk
        return total, {"denoise": float(denoise.detach()), "fk": float(fk.detach())}

    def loss(self, x0: torch.Tensor, pose: torch.Tensor):
        """Return (total_loss, info_dict) for a random timestep per sample."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        pred = self.model(x_t, pose, t)
        return self._losses_from_pred(x0, x_t, t, noise, pred)

    def _eps_from_pred(self, x_t, pose, t, pred=None):
        """Return the noise prediction (handles eps- or x0-parameterization)."""
        if pred is None:
            pred = self.model(x_t, pose, t)
        if self.prediction_type == "eps":
            return pred
        # x0-prediction -> eps = (x_t - sqrt(ab) x0) / sqrt(1-ab)
        sab = self.schedule.sqrt_alpha_bar[t].unsqueeze(-1)
        somab = self.schedule.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return (x_t - sab * pred) / somab

    @torch.no_grad()
    def sample(
        self,
        pose: torch.Tensor,
        n_per_pose: int = 1,
        generator: torch.Generator | None = None,
        sampler: str = "ddpm",
        ddim_steps: int | None = None,
        eta: float = 0.0,
        ztv_last: int = 0,
    ) -> torch.Tensor:
        """Sample joint configurations conditioned on ``pose``.

        Args:
            pose: [P, pose_dim] conditioning poses (normalized).
            n_per_pose: samples per pose (for multimodality).
            sampler: "ddpm" (ancestral, stochastic) or "ddim".
            ddim_steps: number of DDIM steps (<= T); None uses all T.
            eta: DDIM stochasticity (0 = deterministic / zero-terminal-variance).
        Returns:
            x0 samples [P, n_per_pose, dof] (normalized joint space).
        """
        self.model.eval()
        device = pose.device
        P = pose.shape[0]
        B = P * n_per_pose
        cond = pose.repeat_interleave(n_per_pose, dim=0)
        x = self._randn((B, self.dof), device, generator)
        eps_fn = lambda xx, t, x_self=None: self._eps_from_pred(xx, cond, t)  # base MLP: no self-cond

        if sampler == "ddpm":
            x = self._ddpm_sample(x, eps_fn, generator, ztv_last=ztv_last)
        elif sampler == "ddim":
            x = self._ddim_sample(x, eps_fn, generator, ddim_steps, eta)
        else:
            raise ValueError(f"unknown sampler '{sampler}'")
        return x.view(P, n_per_pose, self.dof)

    def _ddpm_sample(self, x, eps_fn, generator, ztv_last: int = 0):
        """eps_fn(x, t_tensor) -> predicted noise. Lets subclasses inject CFG.

        ztv_last>0 = zero terminal variance: stop injecting noise for the final
        ``ztv_last`` low-noise steps (t_ < ztv_last become deterministic), keeping
        early stochasticity (mode diversity) but polishing the endpoint."""
        sch = self.schedule
        B = x.shape[0]
        sc = getattr(self.model, "self_cond", False)
        x_self = None
        for t_ in reversed(range(self.T)):
            t = torch.full((B,), t_, device=x.device, dtype=torch.long)
            eps = eps_fn(x, t, x_self)
            if sc:
                x_self = self._clamp_self(sch.predict_x0_from_eps(x, t, eps))
            beta_t, alpha_t, ab_t = sch.betas[t_], sch.alphas[t_], sch.alpha_bar[t_]
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - ab_t)) * eps)
            if t_ > 0 and t_ >= ztv_last:
                z = self._randn((B, self.dof), x.device, generator)
                x = mean + torch.sqrt(beta_t) * z
            else:
                x = mean
        return x

    def _ddim_sample(self, x, eps_fn, generator, ddim_steps, eta):
        sch = self.schedule
        B = x.shape[0]
        if ddim_steps is None or ddim_steps >= self.T:
            seq = list(range(self.T))
        else:
            seq = torch.linspace(0, self.T - 1, ddim_steps).round().long().unique().tolist()

        sc = getattr(self.model, "self_cond", False)
        x_self = None
        for i in reversed(range(len(seq))):
            t_cur = seq[i]
            t_prev = seq[i - 1] if i > 0 else -1
            t = torch.full((B,), t_cur, device=x.device, dtype=torch.long)
            ab_t = sch.alpha_bar[t_cur]
            ab_prev = sch.alpha_bar[t_prev] if t_prev >= 0 else torch.ones_like(ab_t)

            eps = eps_fn(x, t, x_self)
            x0_hat = (x - torch.sqrt(1.0 - ab_t) * eps) / torch.sqrt(ab_t)
            if sc:
                x_self = self._clamp_self(x0_hat)

            if eta > 0 and t_prev >= 0:
                sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
            else:
                sigma = torch.zeros_like(ab_t)
            c = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0))
            x = torch.sqrt(ab_prev) * x0_hat + c * eps
            if eta > 0 and t_prev >= 0:
                z = self._randn((B, self.dof), x.device, generator)
                x = x + sigma * z
        return x
