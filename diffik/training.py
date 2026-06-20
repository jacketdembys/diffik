"""Training loop with early stopping on a pluggable validation monitor.

The monitor is a no-arg callable (closing over the model) returning a scalar to
MINIMIZE. Callers choose what to monitor: validation *pose error* (metric-aligned,
requires sampling) or validation *loss* (cheap proxy). We found val loss saturates
long before sampled pose accuracy, so pose-error is the correct early-stop signal.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .diffusion import GaussianDiffusion


def _batch_loss(diffusion, batch, device):
    x0 = batch["q"].to(device)
    pose = batch["pose"].to(device)
    if "example" in batch:
        return diffusion.loss(x0, pose, batch["example"].to(device)), x0.shape[0]
    return diffusion.loss(x0, pose), x0.shape[0]


@torch.no_grad()
def compute_val_loss(diffusion, val_loader, device, seed: int = 12345) -> float:
    """Mean validation loss on FIXED timesteps/noise (seed then restore RNG)."""
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    diffusion.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        (loss, _), bs = _batch_loss(diffusion, batch, device)
        total += loss.item() * bs
        n += bs
    diffusion.train()
    torch.set_rng_state(rng_state)
    return total / max(n, 1)


def _state_clone(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_diffusion(
    diffusion: GaussianDiffusion,
    dataset,
    val_dataset=None,
    monitor_fn=None,
    monitor_every: int = 1,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    log_every: int = 0,
    patience: int = 0,
    min_delta: float = 0.0,
    on_epoch=None,
) -> list[dict]:
    """Train; early-stop on ``monitor_fn`` (minimized) checked every ``monitor_every``
    epochs, with ``patience`` counted in CHECKS. Best-monitor weights are restored.

    If ``monitor_fn`` is None but ``val_dataset`` is given, falls back to val loss.
    """
    device = torch.device(device)
    diffusion.to(device)
    opt = torch.optim.AdamW(diffusion.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if monitor_fn is None and val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=max(batch_size, 256), shuffle=False)
        monitor_fn = lambda: compute_val_loss(diffusion, val_loader, device)

    history: list[dict] = []
    best, best_state, best_epoch, no_improve = float("inf"), None, -1, 0

    for epoch in range(epochs):
        diffusion.train()
        total, denoise_acc, fk_acc, n = 0.0, 0.0, 0.0, 0
        for batch in loader:
            (loss, info), bs = _batch_loss(diffusion, batch, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * bs
            denoise_acc += info["denoise"] * bs
            fk_acc += info["fk"] * bs
            n += bs
        rec = {"epoch": epoch, "total": total / n, "denoise": denoise_acc / n, "fk": fk_acc / n}

        do_check = monitor_fn is not None and (epoch % monitor_every == 0 or epoch == epochs - 1)
        if do_check:
            m = monitor_fn()
            rec["val"] = m
            if m < best - min_delta:
                best, best_state, best_epoch, no_improve = m, _state_clone(diffusion.model), epoch, 0
            else:
                no_improve += 1

        history.append(rec)
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            msg = f"  epoch {epoch:4d}  total {rec['total']:.6f}  fk {rec['fk']:.6f}"
            if "val" in rec:
                msg += f"  monitor {rec['val']:.4f}  (best {best:.4f}@{best_epoch})"
            print(msg)
        if on_epoch is not None:
            on_epoch(epoch, rec)

        if patience > 0 and monitor_fn is not None and no_improve >= patience:
            print(f"  early stop at epoch {epoch} (best monitor {best:.4f} @ epoch {best_epoch})")
            break

    if best_state is not None:
        diffusion.model.load_state_dict(best_state)
    return history
