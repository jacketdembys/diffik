"""Config-driven training + evaluation entry point (cluster-ready).

Usage:
    python scripts/train.py --config configs/panda_lbe_trajectory.yaml
    python scripts/train.py --config <cfg> --override train.epochs=50 data.n_trajectories=200

Saves everything needed to regenerate plots/tables WITHOUT rerunning:
    runs/<name>/config.json   - the resolved config
    runs/<name>/dataset.npz   - the generated dataset
    runs/<name>/checkpoint.pth - model weights + normalizers
    runs/<name>/history.json  - per-epoch loss (total/denoise/fk)
    runs/<name>/metrics.json  - final eval summaries
    runs/<name>/errors.npz    - per-pose position(mm)/orientation(deg) errors
    runs/experiments.csv      - one appended summary row per run (the registry)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from diffik.checkpoint import save_checkpoint
from diffik.config import Config, load_config
from diffik.data import build_datasets, build_datasets_lbe, generate_random, generate_trajectory, save_dataset
from diffik.diffusion import GaussianDiffusion, LBEDiffusion, NoiseSchedule
from diffik.eval import evaluate, evaluate_multimodality
from diffik.kinematics import get_robot
from diffik.models import LBEDenoiser, MLPDenoiser
from diffik.training import train_diffusion
from diffik.utils import get_device, set_seed

REGISTRY_FIELDS = [
    "name", "kind", "lbe", "model_type", "n_train", "n_test", "max_epochs", "stopped_epoch",
    "patience", "batch_size", "lr", "fk_loss_weight", "rot_weight", "T", "seeded", "n_params",
    "train_minutes", "pos_mm_avg", "pos_mm_max", "ori_deg_avg", "pct_pos_le_1mm",
    "pct_ori_le_1deg", "diversity", "ms_per_solution", "timestamp",
]


def build_dataset(dc):
    if dc.kind == "trajectory":
        return generate_trajectory(
            robot=dc.robot, n_trajectories=dc.n_trajectories, steps_per_traj=dc.steps_per_traj,
            v_deg=dc.v_deg, v_mm=dc.v_mm, seed=dc.seed,
        )
    return generate_random(robot=dc.robot, n_samples=dc.n_samples, seed=dc.seed)


def build_model(mc, pose_dim, dof):
    if mc.type == "lbe":
        return LBEDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc.hidden_dim,
                           n_layers=mc.n_layers, time_embed_dim=mc.time_embed_dim,
                           pose_embed_dim=mc.pose_embed_dim, example_embed_dim=mc.example_embed_dim,
                           backbone=mc.backbone, dropout=mc.dropout)
    return MLPDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc.hidden_dim, n_layers=mc.n_layers,
                       time_embed_dim=mc.time_embed_dim, pose_embed_dim=mc.pose_embed_dim)


def build_monitor(cfg, diffusion, val, q_norm, device):
    """No-arg callable returning the val metric to MINIMIZE for early stopping.

    val_pose: held-out position error (mm) via sampling on a capped val subset
    (metric-aligned). val_loss: cheap denoising+FK loss proxy.
    """
    if val is None or len(val) == 0:
        return None
    if cfg.train.early_stop_metric == "val_loss":
        from torch.utils.data import DataLoader
        from diffik.training import compute_val_loss
        loader = DataLoader(val, batch_size=256, shuffle=False)
        return lambda: compute_val_loss(diffusion, loader, device)

    # val_pose (default): sample on a capped subset, return position error (mm)
    val_sub = val.head(cfg.train.monitor_cap) if len(val) > cfg.train.monitor_cap else val
    ec = cfg.eval
    kw = dict(n_per_pose=1, sampler=ec.sampler, ddim_steps=ec.ddim_steps, eta=ec.eta)
    if cfg.model.type == "lbe":
        kw["guidance_scale"] = ec.guidance_scale
        if ec.seeded and val_sub.example is not None:
            kw["example"] = val_sub.example.to(device)

    def monitor():
        g = torch.Generator().manual_seed(123)
        r = evaluate(diffusion, val_sub, q_norm, robot=cfg.data.robot, device=device, generator=g, **kw)
        return r.best_of_n.pos_mm_avg

    return monitor


def compute_report_metrics(diffusion, test, q_norm, cfg, device, K=50, mm_cap=512):
    """For each regime (seeded & seedless for LBE; seedless only for MLP):
    - best-of-K accuracy + IROS-style ranges on the FULL test set (chunked evaluate)
    - multimodality shape (diversity/coverage) on a capped subset (valid_diversity is
      O(K^2) per pose, so a subset is used there only).
    Logged to wandb/metrics.json so the report is self-sufficient."""
    regimes = [("seeded", True), ("seedless", False)] if cfg.model.type == "lbe" else [("seedless", False)]
    test_sub = test.head(mm_cap) if len(test) > mm_cap else test
    ec = cfg.eval
    samp = dict(sampler=ec.sampler, ddim_steps=ec.ddim_steps, eta=ec.eta)  # report uses the chosen sampler

    out = {}
    for rname, use_seed in regimes:
        has_ex = getattr(test, "example", None) is not None
        # full-test best-of-K (accuracy + ranges)
        kw = dict(samp)
        if use_seed and has_ex:
            kw["example"] = test.example.to(device)
        g = torch.Generator().manual_seed(cfg.seed)
        res = evaluate(diffusion, test, q_norm, robot=cfg.data.robot, n_per_pose=K,
                       device=device, generator=g, **kw)
        s = res.best_of_n        # per-pose MIN over K
        mn = res.mean            # mean over K
        wr = res.worst_of_n      # per-pose MAX over K
        # multimodality shape on a subset
        kwm = dict(samp)
        if use_seed and test_sub.example is not None:
            kwm["example"] = test_sub.example.to(device)
        gm = torch.Generator().manual_seed(cfg.seed)
        mm = evaluate_multimodality(diffusion, test_sub, q_norm, robot=cfg.data.robot, K=K,
                                    device=device, generator=gm, tol_mm=10.0, tol_deg=5.0, **kwm)
        out[rname] = {
            "K": K, "n_test": len(test), "n_mm": len(test_sub),
            # min / mean / max over the K solutions per pose (averaged over poses)
            "bestK_pos_mm": s.pos_mm_avg, "bestK_ori_deg": s.ori_deg_avg,
            "meanK_pos_mm": mn.pos_mm_avg, "meanK_ori_deg": mn.ori_deg_avg,
            "worstK_pos_mm": wr.pos_mm_avg, "worstK_ori_deg": wr.ori_deg_avg,
            # best-of-K error-distribution buckets
            "bestK_pct_pos_le_1mm": s.pct_pos_le_1mm, "bestK_pct_pos_1_5mm": s.pct_pos_1_5mm,
            "bestK_pct_pos_5_10mm": s.pct_pos_5_10mm, "bestK_pct_pos_gt_10mm": s.pct_pos_gt_10mm,
            "bestK_pct_ori_le_1deg": s.pct_ori_le_1deg, "bestK_pct_ori_1_3deg": s.pct_ori_1_3deg,
            "bestK_pct_ori_gt_3deg": s.pct_ori_gt_3deg,
            "diversity_all": mm.diversity_all, "mean_valid_per_pose": mm.mean_valid_per_pose,
            "frac_poses_multi": mm.frac_poses_multi, "valid_diversity": mm.valid_diversity,
        }
    return out


def append_registry(path, row):
    """Append one summary row to the central CSV registry (creates header once)."""
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in REGISTRY_FIELDS})


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.train.device)
    run_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"[run] {run_dir} | device={device}")

    wb = None
    if cfg.wandb:
        import wandb as wb
        # mode=online passed explicitly overrides any WANDB_MODE=offline set in the image
        wb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.name,
                group=cfg.wandb_group or None, config=cfg.to_dict(), mode=cfg.wandb_mode)

    # --- data ---
    ds = build_dataset(cfg.data)
    save_dataset(ds, os.path.join(run_dir, "dataset.npz"))
    if cfg.data.lbe:
        train, val, test, q_norm, pose_norm = build_datasets_lbe(ds, v_deg=cfg.data.v_deg, v_mm=cfg.data.v_mm, seed=cfg.seed)
    else:
        train, val, test, q_norm, pose_norm = build_datasets(ds, seed=cfg.seed)
    pose_dim, dof = ds.pose.shape[1], ds.q.shape[1]
    print(f"[data] {cfg.data.kind} | train {len(train)} val {len(val)} test {len(test)} | dof {dof} pose {pose_dim}")

    # --- model + diffusion ---
    model = build_model(cfg.model, pose_dim, dof)
    schedule = NoiseSchedule(T=cfg.diffusion.T)
    chain = get_robot(cfg.data.robot)
    common = dict(dof=dof, chain=chain, q_norm=q_norm, fk_loss_weight=cfg.diffusion.fk_loss_weight,
                  rot_weight=cfg.diffusion.rot_weight, fk_weighting=cfg.diffusion.fk_weighting,
                  fk_weight_gamma=cfg.diffusion.fk_weight_gamma, fk_t_window=cfg.diffusion.fk_t_window,
                  prediction_type=cfg.diffusion.prediction_type)
    if cfg.model.type == "lbe":
        diffusion = LBEDiffusion(model, schedule, p_example_dropout=cfg.diffusion.p_example_dropout, **common)
    else:
        diffusion = GaussianDiffusion(model, schedule, **common)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.model.type} | {n_params/1e6:.2f}M params")

    # --- train (capture per-epoch history; periodic checkpoint) ---
    ckpt_path = os.path.join(run_dir, "checkpoint.pth")
    history = []

    def on_epoch(epoch, info):
        history.append({"epoch": epoch, **info})
        if wb is not None:
            wb.log(info, step=epoch)
        if cfg.train.checkpoint_every and (epoch + 1) % cfg.train.checkpoint_every == 0:
            save_checkpoint(ckpt_path, diffusion, q_norm, pose_norm, cfg.to_dict())

    monitor_fn = build_monitor(cfg, diffusion, val, q_norm, device)
    t0 = time.time()
    train_diffusion(diffusion, train, monitor_fn=monitor_fn, monitor_every=cfg.train.monitor_every,
                    epochs=cfg.train.epochs, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
                    device=device, log_every=max(cfg.train.epochs // 10, 1),
                    patience=cfg.train.patience, min_delta=cfg.train.min_delta, on_epoch=on_epoch)
    train_minutes = (time.time() - t0) / 60.0
    stopped_epoch = history[-1]["epoch"] if history else 0
    save_checkpoint(ckpt_path, diffusion, q_norm, pose_norm, cfg.to_dict())
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f)
    print(f"[train] done in {train_minutes:.1f} min | final loss {history[-1]['total']:.6f}")

    # --- eval (held-out test) ---
    ec = cfg.eval
    g = torch.Generator().manual_seed(cfg.seed)
    kw = dict(n_per_pose=ec.n_per_pose, sampler=ec.sampler, ddim_steps=ec.ddim_steps, eta=ec.eta)
    if cfg.model.type == "lbe":
        kw["guidance_scale"] = ec.guidance_scale
        if ec.seeded:
            kw["example"] = test.example.to(device)
    res = evaluate(diffusion, test, q_norm, robot=cfg.data.robot, device=device, generator=g, **kw)
    print(f"[eval] {res}")

    # --- richer report metrics (capped subset): best-of-K + multimodality, both regimes ---
    # evaluate_multimodality returns best-of-K accuracy AND diversity/coverage in one pass.
    report = compute_report_metrics(diffusion, test, q_norm, cfg, device, K=50, mm_cap=512)
    for rn, rm in report.items():
        print(f"[report:{rn}] best-of-{rm['K']} pos {rm['bestK_pos_mm']:.2f}mm ori {rm['bestK_ori_deg']:.2f}deg"
              f" | valid/pose {rm['mean_valid_per_pose']:.1f} | valid_div {rm['valid_diversity']:.3f}")

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"mean": res.mean.as_dict(), "best_of_n": res.best_of_n.as_dict(),
                   "diversity": res.diversity, "ms_per_solution": res.ms_per_solution,
                   "n_params": n_params, "n_train": len(train), "train_minutes": train_minutes,
                   "stopped_epoch": stopped_epoch, "max_epochs": cfg.train.epochs,
                   "report": report}, f, indent=2)
    np.savez(os.path.join(run_dir, "errors.npz"),
             pos_mm=res.pos_mm_per_pose, ori_deg=res.ori_deg_per_pose)

    # --- append to central registry ---
    s = res.best_of_n
    append_registry(os.path.join(cfg.out_dir, "experiments.csv"), {
        "name": cfg.name, "kind": cfg.data.kind, "lbe": cfg.data.lbe, "model_type": cfg.model.type,
        "n_train": len(train), "n_test": len(test), "max_epochs": cfg.train.epochs,
        "stopped_epoch": stopped_epoch, "patience": cfg.train.patience,
        "batch_size": cfg.train.batch_size, "lr": cfg.train.lr,
        "fk_loss_weight": cfg.diffusion.fk_loss_weight, "rot_weight": cfg.diffusion.rot_weight,
        "T": cfg.diffusion.T, "seeded": (cfg.model.type == "lbe" and ec.seeded),
        "n_params": n_params, "train_minutes": round(train_minutes, 2),
        "pos_mm_avg": s.pos_mm_avg, "pos_mm_max": s.pos_mm_max, "ori_deg_avg": s.ori_deg_avg,
        "pct_pos_le_1mm": s.pct_pos_le_1mm, "pct_ori_le_1deg": s.pct_ori_le_1deg,
        "diversity": res.diversity, "ms_per_solution": res.ms_per_solution,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    if wb is not None:
        summary = {"test/pos_mm_avg": res.mean.pos_mm_avg, "test/pos_mm_max": res.mean.pos_mm_max,
                   "test/ori_deg_avg": res.mean.ori_deg_avg, "test/pct_pos_le_1mm": res.mean.pct_pos_le_1mm,
                   "test/pct_ori_le_1deg": res.mean.pct_ori_le_1deg,
                   "n_params": n_params, "n_train": len(train), "stopped_epoch": stopped_epoch}
        for rn, rm in report.items():  # seeded / seedless best-of-K + multimodality
            for k, v in rm.items():
                summary[f"{rn}/{k}"] = v
        wb.summary.update(summary)
        art = wb.Artifact(f"{cfg.name}-ckpt", type="model")
        art.add_file(ckpt_path)
        wb.log_artifact(art)
        wb.finish()

    return {"metrics": res, "n_params": n_params, "train_minutes": train_minutes}


def apply_overrides(cfg, overrides):
    for ov in overrides:
        key, _, val = ov.partition("=")
        if "." in key:                       # section.field
            section, fieldname = key.split(".", 1)
            obj = getattr(cfg, section)
        else:                                # top-level field (out_dir, name, seed, ...)
            obj, fieldname = cfg, key
        cur = getattr(obj, fieldname)
        cast = type(cur) if cur is not None else str
        setattr(obj, fieldname, (val == "true") if cast is bool else cast(val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    run(cfg)


if __name__ == "__main__":
    main()
