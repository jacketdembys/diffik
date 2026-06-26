"""A/B the sampler on an existing LBE checkpoint (inference-only, no retraining):
DDPM (stochastic) vs DDIM eta=0 (deterministic) vs zero-terminal-variance DDPM.
Reports best-of-K precision AND diversity for both regimes (seeded & seedless), so
we can confirm determinism sharpens precision without killing the multi-IK ability.

    WANDB_API_KEY=<key> python scripts/ab_samplers.py --run lbe_n6400_h768_l6_rmlp
"""
from __future__ import annotations

import argparse
import os

import torch
import wandb

from diffik.checkpoint import load_checkpoint
from diffik.data import build_datasets_lbe, generate_trajectory
from diffik.diffusion import LBEDiffusion, NoiseSchedule
from diffik.eval import evaluate
from diffik.models import LBEDenoiser
from diffik.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="jacketdembys")
    ap.add_argument("--project", default="diffik")
    ap.add_argument("--run", default="lbe_n6400_h768_l6_rmlp")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n_poses", type=int, default=256, help="test subset; <=0 = full test set")
    ap.add_argument("--ztv", type=int, default=50, help="zero-terminal-variance: # final deterministic steps")
    ap.add_argument("--regimes", default="seedless,seeded", help="comma list: seedless,seeded")
    ap.add_argument("--samplers", default="ddpm,ddim,ztv", help="comma list: ddpm,ddim,ztv")
    ap.add_argument("--wandb", action="store_true", help="log results to a wandb run")
    ap.add_argument("--wandb_group", default="ab_samplers")
    args = ap.parse_args()
    want_reg = set(args.regimes.split(","))
    want_smp = set(args.samplers.split(","))

    device = get_device()
    api = wandb.Api()
    art = api.artifact(f"{args.entity}/{args.project}/{args.run}-ckpt:latest")
    ckpt = os.path.join(art.download(), "checkpoint.pth")
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    dc, mc = cfg["data"], cfg["model"]

    ds = generate_trajectory(robot=dc["robot"], n_trajectories=dc["n_trajectories"],
                             steps_per_traj=dc["steps_per_traj"], v_deg=dc["v_deg"], seed=cfg["seed"])
    _, _, test, q_norm, _ = build_datasets_lbe(ds, v_deg=dc["v_deg"], v_mm=dc["v_mm"], seed=cfg["seed"])
    if args.n_poses > 0 and len(test) > args.n_poses:
        test = test.head(args.n_poses)
    pose_dim, dof = ds.pose.shape[1], ds.q.shape[1]

    wb = None
    if args.wandb:
        wb = wandb.init(project=args.project, entity=args.entity, group=args.wandb_group,
                        name=f"ab_{args.run}_{args.samplers}", mode="online",
                        config={"run": args.run, "K": args.K, "n_poses": len(test), "ztv": args.ztv})

    model = LBEDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc["hidden_dim"],
                        n_layers=mc["n_layers"], backbone=mc.get("backbone", "plain"))
    diff = LBEDiffusion(model, NoiseSchedule(T=cfg["diffusion"]["T"]), dof=dof)
    diff, q_norm, _, _ = load_checkpoint(ckpt, diff, map_location=device)

    samplers = [
        ("ddpm", "DDPM (stochastic)", {"sampler": "ddpm"}),
        ("ddim", "DDIM eta=0 (determ.)", {"sampler": "ddim", "eta": 0.0}),
        ("ztv", f"ZTV-{args.ztv} (determ. tail)", {"sampler": "ddpm", "ztv_last": args.ztv}),
    ]
    samplers = [(k, n, kw) for (k, n, kw) in samplers if k in want_smp]
    regimes = [(n, s) for (n, s) in [("seedless", False), ("seeded", True)] if n in want_reg]

    print(f"=== {args.run} | A/B samplers | K={args.K} | n_poses={len(test)} | device={device}\n")
    for rname, use_seed in regimes:
        ex = test.example.to(device) if (use_seed and test.example is not None) else None
        print(f"--- {rname} ---")
        print(f"{'sampler':<26}{'bestK_pos':>10}{'bestK_ori':>10}{'meanK_pos':>10}{'diversity':>11}")
        for _skey, sname, kw in samplers:
            g = torch.Generator().manual_seed(0)
            kwargs = dict(kw)
            if ex is not None:
                kwargs["example"] = ex
            res = evaluate(diff, test, q_norm, robot=dc["robot"], n_per_pose=args.K,
                           device=device, generator=g, **kwargs)
            print(f"{sname:<26}{res.best_of_n.pos_mm_avg:>10.2f}{res.best_of_n.ori_deg_avg:>10.2f}"
                  f"{res.mean.pos_mm_avg:>10.2f}{res.diversity:>11.4f}")
            if wb is not None:
                p = f"{rname}/{_skey}"
                wb.summary.update({
                    f"{p}/bestK_pos_mm": res.best_of_n.pos_mm_avg, f"{p}/bestK_ori_deg": res.best_of_n.ori_deg_avg,
                    f"{p}/meanK_pos_mm": res.mean.pos_mm_avg, f"{p}/worstK_pos_mm": res.worst_of_n.pos_mm_avg,
                    f"{p}/diversity": res.diversity,
                    f"{p}/bestK_pct_pos_le_1mm": res.best_of_n.pct_pos_le_1mm,
                    f"{p}/bestK_pct_ori_le_1deg": res.best_of_n.pct_ori_le_1deg,
                })
        print()
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
