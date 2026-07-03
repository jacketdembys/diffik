"""Full-test refinement eval + baselines (cluster). Loads a checkpoint, samples
K candidates per pose (diffusion seeded/seedless) OR random-init, runs damped-GN
refinement, and logs per-step best-of-K / mean / %sub-mm / diversity / distinct
sub-mm modes to wandb. Chunked over poses so the full test set fits.

    WANDB_API_KEY=<k> python scripts/refine_eval.py --run <ckpt> --init diffusion --regime seedless --steps 15
"""
from __future__ import annotations

import argparse
import os

import torch
import wandb

from diffik.checkpoint import load_checkpoint
from diffik.data import build_datasets_lbe, generate_trajectory
from diffik.diffusion import LBEDiffusion, NoiseSchedule
from diffik.models import LBEDenoiser
from diffik.kinematics import get_robot
from diffik.kinematics.dh import forward_kinematics
from diffik.kinematics.pose import pose_error
from diffik.kinematics.refine import dls_step
from diffik.kinematics.robots import PANDA_JOINT_LIMITS
from diffik.utils import get_device

FK = torch.float64
TARGETS = [0, 1, 2, 3, 5, 10, 15]


def distinct_modes(q_Kd, pos_K, tol_mm, joint_tol):
    valid = q_Kd[pos_K <= tol_mm]
    reps = []
    for v in valid:
        if all((v - r).norm() > joint_tol for r in reps):
            reps.append(v)
    return len(reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="jacketdembys"); ap.add_argument("--project", default="diffik")
    ap.add_argument("--run", default="lbe_n6400_h768_l6_rmlp_rw01scoff")
    ap.add_argument("--init", choices=["diffusion", "random"], default="diffusion")
    ap.add_argument("--regime", choices=["seeded", "seedless"], default="seeded")
    ap.add_argument("--K", type=int, default=20); ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--n_poses", type=int, default=0, help="0 = full test set")
    ap.add_argument("--chunk_samples", type=int, default=8192)
    ap.add_argument("--modes_cap", type=int, default=1024, help="poses used for distinct-mode counting")
    ap.add_argument("--joint_tol", type=float, default=0.1)
    ap.add_argument("--wandb", action="store_true"); ap.add_argument("--wandb_group", default="refine_eval")
    args = ap.parse_args()

    os.environ["WANDB_MODE"] = "online"
    device = get_device()
    api = wandb.Api()
    ckpt = os.path.join(api.artifact(f"{args.entity}/{args.project}/{args.run}-ckpt:latest").download(), "checkpoint.pth")
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    dc, mc = cfg["data"], cfg["model"]

    ds = generate_trajectory(robot=dc["robot"], n_trajectories=dc["n_trajectories"],
                             steps_per_traj=dc["steps_per_traj"], v_deg=dc["v_deg"], seed=cfg["seed"])
    _, _, test, q_norm, _ = build_datasets_lbe(ds, v_deg=dc["v_deg"], v_mm=dc["v_mm"], seed=cfg["seed"])
    if args.n_poses > 0 and len(test) > args.n_poses:
        test = test.head(args.n_poses)
    pose_dim, dof = ds.pose.shape[1], ds.q.shape[1]

    model = LBEDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc["hidden_dim"], n_layers=mc["n_layers"],
                        backbone=mc.get("backbone", "plain"), self_cond=mc.get("self_cond", False))
    diff = LBEDiffusion(model, NoiseSchedule(T=cfg["diffusion"]["T"]), dof=dof)
    diff, q_norm, _, _ = load_checkpoint(ckpt, diff, map_location=device)
    diff.to(device); diff.eval()
    chain = get_robot(dc["robot"], dtype=FK)
    lim = torch.tensor(PANDA_JOINT_LIMITS, dtype=FK)
    qmin, qmax = lim[:, 0], lim[:, 1]

    P, K = len(test), args.K
    steps_log = [s for s in TARGETS if s <= args.steps]
    # accumulators per logged step
    acc = {s: {"best_pos": 0.0, "best_ori": 0.0, "sum_pos": 0.0, "n_sol": 0,
               "poses_submm": 0, "div": 0.0, "modes": 0, "modes_poses": 0} for s in steps_log}
    step_poses = 0
    chunk = max(1, args.chunk_samples // K)
    for s0 in range(0, P, chunk):
        sl = slice(s0, min(s0 + chunk, P))
        c = sl.stop - sl.start
        q_true = q_norm.inverse_transform(test.q[sl].cpu()).to(FK)
        T_d = forward_kinematics(q_true, chain)
        T_d_cK = T_d.repeat_interleave(K, dim=0)
        if args.init == "random":
            qf = qmin + (qmax - qmin) * torch.rand(c * K, dof, dtype=FK)
        else:
            ex = None if args.regime == "seedless" else test.example[sl].to(device)
            g = torch.Generator().manual_seed(s0)
            samp = diff.sample(test.pose[sl].to(device), example=ex, n_per_pose=K,
                               sampler="ddim", eta=0.0, generator=g)
            qf = q_norm.inverse_transform(samp.cpu()).to(FK).reshape(c * K, dof)
        for s in range(0, args.steps + 1):
            if s > 0:
                qf = torch.clamp(dls_step(qf, T_d_cK, chain, args.lam), qmin, qmax)
            if s in steps_log:
                Tp = forward_kinematics(qf, chain)
                pos, ori = pose_error(Tp, T_d_cK)
                pos, ori = pos.reshape(c, K), ori.reshape(c, K)
                q_cKd = qf.reshape(c, K, dof)
                bi = pos.argmin(1); r = torch.arange(c)
                a = acc[s]
                a["best_pos"] += float(pos[r, bi].sum()); a["best_ori"] += float(ori[r, bi].sum())
                a["sum_pos"] += float(pos.sum()); a["n_sol"] += c * K
                a["poses_submm"] += int((pos.min(1).values <= 1.0).sum())
                a["div"] += float(q_cKd.std(dim=1, unbiased=False).mean()) * c
                if s0 < args.modes_cap:
                    for j in range(c):
                        a["modes"] += distinct_modes(q_cKd[j], pos[j], 1.0, args.joint_tol)
                        a["modes_poses"] += 1
        step_poses += c

    name = f"refine_{args.init}_{args.regime if args.init=='diffusion' else 'randinit'}"
    print(f"=== {args.run} | {name} | K={K} full_test={P} lam={args.lam} ===")
    wb = wandb.init(project=args.project, entity=args.entity, group=args.wandb_group,
                    name=f"{name}_{args.run}", mode="online",
                    config={"run": args.run, "init": args.init, "regime": args.regime, "K": K,
                            "steps": args.steps, "lam": args.lam, "n_poses": P}) if args.wandb else None
    for s in steps_log:
        a = acc[s]
        bp, bo = a["best_pos"] / P, a["best_ori"] / P
        meanp = a["sum_pos"] / a["n_sol"]; submm = 100.0 * a["poses_submm"] / P
        div = a["div"] / P; modes = a["modes"] / max(a["modes_poses"], 1)
        print(f"  step {s:>2}: best-of-K {bp:7.3f}mm ori {bo:6.3f}deg | mean {meanp:8.3f}mm | "
              f"%<=1mm {submm:5.1f} | div {div:.4f} | distinct sub-mm modes/pose {modes:.2f}")
        if wb is not None:
            wb.summary.update({f"step{s}/bestK_pos_mm": bp, f"step{s}/bestK_ori_deg": bo,
                               f"step{s}/mean_pos_mm": meanp, f"step{s}/pct_poses_submm": submm,
                               f"step{s}/diversity": div, f"step{s}/distinct_modes": modes})
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
