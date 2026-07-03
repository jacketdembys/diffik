"""Validate the refinement thesis WITHOUT retraining: take the diffusion's samples
and run a few analytic damped-least-squares (Gauss-Newton) IK steps toward the target
pose, using the geometric Jacobian. Measure whether error drops to sub-mm and whether
the K solutions stay distinct (diversity preserved).

    WANDB_API_KEY=<key> python scripts/newton_refine_validation.py --run lbe_n6400_h768_l6_rmlp_rw01scoff
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
from diffik.kinematics.robots import PANDA_JOINT_LIMITS
from diffik.utils import get_device

FK = torch.float64


def geo_jacobian(q, chain):
    """Analytic geometric Jacobian [N,6,dof] for a revolute DH chain."""
    _, frames = forward_kinematics(q, chain, return_all=True)   # frames: [N, dof+1, 4, 4]
    o_n = frames[:, -1, :3, 3]
    Jv, Jw = [], []
    for i in range(q.shape[1]):
        z = frames[:, i, :3, 2]          # joint (i+1) axis = z of frame i
        o = frames[:, i, :3, 3]
        Jv.append(torch.linalg.cross(z, o_n - o))
        Jw.append(z)
    return torch.cat([torch.stack(Jv, -1), torch.stack(Jw, -1)], dim=1)   # [N,6,dof]


def pose_error_vec(q, T_d, chain):
    """6D task-space error [p_d - p ; orientation error] -> [N,6]."""
    T = forward_kinematics(q, chain)
    p, R = T[:, :3, 3], T[:, :3, :3]
    p_d, R_d = T_d[:, :3, 3], T_d[:, :3, :3]
    Re = R_d @ R.transpose(1, 2)
    e_rot = 0.5 * torch.stack([Re[:, 2, 1] - Re[:, 1, 2],
                               Re[:, 0, 2] - Re[:, 2, 0],
                               Re[:, 1, 0] - Re[:, 0, 1]], dim=-1)
    return torch.cat([p_d - p, e_rot], dim=-1)


def dls_step(q, T_d, chain, lam):
    e = pose_error_vec(q, T_d, chain)          # [N,6]
    J = geo_jacobian(q, chain)                 # [N,6,dof]
    JJt = J @ J.transpose(1, 2)
    I = torch.eye(6, dtype=q.dtype, device=q.device).expand_as(JJt)
    y = torch.linalg.solve(JJt + lam * I, e.unsqueeze(-1))
    return q + (J.transpose(1, 2) @ y).squeeze(-1)


def report(tag, q_PKd, T_d_PK, chain, K):
    P = q_PKd.shape[0]
    T_pred = forward_kinematics(q_PKd.reshape(P * K, -1), chain)
    pos, ori = pose_error(T_pred, T_d_PK)
    pos, ori = pos.reshape(P, K), ori.reshape(P, K)
    bi = pos.argmin(1); r = torch.arange(P)
    div = float(q_PKd.std(dim=1, unbiased=False).mean())
    print(f"  {tag:<11} best-of-K pos {pos[r,bi].mean():7.3f}mm ori {ori[r,bi].mean():6.3f}deg | "
          f"mean {pos.mean():7.3f}mm | %<=1mm {float((pos.min(1).values<=1).float().mean())*100:5.1f} | div {div:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="jacketdembys"); ap.add_argument("--project", default="diffik")
    ap.add_argument("--run", default="lbe_n6400_h768_l6_rmlp_rw01scoff")
    ap.add_argument("--K", type=int, default=20); ap.add_argument("--n_poses", type=int, default=256)
    ap.add_argument("--steps", type=int, default=10); ap.add_argument("--lam", type=float, default=1e-3)
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
    test = test.head(args.n_poses) if len(test) > args.n_poses else test
    pose_dim, dof = ds.pose.shape[1], ds.q.shape[1]

    model = LBEDenoiser(dof=dof, pose_dim=pose_dim, hidden_dim=mc["hidden_dim"], n_layers=mc["n_layers"],
                        backbone=mc.get("backbone", "plain"), self_cond=mc.get("self_cond", False))
    diff = LBEDiffusion(model, NoiseSchedule(T=cfg["diffusion"]["T"]), dof=dof)
    diff, q_norm, _, _ = load_checkpoint(ckpt, diff, map_location=device)
    diff.to(device); diff.eval()

    chain = get_robot(dc["robot"], dtype=FK)
    P, K = len(test), args.K
    ex = test.example.to(device)
    g = torch.Generator().manual_seed(0)
    samp = diff.sample(test.pose.to(device), example=ex, n_per_pose=K, sampler="ddim", eta=0.0, generator=g)
    q = q_norm.inverse_transform(samp.cpu()).to(FK)                      # [P,K,dof]

    q_true = q_norm.inverse_transform(test.q.cpu()).to(FK)
    T_d = forward_kinematics(q_true, chain)                              # [P,4,4] targets
    T_d_PK = T_d.repeat_interleave(K, dim=0)
    lim = torch.tensor(PANDA_JOINT_LIMITS, dtype=FK)

    print(f"=== {args.run} | Newton refine | K={K} n_poses={P} lam={args.lam} ===")
    report("step 0", q, T_d_PK, chain, K)
    qf = q.reshape(P * K, dof)
    for s in range(1, args.steps + 1):
        qf = dls_step(qf, T_d_PK, chain, args.lam)
        qf = torch.clamp(qf, lim[:, 0], lim[:, 1])
        if s in (1, 2, 3, 5, 10):
            report(f"step {s}", qf.reshape(P, K, dof), T_d_PK, chain, K)


if __name__ == "__main__":
    main()
