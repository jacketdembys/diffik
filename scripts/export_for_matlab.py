"""Export a test split to CSV for the MATLAB numerical-IK baselines.

We export only JOINT vectors (convention-free):
  - q1..qN   : query/ground-truth joints (define the target via MATLAB's own FK)
  - qe1..qeN : example joints = the initial guess (the LBE seed)

The MATLAB driver (matlab/numerical_ik_diffik.m) reads this, runs SD/SVF/MX
seeded from qe toward FK(q), and writes the solved joints back. We then score
those solved joints with our own FK so numerical and DiffIK use one yardstick.

Uses the SAME dataset/seed/split as our DiffIK evaluation so the test poses match.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from diffik.data import add_examples, generate_trajectory, split_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="matlab/diffik_testset.csv")
    ap.add_argument("--robot", default="7DoF-7R-Panda", help="MATLAB getDH_rad robot name")
    ap.add_argument("--n_trajectories", type=int, default=100)
    ap.add_argument("--steps_per_traj", type=int, default=40)
    ap.add_argument("--v_deg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = generate_trajectory(
        n_trajectories=args.n_trajectories, steps_per_traj=args.steps_per_traj,
        v_deg=args.v_deg, seed=args.seed,
    )
    ex_q, _ = add_examples(ds, v_deg=args.v_deg, seed=args.seed)
    _, _, te = split_indices(ds, seed=args.seed)  # same split as DiffIK eval

    q = ds.q[te]          # [M, dof] query/GT joints (target)
    qe = ex_q[te]         # [M, dof] example joints (initial guess / seed)
    dof = q.shape[1]

    cols = {f"q{i+1}": q[:, i] for i in range(dof)}
    cols.update({f"qe{i+1}": qe[:, i] for i in range(dof)})
    df = pd.DataFrame(cols)
    df.to_csv(args.out, index=False)

    print(f"wrote {len(df)} test samples ({dof} DoF) -> {args.out}")
    print(f"robot for MATLAB: {args.robot}")
    print("columns:", list(df.columns))


if __name__ == "__main__":
    main()
