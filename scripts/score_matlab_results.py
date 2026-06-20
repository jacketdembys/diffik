"""Score the MATLAB numerical-IK solved joints with OUR FK + metrics.

Reads:
  - diffik_testset.csv           (q1..qN = target joints)
  - diffik_numerical_results.csv ({INV}_qsol1..N, {INV}_iters/solved/time)

For each inverse, runs our verified FK on the solved joints and on the target
joints and reports position(mm)/orientation(deg) via the same ErrorSummary used
for DiffIK -- so numerical baselines and DiffIK are directly comparable.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
import torch

from diffik.eval import summarize_errors
from diffik.kinematics import forward_kinematics, get_robot, pose_error

DT = torch.float64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="matlab/diffik_testset.csv")
    ap.add_argument("--results", default="matlab/diffik_numerical_results.csv")
    ap.add_argument("--robot", default="panda_7r")
    args = ap.parse_args()

    test = pd.read_csv(args.testset)
    res = pd.read_csv(args.results)

    dof = sum(c.startswith("q") and not c.startswith("qe") for c in test.columns)
    q_target = torch.tensor(test[[f"q{i+1}" for i in range(dof)]].to_numpy(), dtype=DT)

    chain = get_robot(args.robot, dtype=DT)
    T_target = forward_kinematics(q_target, chain)

    inverses = sorted({m.group(1) for c in res.columns if (m := re.match(r"([A-Za-z]+)_qsol1$", c))})
    print(f"robot={args.robot}  samples={len(test)}  inverses={inverses}\n")
    print(f"{'method':<8} {'pos avg(mm)':>11} {'pos max':>9} {'ori avg(deg)':>12} "
          f"{'<=1mm':>7} {'<=1deg':>7} {'solved%':>8} {'iters':>7}")

    for inv in inverses:
        qsol = torch.tensor(res[[f"{inv}_qsol{i+1}" for i in range(dof)]].to_numpy(), dtype=DT)
        T_pred = forward_kinematics(qsol, chain)
        pos_mm, ori_deg = pose_error(T_pred, T_target)
        s = summarize_errors(pos_mm, ori_deg)
        solved = res[f"{inv}_solved"].mean() * 100.0 if f"{inv}_solved" in res else float("nan")
        iters = res[f"{inv}_iters"].mean() if f"{inv}_iters" in res else float("nan")
        print(f"{inv:<8} {s.pos_mm_avg:>11.3f} {s.pos_mm_max:>9.2f} {s.ori_deg_avg:>12.3f} "
              f"{s.pct_pos_le_1mm:>6.1f}% {s.pct_ori_le_1deg:>6.1f}% {solved:>7.1f}% {iters:>7.1f}")


if __name__ == "__main__":
    main()
