"""Pull DiffIK results from wandb and rebuild the report (no local run dirs needed).

Each training job logs to its wandb run summary: single-sample test metrics, and —
for both regimes — best-of-K accuracy + multimodality (keys seeded/* and seedless/*).
This fetches every run via the wandb API and writes a consolidated CSV + scaling plots.

    python scripts/pull_wandb.py --entity jacketdembys --project diffik --out report_wandb
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

SUMMARY_KEYS = [
    "n_train", "n_params", "stopped_epoch",
    "test/pos_mm_avg", "test/pos_mm_max", "test/ori_deg_avg",
    "test/pct_pos_le_1mm", "test/pct_ori_le_1deg",
    "seeded/bestK_pos_mm", "seeded/bestK_ori_deg", "seeded/bestK_pct_pos_le_1mm",
    "seeded/diversity_all", "seeded/mean_valid_per_pose", "seeded/frac_poses_multi", "seeded/valid_diversity",
    "seedless/bestK_pos_mm", "seedless/bestK_ori_deg", "seedless/bestK_pct_pos_le_1mm",
    "seedless/diversity_all", "seedless/mean_valid_per_pose", "seedless/frac_poses_multi", "seedless/valid_diversity",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="jacketdembys")
    ap.add_argument("--project", default="diffik")
    ap.add_argument("--out", default="report_wandb")
    ap.add_argument("--filter", default="", help="substring filter on run name")
    args = ap.parse_args()

    api = wandb.Api()
    rows = []
    for r in api.runs(f"{args.entity}/{args.project}"):
        if args.filter and args.filter not in r.name:
            continue
        row = {"name": r.name, "state": r.state}
        for k in SUMMARY_KEYS:
            row[k.replace("/", "_")] = r.summary.get(k)
        rows.append(row)

    if not rows:
        raise SystemExit("no runs found")
    df = pd.DataFrame(rows).sort_values("n_train")
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "wandb_runs.csv"), index=False)

    # scaling plot: single vs seeded best-of-K vs seedless best-of-K
    d = df.dropna(subset=["n_train"]).sort_values("n_train")
    plt.figure(figsize=(6, 4))
    for col, lbl, st in [("test_pos_mm_avg", "single (seeded)", "o-"),
                         ("seeded_bestK_pos_mm", "seeded best-of-K", "s--"),
                         ("seedless_bestK_pos_mm", "seedless best-of-K", "^:")]:
        if col in d and d[col].notna().any():
            dd = d.dropna(subset=[col])
            plt.plot(dd["n_train"], dd[col], st, label=lbl)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("training samples"); plt.ylabel("position error (mm)")
    plt.title("DiffIK scaling (from wandb)"); plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "scaling_position.png"), dpi=150); plt.close()

    print(df.to_string(index=False))
    print(f"\nwrote {args.out}/wandb_runs.csv + scaling_position.png")


if __name__ == "__main__":
    main()
