"""Generate plots and tables from saved artifacts only (no retraining).

Reads runs/experiments.csv (+ per-run runs/<name>/errors.npz) and writes:
    runs/report/scaling_position.png     - position error vs training-set size
    runs/report/scaling_orientation.png  - orientation error vs size
    runs/report/summary_table.md / .csv  - the registry as a clean table
    runs/report/error_hist_<name>.png    - per-run error distributions (optional)

Safe to run anytime; it only consumes saved files.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scaling_plot(df, ycol, ylabel, out_png, bestk_col=None):
    plt.figure(figsize=(6, 4))
    for kind, g in df.groupby("kind"):
        g = g.sort_values("n_train")
        plt.plot(g["n_train"], g[ycol], "o-", label=f"{kind} (single)")
        if bestk_col and bestk_col in g and g[bestk_col].notna().any():
            gb = g.dropna(subset=[bestk_col])
            plt.plot(gb["n_train"], gb[bestk_col], "s--", label=f"{kind} (best-of-K)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("training samples")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs training-set size")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs_es")
    ap.add_argument("--hist", action="store_true", help="also plot per-run error histograms")
    args = ap.parse_args()

    reg = os.path.join(args.out_dir, "experiments.csv")
    if not os.path.exists(reg):
        raise SystemExit(f"no registry at {reg} -- run a sweep first")
    df = pd.read_csv(reg)
    rep = os.path.join(args.out_dir, "report")
    os.makedirs(rep, exist_ok=True)

    # merge best-of-K results if present (matched on run name)
    bestofk = os.path.join(args.out_dir, "bestofk.csv")
    if os.path.exists(bestofk):
        bk = pd.read_csv(bestofk)[["name", "K", "pos_bestk", "ori_bestk",
                                   "pct_pos_le_1mm_bestk", "pct_ori_le_1deg_bestk"]]
        df = df.merge(bk, on="name", how="left")

    # scaling curves (single + best-of-K overlay if available)
    scaling_plot(df, "pos_mm_avg", "position error (mm)", os.path.join(rep, "scaling_position.png"),
                 bestk_col="pos_bestk" if "pos_bestk" in df else None)
    scaling_plot(df, "ori_deg_avg", "orientation error (deg)", os.path.join(rep, "scaling_orientation.png"),
                 bestk_col="ori_bestk" if "ori_bestk" in df else None)

    # summary table (markdown + csv)
    cols = ["name", "kind", "n_train", "stopped_epoch",
            "pos_mm_avg", "pos_bestk", "ori_deg_avg", "ori_bestk",
            "pct_pos_le_1mm", "pct_pos_le_1mm_bestk",
            "pct_ori_le_1deg", "pct_ori_le_1deg_bestk", "train_minutes"]
    cols = [c for c in cols if c in df.columns]
    tbl = df[cols].sort_values(["kind", "n_train"]).round(3)
    tbl.to_csv(os.path.join(rep, "summary_table.csv"), index=False)
    with open(os.path.join(rep, "summary_table.md"), "w") as f:
        f.write(tbl.to_markdown(index=False))

    if args.hist:
        for name in df["name"]:
            p = os.path.join(args.out_dir, str(name), "errors.npz")
            if not os.path.exists(p):
                continue
            e = np.load(p)
            plt.figure(figsize=(6, 4))
            plt.hist(e["pos_mm"], bins=40, alpha=0.8)
            plt.axvline(1.0, color="r", ls="--", label="1 mm")
            plt.xlabel("position error (mm)"); plt.ylabel("count")
            plt.title(f"{name} (n={len(e['pos_mm'])})"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(rep, f"error_hist_{name}.png"), dpi=150)
            plt.close()

    print(f"wrote report -> {rep}")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
