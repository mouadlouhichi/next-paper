#!/usr/bin/env python3
"""Regenerate Figure 2 (round-6 fixes): HitRate@20 terminology + hatching.

Reads the released C1b five-seed summaries (v4b matched controls) and draws
the 1x4 panel bar figure (ML-1M / Amazon-Book x NDCG@20 / HitRate@20).
Reviewer-requested changes vs. the previous version:
  * y-axis and caption use HitRate@20 (the paper's formal metric; with one
    held-out test item it equals Recall@20) instead of "Recall@20";
  * every family also carries a distinct hatch pattern so the figure remains
    readable in grayscale and for colorblind readers;
  * error bars are +/- 1 SD over the five seeds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path(__file__).resolve().parent.parent / "results" / "journal_runs"
OUT = Path(__file__).resolve().parent.parent.parent / "paper_package" / "assets" / "figures"
OUT2 = Path(__file__).resolve().parent.parent.parent / "springer_latex" / "assets" / "figures"

FAMILIES = ["uniform", "additive-pref", "attention", "heuristic-pop",
            "valid-sim", "valid-linear", "shapley-mc", "loo-marginal"]
LABELS = ["Uniform", "Additive", "Attention", "Pop", "Valid-Sim", "Valid-Lin", "Shapley", "CoalGameRec"]
COLORS = ["#95a5a6", "#9b59b6", "#3498db", "#16a085", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
HATCHES = ["", "//", "\\\\", "xx", "..", "--", "++", "oo"]

DATASETS = [("ml1m", "MovieLens-1M"), ("amazon_books", "Amazon-Book")]
METRICS = [("NDCG@20", "NDCG@20"), ("HitRate@20", "HitRate@20")]


def load(ds_key: str) -> pd.DataFrame:
    p = RUNS / f"{ds_key}_lightgcn_v4b_matched_controls" / "tables" / "summary_by_seed_family.csv"
    df = pd.read_csv(p)
    g = df.groupby("family")[["NDCG@20", "HitRate@20"]].agg(["mean", "std"]).loc[FAMILIES]
    return g


def main():
    data = {k: load(k) for k, _ in DATASETS}
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
    x = range(len(FAMILIES))
    panel = 0
    for ds_key, ds_label in DATASETS:
        for met_key, met_label in METRICS:
            ax = axes[panel]
            g = data[ds_key]
            means = g[(met_key, "mean")].values
            sds = g[(met_key, "std")].values
            bars = ax.bar(x, means, yerr=sds, capsize=3, color=COLORS,
                          edgecolor="black", linewidth=0.6, width=0.78)
            for b, h in zip(bars, HATCHES):
                b.set_hatch(h)
            for xi, v in zip(x, means):
                ax.text(xi, means.max() * 1.18, f"{v:.4f}".lstrip("0"), rotation=90,
                        ha="center", va="bottom", fontsize=6.5)
            ax.set_xticks(list(x))
            ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{ds_label} — {met_label}", fontsize=10)
            ax.set_ylim(0, means.max() * 1.30)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            panel += 1
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, hatch=h, ec="black", lw=0.6)
               for c, h in zip(COLORS, HATCHES)]
    fig.legend(handles, LABELS, loc="upper center", ncol=8, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for outdir in (OUT, OUT2):
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "performance_ndcg_recall.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "performance_ndcg_recall.svg", bbox_inches="tight")
    print("WROTE", OUT / "performance_ndcg_recall.png")


if __name__ == "__main__":
    main()
