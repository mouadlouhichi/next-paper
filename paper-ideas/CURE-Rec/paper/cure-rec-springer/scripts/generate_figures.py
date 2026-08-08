"""Generate the CURE-Rec manuscript figures from checksumed result tables.

All values are read from the reproducibility snapshot.  The script deliberately
uses a restrained black/grey Matplotlib style so the figures remain legible in a
Springer PDF, in greyscale printing, and during referee review.

Run from any directory with:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


PAPER = Path(__file__).resolve().parents[1]
CURE_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = CURE_ROOT / "results" / "reproducibility_snapshot_latest"
FIGURES = PAPER / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLACK = "#1a1a1a"
DARK = "#555555"
MID = "#9a9a9a"
LIGHT = "#d9d9d9"
WHITE = "#ffffff"


def read_csv(name: str) -> list[dict[str, str]]:
    with (SNAPSHOT / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d0d0d0", linestyle="--", linewidth=0.55, zorder=0)


def controlled_recovery() -> None:
    rows = read_csv("regime_selection_summary.csv")
    labels = [
        "Add.", "Comp.", "Red.", "Antag.", "Delay-S",
        "Delay-L", "Repair-B", "Repair-R", "Misspec.",
    ]
    oracle = [float(row["oracle_jaccard"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.3, 3.55))
    x = np.arange(len(labels))
    colors = [LIGHT] * (len(labels) - 1) + [BLACK]
    bars = ax.bar(x, oracle, color=colors, edgecolor=BLACK, linewidth=0.7, zorder=3)
    for bar, val in zip(bars, oracle, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.035, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Oracle portfolio Jaccard")
    clean(ax)
    save(fig, "figure_02_controlled_recovery")


def oat_sensitivity() -> None:
    rows = read_csv("calibration_oat_summary.csv")
    wanted = [
        "baseline", "oat-repeat_threshold-2", "oat-repeat_threshold-4", "oat-horizon-8",
        "oat-horizon-16", "oat-provider_threshold-0p2400", "oat-provider_threshold-0p3200",
        "oat-novelty_delayed_benefit-0p0300", "oat-novelty_delayed_benefit-0p0600",
    ]
    lookup = {row["point_id"]: row for row in rows}
    selected = [lookup[key] for key in wanted]
    labels = ["Base", "$r=2$", "$r=4$", "$T=8$", "$T=16$", "$p=.24$", "$p=.32$", "$n=.03$", "$n=.06$"]
    means = np.asarray([float(row["lower_improvement_mean"]) for row in selected])
    stds = np.asarray([float(row["lower_improvement_std"]) for row in selected])
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=2.5, color=LIGHT, edgecolor=BLACK, linewidth=0.7, zorder=3)
    bars[0].set_facecolor(DARK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.43)
    ax.set_ylabel("Robust lower improvement")
    clean(ax)
    ax.text(0.99, 0.95, "$r$: repeat threshold; $T$: horizon;\n$p$: provider threshold; $n$: novelty drift",
            ha="right", va="top", transform=ax.transAxes, fontsize=7.2, color=DARK)
    save(fig, "figure_03_oat_sensitivity")


def lhs_attribution() -> None:
    rows = read_csv("calibration_lhs_attributions.csv")
    order = ["repeat_cap", "explore_slot", "tail_slot", "diversify", "novel_slot", "provider_balance"]
    values = {name: [] for name in order}
    for row in rows:
        values[row["intervention"]].append(float(row["phi_mean"]))
    means = [float(np.mean(values[name])) for name in order]
    stds = [float(np.std(values[name], ddof=1)) for name in order]
    labels = ["repeat\ncap", "explore\nslot", "tail\nslot", "diversify", "novel\nslot", "provider\nbalance"]
    fig, ax = plt.subplots(figsize=(6.3, 3.25))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=stds, capsize=2.5, color=LIGHT, edgecolor=BLACK, linewidth=0.7, zorder=3)
    bars[0].set_facecolor(DARK)
    ax.axhline(0, color=BLACK, linewidth=0.7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean Shapley value across LHS decisions")
    clean(ax)
    save(fig, "figure_04_lhs_attribution")


def attribution_decision() -> None:
    attribution_rows = read_csv("calibration_lhs_attributions.csv")
    decision_rows = read_csv("calibration_lhs_seed_decisions.csv")
    order = ["repeat_cap", "explore_slot", "tail_slot", "diversify", "novel_slot", "provider_balance"]
    values = {name: [] for name in order}
    for row in attribution_rows:
        values[row["intervention"]].append(float(row["phi_mean"]))
    inclusion = Counter()
    for row in decision_rows:
        selected = row["selected_interventions"]
        for name in order:
            inclusion[name] += int(name in selected)
    means = np.asarray([np.mean(values[name]) for name in order])
    rates = np.asarray([inclusion[name] / len(decision_rows) for name in order])
    fig, ax = plt.subplots(figsize=(5.6, 3.65))
    ax.scatter(means, rates, marker="o", s=48, facecolors=WHITE, edgecolors=BLACK, linewidths=1.0, zorder=3)
    for x, y, label in zip(means, rates, order, strict=True):
        ax.annotate(label.replace("_", " "), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.axvline(0, color=DARK, linewidth=0.65, linestyle="--")
    ax.set_xlabel("Mean Shapley value across LHS decisions")
    ax.set_ylabel("Portfolio inclusion rate")
    ax.set_ylim(-0.04, 1.04)
    clean(ax)
    save(fig, "figure_05_attribution_decision")


def external_ranking() -> None:
    bpr = {row["metric"]: row for row in read_csv("final_bpr_seed_summary.csv")}
    sas = {row["metric"]: row for row in read_csv("final_sasrec_seed_summary.csv")}
    names = ["Popularity", "BPR-MF", "SASRec"]
    recall = [0.049, float(bpr["recall_at_k"]["mean"]), float(sas["recall_at_k"]["mean"])]
    recall_sd = [0.0, float(bpr["recall_at_k"]["std"]), float(sas["recall_at_k"]["std"])]
    ndcg = [0.025520142891504102, float(bpr["ndcg_at_k"]["mean"]), float(sas["ndcg_at_k"]["mean"])]
    ndcg_sd = [0.0, float(bpr["ndcg_at_k"]["std"]), float(sas["ndcg_at_k"]["std"])]
    x = np.arange(len(names)); width = 0.34
    fig, ax = plt.subplots(figsize=(5.9, 3.35))
    ax.bar(x - width / 2, recall, width, yerr=recall_sd, capsize=2.5, label="Recall@10", color=LIGHT, edgecolor=BLACK, linewidth=0.7, zorder=3)
    ax.bar(x + width / 2, ndcg, width, yerr=ndcg_sd, capsize=2.5, label="NDCG@10", color=DARK, edgecolor=BLACK, linewidth=0.7, zorder=3)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.17)
    ax.set_ylabel("Ranking metric")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    clean(ax)
    save(fig, "figure_06_external_ranking")


def main() -> None:
    controlled_recovery()
    oat_sensitivity()
    lhs_attribution()
    attribution_decision()
    external_ranking()
    print(f"Wrote Matplotlib figures to {FIGURES}")


if __name__ == "__main__":
    main()
