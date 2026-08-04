"""Small report emitters used by CLI and quickstart notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt

from cure_rec.game import GameResult
from cure_rec.observability import RunLogger
from cure_rec.planner import PortfolioDecision


def emit_figures(game: GameResult, decision: PortfolioDecision, logger: RunLogger) -> None:
    regions = game.regions.sort_values("phi_mean")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lower_error = regions["phi_mean"] - regions["phi_lower"]
    upper_error = regions["phi_upper"] - regions["phi_mean"]
    colors = ["#2E86AB" if name in decision.selected_interventions else "#A6A6A6" for name in regions["intervention"]]
    ax.bar(regions["intervention"], regions["phi_mean"], color=colors)
    ax.errorbar(regions["intervention"], regions["phi_mean"], yerr=[lower_error, upper_error], fmt="none", color="black", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Shapley improvement contribution")
    ax.set_title("CURE-Rec Shapley regions across configured scenarios")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = logger.figures_dir / "shapley_regions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    table = game.coalition_table.groupby("mask", as_index=False).agg(
        improvement=("improvement", "min"),
        cost=("cost", "first"),
        active_interventions=("active_interventions", "first"),
    ).sort_values("improvement", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [label if label else "base" for label in table["active_interventions"]]
    ax.barh(labels[::-1], table["improvement"].iloc[::-1], color="#2E86AB")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Worst-case improvement over base policy")
    ax.set_title("Top coalition policies by configured robust improvement")
    fig.tight_layout()
    path = logger.figures_dir / "coalition_improvements.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    logger.event("figures_written", files=["shapley_regions.png", "coalition_improvements.png"])
