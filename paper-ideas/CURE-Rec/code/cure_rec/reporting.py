"""Deterministic manuscript-asset generation for CURE-Rec runs.

Every successful CURE-Sim run emits a numbered asset set under `tables/` and
`figures/`, plus an asset registry that distinguishes generated empirical assets
from manuscript assets requiring literature synthesis or later real-log evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cure_rec.config import INTERVENTION_NAMES, Settings
from cure_rec.game import EMPTY_MASK, GameResult
from cure_rec.observability import RunLogger
from cure_rec.planner import PortfolioDecision, decision_to_dict


ASSET_CONTRACT: tuple[dict[str, str], ...] = (
    {"id": "Table 1", "path": "tables/table_01_asset_registry.csv", "scope": "generated", "purpose": "Asset provenance, scope, and readiness"},
    {"id": "Table 2", "path": "tables/table_02_benchmark_configuration.csv", "scope": "generated", "purpose": "CURE-Sim and policy configuration"},
    {"id": "Table 3", "path": "tables/table_03_attribution_regions.csv", "scope": "generated", "purpose": "Full-game Shapley and feasibility-aware semivalue regions"},
    {"id": "Table 4", "path": "tables/table_04_uncertainty_summary.csv", "scope": "generated", "purpose": "Scenario uncertainty widths and attribution signs"},
    {"id": "Table 5", "path": "tables/table_05_portfolio_decision.csv", "scope": "generated", "purpose": "Robust selected portfolio and constraint diagnostics"},
    {"id": "Table 6", "path": "tables/table_06_long_term_tradeoffs.csv", "scope": "generated", "purpose": "Base versus selected policy outcomes by scenario"},
    {"id": "Table 7", "path": "tables/table_07_selection_comparison.csv", "scope": "generated", "purpose": "Base, best-single, full, and robust portfolio comparison"},
    {"id": "Table 8", "path": "tables/table_08_runtime_summary.csv", "scope": "generated", "purpose": "Coalition evaluation runtime by scenario and cardinality"},
    {"id": "Figure 1", "path": "figures/figure_01_framework.png", "scope": "generated", "purpose": "CURE-Rec execution flow"},
    {"id": "Figure 2", "path": "figures/figure_02_shapley_regions.png", "scope": "generated", "purpose": "Shapley regions and selected interventions"},
    {"id": "Figure 3", "path": "figures/figure_03_uncertainty_widths.png", "scope": "generated", "purpose": "Scenario-induced attribution-width profile"},
    {"id": "Figure 4", "path": "figures/figure_04_interaction_heatmap.png", "scope": "generated", "purpose": "Pairwise interaction regions"},
    {"id": "Figure 5", "path": "figures/figure_05_trajectory_comparison.png", "scope": "generated", "purpose": "Selected versus base long-term trajectories"},
    {"id": "Figure 6", "path": "figures/figure_06_decision_card.png", "scope": "generated", "purpose": "Deployment/explanation decision card"},
    {"id": "Figure 7", "path": "figures/figure_07_runtime_by_cardinality.png", "scope": "generated", "purpose": "Exact-game runtime profile"},
    {"id": "Figure 8", "path": "figures/figure_08_scenario_sensitivity.png", "scope": "generated", "purpose": "Policy-improvement sensitivity across scenarios"},
    {"id": "Manual literature table", "path": "manuscript only", "scope": "manual", "purpose": "Prior-work positioning must be literature-reviewed, not generated from a run"},
    {"id": "Real-log OPE assets", "path": "future audited-log run", "scope": "future", "purpose": "Generated only after support and causal-log audit passes"},
)


PALETTE = {
    "selected": "#2E86AB",
    "neutral": "#A6A6A6",
    "positive": "#2A9D8F",
    "negative": "#E76F51",
    "dark": "#264653",
    "accent": "#E9C46A",
}


def _write_table(logger: RunLogger, filename: str, frame: pd.DataFrame) -> Path:
    return logger.write_dataframe(f"tables/{filename}", frame)


def _scenario_values(game: GameResult, mask: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, scenario_game in game.scenario_games.items():
        value = scenario_game.values[mask]
        rows.append({
            "scenario": scenario_name,
            "mask": mask,
            "active_interventions": ";".join(value.active_interventions),
            "cost": value.cost,
            "utility": value.utility,
            "improvement": value.improvement,
            "satisfaction": value.satisfaction,
            "retention": value.retention,
            "fatigue": value.fatigue,
            "relevance": value.relevance,
            "provider_disparity": value.provider_disparity,
            "catalog_coverage": value.catalog_coverage,
            "duration_seconds": value.duration_seconds,
        })
    return pd.DataFrame(rows)


def _robust_mask(game: GameResult, candidate_masks: list[int]) -> int:
    return max(candidate_masks, key=lambda mask: min(scenario.values[mask].improvement for scenario in game.scenario_games.values()))


def _plot_framework(logger: RunLogger) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.axis("off")
    nodes = [
        (0.05, "Base\npolicy"),
        (0.25, "Policy\ninterventions"),
        (0.46, "Scenario /\ncausal models"),
        (0.67, "Exact game\n+ attribution"),
        (0.88, "Robust\ndecision card"),
    ]
    for x, label in nodes:
        ax.text(x, 0.52, label, ha="center", va="center", fontsize=11,
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#EAF4F7", "edgecolor": PALETTE["selected"], "linewidth": 1.6})
    for (x0, _), (x1, _) in zip(nodes, nodes[1:]):
        ax.annotate("", xy=(x1 - 0.075, 0.52), xytext=(x0 + 0.075, 0.52), arrowprops={"arrowstyle": "->", "color": PALETTE["dark"], "lw": 1.7})
    ax.text(0.46, 0.15, "Structured JSONL logs + manifests + tables + figures", ha="center", color=PALETTE["dark"], fontsize=10)
    fig.tight_layout()
    fig.savefig(logger.figures_dir / "figure_01_framework.png", dpi=180)
    plt.close(fig)


def _plot_shapley_regions(game: GameResult, decision: PortfolioDecision, logger: RunLogger) -> None:
    regions = game.regions.sort_values("phi_mean")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    lower_error = regions["phi_mean"] - regions["phi_lower"]
    upper_error = regions["phi_upper"] - regions["phi_mean"]
    colors = [PALETTE["selected"] if name in decision.selected_interventions else PALETTE["neutral"] for name in regions["intervention"]]
    ax.bar(regions["intervention"], regions["phi_mean"], color=colors)
    ax.errorbar(regions["intervention"], regions["phi_mean"], yerr=[lower_error, upper_error], fmt="none", color="black", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Shapley improvement contribution")
    ax.set_title("CURE-Rec full-game Shapley regions")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(logger.figures_dir / "figure_02_shapley_regions.png", dpi=180)
    plt.close(fig)


def _plot_uncertainty_widths(game: GameResult, logger: RunLogger) -> None:
    regions = game.regions.copy()
    regions["phi_width"] = regions["phi_upper"] - regions["phi_lower"]
    regions["psi_width"] = regions["psi_feasible_upper"] - regions["psi_feasible_lower"]
    x = np.arange(len(regions))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.38
    ax.bar(x - width / 2, regions["phi_width"], width=width, label="full-game Shapley", color=PALETTE["selected"])
    ax.bar(x + width / 2, regions["psi_width"], width=width, label="feasible semivalue", color=PALETTE["accent"])
    ax.set_xticks(x, regions["intervention"], rotation=25)
    ax.set_ylabel("Configured scenario-region width")
    ax.set_title("Attribution sensitivity across configured scenarios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(logger.figures_dir / "figure_03_uncertainty_widths.png", dpi=180)
    plt.close(fig)


def _plot_interaction_heatmap(game: GameResult, logger: RunLogger) -> None:
    names = list(INTERVENTION_NAMES)
    matrix = np.zeros((len(names), len(names)))
    for row in game.interaction_table.to_dict(orient="records"):
        i, j = names.index(row["intervention_i"]), names.index(row["intervention_j"])
        matrix[i, j] = matrix[j, i] = row["interaction_mean"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    lim = max(float(np.abs(matrix).max()), 1e-6)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="mean Grabisch–Roubens interaction")
    ax.set_title("Pairwise policy-intervention interactions")
    fig.tight_layout()
    fig.savefig(logger.figures_dir / "figure_04_interaction_heatmap.png", dpi=180)
    plt.close(fig)


def _load_trajectory(logger: RunLogger, scenario: str, mask: int) -> list[dict[str, Any]]:
    path = logger.artifacts_dir / "coalitions" / scenario / f"mask_{mask:02d}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("trajectory", [])


def _plot_trajectory_comparison(game: GameResult, decision: PortfolioDecision, logger: RunLogger) -> None:
    scenario = next(iter(game.scenario_games))
    base_trace = _load_trajectory(logger, scenario, EMPTY_MASK)
    selected_trace = _load_trajectory(logger, scenario, decision.selected_mask)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    metrics = [("mean_satisfaction", "Satisfaction"), ("mean_fatigue", "Fatigue"), ("provider_disparity", "Provider disparity")]
    for axis, (key, title) in zip(axes, metrics):
        if base_trace and selected_trace:
            axis.plot([x["step"] for x in base_trace], [x[key] for x in base_trace], marker="o", label="base")
            axis.plot([x["step"] for x in selected_trace], [x[key] for x in selected_trace], marker="o", label="selected")
        axis.set_title(title)
        axis.set_xlabel("Horizon step")
    axes[0].set_ylabel("Metric value")
    axes[0].legend()
    fig.suptitle(f"Long-term trace: {scenario} scenario")
    fig.tight_layout()
    fig.savefig(logger.figures_dir / "figure_05_trajectory_comparison.png", dpi=180)
    plt.close(fig)


def _plot_decision_card(game: GameResult, decision: PortfolioDecision, logger: RunLogger) -> None:
    selected = game.regions[game.regions["intervention"].isin(decision.selected_interventions)]
    lines = [
        f"Action: {decision.action.upper()}",
        f"Portfolio: {', '.join(decision.selected_interventions) or 'base policy'}",
        f"Worst-case improvement: {decision.lower_improvement:.4f}",
        f"Cost: {decision.cost:.3f}",
        f"Relevance lower delta: {decision.relevance_delta_lower:.4f}",
        f"Provider disparity upper: {decision.provider_disparity_upper:.4f}",
        "",
        "Selected attribution regions:",
    ]
    for row in selected.to_dict(orient="records"):
        lines.append(f"  {row['intervention']}: [{row['phi_lower']:.4f}, {row['phi_upper']:.4f}]")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.text(0.03, 0.95, "CURE-Rec decision card", fontsize=16, fontweight="bold", va="top", color=PALETTE["dark"])
    ax.text(0.03, 0.82, "\n".join(lines), fontsize=11, va="top", family="monospace", bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F7FBFC", "edgecolor": PALETTE["selected"]})
    fig.tight_layout()
    fig.savefig(logger.figures_dir / "figure_06_decision_card.png", dpi=180)
    plt.close(fig)


def _plot_runtime(game: GameResult, logger: RunLogger) -> None:
    frame = game.coalition_table.copy()
    frame["cardinality"] = frame["active_interventions"].apply(lambda text: 0 if not text else len(text.split(";")))
    summary = frame.groupby("cardinality", as_index=False)["duration_seconds"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(summary["cardinality"], summary["duration_seconds"], color=PALETTE["dark"])
    ax.set_xlabel("Coalition cardinality")
    ax.set_ylabel("Mean coalition evaluation time (seconds)")
    ax.set_title("Exact-game runtime by coalition cardinality")
    fig.tight_layout()
    fig.savefig(logger.figures_dir / "figure_07_runtime_by_cardinality.png", dpi=180)
    plt.close(fig)


def _plot_scenario_sensitivity(game: GameResult, decision: PortfolioDecision, logger: RunLogger) -> None:
    rows = _scenario_values(game, decision.selected_mask)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = [PALETTE["positive"] if value > 0 else PALETTE["negative"] for value in rows["improvement"]]
    ax.bar(rows["scenario"], rows["improvement"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Improvement over base policy")
    ax.set_title("Selected-portfolio sensitivity across configured scenarios")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(logger.figures_dir / "figure_08_scenario_sensitivity.png", dpi=180)
    plt.close(fig)


def _tables(game: GameResult, decision: PortfolioDecision, settings: Settings, logger: RunLogger) -> dict[str, pd.DataFrame]:
    benchmark = pd.DataFrame([{
        "n_users": settings.simulator.n_users,
        "n_items": settings.simulator.n_items,
        "n_providers": settings.simulator.n_providers,
        "n_categories": settings.simulator.n_categories,
        "horizon": settings.simulator.horizon,
        "slate_size": settings.simulator.slate_size,
        "n_interventions": len(INTERVENTION_NAMES),
        "n_coalitions": 2 ** len(INTERVENTION_NAMES),
        "scenarios": ";".join(game.scenario_games),
        "config_hash": settings.config_hash(),
    }])
    regions = game.regions.copy()
    uncertainty = regions[["intervention", "phi_lower", "phi_upper", "psi_feasible_lower", "psi_feasible_upper", "phi_psi_sign_agree"]].copy()
    uncertainty["full_shapley_width"] = uncertainty["phi_upper"] - uncertainty["phi_lower"]
    uncertainty["feasible_semivalue_width"] = uncertainty["psi_feasible_upper"] - uncertainty["psi_feasible_lower"]
    decision_table = pd.DataFrame([decision_to_dict(decision)])
    tradeoffs = pd.concat([
        _scenario_values(game, EMPTY_MASK).assign(policy="base"),
        _scenario_values(game, decision.selected_mask).assign(policy="selected"),
    ], ignore_index=True)

    masks = sorted(set(game.coalition_table["mask"]))
    single_masks = [mask for mask in masks if bin(mask).count("1") == 1]
    best_single = _robust_mask(game, single_masks)
    comparison_masks = {
        "base": EMPTY_MASK,
        "best_single": best_single,
        "full_coalition": max(masks),
        "direct_robust": decision.selected_mask,
    }
    comparison_rows: list[dict[str, Any]] = []
    for label, mask in comparison_masks.items():
        values = _scenario_values(game, mask)
        comparison_rows.append({
            "method": label,
            "mask": mask,
            "active_interventions": ";".join(game.scenario_games[next(iter(game.scenario_games))].values[mask].active_interventions),
            "lower_improvement": float(values["improvement"].min()),
            "mean_improvement": float(values["improvement"].mean()),
            "upper_improvement": float(values["improvement"].max()),
            "cost": float(values["cost"].iloc[0]),
        })
    comparisons = pd.DataFrame(comparison_rows)

    runtime = game.coalition_table.copy()
    runtime["cardinality"] = runtime["active_interventions"].apply(lambda text: 0 if not text else len(text.split(";")))
    runtime = runtime.groupby(["scenario", "cardinality"], as_index=False).agg(
        coalition_count=("mask", "count"),
        mean_seconds=("duration_seconds", "mean"),
        max_seconds=("duration_seconds", "max"),
        total_seconds=("duration_seconds", "sum"),
    )
    generated = {
        "table_02_benchmark_configuration.csv": benchmark,
        "table_03_attribution_regions.csv": regions,
        "table_04_uncertainty_summary.csv": uncertainty,
        "table_05_portfolio_decision.csv": decision_table,
        "table_06_long_term_tradeoffs.csv": tradeoffs,
        "table_07_selection_comparison.csv": comparisons,
        "table_08_runtime_summary.csv": runtime,
    }
    for filename, frame in generated.items():
        _write_table(logger, filename, frame)
    return generated


def emit_assets(game: GameResult, decision: PortfolioDecision, settings: Settings, logger: RunLogger) -> None:
    """Generate the complete empirical asset contract for a CURE-Sim run."""
    _tables(game, decision, settings, logger)
    _plot_framework(logger)
    _plot_shapley_regions(game, decision, logger)
    _plot_uncertainty_widths(game, logger)
    _plot_interaction_heatmap(game, logger)
    _plot_trajectory_comparison(game, decision, logger)
    _plot_decision_card(game, decision, logger)
    _plot_runtime(game, logger)
    _plot_scenario_sensitivity(game, decision, logger)

    registry = pd.DataFrame(ASSET_CONTRACT)
    # Materialize Table 1 first. Its own existence cannot be measured before the
    # registry file is written, which previously produced a false negative.
    _write_table(logger, "table_01_asset_registry.csv", registry)
    for index, row in registry.iterrows():
        registry.loc[index, "exists"] = bool(
            row["scope"] == "generated" and (logger.run_dir / row["path"]).exists()
        )
    # Rewrite Table 1 and the machine-readable manifest with final existence flags.
    _write_table(logger, "table_01_asset_registry.csv", registry)
    logger.write_json("artifacts/asset_manifest.json", registry.to_dict(orient="records"))
    logger.event(
        "assets_generated",
        tables=[row["path"] for row in ASSET_CONTRACT if row["path"].startswith("tables/")],
        figures=[row["path"] for row in ASSET_CONTRACT if row["path"].startswith("figures/")],
    )
