"""Multi-seed CURE-Sim execution with paired common-random-number summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cure_rec.config import Settings
from cure_rec.pipeline import run_experiment
from cure_rec.planner import decision_to_dict


@dataclass(frozen=True)
class SeedSweepResult:
    run_dir: Path
    decisions: pd.DataFrame
    attributions: pd.DataFrame
    interactions: pd.DataFrame
    base_feasibility: pd.DataFrame


def run_seed_sweep(settings: Settings, seeds: Iterable[int]) -> SeedSweepResult:
    """Run paired environment seeds and aggregate decision/attribution stability.

    Within each seed, CURE-Sim resets every coalition to the same scenario-specific
    random state. This common-random-number design reduces variance for paired
    coalition differences. Seeds are independent across repetitions.
    """
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("At least one seed is required")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep_root = Path(settings.run.output_root) / f"seed-sweep-{stamp}"
    decision_rows: list[dict] = []
    attribution_rows: list[dict] = []
    interaction_rows: list[dict] = []
    base_rows: list[dict] = []

    for seed in seed_list:
        seeded = settings.model_copy(deep=True)
        seeded.run.seed = int(seed)
        seeded.run.name = f"{settings.run.name}-seed-{seed}"
        seeded.run.output_root = sweep_root / "runs"
        logger, game, decision = run_experiment(seeded)
        decision_rows.append({
            "seed": seed,
            "cure_run_dir": str(logger.run_dir),
            **decision_to_dict(decision),
        })
        for row in game.regions.to_dict(orient="records"):
            attribution_rows.append({"seed": seed, **row})
        for row in game.interaction_table.to_dict(orient="records"):
            interaction_rows.append({"seed": seed, **row})
        base_values = [scenario.values[0] for scenario in game.scenario_games.values()]
        provider_upper = float(max(value.provider_disparity for value in base_values))
        fatigue_upper = float(max(value.fatigue for value in base_values))
        base_rows.append({
            "seed": seed,
            "base_feasible": decision.base_feasible,
            "provider_disparity_upper": provider_upper,
            "provider_margin": settings.constraints.max_provider_disparity - provider_upper,
            "fatigue_upper": fatigue_upper,
            "fatigue_margin": settings.constraints.max_fatigue - fatigue_upper,
            "relevance_margin": -settings.constraints.min_relevance_delta,
            "budget_margin": settings.constraints.budget,
            "provider_failure": provider_upper > settings.constraints.max_provider_disparity,
            "fatigue_failure": fatigue_upper > settings.constraints.max_fatigue,
        })

    sweep_root.mkdir(parents=True, exist_ok=True)
    decisions = pd.DataFrame(decision_rows)
    attributions = pd.DataFrame(attribution_rows)
    interactions = pd.DataFrame(interaction_rows)
    base_feasibility = pd.DataFrame(base_rows)
    decisions.to_csv(sweep_root / "seed_sweep_decisions.csv", index=False)
    attributions.to_csv(sweep_root / "seed_sweep_attributions.csv", index=False)
    interactions.to_csv(sweep_root / "seed_sweep_interactions.csv", index=False)
    base_feasibility.to_csv(sweep_root / "seed_sweep_base_feasibility.csv", index=False)
    attribution_summary = attributions.groupby("intervention", as_index=False).agg(
        phi_mean_mean=("phi_mean", "mean"),
        phi_mean_std=("phi_mean", "std"),
        phi_lower_mean=("phi_lower", "mean"),
        phi_upper_mean=("phi_upper", "mean"),
        positive_sign_rate=("phi_lower", lambda x: float((x > 0).mean())),
    )
    attribution_summary.to_csv(sweep_root / "seed_sweep_attribution_summary.csv", index=False)
    selection_frequency = decisions.assign(portfolio=decisions["selected_interventions"].astype(str)).groupby("portfolio", as_index=False).agg(
        frequency=("seed", "count"),
        lower_improvement_mean=("lower_improvement", "mean"),
    )
    selection_frequency.to_csv(sweep_root / "seed_sweep_selection_frequency.csv", index=False)
    _emit_seed_sweep_assets(sweep_root, decisions, attribution_summary, interactions, base_feasibility, selection_frequency)
    return SeedSweepResult(sweep_root, decisions, attributions, interactions, base_feasibility)


def _emit_seed_sweep_assets(
    sweep_root: Path,
    decisions: pd.DataFrame,
    attribution_summary: pd.DataFrame,
    interactions: pd.DataFrame,
    base_feasibility: pd.DataFrame,
    selection_frequency: pd.DataFrame,
) -> None:
    figures = sweep_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(selection_frequency["portfolio"], selection_frequency["frequency"], color="#2E86AB")
    ax.set_ylabel("Selection frequency")
    ax.set_title("Seed-sweep selected portfolios")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(figures / "seed_figure_portfolio_frequency.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = np.where(decisions["base_feasible"], "#2A9D8F", "#E76F51")
    ax.scatter(decisions["seed"], decisions["lower_improvement"], c=colors, s=55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Seed"); ax.set_ylabel("Lower robust improvement")
    ax.set_title("Seed-level robust improvement (green=improvement, red=repair)")
    fig.tight_layout(); fig.savefig(figures / "seed_figure_lower_improvement.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(attribution_summary["intervention"], attribution_summary["phi_mean_mean"], yerr=attribution_summary["phi_mean_std"].fillna(0.0), fmt="o", capsize=4, color="#264653")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Shapley value ± seed SD")
    ax.set_title("Shapley stability across seeds")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(figures / "seed_figure_shapley_stability.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = np.where(base_feasibility["base_feasible"], "#2A9D8F", "#E76F51")
    ax.scatter(base_feasibility["seed"], base_feasibility["provider_margin"], c=colors, s=55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Seed"); ax.set_ylabel("Base provider feasibility margin")
    ax.set_title("Base-policy feasibility by seed")
    fig.tight_layout(); fig.savefig(figures / "seed_figure_base_feasibility.png", dpi=180); plt.close(fig)

    names = list(attribution_summary["intervention"])
    matrix = np.zeros((len(names), len(names)))
    interaction_mean = interactions.groupby(["intervention_i", "intervention_j"], as_index=False)["interaction_mean"].mean()
    for row in interaction_mean.itertuples(index=False):
        i, j = names.index(row.intervention_i), names.index(row.intervention_j)
        matrix[i, j] = matrix[j, i] = row.interaction_mean
    bound = max(float(np.abs(matrix).max()), 1e-6)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(image, ax=ax, label="Mean interaction across seeds")
    ax.set_title("Seed-aggregated interaction heatmap")
    fig.tight_layout(); fig.savefig(figures / "seed_figure_interaction_heatmap.png", dpi=180); plt.close(fig)


@dataclass(frozen=True)
class AllVariationsResult:
    run_dir: Path
    data_analysis_dir: Path
    variation_summary: pd.DataFrame
    decisions: pd.DataFrame


def _variation_settings(settings: Settings, *, name: str, output_root: Path) -> Settings:
    copied = settings.model_copy(deep=True)
    copied.run.name = name
    copied.run.output_root = output_root
    return copied


def run_all_variations(
    quick_settings: Settings,
    full_settings: Settings,
    *,
    dataset: str,
    source: str | Path,
    download: bool = False,
    run_bpr: bool = True,
    bpr_updates: int = 1_500_000,
    max_eval_users: int = 1_000,
    quick_seeds: Iterable[int] = (42, 43, 44, 45, 46),
    full_seeds: Iterable[int] = (42, 43, 44, 45, 46),
    final_seeds: Iterable[int] = tuple(range(100, 120)),
) -> AllVariationsResult:
    """Run every self-contained experimental variation in a fixed scientific order.

    Order is deliberate: external data/model analysis first, then quick behavioural
    validation, controlled oracle regimes, quick stability, full behavioural run,
    full stability, and final 20-seed inference. This command can run for hours.
    """
    from cure_rec.analysis import analyze_dataset
    from cure_rec.data import load_dataset
    from cure_rec.observability import RunLogger
    from cure_rec.regimes import run_regime_suite

    quick_seed_list = list(quick_seeds)
    full_seed_list = list(full_seeds)
    final_seed_list = list(final_seeds)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    master_root = Path(full_settings.run.output_root) / f"all-variations-{stamp}"
    master_root.mkdir(parents=True, exist_ok=False)
    loaded = load_dataset(dataset, source, download=download)
    analysis = analyze_dataset(
        loaded,
        output_root=master_root / "external_data",
        run_bpr=run_bpr,
        bpr_updates=bpr_updates,
        max_eval_users=max_eval_users,
        seed=full_settings.run.seed,
    )

    variation_rows: list[dict] = []
    decision_frames: list[pd.DataFrame] = []

    def execute_single(label: str, settings: Settings) -> None:
        logger, _, decision = run_experiment(_variation_settings(settings, name=label, output_root=master_root / label))
        variation_rows.append({
            "variation": label,
            "kind": "single_run",
            "run_dir": str(logger.run_dir),
            **decision_to_dict(decision),
        })

    def execute_sweep(label: str, settings: Settings, seeds: Iterable[int]) -> None:
        seed_list = list(seeds)
        result = run_seed_sweep(_variation_settings(settings, name=label, output_root=master_root / label), seed_list)
        variation_rows.append({
            "variation": label,
            "kind": "seed_sweep",
            "run_dir": str(result.run_dir),
            "seed_count": len(seed_list),
        })
        frame = result.decisions.copy()
        frame.insert(0, "variation", label)
        decision_frames.append(frame)

    execute_single("quick_single", quick_settings)

    regime_logger = RunLogger(_variation_settings(quick_settings, name="controlled_regimes", output_root=master_root / "controlled_regimes"))
    try:
        regime_result = run_regime_suite(quick_settings, regime_logger)
        regime_logger.close(status="completed")
    except Exception:
        regime_logger.close(status="failed")
        raise
    variation_rows.append({
        "variation": "controlled_regimes",
        "kind": "oracle_regime_suite",
        "run_dir": str(regime_result.run_dir),
        "estimated_selection_match_rate": float(regime_result.summary["estimated_selection_match"].mean()),
        "oracle_selection_match_rate": float(regime_result.summary["oracle_selection_match"].mean()),
        "mean_shapley_mae": float(regime_result.attribution_recovery["absolute_error"].mean()),
    })

    execute_sweep("quick_five_seed", quick_settings, quick_seed_list)
    execute_single("full_single", full_settings)
    execute_sweep("full_five_seed", full_settings, full_seed_list)
    execute_sweep("full_twenty_seed", full_settings, final_seed_list)

    summary = pd.DataFrame(variation_rows)
    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    summary.to_csv(master_root / "all_variations_summary.csv", index=False)
    decisions.to_csv(master_root / "all_variations_seed_decisions.csv", index=False)
    manifest = {
        "dataset": dataset,
        "source": str(source),
        "data_analysis_dir": str(analysis.run_dir),
        "run_bpr": run_bpr,
        "variations": summary.to_dict(orient="records"),
        "quick_seeds": quick_seed_list,
        "full_seeds": full_seed_list,
        "final_seeds": final_seed_list,
        "execution_order": [
            "external_data_analysis",
            "quick_single",
            "controlled_regimes",
            "quick_five_seed",
            "full_single",
            "full_five_seed",
            "full_twenty_seed",
        ],
    }
    (master_root / "all_variations_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return AllVariationsResult(master_root, analysis.run_dir, summary, decisions)


def postprocess_seed_sweep(run_dir: str | Path, settings: Settings | None = None) -> SeedSweepResult:
    """Regenerate aggregate tables and figures from an existing completed sweep.

    This avoids repeating expensive full CURE-Sim sweeps when only reporting code
    changes. The completed child run directories already hold raw coalition and
    interaction tables.
    """
    sweep_root = Path(run_dir)
    decisions = pd.read_csv(sweep_root / "seed_sweep_decisions.csv")
    attributions = pd.read_csv(sweep_root / "seed_sweep_attributions.csv")
    interaction_rows: list[pd.DataFrame] = []
    base_rows: list[dict] = []
    for decision in decisions.to_dict(orient="records"):
        child = Path(decision["cure_run_dir"])
        interactions_path = child / "tables" / "interaction_regions.csv"
        coalition_path = child / "tables" / "coalition_values.csv"
        if interactions_path.exists():
            frame = pd.read_csv(interactions_path)
            frame.insert(0, "seed", decision["seed"])
            interaction_rows.append(frame)
        if coalition_path.exists():
            coalitions = pd.read_csv(coalition_path)
            base = coalitions[coalitions["mask"] == 0]
            base_rows.append({
                "seed": decision["seed"],
                "base_feasible": bool(decision["base_feasible"]),
                "provider_disparity_upper": float(base["provider_disparity"].max()),
                "provider_margin": (
                    float(settings.constraints.max_provider_disparity - base["provider_disparity"].max())
                    if settings is not None else float("nan")
                ),
                "fatigue_upper": float(base["fatigue"].max()),
                "fatigue_margin": (
                    float(settings.constraints.max_fatigue - base["fatigue"].max())
                    if settings is not None else float("nan")
                ),
                "relevance_margin": -settings.constraints.min_relevance_delta if settings is not None else float("nan"),
                "budget_margin": settings.constraints.budget if settings is not None else float("nan"),
                "provider_failure": (
                    bool(base["provider_disparity"].max() > settings.constraints.max_provider_disparity)
                    if settings is not None else not bool(decision["base_feasible"])
                ),
                "fatigue_failure": (
                    bool(base["fatigue"].max() > settings.constraints.max_fatigue)
                    if settings is not None else False
                ),
            })
    interactions = pd.concat(interaction_rows, ignore_index=True) if interaction_rows else pd.DataFrame()
    # Existing decisions do not store historical constraint thresholds. Recompute
    # visual base feasibility from the recorded planner decision and preserve raw
    # provider values; future sweeps write exact margins directly.
    base_feasibility = pd.DataFrame(base_rows)
    attribution_summary = attributions.groupby("intervention", as_index=False).agg(
        phi_mean_mean=("phi_mean", "mean"),
        phi_mean_std=("phi_mean", "std"),
        phi_lower_mean=("phi_lower", "mean"),
        phi_upper_mean=("phi_upper", "mean"),
        positive_sign_rate=("phi_lower", lambda x: float((x > 0).mean())),
    )
    selection_frequency = decisions.assign(portfolio=decisions["selected_interventions"].astype(str)).groupby("portfolio", as_index=False).agg(
        frequency=("seed", "count"),
        lower_improvement_mean=("lower_improvement", "mean"),
    )
    interactions.to_csv(sweep_root / "seed_sweep_interactions.csv", index=False)
    base_feasibility.to_csv(sweep_root / "seed_sweep_base_feasibility.csv", index=False)
    attribution_summary.to_csv(sweep_root / "seed_sweep_attribution_summary.csv", index=False)
    selection_frequency.to_csv(sweep_root / "seed_sweep_selection_frequency.csv", index=False)
    if not interactions.empty and not base_feasibility.empty:
        _emit_seed_sweep_assets(sweep_root, decisions, attribution_summary, interactions, base_feasibility, selection_frequency)
    return SeedSweepResult(sweep_root, decisions, attributions, interactions, base_feasibility)
