"""Phase E (reviewer revision): held-out selector comparison where selectors disagree.

Reviewer concern: in the archived full-configuration selector study, exact CURE,
best singleton, robust-Shapley-1, robust-Shapley-budget, greedy robust, and the
nominal-scenario selector all choose the same portfolio, so the comparison cannot
establish the value of the exact decision machinery.

This script therefore:

1. screens a predeclared list of configurations (the archived LHS design points
   whose seed-level decisions were NOT always repeat-cap, plus the baseline) with
   one seed each and records whether the six substantive selectors disagree;
2. promotes the first two divergent configurations, in declared order, to the full
   held-out protocol used by the archived selector study: portfolios are selected
   on seeds 42-46 and evaluated on disjoint seeds 200-219;
3. reports paired evaluation-seed statistics with the same clustering rule as the
   archived study (the evaluation seed is the independent unit).

All evidence is simulator-conditional; nothing here is external causal inference.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cure_rec.calibration import CalibrationPoint, _settings_for_point  # noqa: E402
from cure_rec.config import Settings, load_settings  # noqa: E402
from cure_rec.game import (  # noqa: E402
    ALL_MASKS,
    EMPTY_MASK,
    GameResult,
    coalition_names,
    evaluate_coalition,
)
from cure_rec.observability import RunLogger  # noqa: E402
from cure_rec.policies import HistoryAwarePolicy  # noqa: E402
from cure_rec.revision import SelectorChoice, _paired_summary, selector_choices  # noqa: E402
from cure_rec.simulator import CureSim  # noqa: E402

SNAPSHOT = ROOT.parent / "results" / "reproducibility_snapshot_latest" / "calibration_lhs_configurations.csv"
ASSETS = ROOT / "results" / "reviewer_phase_assets" / "divergent_selector_holdout"
BASE_CONFIG = ROOT / "configs" / "curesim_full.yaml"

# Predeclared screening order: baseline first, then the archived LHS design points
# whose five-seed decisions contained more than one distinct portfolio (see
# calibration_lhs_seed_decisions.csv in the reproducibility snapshot). No outcome
# from this script is used to choose the screening list.
SCREENING_ORDER = [
    "baseline",
    "lhs-012",
    "lhs-009",
    "lhs-014",
    "lhs-003",
    "lhs-010",
    "lhs-008",
    "lhs-002",
]
SUBSTANTIVE_SELECTORS = (
    "cure_exact_maximin",
    "best_singleton",
    "robust_shapley_1",
    "robust_shapley_budget",
    "greedy_robust",
    "nominal_scenario",
)
SELECTION_SEEDS = (42, 43, 44, 45, 46)
EVALUATION_SEEDS = tuple(range(200, 220))
MAX_HOLDOUT_CONFIGS = 2


def load_screening_points() -> dict[str, CalibrationPoint]:
    base = load_settings(BASE_CONFIG)
    baseline_values = {}
    from cure_rec.calibration import _PARAMETER_PATHS, _value

    for name in _PARAMETER_PATHS:
        baseline_values[name] = _value(base, name)
    points = {"baseline": CalibrationPoint("baseline", baseline_values, "baseline", None, is_baseline=True)}
    frame = pd.read_csv(SNAPSHOT)
    for point_id in SCREENING_ORDER:
        if point_id == "baseline":
            continue
        row = frame[frame["point_id"] == point_id]
        if row.empty:
            raise KeyError(f"{point_id} missing from archived LHS configuration table")
        values = {
            "fatigue_strength": float(row["parameter_fatigue_strength"].iloc[0]),
            "repeat_threshold": int(row["parameter_repeat_threshold"].iloc[0]),
            "horizon": int(row["parameter_horizon"].iloc[0]),
            "provider_threshold": float(row["parameter_provider_threshold"].iloc[0]),
            "provider_balance_strength": float(row["parameter_provider_balance_strength"].iloc[0]),
            "novelty_delayed_benefit": float(row["parameter_novelty_delayed_benefit"].iloc[0]),
            "exploration_cost": float(row["parameter_exploration_cost"].iloc[0]),
        }
        points[point_id] = CalibrationPoint(point_id, values, "joint_lhs", None)
    return points


def _settings_for(point: CalibrationPoint, seed: int, output_root: Path) -> Settings:
    base = load_settings(BASE_CONFIG)
    configured = _settings_for_point(base, point, output_root)
    configured.run.seed = seed
    configured.run.name = f"{point.point_id}-seed-{seed}"
    return configured


def run_full_game_payload(args: tuple[str, dict, int, str]) -> dict:
    """Worker: one exact game plus the selector choices derived from it."""
    point_id, values, seed, output_root = args
    point = CalibrationPoint(point_id, values, "joint_lhs" if point_id != "baseline" else "baseline", None, point_id == "baseline")
    settings = _settings_for(point, seed, Path(output_root))
    started = time.time()
    from cure_rec.pipeline import run_experiment

    logger, game, decision = run_experiment(settings)
    choices = selector_choices(game, settings, selection_seed=seed)
    duration = time.time() - started
    logger.close(status="completed")
    return {
        "point_id": point_id,
        "seed": seed,
        "run_dir": str(logger.run_dir),
        "duration_seconds": duration,
        "cure_selected_mask": decision.selected_mask,
        "cure_mode": decision.mode.value,
        "choices": [asdict(choice) for choice in choices],
    }


def evaluate_mask_set(settings: Settings, masks: tuple[int, ...], logger: RunLogger) -> dict:
    """Lean evaluation: coalition values for a declared mask set only."""
    robust_values: dict[int, float] = {}
    margins_by_mask: dict[int, dict[str, float]] = {}
    scenario_names = [scenario.name for scenario in settings.scenarios]
    scenario_improvements: dict[int, list[float]] = {mask: [] for mask in masks}
    base_metrics_candidates: list[dict[str, float]] = []
    for scenario in settings.scenarios:
        simulator = CureSim(settings, scenario)
        policy = HistoryAwarePolicy(simulator, settings.policy)
        baseline = evaluate_coalition(simulator, scenario, policy, EMPTY_MASK, 0.0, settings, logger)
        from dataclasses import replace

        baseline = replace(baseline, improvement=0.0)
        values = {EMPTY_MASK: baseline}
        for mask in masks:
            if mask != EMPTY_MASK:
                values[mask] = evaluate_coalition(simulator, scenario, policy, mask, baseline.utility, settings, logger)
        for mask in masks:
            scenario_improvements[mask].append(values[mask].improvement)
        relevance_deltas = [values[mask].relevance - baseline.relevance for mask in masks]
        for mask, delta in zip(masks, relevance_deltas, strict=True):
            margins_by_mask.setdefault(mask, {"relevance_delta_lower": float("inf"), "provider_disparity_upper": float("-inf"), "fatigue_upper": float("-inf")})
            margins_by_mask[mask]["relevance_delta_lower"] = min(margins_by_mask[mask]["relevance_delta_lower"], float(delta))
            margins_by_mask[mask]["provider_disparity_upper"] = max(margins_by_mask[mask]["provider_disparity_upper"], float(values[mask].provider_disparity))
            margins_by_mask[mask]["fatigue_upper"] = max(margins_by_mask[mask]["fatigue_upper"], float(values[mask].fatigue))
            margins_by_mask[mask]["cost"] = float(values[mask].cost)
    for mask in masks:
        robust_values[mask] = float(min(scenario_improvements[mask]))
    constraints = settings.constraints
    feasible_by_mask: dict[int, bool] = {}
    for mask in masks:
        metrics = margins_by_mask[mask]
        feasible_by_mask[mask] = bool(
            metrics["cost"] <= constraints.budget + 1e-12
            and metrics["relevance_delta_lower"] >= constraints.min_relevance_delta
            and metrics["provider_disparity_upper"] <= constraints.max_provider_disparity
            and metrics["fatigue_upper"] <= constraints.max_fatigue
        )
    return {
        "robust_values": robust_values,
        "margins": margins_by_mask,
        "feasible": feasible_by_mask,
        "scenarios": scenario_names,
        "constraints": {
            "budget": constraints.budget,
            "min_relevance_delta": constraints.min_relevance_delta,
            "max_provider_disparity": constraints.max_provider_disparity,
            "max_fatigue": constraints.max_fatigue,
        },
    }


def run_lean_evaluation_payload(args: tuple[str, dict, int, tuple, str]) -> dict:
    point_id, values, seed, masks, output_root = args
    point = CalibrationPoint(point_id, values, "joint_lhs" if point_id != "baseline" else "baseline", None, point_id == "baseline")
    settings = _settings_for(point, seed, Path(output_root))
    logger = RunLogger(settings)
    try:
        result = evaluate_mask_set(settings, tuple(int(mask) for mask in masks), logger)
        logger.close(status="completed")
        return {"point_id": point_id, "seed": seed, **result, "run_dir": str(logger.run_dir)}
    except Exception:
        logger.close(status="failed")
        raise


def divergent_selectors(choices: list[SelectorChoice | dict]) -> tuple[bool, list[str]]:
    masks = []
    for choice in choices:
        selector = choice.selector if isinstance(choice, SelectorChoice) else choice["selector"]
        mask = choice.selected_mask if isinstance(choice, SelectorChoice) else choice["selected_mask"]
        if selector in SUBSTANTIVE_SELECTORS:
            masks.append(mask)
    distinct = sorted(set(masks))
    return len(distinct) > 1, distinct


def holdout_for_point(point: CalibrationPoint, output_root: Path, workers: int) -> Path:
    """Full held-out protocol for one configuration."""
    root = output_root / point.point_id
    root.mkdir(parents=True, exist_ok=True)

    # 1. Selection games (full tables; selectors need exact robust Shapley).
    selection_payloads = [(point.point_id, point.values, seed, str(root / "selection")) for seed in SELECTION_SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        selection_results = list(pool.map(run_full_game_payload, selection_payloads))

    all_choices: list[SelectorChoice] = []
    for result in selection_results:
        all_choices.extend(SelectorChoice(**choice) for choice in result["choices"])
    choice_frame = pd.DataFrame([{**asdict(choice), "selected_interventions": ";".join(choice.selected_interventions)} for choice in all_choices])
    choice_frame.to_csv(root / "selection_choices.csv", index=False)

    masks = tuple(sorted({choice.selected_mask for choice in all_choices}))
    (root / "evaluation_masks.json").write_text(json.dumps({"masks": list(masks), "names": {str(mask): list(coalition_names(mask)) for mask in masks}}, indent=2))

    # 2. Lean evaluation on disjoint seeds.
    evaluation_payloads = [(point.point_id, point.values, seed, masks, str(root / "evaluation")) for seed in EVALUATION_SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        evaluation_results = list(pool.map(run_lean_evaluation_payload, evaluation_payloads))

    rows = []
    for choice in all_choices:
        for result in evaluation_results:
            margins = result["margins"][choice.selected_mask]
            rows.append({
                "selection_seed": choice.selection_seed,
                "evaluation_seed": result["seed"],
                "selector": choice.selector,
                "selected_mask": choice.selected_mask,
                "selected_interventions": ";".join(choice.selected_interventions),
                "robust_lower_improvement": result["robust_values"][choice.selected_mask],
                "feasible": result["feasible"][choice.selected_mask],
                "cost": margins["cost"],
                "relevance_delta_lower": margins["relevance_delta_lower"],
                "provider_disparity_upper": margins["provider_disparity_upper"],
                "fatigue_upper": margins["fatigue_upper"],
            })
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(root / "heldout_selector_evaluations.csv", index=False)
    summary = _paired_summary(evaluation, "cure_exact_maximin")
    summary.to_csv(root / "heldout_selector_summary.csv", index=False)
    return root


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    workers = 2
    points = load_screening_points()
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "screening_order": SCREENING_ORDER,
        "substantive_selectors": list(SUBSTANTIVE_SELECTORS),
        "selection_seeds": list(SELECTION_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "claim_scope": "Simulator-conditional held-out seed comparison; not external causal inference.",
    }

    # Stage 1: screening games, one seed each, in declared order (two workers).
    screening_payloads = [(point_id, points[point_id].values, 42, str(ASSETS / "screening")) for point_id in SCREENING_ORDER]
    screening_results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for payload, result in zip(screening_payloads, pool.map(run_full_game_payload, screening_payloads), strict=True):
            screening_results[payload[0]] = result
            choices = [SelectorChoice(**choice) for choice in result["choices"]]
            divergent, distinct = divergent_selectors(choices)
            print(f"SCREEN {payload[0]}: divergent={divergent} distinct_masks={distinct} ({result['duration_seconds']:.0f}s)", flush=True)

    screening_rows = []
    for point_id in SCREENING_ORDER:
        result = screening_results[point_id]
        for choice in result["choices"]:
            screening_rows.append({"point_id": point_id, **choice, "selected_interventions": ";".join(choice["selected_interventions"])})
    pd.DataFrame(screening_rows).to_csv(ASSETS / "screening_selector_choices.csv", index=False)

    promoted = []
    for point_id in SCREENING_ORDER:
        divergent, distinct = divergent_selectors([SelectorChoice(**choice) for choice in screening_results[point_id]["choices"]])
        if divergent:
            promoted.append((point_id, distinct))
        if len(promoted) >= MAX_HOLDOUT_CONFIGS:
            break
    manifest["promoted_configurations"] = [{"point_id": point_id, "screening_distinct_masks": masks} for point_id, masks in promoted]
    print(f"PROMOTED: {[point_id for point_id, _ in promoted]}", flush=True)

    # Stage 2: full held-out protocol for promoted configurations.
    for point_id, _ in promoted:
        root = holdout_for_point(points[point_id], ASSETS, workers)
        print(f"HOLDOUT COMPLETE: {root}", flush=True)

    (ASSETS / "revision_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("PHASE E DIVERGENCE SUITE DONE", flush=True)


if __name__ == "__main__":
    main()
