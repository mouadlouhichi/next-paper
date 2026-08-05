"""Controlled analytic cooperative regimes for scientific CURE-Rec validation.

The behavioural CURE-Sim environment validates execution. This module adds
transparent oracle games with known additive, complementary, redundant,
antagonistic, delayed, repair, and misspecified structures. It is deliberately
explicit: no regime is inferred post hoc from a favourable simulation result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cure_rec.config import INTERVENTION_NAMES, Settings
from cure_rec.game import ALL_MASKS, CoalitionValue, GameResult, ScenarioGame, coalition_names, exact_interactions, exact_shapley, feasible_semivalue
from cure_rec.observability import RunLogger
from cure_rec.planner import PortfolioDecision, decision_to_dict, select_robust_portfolio


MaskValue = Callable[[int], float]
MaskMetric = Callable[[int], float]


def _has(mask: int, name: str) -> bool:
    return bool(mask & (1 << INTERVENTION_NAMES.index(name)))


def _count(mask: int) -> int:
    return mask.bit_count()


@dataclass(frozen=True)
class RegimeDefinition:
    name: str
    description: str
    expected_status: str
    expected_selected: tuple[str, ...]  # Expected optimizer under the estimated game
    oracle_selected: tuple[str, ...]    # True-game optimum used for oracle recovery
    horizon: int
    estimated_value: MaskValue
    true_value: MaskValue
    provider_disparity: MaskMetric
    relevance_delta: MaskMetric


def _additive(mask: int) -> float:
    effects = {
        "repeat_cap": 0.05,
        "explore_slot": 0.06,
        "tail_slot": 0.05,
        "diversify": 0.04,
        "novel_slot": 0.03,
        "provider_balance": 0.05,
    }
    return sum(value for name, value in effects.items() if _has(mask, name))


def _complementary(mask: int) -> float:
    value = 0.01 * _has(mask, "repeat_cap") + 0.01 * _has(mask, "explore_slot")
    if _has(mask, "repeat_cap") and _has(mask, "explore_slot"):
        value += 0.25
    value -= 0.02 * sum(_has(mask, name) for name in ("tail_slot", "diversify", "novel_slot", "provider_balance"))
    return value


def _redundant(mask: int) -> float:
    value = 0.15 if (_has(mask, "tail_slot") or _has(mask, "novel_slot")) else 0.0
    value -= 0.02 * sum(_has(mask, name) for name in ("repeat_cap", "explore_slot", "diversify", "provider_balance"))
    return value


def _antagonistic(mask: int) -> float:
    value = 0.14 * _has(mask, "explore_slot") + 0.12 * _has(mask, "diversify")
    if _has(mask, "explore_slot") and _has(mask, "diversify"):
        value -= 0.32
    value -= 0.015 * sum(_has(mask, name) for name in ("repeat_cap", "tail_slot", "novel_slot", "provider_balance"))
    return value


def _delayed_short(mask: int) -> float:
    return -0.04 * _has(mask, "repeat_cap") - 0.01 * (_count(mask) - _has(mask, "repeat_cap"))


def _delayed_long(mask: int) -> float:
    return 0.20 * _has(mask, "repeat_cap") - 0.01 * (_count(mask) - _has(mask, "repeat_cap"))


def _provider_repair_balancing(mask: int) -> float:
    # Both repairs are feasible, but provider balancing is the lower-cost utility
    # repair in this regime; combining them introduces a deliberate overlap cost.
    if _has(mask, "repeat_cap") and _has(mask, "provider_balance"):
        return 0.02 - 0.02 * (_count(mask) - 2)
    if _has(mask, "provider_balance"):
        return 0.04 - 0.02 * (_count(mask) - 1)
    if _has(mask, "repeat_cap"):
        return 0.02 - 0.02 * (_count(mask) - 1)
    return -0.02 * _count(mask)


def _provider_repair_repeat_utility(mask: int) -> float:
    # Here repeat suppression is the stronger repair and extra provider balancing
    # adds a mild utility cost despite also restoring feasibility.
    if _has(mask, "repeat_cap") and _has(mask, "provider_balance"):
        return 0.08 - 0.02 * (_count(mask) - 2)
    if _has(mask, "repeat_cap"):
        return 0.10 - 0.02 * (_count(mask) - 1)
    if _has(mask, "provider_balance"):
        return 0.03 - 0.02 * (_count(mask) - 1)
    return -0.02 * _count(mask)


def _provider_nominal(mask: int) -> float:
    return 0.22


def _provider_repair_balance(mask: int) -> float:
    if _has(mask, "provider_balance"):
        return 0.20
    if _has(mask, "repeat_cap"):
        return 0.34
    return 0.42


def _provider_repair_repeat_disparity(mask: int) -> float:
    if _has(mask, "repeat_cap"):
        return 0.20
    if _has(mask, "provider_balance"):
        return 0.27
    return 0.42


def _relevance_ok(mask: int) -> float:
    return -0.01 * _count(mask)


def _misspecified_true(mask: int) -> float:
    return 0.18 * _has(mask, "explore_slot") - 0.01 * (_count(mask) - _has(mask, "explore_slot"))


def _misspecified_estimated(mask: int) -> float:
    return 0.16 * _has(mask, "repeat_cap") - 0.01 * (_count(mask) - _has(mask, "repeat_cap"))


def regime_library() -> tuple[RegimeDefinition, ...]:
    return (
        RegimeDefinition("additive", "Independent positive interventions with zero interactions.", "improve_selected", ("repeat_cap", "explore_slot", "tail_slot", "provider_balance"), ("repeat_cap", "explore_slot", "tail_slot", "provider_balance"), 12, _additive, _additive, _provider_nominal, _relevance_ok),
        RegimeDefinition("complementary", "Repeat cap and exploration are jointly valuable but weak alone.", "improve_selected", ("repeat_cap", "explore_slot"), ("repeat_cap", "explore_slot"), 12, _complementary, _complementary, _provider_nominal, _relevance_ok),
        RegimeDefinition("redundant", "Long-tail and novelty overlap; one should be selected by tie/cost rule.", "improve_selected", ("tail_slot",), ("tail_slot",), 12, _redundant, _redundant, _provider_nominal, _relevance_ok),
        RegimeDefinition("antagonistic", "Exploration and diversity are individually useful but jointly harmful.", "improve_selected", ("explore_slot",), ("explore_slot",), 12, _antagonistic, _antagonistic, _provider_nominal, _relevance_ok),
        RegimeDefinition("delayed_fatigue_short", "Repeat cap is costly at short horizon.", "abstain_keep_base", (), (), 4, _delayed_short, _delayed_short, _provider_nominal, _relevance_ok),
        RegimeDefinition("delayed_fatigue_long", "Repeat cap yields delayed benefit at long horizon.", "improve_selected", ("repeat_cap",), ("repeat_cap",), 12, _delayed_long, _delayed_long, _provider_nominal, _relevance_ok),
        RegimeDefinition("provider_repair_balancing", "Infeasible base repaired most efficiently by provider balancing.", "repair_selected", ("provider_balance",), ("provider_balance",), 12, _provider_repair_balancing, _provider_repair_balancing, _provider_repair_balance, _relevance_ok),
        RegimeDefinition("provider_repair_repeat", "Infeasible base repaired most efficiently by repeat suppression.", "repair_selected", ("repeat_cap",), ("repeat_cap",), 12, _provider_repair_repeat_utility, _provider_repair_repeat_utility, _provider_repair_repeat_disparity, _relevance_ok),
        RegimeDefinition("misspecified_ambiguity", "Estimated model favours repeat cap while true model favours exploration.", "improve_selected", ("repeat_cap",), ("explore_slot",), 12, _misspecified_estimated, _misspecified_true, _provider_nominal, _relevance_ok),
    )


def _game_from_value(settings: Settings, regime: RegimeDefinition, value_fn: MaskValue, label: str) -> GameResult:
    values: dict[int, CoalitionValue] = {}
    for mask in ALL_MASKS:
        improvement = float(value_fn(mask))
        values[mask] = CoalitionValue(
            scenario=label,
            mask=mask,
            active_interventions=coalition_names(mask),
            cost=float(sum(settings.interventions.costs[name] for name in coalition_names(mask))),
            utility=0.5 + improvement,
            improvement=improvement,
            satisfaction=0.5 + max(improvement, -0.25),
            retention=0.5 + max(improvement, -0.25),
            fatigue=0.1 if not _has(mask, "repeat_cap") else 0.05,
            relevance=0.5 + regime.relevance_delta(mask),
            provider_disparity=regime.provider_disparity(mask),
            catalog_coverage=0.2 + 0.01 * _count(mask),
            duration_seconds=0.0,
            intervention_stats={},
        )
    improvements = {mask: value.improvement for mask, value in values.items()}
    game = ScenarioGame(
        scenario=label,
        values=values,
        shapley=exact_shapley(improvements),
        interactions=exact_interactions(improvements),
        feasibility_semivalue=feasible_semivalue(improvements, settings),
    )
    rows = []
    for value in values.values():
        row = asdict(value)
        row["active_interventions"] = ";".join(value.active_interventions)
        rows.append(row)
    regions = pd.DataFrame([
        {
            "intervention": name,
            "phi_lower": game.shapley[name],
            "phi_upper": game.shapley[name],
            "phi_mean": game.shapley[name],
            "psi_feasible_lower": game.feasibility_semivalue[name],
            "psi_feasible_upper": game.feasibility_semivalue[name],
            "phi_psi_sign_agree": np.sign(game.shapley[name]) == np.sign(game.feasibility_semivalue[name]),
        }
        for name in INTERVENTION_NAMES
    ])
    interactions = pd.DataFrame([
        {
            "intervention_i": pair[0],
            "intervention_j": pair[1],
            "interaction_lower": value,
            "interaction_upper": value,
            "interaction_mean": value,
        }
        for pair, value in game.interactions.items()
    ])
    return GameResult({label: game}, regions, pd.DataFrame(rows), interactions)


@dataclass(frozen=True)
class RegimeRunResult:
    run_dir: Path
    summary: pd.DataFrame
    attribution_recovery: pd.DataFrame


def run_regime_suite(settings: Settings, logger: RunLogger) -> RegimeRunResult:
    """Run transparent known-structure regimes and emit recovery assets."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = logger.run_dir / "benchmark_regimes" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_rows: list[dict] = []
    recovery_rows: list[dict] = []
    interaction_rows: list[dict] = []

    for regime in regime_library():
        estimated = _game_from_value(settings, regime, regime.estimated_value, f"{regime.name}_estimated")
        true = _game_from_value(settings, regime, regime.true_value, f"{regime.name}_true")
        estimated_decision = select_robust_portfolio(estimated, settings, logger)
        oracle_decision = select_robust_portfolio(true, settings, logger)
        expected_estimated_set = set(regime.expected_selected)
        observed_set = set(estimated_decision.selected_interventions)
        oracle_set = set(oracle_decision.selected_interventions)
        estimated_union = expected_estimated_set | observed_set
        oracle_union = oracle_set | observed_set
        true_values = next(iter(true.scenario_games.values())).values
        oracle_regret = true_values[oracle_decision.selected_mask].improvement - true_values[estimated_decision.selected_mask].improvement
        summary_rows.append({
            "regime": regime.name,
            "description": regime.description,
            "horizon": regime.horizon,
            "expected_status": regime.expected_status,
            "expected_estimated_selected": ";".join(regime.expected_selected),
            "oracle_selected": ";".join(oracle_decision.selected_interventions),
            "observed_estimated_selected": ";".join(estimated_decision.selected_interventions),
            "observed_status": estimated_decision.status.value,
            "base_feasible": estimated_decision.base_feasible,
            "lower_improvement": estimated_decision.lower_improvement,
            "estimated_selection_match": expected_estimated_set == observed_set,
            "oracle_selection_match": oracle_set == observed_set,
            "estimated_jaccard": len(expected_estimated_set & observed_set) / len(estimated_union) if estimated_union else 1.0,
            "oracle_jaccard": len(oracle_set & observed_set) / len(oracle_union) if oracle_union else 1.0,
            "oracle_regret": oracle_regret,
        })
        true_shapley = next(iter(true.scenario_games.values())).shapley
        est_shapley = next(iter(estimated.scenario_games.values())).shapley
        for name in INTERVENTION_NAMES:
            recovery_rows.append({
                "regime": regime.name,
                "intervention": name,
                "true_phi": true_shapley[name],
                "estimated_phi": est_shapley[name],
                "absolute_error": abs(true_shapley[name] - est_shapley[name]),
                "sign_correct": np.sign(true_shapley[name]) == np.sign(est_shapley[name]),
                "covered_by_estimated_point_region": np.isclose(true_shapley[name], est_shapley[name]),
            })
        for row in estimated.interaction_table.to_dict(orient="records"):
            interaction_rows.append({"regime": regime.name, **row})

    summary = pd.DataFrame(summary_rows)
    recovery = pd.DataFrame(recovery_rows)
    interactions = pd.DataFrame(interaction_rows)
    summary.to_csv(run_dir / "regime_selection_summary.csv", index=False)
    recovery.to_csv(run_dir / "regime_attribution_recovery.csv", index=False)
    interactions.to_csv(run_dir / "regime_interactions.csv", index=False)
    manifest = [
        {
            "name": regime.name,
            "description": regime.description,
            "expected_status": regime.expected_status,
            "expected_estimated_selected": regime.expected_selected,
            "oracle_selected": regime.oracle_selected,
            "horizon": regime.horizon,
            "estimated_value_function": regime.estimated_value.__name__,
            "true_value_function": regime.true_value.__name__,
            "provider_disparity_function": regime.provider_disparity.__name__,
            "relevance_function": regime.relevance_delta.__name__,
        }
        for regime in regime_library()
    ]
    (run_dir / "regime_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(summary["regime"], summary["oracle_jaccard"], color="#2E86AB")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Jaccard similarity to oracle portfolio")
    ax.set_title("Controlled-regime portfolio recovery")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(run_dir / "regime_figure_selection_recovery.png", dpi=180)
    plt.close(fig)

    logger.event("benchmark_regime_suite_completed", run_dir=str(run_dir), regimes=list(summary["regime"]))
    return RegimeRunResult(run_dir, summary, recovery)
