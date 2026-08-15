"""Exact six-player intervention game, attribution regions, and coalition artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from math import factorial
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from cure_rec.config import INTERVENTION_NAMES, ScenarioConfig, Settings
from cure_rec.interventions import CANONICAL_ORDER, Coalition, transform_slate
from cure_rec.observability import RunLogger
from cure_rec.policies import HistoryAwarePolicy
from cure_rec.simulator import CureSim, RolloutSummary

N_PLAYERS = len(INTERVENTION_NAMES)
ALL_MASKS = tuple(range(1 << N_PLAYERS))
EMPTY_MASK = 0
FULL_MASK = (1 << N_PLAYERS) - 1


@dataclass(frozen=True)
class CoalitionValue:
    scenario: str
    mask: int
    active_interventions: tuple[str, ...]
    cost: float
    utility: float
    improvement: float
    satisfaction: float
    retention: float
    fatigue: float
    relevance: float
    provider_disparity: float
    catalog_coverage: float
    duration_seconds: float
    intervention_stats: dict[str, int]


@dataclass
class ScenarioGame:
    scenario: str
    values: dict[int, CoalitionValue]
    shapley: dict[str, float]
    interactions: dict[tuple[str, str], float]
    feasibility_semivalue: dict[str, float]


@dataclass
class GameResult:
    scenario_games: dict[str, ScenarioGame]
    regions: pd.DataFrame
    coalition_table: pd.DataFrame
    interaction_table: pd.DataFrame
    # Attribution of the actual maximin characteristic function, not merely
    # an envelope over scenario-specific attributions.
    robust_improvements: dict[int, float] = field(default_factory=dict)
    robust_shapley: dict[str, float] = field(default_factory=dict)
    robust_interactions: dict[tuple[str, str], float] = field(default_factory=dict)


def coalition_names(mask: int) -> tuple[str, ...]:
    return Coalition.from_mask(mask).names()


def shapley_weights(player_index: int) -> Iterable[tuple[int, float]]:
    without = [index for index in range(N_PLAYERS) if index != player_index]
    for size in range(N_PLAYERS):
        for subset in combinations(without, size):
            mask = sum(1 << index for index in subset)
            weight = factorial(size) * factorial(N_PLAYERS - size - 1) / factorial(N_PLAYERS)
            yield mask, weight


def exact_shapley(improvements: dict[int, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    for player_index, player_name in enumerate(INTERVENTION_NAMES):
        contribution = 0.0
        bit = 1 << player_index
        for mask, weight in shapley_weights(player_index):
            contribution += weight * (improvements[mask | bit] - improvements[mask])
        values[player_name] = float(contribution)
    return values


def interaction_weight(size: int) -> float:
    return factorial(size) * factorial(N_PLAYERS - size - 2) / factorial(N_PLAYERS - 1)


def exact_interactions(improvements: dict[int, float]) -> dict[tuple[str, str], float]:
    interactions: dict[tuple[str, str], float] = {}
    for i, j in combinations(range(N_PLAYERS), 2):
        remaining = [index for index in range(N_PLAYERS) if index not in (i, j)]
        score = 0.0
        for size in range(N_PLAYERS - 1):
            for subset in combinations(remaining, size):
                mask = sum(1 << index for index in subset)
                second_difference = (
                    improvements[mask | (1 << i) | (1 << j)]
                    - improvements[mask | (1 << i)]
                    - improvements[mask | (1 << j)]
                    + improvements[mask]
                )
                score += interaction_weight(size) * second_difference
        interactions[(INTERVENTION_NAMES[i], INTERVENTION_NAMES[j])] = float(score)
    return interactions


def _deterministically_deployable(mask: int, settings: Settings) -> bool:
    coalition = Coalition.from_mask(mask)
    return coalition.cost(settings.interventions) <= settings.constraints.budget


def feasible_semivalue(improvements: dict[int, float], settings: Settings) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, name in enumerate(INTERVENTION_NAMES):
        bit = 1 << index
        predecessor_masks = [
            mask
            for mask in ALL_MASKS
            if not (mask & bit)
            and _deterministically_deployable(mask, settings)
            and _deterministically_deployable(mask | bit, settings)
        ]
        values[name] = float(np.mean([improvements[mask | bit] - improvements[mask] for mask in predecessor_masks])) if predecessor_masks else float("nan")
    return values


def _policy_for_coalition(policy: HistoryAwarePolicy, coalition: Coalition, settings: Settings):
    def policy_fn(state, user_id, rng):
        result = transform_slate(policy, state, user_id, coalition, settings.interventions, rng)
        return result.slate, result.manifest

    return policy_fn


def evaluate_coalition(
    base_simulator: CureSim,
    scenario: ScenarioConfig,
    policy: HistoryAwarePolicy,
    mask: int,
    baseline_utility: float,
    settings: Settings,
    logger: RunLogger,
) -> CoalitionValue:
    coalition = Coalition.from_mask(mask)
    started = perf_counter()
    logger.event("coalition_evaluation_started", scenario=scenario.name, coalition_mask=mask, active_interventions=coalition.names())
    simulator = base_simulator.clone()
    simulator.scenario = scenario
    if not settings.run.common_random_numbers:
        # Preserve the scenario initial state but give this coalition its own
        # reproducible exogenous shock stream for the CRN-on/off ablation.
        simulator.rng_offset = 1_000_003 * mask
        # `clone()` copies the generator state created during simulator
        # construction. Setting rng_offset alone therefore has no effect until
        # the simulator is reset. Without this reset the CRN-off ablation is
        # accidentally identical to CRN-on.
        simulator.reset()
    summary: RolloutSummary = simulator.rollout(_policy_for_coalition(policy, coalition, settings))
    cost = coalition.cost(settings.interventions)
    utility = summary.utility_before_cost - settings.utility.cost_weight * cost
    value = CoalitionValue(
        scenario=scenario.name,
        mask=mask,
        active_interventions=coalition.names(),
        cost=cost,
        utility=float(utility),
        improvement=float(utility - baseline_utility),
        satisfaction=summary.satisfaction,
        retention=summary.retention,
        fatigue=summary.fatigue,
        relevance=summary.relevance,
        provider_disparity=summary.provider_disparity,
        catalog_coverage=summary.catalog_coverage,
        duration_seconds=perf_counter() - started,
        intervention_stats=summary.intervention_stats,
    )
    logger.event("coalition_evaluated", **asdict(value))
    logger.metric("coalition_improvement", value.improvement, scenario=scenario.name, coalition_mask=mask)
    logger.write_json(
        f"artifacts/coalitions/{scenario.name}/mask_{mask:02d}.json",
        {
            "scenario": scenario.name,
            "coalition_mask": mask,
            "active_interventions": coalition.names(),
            "canonical_order": CANONICAL_ORDER,
            "base_policy": "HistoryAwarePolicy",
            "config_hash": settings.config_hash(),
            "operator_parameters": settings.interventions.model_dump(),
            "value": asdict(value),
            "trajectory": summary.step_summaries,
            "sample_transform_traces": [summary["transform_samples"] for summary in summary.step_summaries],
        },
    )
    return value


def run_scenario_game(settings: Settings, scenario: ScenarioConfig, logger: RunLogger) -> ScenarioGame:
    simulator = CureSim(settings, scenario)
    policy = HistoryAwarePolicy(simulator, settings.policy)
    logger.event("simulator_ready", scenario=scenario.name, n_users=settings.simulator.n_users, n_items=settings.simulator.n_items, horizon=settings.simulator.horizon)
    baseline = evaluate_coalition(simulator, scenario, policy, EMPTY_MASK, 0.0, settings, logger)
    # The empty coalition is the deployed base policy. Its improvement is zero
    # by definition, while its raw utility remains available for auditability.
    baseline = replace(baseline, improvement=0.0)
    values: dict[int, CoalitionValue] = {EMPTY_MASK: baseline}
    for mask in ALL_MASKS:
        if mask:
            values[mask] = evaluate_coalition(simulator, scenario, policy, mask, baseline.utility, settings, logger)
    improvements = {mask: value.improvement for mask, value in values.items()}
    shapley = exact_shapley(improvements)
    interactions = exact_interactions(improvements)
    feasible = feasible_semivalue(improvements, settings)
    efficiency_gap = sum(shapley.values()) - improvements[FULL_MASK]
    logger.event("scenario_game_completed", scenario=scenario.name, grand_coalition_improvement=improvements[FULL_MASK], shapley_efficiency_gap=efficiency_gap)
    if not np.isclose(efficiency_gap, 0.0, atol=1e-8):
        raise AssertionError(f"Shapley efficiency failed for {scenario.name}: {efficiency_gap}")
    return ScenarioGame(scenario.name, values, shapley, interactions, feasible)


def run_exact_game(settings: Settings, logger: RunLogger) -> GameResult:
    scenario_games = {scenario.name: run_scenario_game(settings, scenario, logger) for scenario in settings.scenarios}
    coalition_rows: list[dict] = []
    for game in scenario_games.values():
        for value in game.values.values():
            row = asdict(value)
            row["active_interventions"] = ";".join(value.active_interventions)
            row["intervention_stats"] = str(value.intervention_stats)
            coalition_rows.append(row)
    coalition_table = pd.DataFrame(coalition_rows).sort_values(["scenario", "mask"]).reset_index(drop=True)
    region_rows: list[dict] = []
    for name in INTERVENTION_NAMES:
        phi_values = [game.shapley[name] for game in scenario_games.values()]
        psi_values = [game.feasibility_semivalue[name] for game in scenario_games.values()]
        region_rows.append({
            "intervention": name,
            "phi_lower": float(np.min(phi_values)),
            "phi_upper": float(np.max(phi_values)),
            "phi_mean": float(np.mean(phi_values)),
            "psi_feasible_lower": float(np.nanmin(psi_values)),
            "psi_feasible_upper": float(np.nanmax(psi_values)),
            "phi_psi_sign_agree": bool(np.sign(np.mean(phi_values)) == np.sign(np.nanmean(psi_values))),
        })
    regions = pd.DataFrame(region_rows)
    interaction_rows: list[dict] = []
    pairs = next(iter(scenario_games.values())).interactions.keys()
    for pair in pairs:
        pair_values = [game.interactions[pair] for game in scenario_games.values()]
        interaction_rows.append({
            "intervention_i": pair[0],
            "intervention_j": pair[1],
            "interaction_lower": float(np.min(pair_values)),
            "interaction_upper": float(np.max(pair_values)),
            "interaction_mean": float(np.mean(pair_values)),
        })
    interaction_table = pd.DataFrame(interaction_rows)

    # The planner maximizes the worst-scenario improvement. Its explanation must
    # therefore include the Shapley allocation of this robust characteristic
    # function, rather than treating scenario-wise min/max regions as an additive
    # decomposition of a nonlinear minimum.
    robust_improvements = {
        mask: float(min(game.values[mask].improvement for game in scenario_games.values()))
        for mask in ALL_MASKS
    }
    robust_shapley = exact_shapley(robust_improvements)
    robust_interactions = exact_interactions(robust_improvements)
    robust_gap = float(sum(robust_shapley.values()) - robust_improvements[FULL_MASK])
    if not np.isclose(robust_gap, 0.0, atol=1e-8):
        raise AssertionError(f"Robust-game Shapley efficiency failed: {robust_gap}")
    robust_table = pd.DataFrame([
        {
            "intervention": name,
            "robust_phi": robust_shapley[name],
            "scenario_phi_lower": float(regions.loc[regions["intervention"] == name, "phi_lower"].iloc[0]),
            "scenario_phi_upper": float(regions.loc[regions["intervention"] == name, "phi_upper"].iloc[0]),
            "budget_feasible_semivalue_lower": float(regions.loc[regions["intervention"] == name, "psi_feasible_lower"].iloc[0]),
            "budget_feasible_semivalue_upper": float(regions.loc[regions["intervention"] == name, "psi_feasible_upper"].iloc[0]),
        }
        for name in INTERVENTION_NAMES
    ])
    robust_interaction_table = pd.DataFrame([
        {"intervention_i": i, "intervention_j": j, "robust_interaction": value}
        for (i, j), value in robust_interactions.items()
    ])
    logger.event("robust_game_completed", robust_grand_improvement=robust_improvements[FULL_MASK], robust_shapley_efficiency_gap=robust_gap)
    logger.write_dataframe("tables/coalition_values.csv", coalition_table)
    logger.write_dataframe("tables/shapley_regions.csv", regions)
    logger.write_dataframe("tables/interaction_regions.csv", interaction_table)
    logger.write_dataframe("tables/robust_game_attribution.csv", robust_table)
    logger.write_dataframe("tables/robust_game_interactions.csv", robust_interaction_table)
    logger.write_json("artifacts/game_manifest.json", {
        "players": INTERVENTION_NAMES,
        "all_masks": list(ALL_MASKS),
        "canonical_composition": ["repeat_cap", "eligibility_filter", "injection_allocation", "diversify", "provider_balance"],
        "scenarios": list(scenario_games),
        "robust_characteristic_function": "min_scenario_improvement",
        "robust_grand_improvement": robust_improvements[FULL_MASK],
        "robust_shapley_efficiency_gap": robust_gap,
    })
    return GameResult(
        scenario_games, regions, coalition_table, interaction_table,
        robust_improvements=robust_improvements,
        robust_shapley=robust_shapley,
        robust_interactions=robust_interactions,
    )
