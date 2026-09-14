"""Regression tests for the calibration-robust planner (reviewer item 2).

These tests build synthetic coalition tables rather than running CURE-Sim, so
they exercise the decision logic of ``cure_rec.seed_robust`` deterministically
and in milliseconds.  The scientific evidence itself lives in
``results/reviewer_phase_assets/seed_robust_planner/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cure_rec.config import INTERVENTION_NAMES, Settings
from cure_rec.game import CoalitionValue, GameResult, ScenarioGame
from cure_rec.planner import select_robust_portfolio
from cure_rec.seed_robust import (
    _QuietLogger,
    calibration_robust_values,
    games_from_tables,
    select_calibration_robust_portfolio,
    seed_scenario_metrics,
)

# masks: 0 = base, 1 = repeat_cap, 2 = explore_slot
SCENARIOS = ("nominal", "stress")


def _value(scenario: str, mask: int, *, utility: float, relevance: float, disparity: float, fatigue: float) -> CoalitionValue:
    return CoalitionValue(
        scenario=scenario,
        mask=mask,
        active_interventions=tuple(INTERVENTION_NAMES[i] for i in range(len(INTERVENTION_NAMES)) if mask & (1 << i)),
        cost=0.05 if mask else 0.0,
        utility=utility,
        improvement=0.0,  # recomputed below relative to the per-scenario base
        satisfaction=0.5,
        retention=0.5,
        fatigue=fatigue,
        relevance=relevance,
        provider_disparity=disparity,
        catalog_coverage=0.5,
        duration_seconds=0.0,
        intervention_stats={},
    )


def _game(per_scenario: dict[str, dict[int, tuple[float, float, float, float]]]) -> GameResult:
    """Build a GameResult shell; ``improvement`` is set relative to mask 0."""
    scenario_games = {}
    for name, rows in per_scenario.items():
        base_utility = rows[0][0]
        values = {
            mask: _value(name, mask, utility=u, relevance=r, disparity=d, fatigue=f)
            for mask, (u, r, d, f) in rows.items()
        }
        for mask, value in values.items():
            values[mask] = CoalitionValue(**{**value.__dict__, "improvement": value.utility - base_utility})
        scenario_games[name] = ScenarioGame(scenario=name, values=values, shapley={}, interactions={}, feasibility_semivalue={})
    return GameResult(scenario_games=scenario_games, regions=None, coalition_table=None, interaction_table=None)


def _settings(max_disparity: float = 0.30) -> Settings:
    settings = Settings()
    settings.constraints.max_provider_disparity = max_disparity
    settings.constraints.budget = 0.35
    settings.constraints.min_relevance_delta = -0.10
    settings.constraints.max_fatigue = 0.70
    return settings


def _seed_game(disparity_mask1: float) -> GameResult:
    """One seed where mask 1 has the best value but a seed-dependent disparity."""
    return _game(
        {
            "nominal": {0: (0.0, 0.60, 0.20, 0.40), 1: (0.30, 0.60, disparity_mask1, 0.40), 2: (0.10, 0.60, 0.20, 0.40)},
            "stress": {0: (0.0, 0.60, 0.20, 0.45), 1: (0.28, 0.60, disparity_mask1, 0.45), 2: (0.09, 0.60, 0.20, 0.45)},
        }
    )


def test_single_calibration_seed_reduces_to_shipped_planner():
    """With |Z_cal| = 1 the minimax rule must equal planner.select_robust_portfolio."""
    settings = _settings()
    game = _seed_game(0.25)
    shipped = select_robust_portfolio(game, settings, _QuietLogger())
    extended = select_calibration_robust_portfolio({42: game}, settings, _QuietLogger(), rule="minimax")
    assert shipped.selected_mask == extended.decision.selected_mask
    assert shipped.lower_improvement == pytest.approx(extended.decision.lower_improvement, abs=1e-12)


def test_pooled_constraints_reject_a_seed_unstable_portfolio():
    """mask 1 wins on seed 42 but violates disparity on seed 43; pooling must reject it."""
    settings = _settings(max_disparity=0.30)
    games = {42: _seed_game(0.25), 43: _seed_game(0.40)}

    # Single-seed selection picks the high-value but seed-unstable mask.
    single = select_calibration_robust_portfolio({42: games[42]}, settings, _QuietLogger(), rule="minimax")
    assert single.decision.selected_mask == 1

    # Pooled selection must fall back to a portfolio feasible on every seed.
    pooled = select_calibration_robust_portfolio(games, settings, _QuietLogger(), rule="minimax")
    assert pooled.decision.selected_mask == 2
    _, per_seed = seed_scenario_metrics(games, pooled.decision.selected_mask, settings)
    assert all(per_seed.values())
    assert pooled.calibration_feasible_seed_rate == 1.0
    _, unstable = seed_scenario_metrics(games, 1, settings)
    assert unstable[43] is False and unstable[42] is True


def test_calibration_value_is_the_pooled_minimum():
    games = {42: _seed_game(0.25), 43: _seed_game(0.40)}
    values = calibration_robust_values(games)
    # mask 1: worst over both seeds and scenarios is seed 43 / stress: 0.28 - 0 = 0.28
    assert values[1] == pytest.approx(0.28)
    # mask 2 is identical on both seeds, so pooling does not change it
    assert values[2] == pytest.approx(0.09)


def test_chance_constraint_sits_between_the_two_extremes():
    """alpha = 0 behaves like minimax; alpha >= 0.5 admits the 1-of-2-feasible mask."""
    settings = _settings(max_disparity=0.30)
    games = {42: _seed_game(0.25), 43: _seed_game(0.40)}

    strict = select_calibration_robust_portfolio(games, settings, _QuietLogger(), rule="chance", alpha=0.0)
    assert strict.decision.selected_mask == 2

    loose = select_calibration_robust_portfolio(games, settings, _QuietLogger(), rule="chance", alpha=0.5)
    assert loose.decision.selected_mask == 1
    assert loose.calibration_feasible_seed_rate == 0.5


def test_games_from_tables_roundtrip_matches_direct_construction():
    """The live-rollout helper and the archived-table reconstruction must agree."""
    settings = _settings()
    game = _seed_game(0.25)
    tables = {name: sg.values for name, sg in game.scenario_games.items()}
    rebuilt = games_from_tables(tables)
    direct = select_calibration_robust_portfolio({42: game}, settings, _QuietLogger(), rule="minimax")
    via_tables = select_calibration_robust_portfolio({42: rebuilt}, settings, _QuietLogger(), rule="minimax")
    assert direct.decision.selected_mask == via_tables.decision.selected_mask
    assert direct.decision.lower_improvement == pytest.approx(via_tables.decision.lower_improvement, abs=1e-12)


def test_invalid_rule_and_alpha_are_rejected():
    settings = _settings()
    games = {42: _seed_game(0.25)}
    with pytest.raises(ValueError):
        select_calibration_robust_portfolio(games, settings, _QuietLogger(), rule="mean")
    with pytest.raises(ValueError):
        select_calibration_robust_portfolio(games, settings, _QuietLogger(), rule="chance", alpha=1.0)
    with pytest.raises(ValueError):
        select_calibration_robust_portfolio({}, settings, _QuietLogger(), rule="minimax")
