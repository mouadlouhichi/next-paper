from __future__ import annotations

import pandas as pd

from cure_rec.game import CoalitionValue, GameResult, ScenarioGame
from cure_rec.observability import RunLogger
from cure_rec.planner import DecisionMode, DecisionStatus, select_robust_portfolio


def _value(mask: int, improvement: float, provider_disparity: float) -> CoalitionValue:
    return CoalitionValue(
        scenario="synthetic",
        mask=mask,
        active_interventions=() if mask == 0 else ("repeat_cap",),
        cost=0.0 if mask == 0 else 0.05,
        utility=0.5 + improvement,
        improvement=improvement,
        satisfaction=0.5,
        retention=0.5,
        fatigue=0.1,
        relevance=0.5,
        provider_disparity=provider_disparity,
        catalog_coverage=0.2,
        duration_seconds=0.0,
        intervention_stats={},
    )


def _game(base_provider: float, repair_provider: float, repair_improvement: float) -> GameResult:
    values = {
        0: _value(0, 0.0, base_provider),
        1: _value(1, repair_improvement, repair_provider),
    }
    scenario = ScenarioGame("synthetic", values, {}, {}, {})
    return GameResult({"synthetic": scenario}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def _logger(settings, tmp_path) -> RunLogger:
    settings.run.output_root = tmp_path
    return RunLogger(settings)


def test_improvement_mode_abstains_when_base_feasible_and_no_gain(settings, tmp_path):
    game = _game(base_provider=0.1, repair_provider=0.1, repair_improvement=-0.05)
    logger = _logger(settings, tmp_path)
    decision = select_robust_portfolio(game, settings, logger)
    logger.close()
    assert decision.mode is DecisionMode.IMPROVEMENT
    assert decision.status is DecisionStatus.ABSTAIN_KEEP_BASE
    assert decision.selected_mask == 0


def test_repair_mode_selects_feasible_repair_even_if_negative(settings, tmp_path):
    game = _game(base_provider=0.6, repair_provider=0.1, repair_improvement=-0.05)
    logger = _logger(settings, tmp_path)
    decision = select_robust_portfolio(game, settings, logger)
    logger.close()
    assert decision.mode is DecisionMode.REPAIR
    assert decision.status is DecisionStatus.REPAIR_SELECTED
    assert decision.selected_mask == 1
    assert decision.lower_improvement < 0


def test_repair_mode_certifies_no_feasible_portfolio(settings, tmp_path):
    game = _game(base_provider=0.6, repair_provider=0.6, repair_improvement=0.1)
    logger = _logger(settings, tmp_path)
    decision = select_robust_portfolio(game, settings, logger)
    logger.close()
    assert decision.mode is DecisionMode.REPAIR
    assert decision.status is DecisionStatus.NO_FEASIBLE_PORTFOLIO
    assert decision.feasible is False
