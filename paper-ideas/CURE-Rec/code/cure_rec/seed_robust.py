"""Calibration-robust portfolio planning over scenarios AND initialization seeds.

The shipped planner (:func:`cure_rec.planner.select_robust_portfolio`) maximizes

.. math::

    v^{rob}(S) = \\min_{m \\in \\mathcal{M}} \\Delta V_m(S)

and tests feasibility against the same finite scenario set of the *single* seed
that built the coalition table.  A portfolio can therefore be robust to the
declared scenario ambiguity and still fail on an unseen initialization seed
``zeta``: the archived disagreement studies select portfolios whose held-out
feasibility is 0.70 and 0.47 even though they are exactly feasible at selection
time.

This module extends both objects to a calibration seed set ``Z_cal``:

.. math::

    \\underline{\\Delta}_{cal}(S) &= \\min_{m \\in \\mathcal{M},\\, \\zeta \\in Z_{cal}}
        \\bigl[V_{m,\\zeta}(S) - V_{m,\\zeta}(\\varnothing)\\bigr] \\\\
    1_{F,cal}(S) &= 1\\{c(S) \\le B,\\;
        \\min_{m,\\zeta} \\Delta Rel_{m,\\zeta}(S) \\ge r_{min},\\;
        \\max_{m,\\zeta} Disp_{m,\\zeta}(S) \\le d_{max},\\;
        \\max_{m,\\zeta} Fat_{m,\\zeta}(S) \\le f_{max}\\}

and adds the less conservative empirical chance-constrained alternative

.. math::

    \\frac{1}{|Z_{cal}|} \\sum_{\\zeta \\in Z_{cal}} 1_F(S;\\zeta) \\ge 1 - \\alpha,

whose value is the seed-averaged worst-scenario improvement.  Construction uses
calibration seeds only; held-out seeds are used exclusively to score frozen
decisions and never re-enter selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from cure_rec.config import Settings
from cure_rec.game import (
    EMPTY_MASK,
    CoalitionValue,
    GameResult,
    coalition_names,
    evaluate_coalition,
)
from cure_rec.observability import RunLogger
from cure_rec.planner import DecisionMode, DecisionStatus, PortfolioDecision
from cure_rec.policies import HistoryAwarePolicy
from cure_rec.simulator import CureSim

TOL = 1e-12


class _QuietLogger:
    """Drop per-coalition artifacts; the driver archives compact summaries."""

    def event(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def write_json(self, *args, **kwargs):
        return None


@dataclass(frozen=True)
class SeedRobustSelection:
    """A planner decision together with the calibration evidence behind it."""

    decision: PortfolioDecision
    rule: str
    alpha: float
    calibration_seeds: tuple[int, ...]
    calibration_lower_improvement: float
    calibration_feasible_seed_rate: float
    per_seed_feasible: dict[int, bool] = field(default_factory=dict)
    runner_up_mask: int | None = None
    runner_up_gap: float = float("nan")


def evaluate_seed_masks(
    settings: Settings,
    seed: int,
    masks: Sequence[int],
    logger: RunLogger | None = None,
) -> dict[str, dict[int, CoalitionValue]]:
    """Evaluate ``masks`` for one initialization seed across all scenarios.

    Returns ``{scenario_name: {mask: CoalitionValue}}``.  The empty coalition is
    always evaluated first because every improvement is defined relative to the
    base policy of that same seed and scenario.
    """
    log = logger if logger is not None else _QuietLogger()
    # CureSim seeds its generator from settings.run.seed, so the initialization
    # seed must be applied to a private copy rather than assumed by the caller.
    cfg = settings.model_copy(deep=True)
    cfg.run.seed = int(seed)
    tables: dict[str, dict[int, CoalitionValue]] = {}
    wanted = [EMPTY_MASK] + [mask for mask in masks if mask != EMPTY_MASK]
    for scenario in cfg.scenarios:
        simulator = CureSim(cfg, scenario)
        policy = HistoryAwarePolicy(simulator, cfg.policy)
        base = evaluate_coalition(simulator, scenario, policy, EMPTY_MASK, 0.0, cfg, log)
        base = _zero_improvement(base)
        values: dict[int, CoalitionValue] = {EMPTY_MASK: base}
        for mask in wanted:
            if mask == EMPTY_MASK:
                continue
            values[mask] = evaluate_coalition(simulator, scenario, policy, mask, base.utility, cfg, log)
        tables[scenario.name] = values
    return tables


def _zero_improvement(value: CoalitionValue) -> CoalitionValue:
    from dataclasses import replace

    return replace(value, improvement=0.0)


def games_from_tables(tables: dict[str, dict[int, CoalitionValue]]) -> GameResult:
    """Wrap per-scenario value tables in a ``GameResult`` shell.

    Only ``scenario_games[...].values`` is populated, which is all the
    constraint/value machinery needs.  Attribution objects are left empty
    because calibration-robust selection is a decision rule, not an explainer.
    """
    from cure_rec.game import ScenarioGame

    scenario_games = {
        name: ScenarioGame(scenario=name, values=values, shapley={}, interactions={}, feasibility_semivalue={})
        for name, values in tables.items()
    }
    return GameResult(
        scenario_games=scenario_games,
        regions=None,
        coalition_table=None,
        interaction_table=None,
    )


def _masks(games: dict[int, GameResult]) -> list[int]:
    return sorted({mask for game in games.values() for scenario in game.scenario_games.values() for mask in scenario.values})


def calibration_robust_values(games: dict[int, GameResult]) -> dict[int, float]:
    """``min`` over scenarios *and* calibration seeds of the improvement."""
    values: dict[int, float] = {}
    for mask in _masks(games):
        values[mask] = float(
            min(
                scenario.values[mask].improvement
                for game in games.values()
                for scenario in game.scenario_games.values()
            )
        )
    return values


def seed_scenario_metrics(
    games: dict[int, GameResult], mask: int, settings: Settings
) -> tuple[dict[str, float], dict[int, bool]]:
    """Constraint margins pooled over seeds, plus per-seed feasibility flags."""
    constraints = settings.constraints
    relevance = [
        scenario.values[mask].relevance - scenario.values[EMPTY_MASK].relevance
        for game in games.values()
        for scenario in game.scenario_games.values()
    ]
    disparity = [
        scenario.values[mask].provider_disparity
        for game in games.values()
        for scenario in game.scenario_games.values()
    ]
    fatigue = [
        scenario.values[mask].fatigue for game in games.values() for scenario in game.scenario_games.values()
    ]
    cost = float(next(iter(next(iter(games.values())).scenario_games.values())).values[mask].cost)
    metrics = {
        "cost": cost,
        "relevance_delta_lower": float(np.min(relevance)),
        "provider_disparity_upper": float(np.max(disparity)),
        "fatigue_upper": float(np.max(fatigue)),
    }
    per_seed: dict[int, bool] = {}
    for seed, game in games.items():
        ok = cost <= constraints.budget + TOL
        for scenario in game.scenario_games.values():
            delta = scenario.values[mask].relevance - scenario.values[EMPTY_MASK].relevance
            ok = (
                ok
                and delta >= constraints.min_relevance_delta - TOL
                and scenario.values[mask].provider_disparity <= constraints.max_provider_disparity + TOL
                and scenario.values[mask].fatigue <= constraints.max_fatigue + TOL
            )
        per_seed[int(seed)] = bool(ok)
    return metrics, per_seed


def _all_feasible(metrics: dict[str, float], settings: Settings) -> bool:
    constraints = settings.constraints
    return (
        metrics["cost"] <= constraints.budget + TOL
        and metrics["relevance_delta_lower"] >= constraints.min_relevance_delta - TOL
        and metrics["provider_disparity_upper"] <= constraints.max_provider_disparity + TOL
        and metrics["fatigue_upper"] <= constraints.max_fatigue + TOL
    )


def seed_averaged_worst_scenario_values(games: dict[int, GameResult]) -> dict[int, float]:
    """Mean over calibration seeds of the worst-scenario improvement."""
    values: dict[int, float] = {}
    for mask in _masks(games):
        per_seed = [
            min(scenario.values[mask].improvement for scenario in game.scenario_games.values())
            for game in games.values()
        ]
        values[mask] = float(np.mean(per_seed))
    return values


def select_calibration_robust_portfolio(
    games: dict[int, GameResult],
    settings: Settings,
    logger: RunLogger | None = None,
    *,
    rule: str = "minimax",
    alpha: float = 0.2,
) -> SeedRobustSelection:
    """Select a portfolio under scenario *and* initialization-seed robustness.

    ``rule="minimax"``  requires every constraint to hold for every
    (scenario, calibration seed) pair and ranks by the pooled worst-case
    improvement -- the conservative extension of the shipped planner.

    ``rule="chance"`` requires only a ``1 - alpha`` fraction of calibration seeds
    to be fully scenario-feasible and ranks by the seed-averaged worst-scenario
    improvement -- the less conservative alternative.

    Improvement/repair/abstention semantics are unchanged from
    :func:`cure_rec.planner.select_robust_portfolio`; only the uncertainty set
    over which ``v`` and ``1_F`` are computed is enlarged.
    """
    if rule not in ("minimax", "chance"):
        raise ValueError(f"unknown rule {rule!r}; expected 'minimax' or 'chance'")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    if not games:
        raise ValueError("at least one calibration game is required")
    log = logger if logger is not None else _QuietLogger()

    if rule == "minimax":
        values = calibration_robust_values(games)
    else:
        values = seed_averaged_worst_scenario_values(games)
    masks = _masks(games)

    base_metrics, base_per_seed = seed_scenario_metrics(games, EMPTY_MASK, settings)
    base_ok = _all_feasible(base_metrics, settings) if rule == "minimax" else _seed_rate_ok(base_per_seed, alpha)
    mode = DecisionMode.IMPROVEMENT if base_ok else DecisionMode.REPAIR

    candidates: list[tuple[float, int, dict[str, float], dict[int, bool]]] = []
    for mask in masks:
        metrics, per_seed = seed_scenario_metrics(games, mask, settings)
        ok = _all_feasible(metrics, settings) if rule == "minimax" else _seed_rate_ok(per_seed, alpha)
        if not ok:
            continue
        candidates.append((values[mask], mask, metrics, per_seed))

    def bounds(mask: int) -> tuple[float, float]:
        pool = [
            scenario.values[mask].improvement
            for game in games.values()
            for scenario in game.scenario_games.values()
        ]
        return float(np.min(pool)), float(np.max(pool))

    lower_sel, upper_sel = bounds(EMPTY_MASK)
    runner_up: int | None = None
    gap = float("nan")

    if mode is DecisionMode.IMPROVEMENT:
        ordered = sorted(candidates, key=lambda row: (row[0], -row[1]), reverse=True)
        value, mask, metrics, per_seed = ordered[0]
        if len(ordered) > 1:
            runner_up = ordered[1][1]
            gap = float(ordered[0][0] - ordered[1][0])
        lower_sel, upper_sel = bounds(mask)
        if value <= 0.0:
            decision = PortfolioDecision(
                mode=mode,
                status=DecisionStatus.ABSTAIN_KEEP_BASE,
                base_feasible=True,
                selected_mask=EMPTY_MASK,
                selected_interventions=coalition_names(EMPTY_MASK),
                lower_improvement=lower_sel,
                upper_improvement=upper_sel,
                feasible=True,
                reason=(
                    f"Base policy is feasible under the {rule} calibration rule "
                    f"and no feasible portfolio has positive worst-case improvement."
                ),
                **base_metrics,
            )
        else:
            decision = PortfolioDecision(
                mode=mode,
                status=DecisionStatus.IMPROVE_SELECTED,
                base_feasible=True,
                selected_mask=mask,
                selected_interventions=coalition_names(mask),
                lower_improvement=lower_sel,
                upper_improvement=upper_sel,
                feasible=True,
                reason=(
                    f"Base policy is feasible; selected the {rule} calibration-robust "
                    f"portfolio over scenarios and seeds {sorted(games)}."
                ),
                **metrics,
            )
    else:
        repairs = sorted(
            (row for row in candidates if row[1] != EMPTY_MASK), key=lambda row: (row[0], -row[1]), reverse=True
        )
        if not repairs:
            decision = PortfolioDecision(
                mode=mode,
                status=DecisionStatus.NO_FEASIBLE_PORTFOLIO,
                base_feasible=False,
                selected_mask=EMPTY_MASK,
                selected_interventions=coalition_names(EMPTY_MASK),
                lower_improvement=lower_sel,
                upper_improvement=upper_sel,
                feasible=False,
                reason=(
                    f"Base policy violates the {rule} calibration constraints and no feasible "
                    f"intervention portfolio exists on any calibration seed."
                ),
                **base_metrics,
            )
        else:
            value, mask, metrics, per_seed = repairs[0]
            if len(repairs) > 1:
                runner_up = repairs[1][1]
                gap = float(repairs[0][0] - repairs[1][0])
            lower_sel, upper_sel = bounds(mask)
            decision = PortfolioDecision(
                mode=mode,
                status=DecisionStatus.REPAIR_SELECTED,
                base_feasible=False,
                selected_mask=mask,
                selected_interventions=coalition_names(mask),
                lower_improvement=lower_sel,
                upper_improvement=upper_sel,
                feasible=True,
                reason=(
                    f"Base policy violates the {rule} calibration constraints; selected the best "
                    f"feasible repair over scenarios and seeds {sorted(games)}."
                ),
                **metrics,
            )

    log.event(
        "calibration_robust_selected",
        rule=rule,
        alpha=alpha,
        calibration_seeds=sorted(games),
        selected_mask=decision.selected_mask,
        status=decision.status.value,
    )
    return SeedRobustSelection(
        decision=decision,
        rule=rule,
        alpha=alpha,
        calibration_seeds=tuple(sorted(games)),
        calibration_lower_improvement=float(values.get(decision.selected_mask, lower_sel)),
        calibration_feasible_seed_rate=float(np.mean(list(per_seed.values()))) if per_seed else float("nan"),
        per_seed_feasible=per_seed,
        runner_up_mask=runner_up,
        runner_up_gap=gap,
    )


def _seed_rate_ok(per_seed: dict[int, bool], alpha: float) -> bool:
    if not per_seed:
        return False
    return float(np.mean(list(per_seed.values()))) >= 1.0 - alpha - TOL


def heldout_evaluation(
    settings: Settings,
    mask: int,
    evaluation_seeds: Iterable[int],
    logger: RunLogger | None = None,
) -> list[dict]:
    """Score one frozen portfolio on unseen seeds.  Never used for selection."""
    rows: list[dict] = []
    for seed in evaluation_seeds:
        tables = evaluate_seed_masks(settings, int(seed), [mask], logger=logger)
        game = games_from_tables(tables)
        metrics, _ = seed_scenario_metrics({int(seed): game}, mask, settings)
        feasible = _all_feasible(metrics, settings)
        pool = [scenario.values[mask].improvement for scenario in game.scenario_games.values()]
        rows.append(
            {
                "evaluation_seed": int(seed),
                "selected_mask": int(mask),
                "selected_interventions": ";".join(coalition_names(mask)),
                "feasible": bool(feasible),
                "robust_lower_improvement": float(np.min(pool)),
                "robust_upper_improvement": float(np.max(pool)),
                **metrics,
            }
        )
    return rows
