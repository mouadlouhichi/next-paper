"""Direct robust-improvement selection with explicit improvement and repair modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from cure_rec.config import Settings
from cure_rec.game import EMPTY_MASK, GameResult, coalition_names
from cure_rec.observability import RunLogger


class DecisionMode(str, Enum):
    """Whether the deployed base policy is already robustly feasible."""

    IMPROVEMENT = "improvement"
    REPAIR = "repair"


class DecisionStatus(str, Enum):
    """Outcome semantics for a robust portfolio decision."""

    IMPROVE_SELECTED = "improve_selected"
    ABSTAIN_KEEP_BASE = "abstain_keep_base"
    REPAIR_SELECTED = "repair_selected"
    NO_FEASIBLE_PORTFOLIO = "no_feasible_portfolio"


@dataclass(frozen=True)
class PortfolioDecision:
    mode: DecisionMode
    status: DecisionStatus
    base_feasible: bool
    selected_mask: int
    selected_interventions: tuple[str, ...]
    lower_improvement: float
    upper_improvement: float
    cost: float
    relevance_delta_lower: float
    provider_disparity_upper: float
    fatigue_upper: float
    feasible: bool
    reason: str

    @property
    def action(self) -> str:
        """Backward-compatible concise status for notebooks and CLI summaries."""
        return self.status.value


def decision_to_dict(decision: PortfolioDecision) -> dict:
    """Serialize enum-backed decision semantics consistently in JSON and CSV assets."""
    record = asdict(decision)
    record["mode"] = decision.mode.value
    record["status"] = decision.status.value
    record["action"] = decision.action
    return record


def _constraints_for_mask(game: GameResult, mask: int, settings: Settings) -> tuple[bool, dict[str, float]]:
    scenario_values = [scenario.values[mask] for scenario in game.scenario_games.values()]
    relevance_deltas = [
        scenario.values[mask].relevance - scenario.values[EMPTY_MASK].relevance
        for scenario in game.scenario_games.values()
    ]
    metrics = {
        "cost": float(scenario_values[0].cost),
        "relevance_delta_lower": float(np.min(relevance_deltas)),
        "provider_disparity_upper": float(np.max([value.provider_disparity for value in scenario_values])),
        "fatigue_upper": float(np.max([value.fatigue for value in scenario_values])),
    }
    feasible = (
        metrics["cost"] <= settings.constraints.budget
        and metrics["relevance_delta_lower"] >= settings.constraints.min_relevance_delta
        and metrics["provider_disparity_upper"] <= settings.constraints.max_provider_disparity
        and metrics["fatigue_upper"] <= settings.constraints.max_fatigue
    )
    return feasible, metrics


def _improvement_bounds(game: GameResult, mask: int) -> tuple[float, float]:
    improvements = [scenario.values[mask].improvement for scenario in game.scenario_games.values()]
    return float(np.min(improvements)), float(np.max(improvements))


def _decision(
    *,
    mode: DecisionMode,
    status: DecisionStatus,
    base_feasible: bool,
    mask: int,
    lower: float,
    upper: float,
    metrics: dict[str, float],
    reason: str,
    feasible: bool,
) -> PortfolioDecision:
    return PortfolioDecision(
        mode=mode,
        status=status,
        base_feasible=base_feasible,
        selected_mask=mask,
        selected_interventions=coalition_names(mask),
        lower_improvement=lower,
        upper_improvement=upper,
        feasible=feasible,
        reason=reason,
        **metrics,
    )


def select_robust_portfolio(game: GameResult, settings: Settings, logger: RunLogger) -> PortfolioDecision:
    """Select a policy using direct maximin improvement, never Shapley sums.

    Improvement mode retains a safe base policy if no intervention can robustly
    improve it. Repair mode applies when the base violates constraints: in that
    setting returning EMPTY is not permitted unless no safety constraint exists.
    """
    masks = sorted({mask for scenario in game.scenario_games.values() for mask in scenario.values})
    base_feasible, base_metrics = _constraints_for_mask(game, EMPTY_MASK, settings)
    base_lower, base_upper = _improvement_bounds(game, EMPTY_MASK)
    mode = DecisionMode.IMPROVEMENT if base_feasible else DecisionMode.REPAIR
    logger.event(
        "planner_mode_resolved",
        mode=mode.value,
        base_feasible=base_feasible,
        base_lower_improvement=base_lower,
        **base_metrics,
    )

    candidates: list[tuple[float, float, int, dict[str, float]]] = []
    for mask in masks:
        feasible, metrics = _constraints_for_mask(game, mask, settings)
        lower, upper = _improvement_bounds(game, mask)
        if not feasible:
            logger.event(
                "portfolio_rejected",
                mode=mode.value,
                coalition_mask=mask,
                active_interventions=coalition_names(mask),
                lower_improvement=lower,
                **metrics,
            )
            continue
        candidates.append((lower, upper, mask, metrics))

    if mode is DecisionMode.IMPROVEMENT:
        # Empty is feasible by definition in this mode, so abstention is safe.
        lower, upper, mask, metrics = max(candidates, key=lambda row: (row[0], -row[2]))
        if lower <= 0.0:
            decision = _decision(
                mode=mode,
                status=DecisionStatus.ABSTAIN_KEEP_BASE,
                base_feasible=True,
                mask=EMPTY_MASK,
                lower=base_lower,
                upper=base_upper,
                metrics=base_metrics,
                feasible=True,
                reason="Base policy is feasible and no feasible portfolio has positive worst-case improvement.",
            )
        else:
            decision = _decision(
                mode=mode,
                status=DecisionStatus.IMPROVE_SELECTED,
                base_feasible=True,
                mask=mask,
                lower=lower,
                upper=upper,
                metrics=metrics,
                feasible=True,
                reason="Base policy is feasible; selected the exact maximin-improvement portfolio.",
            )
    else:
        # Empty is unsafe. Choose the best feasible repair even if its robust
        # improvement is negative, or explicitly certify infeasibility.
        repair_candidates = [candidate for candidate in candidates if candidate[2] != EMPTY_MASK]
        if not repair_candidates:
            decision = _decision(
                mode=mode,
                status=DecisionStatus.NO_FEASIBLE_PORTFOLIO,
                base_feasible=False,
                mask=EMPTY_MASK,
                lower=base_lower,
                upper=base_upper,
                metrics=base_metrics,
                feasible=False,
                reason="Base policy violates robust constraints and no feasible intervention portfolio exists.",
            )
        else:
            lower, upper, mask, metrics = max(repair_candidates, key=lambda row: (row[0], -row[2]))
            decision = _decision(
                mode=mode,
                status=DecisionStatus.REPAIR_SELECTED,
                base_feasible=False,
                mask=mask,
                lower=lower,
                upper=upper,
                metrics=metrics,
                feasible=True,
                reason="Base policy violates robust constraints; selected the best feasible repair by maximin improvement.",
            )

    record = decision_to_dict(decision)
    logger.event("portfolio_selected", **record)
    logger.write_json("artifacts/portfolio_decision.json", record)
    return decision


def build_explanation_card(game: GameResult, decision: PortfolioDecision, logger: RunLogger) -> dict:
    selected = set(decision.selected_interventions)
    region_rows = game.regions.to_dict(orient="records")
    interactions = game.interaction_table.to_dict(orient="records")
    card = {
        "decision": decision_to_dict(decision),
        "selected_attributions": [row for row in region_rows if row["intervention"] in selected],
        "rejected_or_deferred": [row for row in region_rows if row["intervention"] not in selected],
        "relevant_interactions": [
            row for row in interactions
            if row["intervention_i"] in selected or row["intervention_j"] in selected
        ],
        "interpretation": {
            "positive_certificate": "phi_lower > 0 means positive order-averaged marginal contribution across configured scenarios.",
            "improvement_mode": "A feasible base policy may be retained when no portfolio robustly improves it.",
            "repair_mode": "An infeasible base policy cannot be retained; select a feasible repair or certify no feasible portfolio.",
            "selection_rule": "Portfolio selection used direct robust improvement, not summed Shapley lower bounds.",
        },
    }
    logger.write_json("artifacts/explanation_card.json", card)
    logger.event("explanation_card_written", selected_mask=decision.selected_mask, status=decision.status.value)
    return card
