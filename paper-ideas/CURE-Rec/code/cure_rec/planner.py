"""Direct robust-improvement selection and explanation cards."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from cure_rec.config import Settings
from cure_rec.game import EMPTY_MASK, GameResult, coalition_names
from cure_rec.observability import RunLogger


@dataclass(frozen=True)
class PortfolioDecision:
    selected_mask: int
    selected_interventions: tuple[str, ...]
    action: str  # deploy | abstain
    lower_improvement: float
    upper_improvement: float
    cost: float
    relevance_delta_lower: float
    provider_disparity_upper: float
    fatigue_upper: float
    feasible: bool
    reason: str


def _constraints_for_mask(game: GameResult, mask: int, settings: Settings) -> tuple[bool, dict[str, float]]:
    values = [scenario.values[mask] for scenario in game.scenario_games.values()]
    cost = values[0].cost
    # Relevance is measured against the scenario's empty-policy value.
    relevance_deltas = [
        value.relevance - scenario.values[EMPTY_MASK].relevance
        for scenario, value in ((game.scenario_games[name], game.scenario_games[name].values[mask]) for name in game.scenario_games)
    ]
    metrics = {
        "cost": cost,
        "relevance_delta_lower": float(np.min(relevance_deltas)),
        "provider_disparity_upper": float(np.max([value.provider_disparity for value in values])),
        "fatigue_upper": float(np.max([value.fatigue for value in values])),
    }
    feasible = (
        metrics["cost"] <= settings.constraints.budget
        and metrics["relevance_delta_lower"] >= settings.constraints.min_relevance_delta
        and metrics["provider_disparity_upper"] <= settings.constraints.max_provider_disparity
        and metrics["fatigue_upper"] <= settings.constraints.max_fatigue
    )
    return feasible, metrics


def select_robust_portfolio(game: GameResult, settings: Settings, logger: RunLogger) -> PortfolioDecision:
    """Enumerate portfolios using direct worst-case improvement, never Shapley sums."""
    candidates: list[tuple[float, float, int, dict[str, float]]] = []
    for mask in sorted({mask for scenario in game.scenario_games.values() for mask in scenario.values}):
        feasible, metrics = _constraints_for_mask(game, mask, settings)
        if not feasible:
            logger.event("portfolio_rejected", coalition_mask=mask, active_interventions=coalition_names(mask), **metrics)
            continue
        improvements = [scenario.values[mask].improvement for scenario in game.scenario_games.values()]
        candidates.append((float(np.min(improvements)), float(np.max(improvements)), mask, metrics))
    if not candidates:
        return PortfolioDecision(
            selected_mask=EMPTY_MASK,
            selected_interventions=(),
            action="abstain",
            lower_improvement=0.0,
            upper_improvement=0.0,
            cost=0.0,
            relevance_delta_lower=0.0,
            provider_disparity_upper=0.0,
            fatigue_upper=0.0,
            feasible=True,
            reason="No portfolio satisfies robust operational constraints.",
        )
    lower, upper, mask, metrics = max(candidates, key=lambda row: (row[0], -row[2]))
    if lower <= 0.0:
        decision = PortfolioDecision(
            selected_mask=EMPTY_MASK,
            selected_interventions=(),
            action="abstain",
            lower_improvement=lower,
            upper_improvement=upper,
            feasible=True,
            reason="No feasible portfolio has positive worst-case improvement over the base policy.",
            **metrics,
        )
    else:
        decision = PortfolioDecision(
            selected_mask=mask,
            selected_interventions=coalition_names(mask),
            action="deploy",
            lower_improvement=lower,
            upper_improvement=upper,
            feasible=True,
            reason="Selected by exact direct maximin improvement across configured scenarios.",
            **metrics,
        )
    logger.event("portfolio_selected", **asdict(decision))
    logger.write_json("artifacts/portfolio_decision.json", asdict(decision))
    return decision


def build_explanation_card(game: GameResult, decision: PortfolioDecision, logger: RunLogger) -> dict:
    selected = set(decision.selected_interventions)
    region_rows = game.regions.to_dict(orient="records")
    interactions = game.interaction_table.to_dict(orient="records")
    card = {
        "decision": asdict(decision),
        "selected_attributions": [row for row in region_rows if row["intervention"] in selected],
        "rejected_or_deferred": [row for row in region_rows if row["intervention"] not in selected],
        "relevant_interactions": [
            row for row in interactions
            if row["intervention_i"] in selected or row["intervention_j"] in selected
        ],
        "interpretation": {
            "positive_certificate": "phi_lower > 0 means positive order-averaged marginal contribution across configured scenarios.",
            "abstention": "A non-positive robust improvement returns the base policy unchanged.",
            "selection_rule": "Portfolio selection used direct robust improvement, not summed Shapley lower bounds.",
        },
    }
    logger.write_json("artifacts/explanation_card.json", card)
    logger.event("explanation_card_written", selected_mask=decision.selected_mask)
    return card
