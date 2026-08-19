"""Phase E companion (reviewer revision): objective and constraint sensitivity.

Reviewer concerns: (a) the planner's utility weights (w_s, w_r, w_f, w_c) are
normative and were never varied; (b) the constraint surface (budget B, relevance
floor r_min, disparity cap d_max, fatigue cap f_max) deserves a joint frontier.

Both questions can be answered WITHOUT new rollouts: a stored exact-game coalition
table retains, for every scenario and coalition, the raw discounted satisfaction,
retention, fatigue, relevance, provider disparity, and cost. Utility under any
weight vector is a linear recombination, and feasibility under any constraint
vector is a threshold test on the stored margins. This script therefore re-derives
selection outcomes offline from one archived exact game.

Claim scope: sensitivity of the disclosed CURE-Sim decision to declared objective
and constraint choices. Not external causal evidence.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cure_rec.config import INTERVENTION_NAMES, load_settings  # noqa: E402
from cure_rec.game import ALL_MASKS, EMPTY_MASK, FULL_MASK, coalition_names  # noqa: E402

ASSETS = ROOT / "results" / "reviewer_phase_assets" / "objective_constraint_sweeps"
BASE_CONFIG = ROOT / "configs" / "curesim_full.yaml"
COALITION_TABLE_GLOB = ROOT / "results" / "reviewer_phase_assets" / "divergent_selector_holdout" / "screening" / "baseline"

# Pre-specified utility-weight design. The baseline is (0.70, 0.30, 0.35, 1.00).
UTILITY_GRID: list[dict[str, float]] = [
    {"label": "baseline", "w_s": 0.70, "w_r": 0.30, "w_f": 0.35, "w_c": 1.00},
    {"label": "satisfaction_half", "w_s": 0.35, "w_r": 0.30, "w_f": 0.35, "w_c": 1.00},
    {"label": "satisfaction_double", "w_s": 1.40, "w_r": 0.30, "w_f": 0.35, "w_c": 1.00},
    {"label": "retention_half", "w_s": 0.70, "w_r": 0.15, "w_f": 0.35, "w_c": 1.00},
    {"label": "retention_double", "w_s": 0.70, "w_r": 0.60, "w_f": 0.35, "w_c": 1.00},
    {"label": "fatigue_half", "w_s": 0.70, "w_r": 0.30, "w_f": 0.175, "w_c": 1.00},
    {"label": "fatigue_double", "w_s": 0.70, "w_r": 0.30, "w_f": 0.70, "w_c": 1.00},
    {"label": "cost_half", "w_s": 0.70, "w_r": 0.30, "w_f": 0.35, "w_c": 0.50},
    {"label": "cost_double", "w_s": 0.70, "w_r": 0.30, "w_f": 0.35, "w_c": 2.00},
    {"label": "satisfaction_only", "w_s": 1.00, "w_r": 0.00, "w_f": 0.00, "w_c": 1.00},
    {"label": "retention_only", "w_s": 0.00, "w_r": 1.00, "w_f": 0.00, "w_c": 1.00},
    {"label": "fatigue_aversion_only", "w_s": 0.00, "w_r": 0.00, "w_f": 1.00, "w_c": 1.00},
    {"label": "equal_stakeholder", "w_s": 0.45, "w_r": 0.45, "w_f": 0.45, "w_c": 1.00},
    {"label": "cost_insensitive", "w_s": 0.70, "w_r": 0.30, "w_f": 0.35, "w_c": 0.00},
]

# Pre-specified constraint frontier. Each axis is varied jointly in a full grid.
CONSTRAINT_GRID = {
    "budget": (0.15, 0.25, 0.35, 0.45, 0.55),
    "min_relevance_delta": (-0.02, -0.08, -0.15),
    "max_provider_disparity": (0.22, 0.28, 0.34),
    "max_fatigue": (0.55, 0.65, 0.75),
}


def load_coalition_table() -> pd.DataFrame:
    matches = sorted(COALITION_TABLE_GLOB.glob("*/tables/coalition_values.csv"))
    if not matches:
        raise FileNotFoundError(
            "Baseline screening game not found. Run scripts_review/phase_e_divergence.py first."
        )
    frame = pd.read_csv(matches[0])
    return frame, matches[0]


def rebuild_utilities(frame: pd.DataFrame, weights: dict[str, float]) -> dict[str, dict[int, float]]:
    """Per-scenario net utility U_m(S) under a weight vector."""
    utilities: dict[str, dict[int, float]] = {}
    for scenario, part in frame.groupby("scenario"):
        values = {}
        for row in part.itertuples(index=False):
            utility = (
                weights["w_s"] * row.satisfaction
                + weights["w_r"] * row.retention
                - weights["w_f"] * row.fatigue
                - weights["w_c"] * row.cost
            )
            values[int(row.mask)] = float(utility)
        utilities[scenario] = values
    return utilities


def scenario_margins(frame: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Weight-independent feasibility margins per coalition."""
    margins: dict[int, dict[str, float]] = {}
    base_relevance = {scenario: part.set_index("mask").loc[EMPTY_MASK, "relevance"] for scenario, part in frame.groupby("scenario")}
    for mask in ALL_MASKS:
        rows = frame[frame["mask"] == mask]
        relevance_delta = min(rows.loc[rows["scenario"] == scenario, "relevance"].iloc[0] - base_relevance[scenario] for scenario in base_relevance)
        margins[mask] = {
            "cost": float(rows["cost"].iloc[0]),
            "relevance_delta_lower": float(relevance_delta),
            "provider_disparity_upper": float(rows["provider_disparity"].max()),
            "fatigue_upper": float(rows["fatigue"].max()),
        }
    return margins


def select_portfolio(
    utilities: dict[str, dict[int, float]],
    margins: dict[int, dict[str, float]],
    *,
    budget: float,
    min_relevance_delta: float,
    max_provider_disparity: float,
    max_fatigue: float,
) -> dict:
    """Mirror the planner's casewise rule on precomputed tables."""

    def feasible(mask: int) -> bool:
        m = margins[mask]
        return (
            m["cost"] <= budget + 1e-12
            and m["relevance_delta_lower"] >= min_relevance_delta
            and m["provider_disparity_upper"] <= max_provider_disparity
            and m["fatigue_upper"] <= max_fatigue
        )

    def lower(mask: int) -> float:
        return min(values[mask] - values[EMPTY_MASK] for values in utilities.values())

    base_ok = feasible(EMPTY_MASK)
    feasible_masks = [mask for mask in ALL_MASKS if feasible(mask)]
    if base_ok:
        best = max(feasible_masks, key=lambda mask: (lower(mask), -mask))
        if lower(best) <= 0:
            return {"mode": "improvement", "status": "abstain_keep_base", "mask": EMPTY_MASK, "lower": lower(EMPTY_MASK), "base_feasible": True}
        return {"mode": "improvement", "status": "improve_selected", "mask": best, "lower": lower(best), "base_feasible": True}
    repairs = [mask for mask in feasible_masks if mask != EMPTY_MASK]
    if not repairs:
        return {"mode": "repair", "status": "no_feasible_portfolio", "mask": EMPTY_MASK, "lower": lower(EMPTY_MASK), "base_feasible": False}
    best = max(repairs, key=lambda mask: (lower(mask), -mask))
    return {"mode": "repair", "status": "repair_selected", "mask": best, "lower": lower(best), "base_feasible": False}


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    settings = load_settings(BASE_CONFIG)
    frame, source = load_coalition_table()
    margins = scenario_margins(frame)

    # ---------------- utility-weight sensitivity ----------------
    utility_rows = []
    for weights in UTILITY_GRID:
        utilities = rebuild_utilities(frame, weights)
        decision = select_portfolio(
            utilities,
            margins,
            budget=settings.constraints.budget,
            min_relevance_delta=settings.constraints.min_relevance_delta,
            max_provider_disparity=settings.constraints.max_provider_disparity,
            max_fatigue=settings.constraints.max_fatigue,
        )
        utility_rows.append({
            **weights,
            "selected_mask": decision["mask"],
            "selected_portfolio": ";".join(coalition_names(decision["mask"])) or "abstain",
            "mode": decision["mode"],
            "status": decision["status"],
            "robust_lower_improvement": decision["lower"],
            "grand_coalition_lower": min(values[FULL_MASK] - values[EMPTY_MASK] for values in utilities.values()),
        })
    utility_table = pd.DataFrame(utility_rows)
    utility_table.to_csv(ASSETS / "utility_weight_sensitivity.csv", index=False)

    # ---------------- constraint frontier ----------------
    base_utilities = rebuild_utilities(frame, UTILITY_GRID[0])
    frontier_rows = []
    keys = list(CONSTRAINT_GRID)
    for combo in product(*(CONSTRAINT_GRID[key] for key in keys)):
        budget, min_rel, max_disp, max_fat = combo
        decision = select_portfolio(base_utilities, margins, budget=budget, min_relevance_delta=min_rel, max_provider_disparity=max_disp, max_fatigue=max_fat)
        frontier_rows.append({
            "budget": budget,
            "min_relevance_delta": min_rel,
            "max_provider_disparity": max_disp,
            "max_fatigue": max_fat,
            "mode": decision["mode"],
            "status": decision["status"],
            "base_feasible": decision["base_feasible"],
            "selected_mask": decision["mask"],
            "selected_portfolio": ";".join(coalition_names(decision["mask"])) or "abstain",
            "robust_lower_improvement": decision["lower"],
        })
    frontier_table = pd.DataFrame(frontier_rows)
    frontier_table.to_csv(ASSETS / "constraint_frontier.csv", index=False)

    status_counts = frontier_table["status"].value_counts().to_dict()
    grand_feasible_rate = float((frontier_table["budget"] >= margins[FULL_MASK]["cost"]).mean())

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_coalition_table": str(source),
        "method": "offline recombination of archived exact-game outcomes; no new rollouts",
        "utility_grid": UTILITY_GRID,
        "constraint_grid": CONSTRAINT_GRID,
        "frontier_status_counts": status_counts,
        "grand_coalition_cost": margins[FULL_MASK]["cost"],
        "grand_coalition_budget_feasible_fraction_of_frontier": grand_feasible_rate,
        "claim_scope": "CURE-Sim decision sensitivity; not external causal evidence",
    }
    (ASSETS / "revision_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("UTILITY SWEEP")
    print(utility_table.to_string(index=False))
    print("CONSTRAINT FRONTIER STATUS COUNTS:", status_counts)
    print("OFFLINE SWEEPS DONE")


if __name__ == "__main__":
    main()
