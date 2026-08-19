"""Render the lhs-009 held-out selector table rows for the manuscript."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POINT = sys.argv[1] if len(sys.argv) > 1 else "lhs-009"

ORDER = [
    "cure_exact_maximin",
    "best_singleton",
    "robust_shapley_1",
    "nominal_scenario",
    "robust_shapley_budget",
    "greedy_robust",
    "random_feasible",
    "grand_coalition_diagnostic",
]
NAMES = {
    "cure_exact_maximin": "CURE exact maximin",
    "best_singleton": "Best singleton",
    "robust_shapley_1": "Robust Shapley-1",
    "nominal_scenario": "Nominal-scenario selector",
    "robust_shapley_budget": "Robust Shapley-budget",
    "greedy_robust": "Greedy robust",
    "random_feasible": "Random feasible",
    "grand_coalition_diagnostic": "Grand-coalition diagnostic",
}

summary = pd.read_csv(ROOT / "results" / "reviewer_phase_assets" / "divergent_selector_holdout" / POINT / "heldout_selector_summary.csv")
choices = pd.read_csv(ROOT / "results" / "reviewer_phase_assets" / "divergent_selector_holdout" / POINT / "selection_choices.csv")

sel_counts = (
    choices[choices["selector"].isin(ORDER)]
    .groupby("selector")["selected_interventions"]
    .agg(lambda s: "; ".join(f"{name or 'empty'} ({count})" for name, count in s.value_counts().items()))
)

summary = summary.set_index("selector")
for selector in ORDER:
    if selector not in summary.index:
        continue
    row = summary.loc[selector]
    mean = row["robust_lower_improvement_mean"]
    sd = row["robust_lower_improvement_std"]
    fr = row["feasible_rate"]
    diff = row["paired_difference_vs_cure_mean"]
    p = row["exact_sign_test_p"]
    if selector == "cure_exact_maximin":
        diff_s, p_s = "---", "---"
    else:
        diff_s = f"${diff:+.6f}$"
        p_s = f"{p:.6f}"
    print(
        f"{NAMES[selector]} & ${mean:.6f}$ & {sd:.6f} & {fr:.2f} & {diff_s} & {p_s}\\\\%"
        f" selections: {sel_counts.get(selector, '?')}"
    )
