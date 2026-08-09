"""Reviewer-revision selector baselines and held-out portfolio evaluation.

This module is deliberately separate from the original paper results.  It creates
new evidence for the question reviewers correctly ask: does the exact constrained
CURE-Rec planner make better decisions than plausible simpler selectors when
portfolios are selected on one set of seeds and evaluated on disjoint seeds?
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import comb, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cure_rec.config import INTERVENTION_NAMES, Settings
from cure_rec.game import ALL_MASKS, EMPTY_MASK, FULL_MASK, GameResult, coalition_names
from cure_rec.observability import RunLogger
from cure_rec.pipeline import run_experiment
from cure_rec.planner import _constraints_for_mask, _improvement_bounds, select_robust_portfolio


@dataclass(frozen=True)
class SelectorChoice:
    selector: str
    selected_mask: int
    selected_interventions: tuple[str, ...]
    robust_objective: float
    feasible: bool
    selection_seed: int


def _robust_values(game: GameResult) -> dict[int, float]:
    if game.robust_improvements:
        return game.robust_improvements
    return {mask: float(min(s.values[mask].improvement for s in game.scenario_games.values())) for mask in ALL_MASKS}


def _feasible_masks(game: GameResult, settings: Settings) -> list[int]:
    return [mask for mask in ALL_MASKS if _constraints_for_mask(game, mask, settings)[0]]


def _tie_key(mask: int, values: dict[int, float], settings: Settings) -> tuple[float, float, int, int]:
    # Larger utility first; then lower cost, fewer players, stable canonical mask.
    cost = sum(settings.interventions.costs[name] for name in coalition_names(mask))
    return (values[mask], -cost, -mask.bit_count(), -mask)


def _best(candidates: Iterable[int], values: dict[int, float], settings: Settings) -> int:
    candidates = list(candidates)
    if not candidates:
        return EMPTY_MASK
    return max(candidates, key=lambda mask: _tie_key(mask, values, settings))


def selector_choices(game: GameResult, settings: Settings, *, selection_seed: int) -> list[SelectorChoice]:
    """Return transparent alternative portfolio rules on one exact game table.

    These are baselines, not alternative attributions: every rule receives the
    same complete coalition table and declared constraints.  This isolates the
    value of the CURE-Rec decision rule from Monte Carlo/environment differences.
    """
    values = _robust_values(game)
    feasible = _feasible_masks(game, settings)
    base_feasible = EMPTY_MASK in feasible
    exact = select_robust_portfolio(game, settings, _NullLogger())

    singletons = [m for m in feasible if m.bit_count() == 1]
    best_single = _best(singletons, values, settings)

    # Shapley-1: highest robust attribution that is feasible alone.
    ranked = sorted(INTERVENTION_NAMES, key=lambda n: (-game.robust_shapley.get(n, float("-inf")), n))
    shapley_one = next((1 << INTERVENTION_NAMES.index(name) for name in ranked if (1 << INTERVENTION_NAMES.index(name)) in singletons), EMPTY_MASK)

    # Budgeted Shapley prefix: add robust-positive players in attribution order
    # while the current prefix remains feasible; choose the best feasible prefix.
    current = EMPTY_MASK
    prefixes = [EMPTY_MASK]
    for name in ranked:
        if game.robust_shapley.get(name, 0.0) <= 0:
            continue
        proposal = current | (1 << INTERVENTION_NAMES.index(name))
        if proposal in feasible:
            current = proposal
            prefixes.append(current)
    shapley_budget = _best(prefixes, values, settings)

    # Greedy robust: choose feasible addition with largest robust objective gain.
    greedy = EMPTY_MASK
    while True:
        additions = [greedy | (1 << i) for i in range(len(INTERVENTION_NAMES)) if not (greedy & (1 << i))]
        additions = [m for m in additions if m in feasible]
        if not additions:
            break
        candidate = _best(additions, values, settings)
        if values[candidate] <= values.get(greedy, 0.0):
            break
        greedy = candidate

    # Nominal-only selector, constrained by the same hard feasibility criteria.
    nominal_game = next(iter(game.scenario_games.values()))
    nominal_values = {m: nominal_game.values[m].improvement for m in ALL_MASKS}
    nominal = _best(feasible, nominal_values, settings)

    rng = np.random.default_rng(selection_seed + 9173)
    random_feasible = int(rng.choice(feasible)) if feasible else EMPTY_MASK
    grand_feasible = FULL_MASK in feasible

    choices = {
        "cure_exact_maximin": exact.selected_mask,
        "best_singleton": best_single,
        "robust_shapley_1": shapley_one,
        "robust_shapley_budget": shapley_budget,
        "greedy_robust": greedy,
        "nominal_scenario": nominal,
        "random_feasible": random_feasible,
        "grand_coalition_diagnostic": FULL_MASK if grand_feasible else EMPTY_MASK,
    }
    return [SelectorChoice(
        selector=name,
        selected_mask=mask,
        selected_interventions=coalition_names(mask),
        robust_objective=float(values[mask]),
        feasible=mask in feasible,
        selection_seed=selection_seed,
    ) for name, mask in choices.items()]


class _NullLogger:
    """Suppress planner artifacts while deriving selectors from an existing game."""
    def event(self, *args, **kwargs):
        return None
    def write_json(self, *args, **kwargs):
        return None


def _paired_summary(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows = []
    ref = frame[frame["selector"] == reference][["selection_seed", "evaluation_seed", "robust_lower_improvement"]].rename(columns={"robust_lower_improvement": "reference"})
    for selector, part in frame.groupby("selector"):
        merged = part.merge(ref, on=["selection_seed", "evaluation_seed"], how="inner", validate="one_to_one")
        # Selection choices are evaluated on the same held-out seed table.  The
        # independent unit for inference is therefore evaluation_seed (n=20), not
        # the 5 x 20 cross-product rows.  Average across selection seeds before
        # calculating paired effects or sign tests to avoid pseudoreplication.
        per_evaluation_seed = merged.groupby("evaluation_seed", as_index=False).agg(
            robust_lower_improvement=("robust_lower_improvement", "mean"),
            reference=("reference", "mean"),
        )
        diff = (per_evaluation_seed["robust_lower_improvement"] - per_evaluation_seed["reference"]).to_numpy(dtype=float)
        mean = float(diff.mean()) if len(diff) else float("nan")
        sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
        # Exact two-sided sign test; zeros are excluded.
        nonzero = diff[~np.isclose(diff, 0.0)]
        positives = int((nonzero > 0).sum())
        n = len(nonzero)
        sign_p = min(1.0, 2 * sum(comb(n, k) for k in range(positives + 1)) / (2**n)) if n else 1.0
        rows.append({
            "selector": selector,
            "selection_evaluation_pairs": int(len(merged)),
            "independent_evaluation_seeds": int(len(per_evaluation_seed)),
            "robust_lower_improvement_mean": float(part["robust_lower_improvement"].mean()),
            "robust_lower_improvement_std": float(part["robust_lower_improvement"].std(ddof=1)) if len(part) > 1 else 0.0,
            "feasible_rate": float(part["feasible"].mean()),
            "paired_difference_vs_cure_mean": mean,
            "paired_difference_vs_cure_sd": sd,
            "paired_effect_dz": mean / sd if sd > 1e-12 else float("nan"),
            "exact_sign_test_p": sign_p,
        })
    return pd.DataFrame(rows)


def run_selector_holdout_study(
    settings: Settings,
    *,
    selection_seeds: Iterable[int] = (42, 43, 44, 45, 46),
    evaluation_seeds: Iterable[int] = tuple(range(200, 220)),
) -> Path:
    """Select with one seed set and evaluate frozen masks on disjoint seeds.

    This is intentionally expensive. Selection games use all coalitions; evaluation
    games are also exact so every selector's frozen mask can be scored under the
    same unseen scenario/seed table.  It never uses evaluation seeds to choose a
    portfolio.
    """
    selection_seeds, evaluation_seeds = tuple(selection_seeds), tuple(evaluation_seeds)
    if set(selection_seeds) & set(evaluation_seeds):
        raise ValueError("Selection and evaluation seeds must be disjoint")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(settings.run.output_root) / f"reviewer-selector-holdout-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    choices: list[SelectorChoice] = []
    for seed in selection_seeds:
        cfg = settings.model_copy(deep=True)
        cfg.run.seed = seed; cfg.run.name = f"selector-selection-{seed}"; cfg.run.output_root = root / "selection"
        _, game, _ = run_experiment(cfg)
        choices.extend(selector_choices(game, cfg, selection_seed=seed))
    choice_frame = pd.DataFrame([{**asdict(c), "selected_interventions": ";".join(c.selected_interventions)} for c in choices])
    choice_frame.to_csv(root / "selection_choices.csv", index=False)

    evaluation_rows = []
    for seed in evaluation_seeds:
        cfg = settings.model_copy(deep=True)
        cfg.run.seed = seed; cfg.run.name = f"selector-evaluation-{seed}"; cfg.run.output_root = root / "evaluation"
        _, game, _ = run_experiment(cfg)
        robust = _robust_values(game)
        for choice in choices:
            feasible, margins = _constraints_for_mask(game, choice.selected_mask, cfg)
            lower, upper = _improvement_bounds(game, choice.selected_mask)
            evaluation_rows.append({
                "selection_seed": choice.selection_seed,
                "evaluation_seed": seed,
                "selector": choice.selector,
                "selected_mask": choice.selected_mask,
                "selected_interventions": ";".join(choice.selected_interventions),
                "robust_lower_improvement": robust[choice.selected_mask],
                "scenario_lower_improvement": lower,
                "scenario_upper_improvement": upper,
                "feasible": feasible,
                **margins,
            })
    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_csv(root / "heldout_selector_evaluations.csv", index=False)
    summary = _paired_summary(evaluation, "cure_exact_maximin")
    summary.to_csv(root / "heldout_selector_summary.csv", index=False)
    (root / "revision_manifest.json").write_text(json.dumps({
        "selection_seeds": selection_seeds,
        "evaluation_seeds": evaluation_seeds,
        "selectors": sorted(set(choice.selector for choice in choices)),
        "claim": "Held-out simulation-seed comparison; not external real-world causal inference.",
    }, indent=2), encoding="utf-8")
    return root


def recompute_selector_summary(revision_dir: str | Path) -> pd.DataFrame:
    """Recompute clustered held-out statistics from an existing expensive run."""
    root = Path(revision_dir)
    evaluation = pd.read_csv(root / "heldout_selector_evaluations.csv")
    summary = _paired_summary(evaluation, "cure_exact_maximin")
    summary.to_csv(root / "heldout_selector_summary.csv", index=False)
    return summary
