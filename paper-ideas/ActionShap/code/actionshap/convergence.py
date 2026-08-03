"""Monte Carlo convergence diagnostics required by the ActionShap protocol."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr

from .recommendation import Utility, mc_shapley, select_downweight_action


def _top_set(values: np.ndarray, k: int) -> set[int]:
    """Signed beneficial downweight action, including possible abstention."""
    k = min(k, values.size)
    return set(select_downweight_action(values, k, allow_abstain=True))


def convergence_table(
    utility: Utility,
    n_players: int,
    budgets: tuple[int, ...] = (25, 50, 100, 250, 500, 1000),
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    reference: int = 1000,
    reference_seed_offset: int = 100_000,
    estimator: Callable[[int, int], tuple[np.ndarray, float]] | None = None,
) -> list[dict[str, float | int | None]]:
    """Compare estimates with an independent high-permutation reference.

    Reference draws use disjoint seeds, avoiding the optimistic comparison that
    arises when an ``M`` estimate is literally a prefix of its reference run.
    Alongside rank correlation, the table reports the action-agreement criterion
    required to choose the minimum usable number of permutations.
    """
    if n_players < 1:
        raise ValueError("n_players must be positive")
    if reference < max(budgets):
        raise ValueError("reference must be at least the largest evaluated budget")
    if not seeds:
        raise ValueError("at least one seed is required")
    estimate = estimator or (
        lambda permutations, seed: mc_shapley(utility, n_players, permutations, seed)
    )
    references = [
        estimate(reference, reference_seed_offset + seed)[0] for seed in seeds
    ]
    reference_values = np.mean(references, axis=0)
    reference_top1 = _top_set(reference_values, 1)
    reference_top2 = _top_set(reference_values, 2)
    reference_top2_exact = _top_set(reference_values, 2)

    rows: list[dict[str, float | int | None]] = []
    for permutations in budgets:
        correlations: list[float] = []
        top1: list[float] = []
        top2_jaccard: list[float] = []
        top2_exact_agreement: list[float] = []
        errors: list[float] = []
        for seed in seeds:
            values, efficiency_error = estimate(permutations, seed)
            correlation = (
                spearmanr(values, reference_values).statistic
                if np.std(values) and np.std(reference_values)
                else np.nan
            )
            correlations.append(float(correlation))
            estimate_top1 = _top_set(values, 1)
            estimate_top2 = _top_set(values, 2)
            union = estimate_top2 | reference_top2
            top1.append(float(estimate_top1 == reference_top1))
            top2_jaccard.append(
                1.0 if not union else len(estimate_top2 & reference_top2) / len(union)
            )
            estimate_top2_exact = _top_set(values, 2)
            top2_exact_agreement.append(
                float(estimate_top2_exact == reference_top2_exact)
            )
            errors.append(float(efficiency_error))
        valid_correlations = np.asarray(correlations, dtype=float)
        valid_correlations = valid_correlations[np.isfinite(valid_correlations)]
        rows.append(
            {
                "permutations": int(permutations),
                "reference_permutations": int(reference),
                "mean_rank_correlation_to_reference": (
                    float(valid_correlations.mean())
                    if valid_correlations.size
                    else None
                ),
                "std_rank_correlation_to_reference": (
                    float(valid_correlations.std()) if valid_correlations.size else None
                ),
                "valid_rank_seeds": int(valid_correlations.size),
                "mean_top1_agreement": float(np.mean(top1)),
                "mean_top2_jaccard": float(np.mean(top2_jaccard)),
                "mean_top2_exact_agreement": float(np.mean(top2_exact_agreement)),
                "mean_efficiency_error": float(np.mean(errors)),
            }
        )
    return rows


def minimum_usable_permutations(
    rows: list[dict[str, float | int | None]],
    *,
    correlation_threshold: float = 0.95,
    action_threshold: float = 0.80,
) -> int | None:
    """Return the first budget satisfying rank and B=2 set-overlap criteria."""
    eligible = [
        int(row["permutations"])
        for row in rows
        if row.get("mean_rank_correlation_to_reference") is not None
        and float(row["mean_rank_correlation_to_reference"]) >= correlation_threshold
        and float(row["mean_top2_jaccard"]) >= action_threshold
    ]
    return min(eligible) if eligible else None
