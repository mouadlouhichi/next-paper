"""Monte Carlo convergence diagnostics required by the ActionShap protocol."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .recommendation import Utility, mc_shapley


def convergence_table(
    utility: Utility,
    n_players: int,
    budgets: tuple[int, ...] = (25, 50, 100, 250, 500, 1000),
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    reference: int = 1000,
) -> list[dict[str, float | int]]:
    """Compare estimates against a high-budget reference.

    The prefix-walk efficiency error is reported separately but is deliberately
    not used as a convergence criterion, because it telescopes exactly.
    """
    refs = [mc_shapley(utility, n_players, reference, seed)[0] for seed in seeds]
    reference_values = np.mean(refs, axis=0)
    rows = []
    for m in budgets:
        correlations = []
        errors = []
        for seed in seeds:
            values, efficiency_error = mc_shapley(utility, n_players, m, seed)
            corr = spearmanr(values, reference_values).statistic if np.std(values) and np.std(reference_values) else np.nan
            correlations.append(corr)
            errors.append(efficiency_error)
        rows.append({
            "permutations": int(m),
            "mean_rank_correlation_to_reference": float(np.nanmean(correlations)),
            "std_rank_correlation_to_reference": float(np.nanstd(correlations)),
            "mean_efficiency_error": float(np.mean(errors)),
        })
    return rows
