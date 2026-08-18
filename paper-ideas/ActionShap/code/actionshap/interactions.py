"""Pairwise interaction diagnostics for the bounded-intervention action space.

Review 3, issue 4: the additive predicted benefit of Eq. (21) ignores pair
interactions. For every unordered pair we compute the realized interaction

    I_pq = Delta({p,q}) - Delta({p}) - Delta({q})

from bounded singleton/pair effects, and compare additive-action selection with
the interaction-aware selection that uses realized pair effects.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def pair_interactions(
    singleton_effects: np.ndarray, pair_effects: dict[tuple[int, int], float]
) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for (p, q), joint in pair_effects.items():
        out[(p, q)] = float(joint - singleton_effects[p] - singleton_effects[q])
    return out


def interaction_summary(
    singleton_effects: np.ndarray,
    pair_effects: dict[tuple[int, int], float],
) -> dict[str, float]:
    ints = np.array(list(pair_interactions(singleton_effects, pair_effects).values()))
    scale = np.abs(singleton_effects).mean() or 1.0
    return {
        "mean_abs_interaction": float(np.abs(ints).mean()),
        "max_abs_interaction": float(np.abs(ints).max()),
        "interaction_to_singleton_ratio": float(np.abs(ints).mean() / scale),
        "fraction_above_half_singleton": float(
            np.mean(np.abs(ints) > 0.5 * scale)
        ),
    }


def additive_vs_realized_rank(
    attribution: np.ndarray,
    singleton_effects: np.ndarray,
    pair_effects: dict[tuple[int, int], float],
    budget: int = 2,
) -> dict[str, float]:
    """Compare the additive action score with realized joint effects over pairs."""
    additive: list[float] = []
    realized: list[float] = []
    for p, q in combinations(range(attribution.size), 2):
        additive.append(-attribution[p] - attribution[q])
        realized.append(pair_effects[(p, q)])
    additive_arr = np.array(additive)
    realized_arr = np.array(realized)
    if np.std(additive_arr) == 0 or np.std(realized_arr) == 0:
        return {"spearman_additive_realized": float("nan"), "budget": budget}
    from scipy.stats import spearmanr

    return {
        "spearman_additive_realized": float(
            spearmanr(additive_arr, realized_arr).statistic
        ),
        "budget": budget,
    }
