"""Exact Shapley values by coalition enumeration for tractable player counts.

Used to validate the reverse-paired Monte Carlo estimator (review 3, issue 5):
for users with ``n_u <= exact_max`` the characteristic function is evaluated on
all ``2^n`` coalitions (with caching) and the factorial-weighted Shapley value
is computed directly.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

Utility = object  # Callable[[frozenset[int]], float]


def exact_shapley(utility, n_players: int, max_players: int = 12) -> np.ndarray:
    if n_players < 1 or n_players > max_players:
        raise ValueError("exact enumeration requires 1 <= n <= max_players")
    cache: dict[frozenset[int], float] = {}

    def v(coalition: frozenset[int]) -> float:
        if coalition not in cache:
            cache[coalition] = float(utility(coalition))
        return cache[coalition]

    n = n_players
    values = np.zeros(n)
    for p in range(n):
        others = [i for i in range(n) if i != p]
        for size in range(0, n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            marginal = 0.0
            count = 0
            for combo in combinations(others, size):
                s = frozenset(combo)
                marginal += v(s | {p}) - v(s)
                count += 1
            if count:
                values[p] += weight * marginal
    return values


def mc_error_report(
    exact: np.ndarray, estimate: np.ndarray
) -> dict[str, float]:
    diff = estimate - exact
    denom = np.abs(exact).max() if np.abs(exact).max() > 0 else 1.0
    return {
        "max_abs_error": float(np.abs(diff).max()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "rank_spearman": _spearman(exact, estimate),
        "relative_max_error": float(np.abs(diff).max() / denom),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)
