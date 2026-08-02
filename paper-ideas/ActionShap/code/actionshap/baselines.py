"""User-level attribution baselines for the recommendation game."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.linear_model import Ridge

from .recommendation import Utility, mc_shapley


def leave_one_out(utility: Utility, n_players: int) -> np.ndarray:
    """Single-player deletion attribution; oracle only for B=1 masking."""
    full = utility(frozenset(range(n_players)))
    return np.array(
        [full - utility(frozenset(set(range(n_players)) - {p})) for p in range(n_players)],
        dtype=float,
    )


def permutation_importance(utility: Utility, n_players: int) -> np.ndarray:
    """Alias with explicit semantics for the baseline table."""
    return leave_one_out(utility, n_players)


def lime_attribution(
    utility: Utility,
    n_players: int,
    samples: int = 256,
    seed: int = 0,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Binary local surrogate attribution over interaction masks.

    The full coalition is always included. Sampling is deterministic for a
    seed and uses coalition size-balanced Bernoulli masks approximately, which
    avoids a dataset-specific explanation hidden in the baseline.
    """
    if n_players < 1 or samples < 2:
        raise ValueError("n_players must be positive and samples must exceed one")
    rng = np.random.default_rng(seed)
    masks = rng.binomial(1, 0.5, size=(samples - 1, n_players)).astype(float)
    masks = np.vstack([masks, np.ones((1, n_players), dtype=float)])
    y = np.array([utility(frozenset(np.flatnonzero(row))) for row in masks])
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(masks, y)
    return np.asarray(model.coef_, dtype=float)


def random_attribution(n_players: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n_players)


def monte_carlo_attribution(
    utility: Utility,
    n_players: int,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    return mc_shapley(utility, n_players, permutations=permutations, seed=seed)
