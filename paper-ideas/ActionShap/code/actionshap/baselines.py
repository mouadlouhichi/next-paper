"""User-level attribution baselines for the recommendation game."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .recommendation import Utility, mc_shapley


def leave_one_out(utility: Utility, n_players: int) -> np.ndarray:
    """Signed full-coalition deletion attribution.

    It is an algebraic oracle only for a single-player deletion intervention.
    It must be labelled simply ``LOO`` for bounded or joint interventions.
    """
    if n_players < 1:
        raise ValueError("n_players must be positive")
    full_set = frozenset(range(n_players))
    full = float(utility(full_set))
    return np.array(
        [full - float(utility(full_set - {player})) for player in range(n_players)],
        dtype=float,
    )


def permutation_importance(utility: Utility, n_players: int) -> np.ndarray:
    """Deletion-based local importance (the LOO baseline)."""
    return leave_one_out(utility, n_players)


def lime_attribution(
    utility: Utility,
    n_players: int,
    samples: int = 512,
    seed: int = 0,
    ridge_alpha: float = 1.0,
    kernel_width: float = 0.25,
) -> np.ndarray:
    """Locally weighted binary-mask surrogate around the full history.

    The pilot used an unweighted ridge model over globally random masks.  This
    implementation follows LIME's local-surrogate principle: the full mask and
    every leave-one-out neighbour are included deterministically, random masks
    fill the remaining budget, and normalized Hamming distance from the full
    profile determines the sample weight.
    """
    if n_players < 1:
        raise ValueError("n_players must be positive")
    minimum = n_players + 2  # full, empty, and every one-deletion neighbour
    if samples < minimum:
        raise ValueError(f"samples must be at least {minimum} for {n_players} players")
    if ridge_alpha < 0 or kernel_width <= 0:
        raise ValueError("ridge_alpha must be non-negative and kernel_width positive")
    rng = np.random.default_rng(seed)
    full = np.ones((1, n_players), dtype=float)
    local = np.ones((n_players, n_players), dtype=float)
    local[np.arange(n_players), np.arange(n_players)] = 0.0
    empty = np.zeros((1, n_players), dtype=float)
    random_count = samples - minimum
    random_masks = rng.binomial(1, 0.5, size=(random_count, n_players)).astype(float)
    masks = np.vstack((full, local, empty, random_masks))
    outcomes = np.array(
        [float(utility(frozenset(np.flatnonzero(mask)))) for mask in masks], dtype=float
    )
    distance = np.mean(1.0 - masks, axis=1)
    sample_weight = np.exp(-np.square(distance) / np.square(kernel_width))
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(masks, outcomes, sample_weight=sample_weight)
    return np.asarray(model.coef_, dtype=float)


def greedy_counterfactual_attribution(utility: Utility, n_players: int) -> np.ndarray:
    """Sequential-deletion counterfactual baseline.

    At each step, remove the remaining player with the largest signed utility
    improvement, then recompute all marginal effects in the modified coalition.
    Returned values follow the package sign convention: ``-phi`` predicts the
    benefit of downweighting/removal.  This is a search baseline, not an oracle
    for bounded downweighting, because it uses deletion and never observes the
    measured feasible-action effects.
    """
    if n_players < 1:
        raise ValueError("n_players must be positive")
    remaining = set(range(n_players))
    current = frozenset(remaining)
    current_value = float(utility(current))
    attribution = np.zeros(n_players, dtype=float)
    while remaining:
        candidates: list[tuple[float, int, float]] = []
        for player in sorted(remaining):
            value_without = float(utility(current - {player}))
            deletion_benefit = value_without - current_value
            candidates.append((deletion_benefit, player, value_without))
        benefit, selected, next_value = max(
            candidates, key=lambda row: (row[0], -row[1])
        )
        attribution[selected] = -benefit
        remaining.remove(selected)
        current = current - {selected}
        current_value = next_value
    return attribution


def random_attribution(n_players: int, seed: int = 0) -> np.ndarray:
    if n_players < 1:
        raise ValueError("n_players must be positive")
    return np.random.default_rng(seed).normal(size=n_players)


def monte_carlo_attribution(
    utility: Utility,
    n_players: int,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    return mc_shapley(utility, n_players, permutations=permutations, seed=seed)
