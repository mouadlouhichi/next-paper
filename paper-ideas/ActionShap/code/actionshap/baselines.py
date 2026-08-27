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
    seed: int | tuple = 0,
    ridge_alpha: float = 1.0,
    kernel_width: float = 0.25,
    mask_design: str = "bernoulli",
) -> np.ndarray:
    """Locally weighted binary-mask surrogate around the full history.

    The pilot used an unweighted ridge model over globally random masks.  This
    implementation follows LIME's local-surrogate principle: the full mask and
    every leave-one-out neighbour are included deterministically, random masks
    fill the remaining budget, and normalized Hamming distance from the full
    profile determines the sample weight.

    ``mask_design`` selects how the random rows are drawn:

    * ``"bernoulli"`` (default, primary protocol): independent Bernoulli(0.5)
      rows with replacement; repeated masks are implicitly re-weighted by the
      ridge fit.
    * ``"unique"``: random masks are drawn without replacement over distinct
      masks; when the remaining budget exceeds the number of distinct masks
      not yet used, all remaining distinct masks are enumerated.
    * ``"enumerate"``: every one of the ``2**n_players`` distinct masks is
      used exactly once (requires ``2**n_players <= samples``).
    """
    if n_players < 1:
        raise ValueError("n_players must be positive")
    if mask_design not in ("bernoulli", "unique", "enumerate"):
        raise ValueError("mask_design must be bernoulli, unique, or enumerate")
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
    deterministic = np.vstack((full, local, empty))
    random_count = samples - minimum
    if mask_design == "enumerate":
        total = 2 ** n_players
        if total > samples:
            raise ValueError(
                f"enumerate requires samples >= 2**n_players = {total}"
            )
        masks = np.array(
            [[(i >> j) & 1 for j in range(n_players)] for i in range(total)],
            dtype=float,
        )
    else:
        if mask_design == "unique":
            seen = {tuple(row) for row in deterministic}
            rows: list[np.ndarray] = []
            attempts = 0
            while len(rows) < random_count and attempts < 20 * random_count + 1000:
                attempts += 1
                row = rng.binomial(1, 0.5, size=n_players).astype(float)
                key = tuple(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
            if len(rows) < random_count:
                # Distinct-mask space exhausted: add every remaining distinct
                # mask not yet used, then stop (budget cannot be filled with
                # unique masks).
                for i in range(2 ** n_players):
                    row = np.array([(i >> j) & 1 for j in range(n_players)], float)
                    if tuple(row) not in seen:
                        seen.add(tuple(row))
                        rows.append(row)
                        if len(rows) >= random_count:
                            break
                rows = rows[:random_count] if len(rows) > random_count else rows
            random_masks = (
                np.vstack(rows) if rows else np.zeros((0, n_players), dtype=float)
            )
        else:
            random_masks = rng.binomial(1, 0.5, size=(random_count, n_players)).astype(
                float
            )
        masks = np.vstack((deterministic, random_masks)) if len(random_masks) else deterministic
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


def random_attribution(n_players: int, seed: int | tuple = 0) -> np.ndarray:
    """IID standard-normal control scores.

    ``seed`` may be an integer or a tuple of integers. Tuple seeds are passed
    straight to :class:`numpy.random.SeedSequence`, which mixes every word of
    the entropy. Callers should therefore derive the random-control stream as a
    tuple (e.g. ``(experiment_seed, user_id, stream_tag)``) rather than by
    integer addition, because integer offsets can collide for adjacent
    ``(user, seed)`` pairs.
    """
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
