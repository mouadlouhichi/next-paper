"""Recommendation-only ActionShap core utilities.

These functions implement the model-independent parts of the revised
specification: fixed-candidate utility, Monte Carlo Shapley over user history
players, and budgeted joint-action scoring. Dataset loading and model training
remain experiment scripts rather than hidden inside these pure functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, Iterable

import numpy as np


Utility = Callable[[frozenset[int]], float]


@dataclass(frozen=True)
class UserGame:
    """One user's frozen game on a fixed candidate set."""

    players: np.ndarray
    candidate_items: np.ndarray
    target_item: int
    tie_break: np.ndarray

    def __post_init__(self) -> None:
        players = np.asarray(self.players, dtype=int)
        candidates = np.asarray(self.candidate_items, dtype=int)
        tie = np.asarray(self.tie_break, dtype=int)
        if players.ndim != 1 or candidates.ndim != 1 or tie.ndim != 1:
            raise ValueError("players, candidate_items, and tie_break must be 1-D")
        if candidates.size == 0 or tie.shape != candidates.shape:
            raise ValueError("candidate_items and tie_break must be non-empty and aligned")
        if self.target_item not in set(candidates.tolist()):
            raise ValueError("target_item must be in the fixed candidate set")
        object.__setattr__(self, "players", players)
        object.__setattr__(self, "candidate_items", candidates)
        object.__setattr__(self, "tie_break", tie)


def ndcg_at_k(scores: np.ndarray, candidate_items: np.ndarray, target_item: int, k: int = 10, tie_break: np.ndarray | None = None) -> float:
    """NDCG for one held-out target on a fixed candidate set."""
    scores = np.asarray(scores, dtype=float)
    items = np.asarray(candidate_items, dtype=int)
    if scores.shape != items.shape:
        raise ValueError("scores and candidate_items must have the same shape")
    if k < 1:
        raise ValueError("k must be positive")
    if tie_break is None:
        tie_break = np.arange(items.size, dtype=int)
    tie_break = np.asarray(tie_break, dtype=int)
    if tie_break.shape != items.shape:
        raise ValueError("tie_break must align with candidate_items")
    # Higher scores first; lower tie-break value first. This is deterministic.
    order = np.lexsort((tie_break, -scores))[:k]
    hits = np.flatnonzero(items[order] == target_item)
    if hits.size == 0:
        return 0.0
    return float(1.0 / np.log2(int(hits[0]) + 2))


def profile_utility(
    model,
    game: UserGame,
    coalition: frozenset[int],
    k: int = 10,
) -> float:
    """Evaluate v_u(S), including the deterministic zero-profile null game."""
    ids = game.players
    mask = np.fromiter((i in coalition for i in range(ids.size)), dtype=bool, count=ids.size)
    scores = model.score_masked(ids, game.candidate_items, mask)
    return ndcg_at_k(scores, game.candidate_items, game.target_item, k, game.tie_break)



def target_margin_utility(
    model,
    game: UserGame,
    coalition: frozenset[int],
    k: int = 10,
    competitor_count: int = 10,
) -> float:
    """Continuous target-vs-competitors utility on the fixed evaluation set."""
    ids = game.players
    mask = np.fromiter((i in coalition for i in range(ids.size)), dtype=bool, count=ids.size)
    scores = model.score_masked(ids, game.candidate_items, mask)
    target_index = int(np.flatnonzero(game.candidate_items == game.target_item)[0])
    competitors = np.delete(scores, target_index)
    top = np.sort(competitors)[-min(competitor_count, competitors.size):]
    margin = float(scores[target_index] - top.mean())
    return float(1.0 / (1.0 + np.exp(-np.clip(margin, -40.0, 40.0))))

def mc_shapley(
    utility: Utility,
    n_players: int,
    permutations: int = 500,
    seed: int = 0,
    antithetic: bool = True,
) -> tuple[np.ndarray, float]:
    """Estimate per-player Shapley values with a prefix-walk estimator.

    Returns ``(values, efficiency_error)``. With the prefix walk, efficiency
    is expected to hold up to floating-point error by telescoping; the returned
    error is therefore a numerical-stability diagnostic, not a convergence
    diagnostic. The utility receives index coalitions ``frozenset(range(n))``.
    """
    if n_players < 0 or permutations < 1:
        raise ValueError("n_players must be non-negative and permutations must be positive")
    rng = np.random.default_rng(seed)
    values = np.zeros(n_players, dtype=float)
    full = frozenset(range(n_players))
    empty = utility(frozenset())
    for _ in range(permutations):
        order = rng.permutation(n_players)
        current = frozenset()
        previous = empty
        for player in order:
            current = current | {int(player)}
            now = utility(current)
            values[player] += now - previous
            previous = now
        if antithetic and n_players:
            current = frozenset()
            previous = empty
            for player in order[::-1]:
                current = current | {int(player)}
                now = utility(current)
                values[player] += now - previous
                previous = now
    divisor = permutations * (2 if antithetic and n_players else 1)
    values /= divisor
    error = abs(float(values.sum()) - float(utility(full) - empty))
    return values, error


def joint_attribution_score(values: np.ndarray, action: tuple[int, ...], signed: bool = False) -> float:
    """Convert per-player attribution into the prescribed joint-action score."""
    values = np.asarray(values, dtype=float)
    if any(i < 0 or i >= values.size for i in action):
        raise ValueError("action contains an invalid player index")
    selected = values[list(action)] if action else np.empty(0)
    return float(selected.sum() if signed else np.abs(selected).sum())


def select_joint_action(values: np.ndarray, budget: int, signed: bool = False) -> tuple[int, ...]:
    """Select the top-|budget| joint action using the spec's aggregation rule."""
    values = np.asarray(values, dtype=float)
    if budget < 1 or budget > values.size:
        raise ValueError("budget must be between 1 and the number of players")
    order = np.argsort(-(values if signed else np.abs(values)), kind="stable")
    return tuple(int(i) for i in order[:budget])


def exhaustive_oracle(
    utility: Utility,
    n_players: int,
    budget: int,
    rho_grid: Iterable[float] = (0.0,),
    apply_action: Callable[[tuple[int, ...], tuple[float, ...]], float] | None = None,
) -> tuple[tuple[int, ...], tuple[float, ...], float]:
    """Exhaustive best joint action for small games.

    ``apply_action`` must return the signed intervention effect for a player
    tuple and matching rho tuple. It is deliberately injected so the oracle
    cannot accidentally use an attribution or test outcome.
    """
    if apply_action is None:
        raise ValueError("apply_action is required for an intervention oracle")
    if budget < 1 or budget > n_players:
        raise ValueError("invalid budget")
    best = None
    for action in combinations(range(n_players), budget):
        for rhos in product(tuple(rho_grid), repeat=budget):
            effect = float(apply_action(action, rhos))
            candidate = (effect, action, rhos)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    return best[1], best[2], best[0]
