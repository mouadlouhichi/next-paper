"""Core recommendation-game utilities for ActionShap.

The module is intentionally model independent.  It defines a frozen per-user
coalition game, ranking utilities, a cached Monte Carlo Shapley estimator, and
separate action-selection rules for two different questions:

* ``select_joint_action`` ranks factors by attribution magnitude.  It is useful
  for change-prediction diagnostics and retains the original exact-budget API.
* ``select_downweight_action`` predicts *beneficial* downweighting from signed
  attributions and may abstain.  This is the rule used for intervention regret.

Keeping the two rules separate prevents a high magnitude-only alignment score
from being mistaken for a recommendation to take an action in a known direction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

Utility = Callable[[frozenset[int]], float]


@dataclass(frozen=True)
class UserGame:
    """One user's frozen game on one fixed evaluation set.

    ``players`` are item IDs from the retained training history.  Coalitions
    elsewhere in the package are represented by *positions* in this array so
    repeated interactions, if present in another dataset, remain distinct.
    ``tie_break`` contains deterministic global item priorities aligned with
    ``candidate_items``; lower values rank first when scores tie.
    """

    players: np.ndarray
    candidate_items: np.ndarray
    target_item: int
    tie_break: np.ndarray

    def __post_init__(self) -> None:
        players = np.asarray(self.players, dtype=int)
        candidates = np.asarray(self.candidate_items, dtype=int)
        tie = np.asarray(self.tie_break, dtype=np.int64)
        if players.ndim != 1 or candidates.ndim != 1 or tie.ndim != 1:
            raise ValueError("players, candidate_items, and tie_break must be 1-D")
        if candidates.size == 0 or tie.shape != candidates.shape:
            raise ValueError(
                "candidate_items and tie_break must be non-empty and aligned"
            )
        if np.unique(candidates).size != candidates.size:
            raise ValueError("candidate_items must not contain duplicates")
        if np.count_nonzero(candidates == int(self.target_item)) != 1:
            raise ValueError(
                "target_item must occur exactly once in the fixed evaluation set"
            )
        if np.unique(tie).size != tie.size:
            raise ValueError(
                "tie_break priorities must be unique within an evaluation set"
            )
        object.__setattr__(self, "players", players)
        object.__setattr__(self, "candidate_items", candidates)
        object.__setattr__(self, "target_item", int(self.target_item))
        object.__setattr__(self, "tie_break", tie)


def ranking_order(scores: np.ndarray, tie_break: np.ndarray) -> np.ndarray:
    """Return deterministic descending-score order."""
    values = np.asarray(scores, dtype=float)
    tie = np.asarray(tie_break, dtype=np.int64)
    if values.ndim != 1 or values.shape != tie.shape:
        raise ValueError("scores and tie_break must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    return np.lexsort((tie, -values))


def target_rank(
    scores: np.ndarray,
    candidate_items: np.ndarray,
    target_item: int,
    tie_break: np.ndarray | None = None,
) -> int:
    """One-indexed deterministic rank of the held-out target."""
    scores = np.asarray(scores, dtype=float)
    items = np.asarray(candidate_items, dtype=int)
    if scores.shape != items.shape:
        raise ValueError("scores and candidate_items must have the same shape")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if np.count_nonzero(items == int(target_item)) != 1:
        raise ValueError("target_item must occur exactly once")
    tie = (
        np.arange(items.size, dtype=np.int64)
        if tie_break is None
        else np.asarray(tie_break, dtype=np.int64)
    )
    if tie.shape != items.shape:
        raise ValueError("tie_break must align with candidate_items")
    target_position = int(np.flatnonzero(items == int(target_item))[0])
    target_score = scores[target_position]
    target_priority = tie[target_position]
    strictly_higher = np.count_nonzero(scores > target_score)
    tied_ahead = np.count_nonzero((scores == target_score) & (tie < target_priority))
    return int(1 + strictly_higher + tied_ahead)


def ndcg_at_k(
    scores: np.ndarray,
    candidate_items: np.ndarray,
    target_item: int,
    k: int = 10,
    tie_break: np.ndarray | None = None,
) -> float:
    """NDCG@k for exactly one held-out relevant item."""
    if k < 1:
        raise ValueError("k must be positive")
    rank = target_rank(scores, candidate_items, target_item, tie_break)
    return 0.0 if rank > k else float(1.0 / np.log2(rank + 1))


def reciprocal_rank(
    scores: np.ndarray,
    candidate_items: np.ndarray,
    target_item: int,
    tie_break: np.ndarray | None = None,
) -> float:
    """Reciprocal rank of the single held-out target."""
    return 1.0 / target_rank(scores, candidate_items, target_item, tie_break)


def profile_utility(
    model, game: UserGame, coalition: frozenset[int], k: int = 10
) -> float:
    """Evaluate the paper's NDCG characteristic function ``v_u(S)``."""
    ids = game.players
    mask = np.fromiter(
        (i in coalition for i in range(ids.size)), dtype=bool, count=ids.size
    )
    scores = model.score_masked(ids, game.candidate_items, mask)
    return ndcg_at_k(scores, game.candidate_items, game.target_item, k, game.tie_break)


def target_margin_from_scores(
    scores: np.ndarray,
    game: UserGame,
    competitor_count: int = 10,
) -> float:
    """Continuous target-vs-top-competitors utility.

    This is a smooth attribution utility, not NDCG.  It may be used in a
    predeclared sensitivity analysis, but its intervention effects must always
    be labelled ``target-margin`` and never as changes in NDCG.
    """
    if competitor_count < 1:
        raise ValueError("competitor_count must be positive")
    values = np.asarray(scores, dtype=float)
    if values.shape != game.candidate_items.shape:
        raise ValueError("scores must align with the fixed evaluation set")
    target_index = int(np.flatnonzero(game.candidate_items == game.target_item)[0])
    competitors = np.delete(values, target_index)
    if competitors.size == 0:
        raise ValueError("target-margin utility requires at least one competitor")
    count = min(competitor_count, competitors.size)
    top = np.partition(competitors, competitors.size - count)[-count:]
    margin = float(values[target_index] - top.mean())
    return float(1.0 / (1.0 + np.exp(-np.clip(margin, -40.0, 40.0))))


def target_margin_utility(
    model,
    game: UserGame,
    coalition: frozenset[int],
    competitor_count: int = 10,
) -> float:
    """Continuous target-margin utility on the frozen evaluation set."""
    ids = game.players
    mask = np.fromiter(
        (i in coalition for i in range(ids.size)), dtype=bool, count=ids.size
    )
    scores = model.score_masked(ids, game.candidate_items, mask)
    return target_margin_from_scores(scores, game, competitor_count)


def mc_shapley(
    utility: Utility,
    n_players: int,
    permutations: int = 500,
    seed: int | tuple = 0,
    antithetic: bool = True,
) -> tuple[np.ndarray, float]:
    """Estimate Shapley values with cached antithetic permutation prefix walks.

    ``permutations`` is explicitly the number of sampled *base* orders,
    denoted :math:`M_{pair}` in the manuscript.  When ``antithetic`` is true,
    each base order and its reverse are evaluated, so the total number of
    prefix walks is ``T = 2 * M_pair``.  Prefix-walk efficiency telescopes
    exactly and is returned only as a numerical-stability check.
    """
    if n_players < 0 or permutations < 1:
        raise ValueError(
            "n_players must be non-negative and permutations must be positive"
        )
    rng = np.random.default_rng(seed)
    values = np.zeros(n_players, dtype=float)
    cache: dict[frozenset[int], float] = {}

    def cached(coalition: frozenset[int]) -> float:
        if coalition not in cache:
            value = float(utility(coalition))
            if not np.isfinite(value):
                raise ValueError("utility returned a non-finite value")
            cache[coalition] = value
        return cache[coalition]

    empty_set = frozenset()
    full = frozenset(range(n_players))
    empty = cached(empty_set)
    walks = 0
    for _ in range(permutations):
        sampled = rng.permutation(n_players)
        orders = (sampled, sampled[::-1]) if antithetic and n_players else (sampled,)
        for order in orders:
            current = empty_set
            previous = empty
            for player in order:
                p = int(player)
                current = current | {p}
                now = cached(current)
                values[p] += now - previous
                previous = now
            walks += 1
    values /= walks
    error = abs(float(values.sum()) - float(cached(full) - empty))
    return values, error




def mc_shapley_with_se(
    utility: Utility,
    n_players: int,
    permutations: int = 500,
    seed: int | tuple = 0,
    antithetic: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Monte Carlo Shapley with per-player Monte Carlo standard errors.

    Identical sampling scheme to :func:`mc_shapley`; additionally accumulates
    the per-player squared marginal contributions so that the Monte Carlo
    standard error of each player's value is
    ``sqrt(max(0, E[m^2] - E[m]^2) / (T - 1))`` over the ``T`` evaluated
    orders. Returns ``(values, standard_errors, efficiency_error)``.
    """
    if n_players < 0 or permutations < 1:
        raise ValueError(
            "n_players must be non-negative and permutations must be positive"
        )
    rng = np.random.default_rng(seed)
    values = np.zeros(n_players, dtype=float)
    squares = np.zeros(n_players, dtype=float)
    cache: dict[frozenset[int], float] = {}

    def cached(coalition: frozenset[int]) -> float:
        if coalition not in cache:
            value = float(utility(coalition))
            if not np.isfinite(value):
                raise ValueError("utility returned a non-finite value")
            cache[coalition] = value
        return cache[coalition]

    empty_set = frozenset()
    full = frozenset(range(n_players))
    empty = cached(empty_set)
    walks = 0
    for _ in range(permutations):
        sampled = rng.permutation(n_players)
        orders = (sampled, sampled[::-1]) if antithetic and n_players else (sampled,)
        for order in orders:
            current = empty_set
            previous = empty
            for player in order:
                p = int(player)
                current = current | {p}
                now = cached(current)
                d = now - previous
                values[p] += d
                squares[p] += d * d
                previous = now
            walks += 1
    mean = values / walks
    var = np.maximum(squares / walks - mean * mean, 0.0)
    se = np.sqrt(var / max(walks - 1, 1))
    error = abs(float(values.sum() / walks) - float(cached(full) - empty))
    return mean, se, error


def joint_attribution_score(
    values: np.ndarray, action: tuple[int, ...], signed: bool = False
) -> float:
    """Aggregate individual attribution values for a joint action."""
    values = np.asarray(values, dtype=float)
    if any(i < 0 or i >= values.size for i in action):
        raise ValueError("action contains an invalid player index")
    selected = values[list(action)] if action else np.empty(0)
    return float(selected.sum() if signed else np.abs(selected).sum())


def select_joint_action(
    values: np.ndarray, budget: int, signed: bool = False
) -> tuple[int, ...]:
    """Select exactly ``budget`` factors by magnitude (or positive signed score).

    This function is retained for magnitude-prediction diagnostics.  Benefit-
    seeking downweight interventions must use :func:`select_downweight_action`.
    """
    values = np.asarray(values, dtype=float)
    if budget < 1 or budget > values.size:
        raise ValueError("budget must be between 1 and the number of players")
    order = np.argsort(-(values if signed else np.abs(values)), kind="stable")
    return tuple(int(i) for i in order[:budget])


def select_downweight_action(
    values: np.ndarray,
    budget: int,
    *,
    allow_abstain: bool = True,
    tolerance: float = 0.0,
) -> tuple[int, ...]:
    """Select up to ``budget`` interactions predicted to benefit from downweighting.

    For signed coalition attributions, a negative value means the interaction
    lowers the utility relative to the null profile.  Downweighting it is thus
    predicted to have benefit ``-value``.  Only strictly positive predicted
    benefits above ``tolerance`` are selected.  If none exist, the method may
    abstain by returning the empty action.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional array")
    if budget < 1 or budget > values.size:
        raise ValueError("budget must be between 1 and the number of players")
    predicted_benefit = -values
    order = np.argsort(-predicted_benefit, kind="stable")
    selected = [int(i) for i in order if predicted_benefit[i] > tolerance][:budget]
    if selected or allow_abstain:
        return tuple(selected)
    return (int(order[0]),)


def exhaustive_oracle(
    utility: Utility,
    n_players: int,
    budget: int,
    rho_grid: Iterable[float] = (0.5,),
    apply_action: Callable[[tuple[int, ...], tuple[float, ...]], float] | None = None,
    *,
    allow_abstain: bool = True,
) -> tuple[tuple[int, ...], tuple[float, ...], float]:
    """Exhaustive best action of size at most ``budget`` for a small game.

    The no-action option has effect zero when ``allow_abstain`` is true.  Each
    selected player may independently take any value in ``rho_grid``.
    ``utility`` remains in the signature for backwards compatibility; action
    effects are deliberately supplied through ``apply_action`` so an oracle
    cannot accidentally inspect an attribution vector.
    """
    del utility
    if apply_action is None:
        raise ValueError("apply_action is required for an intervention oracle")
    if budget < 1 or budget > n_players:
        raise ValueError("invalid budget")
    rhos = tuple(float(r) for r in rho_grid)
    if not rhos or any(not 0.0 <= r <= 1.0 for r in rhos):
        raise ValueError("rho_grid must contain values in [0, 1]")
    best: tuple[float, tuple[int, ...], tuple[float, ...]] | None = (
        (0.0, (), ()) if allow_abstain else None
    )
    for size in range(1, budget + 1):
        for action in combinations(range(n_players), size):
            for action_rhos in product(rhos, repeat=size):
                effect = float(apply_action(action, action_rhos))
                candidate = (effect, action, action_rhos)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    assert best is not None
    return best[1], best[2], best[0]
