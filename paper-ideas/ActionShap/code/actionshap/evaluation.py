"""Leakage-safe evaluation helpers for recommendation-only ActionShap."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from .baselines import lime_attribution, monte_carlo_attribution, permutation_importance
from .recommendation import UserGame, ndcg_at_k, profile_utility, target_margin_utility


def _value(model, game: UserGame, coalition: frozenset[int], k: int, utility: str) -> float:
    if utility == "target_margin":
        return target_margin_utility(model, game, coalition, k)
    if utility == "ndcg":
        return profile_utility(model, game, coalition, k)
    raise ValueError("utility must be 'target_margin' or 'ndcg'")


def _value_from_scores(scores: np.ndarray, game: UserGame, k: int, utility: str) -> float:
    if utility == "ndcg":
        return ndcg_at_k(scores, game.candidate_items, game.target_item, k, game.tie_break)
    if utility == "target_margin":
        target_index = int(np.flatnonzero(game.candidate_items == game.target_item)[0])
        competitors = np.delete(scores, target_index)
        top = np.sort(competitors)[-min(10, competitors.size):]
        margin = scores[target_index] - top.mean()
        return float(1.0 / (1.0 + np.exp(-np.clip(margin, -40.0, 40.0))))
    raise ValueError("utility must be 'target_margin' or 'ndcg'")


def single_player_effects(
    model,
    game: UserGame,
    k: int = 10,
    rho: float = 0.0,
    utility: str = "target_margin",
) -> np.ndarray:
    """Signed effects of downweighting each retained history interaction."""
    base = _value(model, game, frozenset(range(game.players.size)), k, utility)
    effects = np.empty(game.players.size, dtype=float)
    for p in range(game.players.size):
        weights = np.ones(game.players.size, dtype=float)
        weights[p] = rho
        scores = model.score_downweighted(game.players, game.candidate_items, weights)
        effects[p] = _value_from_scores(scores, game, k, utility) - base
    return effects


def joint_effect(
    model,
    game: UserGame,
    action: tuple[int, ...],
    rho: float | tuple[float, ...] = 0.0,
    k: int = 10,
    utility: str = "target_margin",
) -> float:
    base = _value(model, game, frozenset(range(game.players.size)), k, utility)
    weights = np.ones(game.players.size, dtype=float)
    if isinstance(rho, tuple):
        if len(rho) != len(action):
            raise ValueError("rho tuple must align with action")
        weights[list(action)] = np.asarray(rho, dtype=float)
    else:
        weights[list(action)] = float(rho)
    scores = model.score_downweighted(game.players, game.candidate_items, weights)
    return float(_value_from_scores(scores, game, k, utility) - base)


def exhaustive_best_joint(
    model,
    game: UserGame,
    budget: int,
    rho_grid=(0.0,),
    k: int = 10,
    utility: str = "target_margin",
):
    """Best positive-effect joint action; intended for B=1/B=2."""
    best = None
    for action in combinations(range(game.players.size), budget):
        for rho in rho_grid:
            effect = joint_effect(model, game, action, rho=float(rho), k=k, utility=utility)
            candidate = (effect, action, float(rho))
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise ValueError("no feasible joint action")
    return best[1], best[2], best[0]


def greedy_best_joint(
    model,
    game: UserGame,
    budget: int,
    rho_grid=(0.0,),
    k: int = 10,
    utility: str = "target_margin",
):
    """Greedy forward oracle for budgets too large for exhaustive search."""
    chosen: list[int] = []
    chosen_rho: list[float] = []
    base = _value(model, game, frozenset(range(game.players.size)), k, utility)
    for _ in range(budget):
        best = None
        for p in range(game.players.size):
            if p in chosen:
                continue
            for rho in rho_grid:
                action = tuple(chosen + [p])
                weights = np.ones(game.players.size, dtype=float)
                weights[list(action)] = rho
                scores = model.score_downweighted(game.players, game.candidate_items, weights)
                effect = _value_from_scores(scores, game, k, utility) - base
                if best is None or effect > best[0]:
                    best = (float(effect), p, float(rho))
        if best is None:
            break
        _, p, rho = best
        chosen.append(p)
        chosen_rho.append(rho)
    return tuple(chosen), tuple(chosen_rho), joint_effect(
        model, game, tuple(chosen), tuple(chosen_rho), k, utility
    )


def aia(attribution: np.ndarray, effects: np.ndarray) -> float:
    a = np.abs(np.asarray(attribution, dtype=float))
    d = np.abs(np.asarray(effects, dtype=float))
    if a.shape != d.shape or a.size < 2 or a.std() == 0 or d.std() == 0:
        return float("nan")
    return float(spearmanr(a, d).statistic)


def within_user_aia_null(
    attribution: np.ndarray,
    effects: np.ndarray,
    draws: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.abs(np.asarray(effects, dtype=float)).copy()
    values = []
    for _ in range(draws):
        rng.shuffle(shuffled)
        value = aia(attribution, shuffled)
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def attribution_methods(utility, n_players: int, permutations: int, seed: int) -> dict[str, np.ndarray]:
    shap, _ = monte_carlo_attribution(utility, n_players, permutations, seed)
    return {
        "shapley_mc": shap,
        "loo_oracle": permutation_importance(utility, n_players),
        "lime": lime_attribution(utility, n_players, seed=seed),
    }
