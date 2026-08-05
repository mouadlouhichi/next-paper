"""Evaluation helpers for recommendation-only ActionShap."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations, product

import numpy as np
from scipy.stats import rankdata, spearmanr

from .baselines import (
    greedy_counterfactual_attribution,
    lime_attribution,
    monte_carlo_attribution,
    permutation_importance,
    random_attribution,
)
from .recommendation import (
    UserGame,
    ndcg_at_k,
    profile_utility,
    ranking_order,
    reciprocal_rank,
    target_margin_from_scores,
    target_margin_utility,
    target_rank,
)

SUPPORTED_UTILITIES = ("ndcg", "target_margin")


def _value(
    model, game: UserGame, coalition: frozenset[int], k: int, utility: str
) -> float:
    if utility == "ndcg":
        return profile_utility(model, game, coalition, k)
    if utility == "target_margin":
        return target_margin_utility(model, game, coalition)
    raise ValueError(f"utility must be one of {SUPPORTED_UTILITIES}")


def _value_from_scores(
    scores: np.ndarray, game: UserGame, k: int, utility: str
) -> float:
    if utility == "ndcg":
        return ndcg_at_k(
            scores, game.candidate_items, game.target_item, k, game.tie_break
        )
    if utility == "target_margin":
        return target_margin_from_scores(scores, game)
    raise ValueError(f"utility must be one of {SUPPORTED_UTILITIES}")


def _batch_values_from_scores(
    score_matrix: np.ndarray,
    game: UserGame,
    k: int,
    utility: str,
) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != game.candidate_items.size:
        raise ValueError(
            "score_matrix must have one row per action and one candidate column"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("score_matrix must be finite")
    target_index = int(np.flatnonzero(game.candidate_items == game.target_item)[0])
    target_scores = scores[:, target_index]
    if utility == "ndcg":
        target_priority = game.tie_break[target_index]
        ranks = (
            1
            + np.count_nonzero(scores > target_scores[:, None], axis=1)
            + np.count_nonzero(
                (scores == target_scores[:, None])
                & (game.tie_break[None, :] < target_priority),
                axis=1,
            )
        )
        return np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).astype(float)
    if utility == "target_margin":
        competitors = np.delete(scores, target_index, axis=1)
        if competitors.shape[1] == 0:
            raise ValueError("target-margin utility requires at least one competitor")
        count = min(10, competitors.shape[1])
        top = np.partition(competitors, competitors.shape[1] - count, axis=1)[
            :, -count:
        ]
        margins = target_scores - top.mean(axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(margins, -40.0, 40.0)))
    raise ValueError(f"utility must be one of {SUPPORTED_UTILITIES}")


def ranking_metrics_from_scores(
    scores: np.ndarray, game: UserGame, k: int = 10
) -> dict[str, float | int]:
    """Standard single-target ranking metrics from an aligned score vector."""
    rank = target_rank(scores, game.candidate_items, game.target_item, game.tie_break)
    return {
        "target_rank": int(rank),
        f"ndcg@{k}": ndcg_at_k(
            scores, game.candidate_items, game.target_item, k, game.tie_break
        ),
        f"recall@{k}": float(rank <= k),
        "mrr": reciprocal_rank(
            scores, game.candidate_items, game.target_item, game.tie_break
        ),
    }


def recommendation_metrics(
    model, game: UserGame, k: int = 10
) -> dict[str, float | int]:
    """Standard full-profile ranking metrics for one held-out target."""
    return ranking_metrics_from_scores(
        model.score(game.players, game.candidate_items), game, k
    )


def model_mc_shapley(
    model,
    game: UserGame,
    permutations: int,
    seed: int,
    *,
    k: int = 10,
    utility: str = "ndcg",
    antithetic: bool = True,
    permutation_batch_size: int = 16,
) -> tuple[np.ndarray, float]:
    """Batched prefix-walk Shapley estimator.

    ``permutations`` is the number of base permutations.  With the default
    antithetic reverse walk, the estimator evaluates twice that many orders;
    callers should report both values rather than using one ambiguous ``M``.
    """
    n_players = game.players.size
    if permutations < 1 or permutation_batch_size < 1:
        raise ValueError("permutation counts and batch size must be positive")
    if not hasattr(model, "score_downweighted_batch"):
        from .recommendation import mc_shapley

        value = lambda coalition: _value(model, game, coalition, k, utility)
        return mc_shapley(value, n_players, permutations, seed, antithetic)

    rng = np.random.default_rng(seed)
    sampled_orders = [rng.permutation(n_players) for _ in range(permutations)]
    orders = [
        order_variant
        for order in sampled_orders
        for order_variant in ((order, order[::-1]) if antithetic else (order,))
    ]
    values = np.zeros(n_players, dtype=float)
    rows_per_walk = n_players + 1
    for start in range(0, len(orders), permutation_batch_size):
        batch_orders = orders[start : start + permutation_batch_size]
        weights = np.zeros((len(batch_orders) * rows_per_walk, n_players), dtype=float)
        for walk, order in enumerate(batch_orders):
            offset = walk * rows_per_walk
            for step, player in enumerate(order, start=1):
                weights[offset + step] = weights[offset + step - 1]
                weights[offset + step, int(player)] = 1.0
        score_matrix = model.score_downweighted_batch(
            game.players, game.candidate_items, weights
        )
        coalition_values = _batch_values_from_scores(score_matrix, game, k, utility)
        for walk, order in enumerate(batch_orders):
            offset = walk * rows_per_walk
            marginals = np.diff(coalition_values[offset : offset + rows_per_walk])
            values[order] += marginals
    values /= len(orders)
    empty = _value(model, game, frozenset(), k, utility)
    full = _value(model, game, frozenset(range(n_players)), k, utility)
    efficiency_error = abs(float(values.sum()) - float(full - empty))
    return values, efficiency_error


def single_player_effects(
    model,
    game: UserGame,
    k: int = 10,
    rho: float = 0.0,
    utility: str = "ndcg",
) -> np.ndarray:
    """Signed effects of downweighting each retained history interaction."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    full = frozenset(range(game.players.size))
    base = _value(model, game, full, k, utility)

    # Keep the deletion diagnostic on the exact same coalition-evaluation path
    # as LOO.  The weighted-zero implementation is mathematically equivalent,
    # but it can differ by a few ulps because it multiplies by zero and sums a
    # different sparse block.  Those ulps can reorder tied magnitudes and make
    # the formally identical LOO/deletion AIA appear as 0.9999 instead of 1.
    # rho=0 is therefore evaluated as v(P\{p}) - v(P) directly.
    if rho == 0.0:
        return np.array(
            [
                _value(model, game, full - {player}, k, utility) - base
                for player in range(game.players.size)
            ],
            dtype=float,
        )

    weights = np.ones((game.players.size, game.players.size), dtype=float)
    weights[np.arange(game.players.size), np.arange(game.players.size)] = rho
    if hasattr(model, "score_downweighted_batch"):
        scores = model.score_downweighted_batch(
            game.players, game.candidate_items, weights
        )
    else:
        scores = np.vstack(
            [
                model.score_downweighted(game.players, game.candidate_items, row)
                for row in weights
            ]
        )
    return _batch_values_from_scores(scores, game, k, utility) - base


def joint_effect(
    model,
    game: UserGame,
    action: tuple[int, ...],
    rho: float | tuple[float, ...] = 0.5,
    k: int = 10,
    utility: str = "ndcg",
) -> float:
    """Signed utility effect of one joint downweighting action."""
    if len(set(action)) != len(action) or any(
        player < 0 or player >= game.players.size for player in action
    ):
        raise ValueError("action must contain unique valid player positions")
    if not action:
        return 0.0
    base = _value(model, game, frozenset(range(game.players.size)), k, utility)
    weights = np.ones(game.players.size, dtype=float)
    if isinstance(rho, tuple):
        if len(rho) != len(action):
            raise ValueError("rho tuple must align with action")
        action_rhos = np.asarray(rho, dtype=float)
    else:
        action_rhos = np.full(len(action), float(rho), dtype=float)
    if np.any((action_rhos < 0) | (action_rhos > 1)):
        raise ValueError("rho values must lie in [0, 1]")
    if action:
        weights[list(action)] = action_rhos
    scores = model.score_downweighted(game.players, game.candidate_items, weights)
    return float(_value_from_scores(scores, game, k, utility) - base)


def exhaustive_best_joint_multi(
    model,
    game: UserGame,
    budget: int,
    rho_grid: Iterable[float] = (0.5,),
    k: int = 10,
    utilities: tuple[str, ...] = ("target_margin", "ndcg"),
    *,
    allow_abstain: bool = True,
) -> dict[str, tuple[tuple[int, ...], tuple[float, ...], float]]:
    """Evaluate one exact action space against several utilities in one score pass."""
    if budget < 1 or budget > game.players.size:
        raise ValueError("budget must be between one and the number of players")
    if not utilities or any(
        utility not in SUPPORTED_UTILITIES for utility in utilities
    ):
        raise ValueError(f"utilities must be drawn from {SUPPORTED_UTILITIES}")
    rhos = tuple(float(rho) for rho in rho_grid)
    if not rhos or any(not 0.0 <= rho <= 1.0 for rho in rhos):
        raise ValueError("rho_grid must contain values in [0, 1]")
    actions: list[tuple[int, ...]] = [()] if allow_abstain else []
    action_rhos_list: list[tuple[float, ...]] = [()] if allow_abstain else []
    for size in range(1, budget + 1):
        for action in combinations(range(game.players.size), size):
            for action_rhos in product(rhos, repeat=size):
                actions.append(action)
                action_rhos_list.append(action_rhos)

    weight_matrix = np.ones((len(actions), game.players.size), dtype=float)
    for row, (action, action_rhos) in enumerate(zip(actions, action_rhos_list)):
        if action:
            weight_matrix[row, list(action)] = action_rhos
    if hasattr(model, "score_downweighted_batch"):
        score_matrix = model.score_downweighted_batch(
            game.players, game.candidate_items, weight_matrix
        )
    else:
        score_matrix = np.vstack(
            [
                model.score_downweighted(game.players, game.candidate_items, weights)
                for weights in weight_matrix
            ]
        )

    output: dict[str, tuple[tuple[int, ...], tuple[float, ...], float]] = {}
    full = frozenset(range(game.players.size))
    for utility in dict.fromkeys(utilities):
        base = _value(model, game, full, k, utility)
        effects = _batch_values_from_scores(score_matrix, game, k, utility) - base
        if allow_abstain and actions[0] == ():
            effects[0] = 0.0
        best_index = int(np.argmax(effects))
        output[utility] = (
            actions[best_index],
            action_rhos_list[best_index],
            float(effects[best_index]),
        )
    return output


def exhaustive_best_joint(
    model,
    game: UserGame,
    budget: int,
    rho_grid: Iterable[float] = (0.5,),
    k: int = 10,
    utility: str = "ndcg",
    *,
    allow_abstain: bool = True,
) -> tuple[tuple[int, ...], tuple[float, ...], float]:
    """Exact best action of size at most ``budget``, including no action."""
    return exhaustive_best_joint_multi(
        model,
        game,
        budget,
        rho_grid,
        k,
        (utility,),
        allow_abstain=allow_abstain,
    )[utility]


def greedy_best_joint(
    model,
    game: UserGame,
    budget: int,
    rho_grid: Iterable[float] = (0.5,),
    k: int = 10,
    utility: str = "ndcg",
    *,
    allow_abstain: bool = True,
) -> tuple[tuple[int, ...], tuple[float, ...], float]:
    """Greedy forward action with early stopping when no step improves utility."""
    if budget < 1 or budget > game.players.size:
        raise ValueError("invalid budget")
    rhos = tuple(float(rho) for rho in rho_grid)
    chosen: list[int] = []
    chosen_rhos: list[float] = []
    current_effect = 0.0
    for _ in range(budget):
        best_step: tuple[float, int, float] | None = None
        for player in range(game.players.size):
            if player in chosen:
                continue
            for rho in rhos:
                action = tuple(chosen + [player])
                action_rhos = tuple(chosen_rhos + [rho])
                effect = joint_effect(model, game, action, action_rhos, k, utility)
                if best_step is None or effect > best_step[0]:
                    best_step = (float(effect), player, rho)
        if best_step is None:
            break
        effect, player, rho = best_step
        if allow_abstain and effect <= current_effect:
            break
        chosen.append(player)
        chosen_rhos.append(rho)
        current_effect = effect
    return tuple(chosen), tuple(chosen_rhos), float(current_effect)


def aia(attribution: np.ndarray, effects: np.ndarray) -> float:
    """Magnitude-only Attribution--Intervention Alignment."""
    attribution = np.abs(np.asarray(attribution, dtype=float))
    effects = np.abs(np.asarray(effects, dtype=float))
    if (
        attribution.shape != effects.shape
        or attribution.size < 2
        or attribution.std() == 0
        or effects.std() == 0
    ):
        return float("nan")
    return float(spearmanr(attribution, effects).statistic)


def signed_alignment(attribution: np.ndarray, effects: np.ndarray) -> float:
    """Alignment between predicted downweight benefit ``-phi`` and signed effect."""
    predicted = -np.asarray(attribution, dtype=float)
    observed = np.asarray(effects, dtype=float)
    if (
        predicted.shape != observed.shape
        or predicted.size < 2
        or predicted.std() == 0
        or observed.std() == 0
    ):
        return float("nan")
    return float(spearmanr(predicted, observed).statistic)


def direction_accuracy(attribution: np.ndarray, effects: np.ndarray) -> float:
    """Fraction of non-zero effects whose downweight direction is predicted."""
    predicted = -np.asarray(attribution, dtype=float)
    observed = np.asarray(effects, dtype=float)
    if predicted.shape != observed.shape:
        raise ValueError("attribution and effects must align")
    informative = observed != 0
    if not np.any(informative):
        return float("nan")
    return float(
        np.mean(np.sign(predicted[informative]) == np.sign(observed[informative]))
    )


def topk_intervention_precision(
    attribution: np.ndarray,
    effects: np.ndarray,
    k: int,
) -> float:
    """Overlap precision between predicted and realized beneficial top-k actions."""
    attribution = np.asarray(attribution, dtype=float)
    effects = np.asarray(effects, dtype=float)
    if attribution.shape != effects.shape or k < 1:
        raise ValueError("aligned vectors and a positive k are required")
    k = min(k, attribution.size)
    predicted_order = np.argsort(attribution, kind="stable")  # most negative first
    true_order = np.argsort(-effects, kind="stable")
    predicted = {int(i) for i in predicted_order if attribution[i] < 0}
    truth = {int(i) for i in true_order if effects[i] > 0}
    predicted_top = set(
        list(predicted_order[np.isin(predicted_order, list(predicted))])[:k]
    )
    true_top = set(list(true_order[np.isin(true_order, list(truth))])[:k])
    return float(len(predicted_top & true_top) / k)


def within_user_aia_null(
    attribution: np.ndarray,
    effects: np.ndarray,
    draws: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Within-user magnitude-AIA permutation null."""
    if draws < 1:
        raise ValueError("draws must be positive")
    attribution_values = np.abs(np.asarray(attribution, dtype=float))
    effect_values = np.abs(np.asarray(effects, dtype=float))
    if (
        attribution_values.shape != effect_values.shape
        or attribution_values.size < 2
        or attribution_values.std() == 0
        or effect_values.std() == 0
    ):
        return np.empty(0, dtype=float)
    left = rankdata(attribution_values)
    right = rankdata(effect_values)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    rng = np.random.default_rng(seed)
    orders = np.argsort(rng.random((draws, effect_values.size)), axis=1)
    return (right[orders] @ left) / denominator


def aia_null_summary(
    attribution: np.ndarray,
    effects: np.ndarray,
    draws: int = 1000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Observed AIA, calibrated null, and finite-resolution permutation p-value."""
    observed = aia(attribution, effects)
    null = within_user_aia_null(attribution, effects, draws, seed)
    if not np.isfinite(observed) or null.size == 0:
        return {
            "observed": None,
            "null_mean": None,
            "null_p95": None,
            "p_value": None,
            "draws": int(null.size),
        }
    exceedances = int(np.count_nonzero(null >= observed))
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float((exceedances + 1) / (null.size + 1)),
        "draws": int(null.size),
    }


def masking_sensitivity_gate(
    model,
    games: dict[int, UserGame],
    users: list[int],
    *,
    k: int = 10,
    seed: int = 42,
    minimum_changed_fraction: float = 0.5,
    minimum_mean_abs_ndcg: float = 1e-3,
) -> dict[str, float | int | bool]:
    """Run the mandatory gate on real evaluation games.

    The static control freezes each user's full-profile candidate scores and
    therefore must be exactly invariant to the same mask operation.
    """
    if len(users) < 1:
        raise ValueError("at least one gate user is required")
    rng = np.random.default_rng(seed)
    changed = 0
    ndcg_changes: list[float] = []
    static_changed = 0
    static_ndcg_changes: list[float] = []
    for user in users:
        game = games[user]
        if game.players.size < 1:
            continue
        full_scores = model.score(game.players, game.candidate_items)
        full_order = ranking_order(full_scores, game.tie_break)[:k]
        full_ndcg = ndcg_at_k(
            full_scores, game.candidate_items, game.target_item, k, game.tie_break
        )
        mask = np.ones(game.players.size, dtype=bool)
        mask[int(rng.integers(game.players.size))] = False
        masked_scores = model.score_masked(game.players, game.candidate_items, mask)
        masked_order = ranking_order(masked_scores, game.tie_break)[:k]
        masked_ndcg = ndcg_at_k(
            masked_scores, game.candidate_items, game.target_item, k, game.tie_break
        )
        changed += int(not np.array_equal(full_order, masked_order))
        ndcg_changes.append(abs(masked_ndcg - full_ndcg))

        # Frozen-score control: the history argument cannot influence scores.
        static_order = ranking_order(full_scores.copy(), game.tie_break)[:k]
        static_ndcg = ndcg_at_k(
            full_scores.copy(),
            game.candidate_items,
            game.target_item,
            k,
            game.tie_break,
        )
        static_changed += int(not np.array_equal(full_order, static_order))
        static_ndcg_changes.append(abs(static_ndcg - full_ndcg))

    n = len(ndcg_changes)
    if n == 0:
        raise ValueError("no gate user had an interaction to mask")
    changed_fraction = changed / n
    mean_abs_ndcg = float(np.mean(ndcg_changes))
    static_fraction = static_changed / n
    static_mean_abs_ndcg = float(np.mean(static_ndcg_changes))
    return {
        "users": n,
        "changed_fraction": float(changed_fraction),
        "mean_abs_ndcg_change": mean_abs_ndcg,
        "static_changed_fraction": float(static_fraction),
        "static_mean_abs_ndcg_change": static_mean_abs_ndcg,
        "dynamic_pass": bool(
            changed_fraction >= minimum_changed_fraction
            and mean_abs_ndcg >= minimum_mean_abs_ndcg
        ),
        "static_pass": bool(static_fraction == 0.0 and static_mean_abs_ndcg == 0.0),
        "passed": bool(
            changed_fraction >= minimum_changed_fraction
            and mean_abs_ndcg >= minimum_mean_abs_ndcg
            and static_fraction == 0.0
            and static_mean_abs_ndcg == 0.0
        ),
    }


def attribution_methods(
    utility,
    n_players: int,
    permutations: int,
    seed: int,
    *,
    lime_samples: int = 512,
) -> dict[str, np.ndarray]:
    """All required methods on one shared game."""
    shapley, _ = monte_carlo_attribution(utility, n_players, permutations, seed)
    return {
        "shapley_mc": shapley,
        "loo": permutation_importance(utility, n_players),
        "lime": lime_attribution(utility, n_players, samples=lime_samples, seed=seed),
        "greedy_cf": greedy_counterfactual_attribution(utility, n_players),
        "random": random_attribution(n_players, seed=seed + 1_000_000),
    }
