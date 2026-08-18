"""Synthetic-data tests for the review-3 experiment modules."""
from __future__ import annotations

import numpy as np

from actionshap.ablations import (
    select_forced,
    select_interaction_aware,
    select_magnitude,
)
from actionshap.bounded_baselines import (
    bounded_lime,
    finite_difference,
    integrated_gradients,
)
from actionshap.exact_shapley import exact_shapley, mc_error_report
from actionshap.interactions import (
    additive_vs_realized_rank,
    interaction_summary,
    pair_interactions,
)
from actionshap.recommendation import UserGame, mc_shapley


class LinearWeightModel:
    """score_i = sum_p w_p A[i,p]; exposes the ActionShap weight interface."""

    def __init__(self, A: np.ndarray):
        self.A = A

    def score(self, history, candidates, weights=None):
        w = np.ones(self.A.shape[1]) if weights is None else np.asarray(weights)
        return (self.A @ w)[np.asarray(candidates, dtype=int)]


def _game(A: np.ndarray, target: int = 0) -> UserGame:
    m = A.shape[1]
    return UserGame(
        players=np.arange(m),
        candidate_items=np.arange(A.shape[0]),
        target_item=target,
        tie_break=np.arange(A.shape[0], dtype=np.int64),
    )


def test_exact_shapley_additive_game():
    weights = np.array([1.0, 2.0, -3.0, 0.5])
    utility = lambda s: sum(weights[i] for i in s)
    exact = exact_shapley(utility, 4)
    assert np.allclose(exact, weights, atol=1e-9)


def test_exact_shapley_validates_mc():
    rng = np.random.default_rng(0)
    weights = rng.normal(size=6)
    utility = lambda s: sum(weights[i] for i in s)
    exact = exact_shapley(utility, 6)
    est, _ = mc_shapley(utility, 6, permutations=200, seed=1)
    rep = mc_error_report(exact, est)
    assert rep["max_abs_error"] < 0.25
    assert rep["rank_spearman"] > 0.9


def test_exact_shapley_rejects_large_games():
    import pytest

    with pytest.raises(ValueError):
        exact_shapley(lambda s: 0.0, 13)


def test_pair_interactions_zero_when_additive():
    effects = np.array([1.0, 2.0, 3.0])
    pairs = {(0, 1): 3.0, (0, 2): 4.0, (1, 2): 5.0}
    ints = pair_interactions(effects, pairs)
    assert np.allclose(list(ints.values()), 0.0)
    summ = interaction_summary(effects, pairs)
    assert summ["mean_abs_interaction"] == 0.0


def test_additive_vs_realized_rank_perfect_when_additive():
    attribution = np.array([-1.0, -2.0, -3.0])
    effects = np.array([1.0, 2.0, 3.0])
    pairs = {(0, 1): 3.0, (0, 2): 4.0, (1, 2): 5.0}
    rep = additive_vs_realized_rank(attribution, effects, pairs)
    assert rep["spearman_additive_realized"] > 0.99


def test_bounded_baselines_linear_utility_signs():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 4))
    # make target clearly sensitive to player 0
    A[0] += np.array([4.0, 0.0, 0.0, 0.0])
    model = LinearWeightModel(A)
    game = _game(A, target=0)
    fd = finite_difference(model, game)
    ig = integrated_gradients(model, game, steps=8)
    bl = bounded_lime(model, game, continuous=True, samples=300, seed=5)
    # downweighting player 0 hurts the target margin most -> -phi predicts harm
    assert fd.argmax() == 0
    assert ig.argmax() == 0
    assert bl.argmax() == 0


def test_ablation_selectors():
    attribution = np.array([-2.0, -1.0, 3.0])
    assert select_forced(attribution, 2) == (0, 1)
    assert select_magnitude(attribution, 2) == (2, 0)
    best = select_interaction_aware(
        np.array([1.0, 1.0]), {(0, 1): 5.0})
    assert best == (0, 1)
    best_single = select_interaction_aware(np.array([2.0, 1.0]), {(0, 1): 1.0})
    assert best_single == (0,)


def test_runtime_bench_keys():
    from actionshap.runtime_bench import bench

    rep = bench(lambda: np.dot(np.ones(10), np.ones(10)), repeat=1)
    assert {"median_wall_seconds", "peak_rss_mib"} <= set(rep)
