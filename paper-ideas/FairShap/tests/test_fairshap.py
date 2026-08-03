"""Tests for the FairShap package."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairshap import metrics as M
from fairshap import game as G
from fairshap import rerank as RR


def test_ndcg_ranked_first_is_one():
    assert M.ndcg_at_k([1, 2, 3, 4], [1], k=4) == pytest.approx(1.0)


def test_recall():
    assert M.recall_at_k([1, 2, 3], [1, 2], k=3) == pytest.approx(1.0)
    assert M.recall_at_k([1, 2, 3], [9], k=3) == 0.0


def test_gini_perfect_equal_is_zero():
    assert M.gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert M.gini([1, 0, 0, 0]) > 0.5


def test_exposure_gini():
    lists = [[1, 2], [1, 2], [1, 2], [3, 4]]
    assert 0.0 <= M.exposure_gini(lists, k=2) <= 1.0


def test_arp():
    pop = {1: 10, 2: 5, 3: 1}
    assert M.arp([[1, 2]], pop, k=2) == pytest.approx(7.5)


def test_coalition_value_bounds():
    exposure = {1: 0.5, 2: 0.3, 3: 0.1, 4: 0.1}
    sim = {1: {2: 0.5}, 2: {1: 0.5}}
    v = G.coalition_value(exposure, sim, [1, 2])
    assert 0.0 <= v <= 1.0


def test_exposure_shapley_values_float():
    phi = G.exposure_shapley({1: 0.5, 2: 0.3, 3: 0.2}, {}, [1, 2, 3], M=100, seed=0)
    assert set(phi.keys()) == {1, 2, 3}
    assert all(isinstance(v, float) for v in phi.values())


def test_mc_shapley_additive_recovers():
    w = {0: 2.0, 1: 3.0, 2: 1.0}
    def v(S):
        return sum(w[i] for i in S)
    phi = G.mc_shapley(v, [0, 1, 2], M=2000, seed=0)
    for i in [0, 1, 2]:
        assert abs(phi[i] - w[i]) < 0.05


def test_fair_rerank_returns_permutation():
    scores = {1: 1.0, 2: 0.8, 3: 0.5}
    phi = {1: 0.1, 2: 0.9, 3: 0.3}
    assert set(RR.fair_rerank(scores, phi, gamma=0.5)) == set(scores.keys())


def test_fair_rerank_gamma_controls_tradeoff():
    scores = {1: 1.0, 2: 0.0}
    phi = {1: 0.0, 2: 1.0}
    assert RR.fair_rerank(scores, phi, gamma=0.0)[0] == 1
    assert RR.fair_rerank(scores, phi, gamma=1.0)[0] == 2


def test_hypergraph_trains():
    from fairshap.model import train_hypergraph
    users_items = {0: [1, 2], 1: [2, 3], 2: [1, 3], 3: [4, 5]}
    Q = train_hypergraph(users_items, n_items=6, n_users=4, dim=8, epochs=2,
                         batch_size=8, seed=0, verbose=False)
    assert Q.shape == (6, 8)
    assert np.allclose(np.linalg.norm(Q, axis=1), 1.0, atol=1e-5)
