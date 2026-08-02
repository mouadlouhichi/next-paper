"""Tests for the recommendation-only ActionShap core."""

from __future__ import annotations

import numpy as np
import pytest

from actionshap.models.profile import ProfileAggregationModel
from actionshap.recommendation import (
    UserGame,
    joint_attribution_score,
    mc_shapley,
    ndcg_at_k,
    profile_utility,
    select_joint_action,
)


def test_empty_profile_is_zero_and_masking_changes_scores():
    model = ProfileAggregationModel(np.eye(4))
    candidates = np.array([0, 1, 2, 3])
    full = model.score(np.array([0]), candidates)
    empty = model.score(np.array([], dtype=int), candidates)
    masked = model.score_masked(np.array([0, 1]), candidates, np.array([True, False]))
    assert np.all(empty == 0.0)
    assert not np.allclose(full, masked)


def test_ndcg_is_deterministic_under_ties():
    items = np.array([2, 1, 3])
    tie = np.array([1, 0, 2])
    assert ndcg_at_k(np.zeros(3), items, 1, k=1, tie_break=tie) == pytest.approx(1.0)


def test_profile_utility_full_and_empty_are_defined():
    model = ProfileAggregationModel(np.eye(4))
    game = UserGame(
        players=np.array([0, 1]),
        candidate_items=np.array([0, 1, 2, 3]),
        target_item=0,
        tie_break=np.arange(4),
    )
    empty = profile_utility(model, game, frozenset())
    full = profile_utility(model, game, frozenset({0, 1}))
    assert 0.0 <= empty <= 1.0
    assert 0.0 <= full <= 1.0


def test_mc_shapley_prefix_walk_satisfies_efficiency():
    def utility(coalition):
        s = set(coalition)
        return float(len(s) + (2 if {0, 1}.issubset(s) else 0))

    values, error = mc_shapley(utility, n_players=3, permutations=20, seed=7)
    assert values[0] > 0 and values[1] > 0
    assert error < 1e-12


def test_joint_selection_and_aggregation():
    values = np.array([-0.9, 0.2, 0.7])
    action = select_joint_action(values, budget=2)
    assert action == (0, 2)
    assert joint_attribution_score(values, action) == pytest.approx(1.6)
    assert joint_attribution_score(values, action, signed=True) == pytest.approx(-0.2)


def test_joint_selection_rejects_invalid_budget():
    with pytest.raises(ValueError):
        select_joint_action(np.ones(2), budget=3)
