"""Tests for the recommendation-only ActionShap core."""

from __future__ import annotations

import numpy as np
import pytest
from actionshap.candidates import (
    fixed_evaluation_sets,
    full_unseen_evaluation_sets,
    global_item_priorities,
    tie_break_for_candidates,
)
from actionshap.evaluation import (
    exhaustive_best_joint,
    exhaustive_best_joint_multi,
    joint_effect,
    model_mc_shapley,
)
from actionshap.models.itemknn import fit_item_knn
from actionshap.models.profile import ProfileAggregationModel
from actionshap.recommendation import (
    UserGame,
    exhaustive_oracle,
    joint_attribution_score,
    mc_shapley,
    ndcg_at_k,
    profile_utility,
    select_downweight_action,
    select_joint_action,
    target_margin_utility,
    target_rank,
)


def _game() -> tuple[ProfileAggregationModel, UserGame]:
    model = ProfileAggregationModel(np.eye(5))
    candidates = np.array([0, 1, 2, 3, 4])
    priorities = global_item_priorities(5, seed=9)
    game = UserGame(
        players=np.array([0, 1]),
        candidate_items=candidates,
        target_item=0,
        tie_break=tie_break_for_candidates(candidates, priorities),
    )
    return model, game


def test_profile_batch_scoring_matches_individual_actions():
    model = ProfileAggregationModel(np.eye(4))
    history = np.array([0, 1, 2])
    candidates = np.arange(4)
    weights = np.array([[1.0, 1.0, 1.0], [0.5, 1.0, 1.0], [0.0, 0.0, 0.0]])
    batch = model.score_downweighted_batch(history, candidates, weights)
    individual = np.vstack(
        [model.score_downweighted(history, candidates, row) for row in weights]
    )
    assert np.allclose(batch, individual)


def test_itemknn_batch_scoring_matches_individual_actions():
    model = fit_item_knn(
        {0: np.array([0, 1]), 1: np.array([1, 2]), 2: np.array([2, 3])},
        n_items=4,
        neighbours=3,
    )
    history = np.array([0, 1, 2])
    candidates = np.arange(4)
    weights = np.array([[1.0, 1.0, 1.0], [0.5, 1.0, 1.0], [0.0, 0.0, 0.0]])
    batch = model.score_downweighted_batch(history, candidates, weights)
    individual = np.vstack(
        [model.score_downweighted(history, candidates, row) for row in weights]
    )
    assert np.allclose(batch, individual)


def test_empty_profile_is_zero_and_masking_changes_scores():
    model = ProfileAggregationModel(np.eye(4))
    candidates = np.array([0, 1, 2, 3])
    full = model.score(np.array([0, 1]), candidates)
    empty = model.score(np.array([], dtype=int), candidates)
    masked = model.score_masked(np.array([0, 1]), candidates, np.array([True, False]))
    assert np.all(empty == 0.0)
    assert not np.allclose(full, masked)


def test_ndcg_is_deterministic_under_ties():
    items = np.array([2, 1, 3])
    tie = np.array([1, 0, 2])
    assert ndcg_at_k(np.zeros(3), items, 1, k=1, tie_break=tie) == pytest.approx(1.0)


def test_linear_target_rank_matches_full_lexicographic_sort():
    rng = np.random.default_rng(8)
    items = np.arange(40)
    tie = rng.permutation(40)
    for _ in range(50):
        # Rounded scores deliberately create ties.
        scores = np.round(rng.normal(size=40), 1)
        target = int(rng.choice(items))
        expected_order = np.lexsort((tie, -scores))
        expected = int(np.flatnonzero(expected_order == target)[0]) + 1
        assert target_rank(scores, items, target, tie) == expected


def test_global_tie_break_is_seeded_and_user_independent():
    priorities = global_item_priorities(10, seed=4)
    candidates = np.array([2, 4, 7])
    assert np.array_equal(
        tie_break_for_candidates(candidates, priorities), priorities[candidates]
    )
    assert np.unique(priorities).size == 10


def test_profile_utility_full_and_empty_are_defined():
    model, game = _game()
    empty = profile_utility(model, game, frozenset())
    full = profile_utility(model, game, frozenset({0, 1}))
    assert 0.0 <= empty <= 1.0
    assert 0.0 <= full <= 1.0


@pytest.mark.parametrize("utility_name", ["ndcg", "target_margin"])
def test_batched_model_shapley_matches_generic_prefix_walk(utility_name):
    model, game = _game()
    utility = (
        (lambda coalition: profile_utility(model, game, coalition))
        if utility_name == "ndcg"
        else (lambda coalition: target_margin_utility(model, game, coalition))
    )
    generic, generic_error = mc_shapley(
        utility, n_players=game.players.size, permutations=20, seed=7
    )
    batched, batched_error = model_mc_shapley(
        model,
        game,
        permutations=20,
        seed=7,
        utility=utility_name,
        permutation_batch_size=3,
    )
    assert np.allclose(batched, generic)
    assert generic_error < 1e-12
    assert batched_error < 1e-12


def test_mc_shapley_prefix_walk_satisfies_efficiency():
    def utility(coalition):
        players = set(coalition)
        return float(len(players) + (2 if {0, 1}.issubset(players) else 0))

    values, error = mc_shapley(utility, n_players=3, permutations=20, seed=7)
    assert values[0] > 0 and values[1] > 0
    assert error < 1e-12


def test_joint_selection_and_aggregation():
    values = np.array([-0.9, 0.2, 0.7])
    action = select_joint_action(values, budget=2)
    assert action == (0, 2)
    assert joint_attribution_score(values, action) == pytest.approx(1.6)
    assert joint_attribution_score(values, action, signed=True) == pytest.approx(-0.2)


def test_benefit_selection_uses_sign_and_can_abstain():
    assert select_downweight_action(np.array([0.5, -0.2, -0.8]), 2) == (2, 1)
    assert select_downweight_action(np.array([0.5, 0.2, 0.8]), 2) == ()
    assert select_downweight_action(
        np.array([0.5, 0.2, 0.8]), 2, allow_abstain=False
    ) == (1,)


def test_joint_selection_rejects_invalid_budget():
    with pytest.raises(ValueError):
        select_joint_action(np.ones(2), budget=3)


def test_fixed_evaluation_sets_include_target_and_exclude_all_seen_items():
    seen = {0: np.array([0, 1, 2]), 1: np.array([2, 3, 4])}
    tests = {0: 4, 1: 5}
    sets, coverage = fixed_evaluation_sets(seen, tests, n_items=10, size=5, seed=4)
    assert coverage == 1.0
    assert all(tests[user] in sets[user] for user in tests)
    assert all(len(sets[user]) == 5 for user in tests)
    assert not (set(sets[0]) - {tests[0]}) & set(seen[0])
    assert not (set(sets[1]) - {tests[1]}) & set(seen[1])


def test_full_catalogue_means_full_unseen_catalogue():
    seen = {0: np.array([0, 1, 2])}
    tests = {0: 4}
    sets, coverage = full_unseen_evaluation_sets(seen, tests, n_items=7)
    assert coverage == 1.0
    assert set(sets[0]) == {3, 4, 5, 6}


def test_target_margin_is_continuous_and_distinct_from_ndcg():
    model, game = _game()
    margin = target_margin_utility(model, game, frozenset({0}))
    ndcg = profile_utility(model, game, frozenset({0}))
    assert np.isfinite(margin)
    assert 0.0 < margin < 1.0
    assert margin != ndcg


def test_exact_oracle_includes_no_action_and_smaller_actions():
    # Every intervention is harmful, so the corrected oracle abstains.
    action, rhos, effect = exhaustive_oracle(
        lambda coalition: float(len(coalition)),
        n_players=3,
        budget=2,
        apply_action=lambda action, rhos: -float(len(action)),
        allow_abstain=True,
    )
    assert action == ()
    assert rhos == ()
    assert effect == 0.0


def test_evaluation_oracle_and_joint_effect_accept_empty_action():
    model, game = _game()
    assert joint_effect(
        model, game, (), rho=0.5, utility="target_margin"
    ) == pytest.approx(0.0)
    action, rhos, effect = exhaustive_best_joint(
        model,
        game,
        budget=2,
        rho_grid=(0.5,),
        utility="target_margin",
        allow_abstain=True,
    )
    assert len(action) <= 2
    assert len(action) == len(rhos)
    assert effect >= 0.0


def test_multi_utility_oracle_matches_separate_exact_oracles():
    model, game = _game()
    combined = exhaustive_best_joint_multi(
        model,
        game,
        budget=2,
        rho_grid=(0.5,),
        utilities=("target_margin", "ndcg"),
    )
    for utility in ("target_margin", "ndcg"):
        separate = exhaustive_best_joint(
            model, game, budget=2, rho_grid=(0.5,), utility=utility
        )
        assert combined[utility][0] == separate[0]
        assert combined[utility][1] == separate[1]
        assert combined[utility][2] == pytest.approx(separate[2])
