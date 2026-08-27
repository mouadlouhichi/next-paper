"""Tests for the review-9 construct-validity fixes.

Covers the fixed-denominator (pure-suppression) ItemKNN adapter used by the
relative-reweighting ablation, and the collision-resistant tuple-seed
derivation for the random control.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from actionshap.baselines import random_attribution
from actionshap.models.itemknn import FixedDenominatorItemKNN, ItemKNNModel


def _model() -> ItemKNNModel:
    similarities = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.8, 0.1, 0.0],
                [0.8, 0.0, 0.5, 0.2],
                [0.1, 0.5, 0.0, 0.9],
                [0.0, 0.2, 0.9, 0.0],
            ]
        )
    )
    return ItemKNNModel(similarities)


def test_fixed_denominator_matches_normalized_at_full_profile():
    base = _model()
    fixed = FixedDenominatorItemKNN(base)
    history = np.array([0, 1, 2])
    candidates = np.array([1, 2, 3])
    np.testing.assert_allclose(
        fixed.score(history, candidates), base.score(history, candidates)
    )
    np.testing.assert_allclose(
        fixed.score_downweighted(history, candidates, np.ones(3)),
        base.score(history, candidates),
    )


def test_fixed_denominator_suppresses_without_reallocation():
    base = _model()
    fixed = FixedDenominatorItemKNN(base)
    history = np.array([0, 1, 2])
    candidates = np.array([1, 2, 3])
    full = base.score(history, candidates)
    rho = 0.5
    weights = np.array([rho, 1.0, 1.0])
    # Pure suppression: only player 0's contribution changes, and it is
    # scaled by rho against the *full-window* denominator n_u = 3.
    weights_zero = np.array([0.0, 1.0, 1.0])
    suppressed = fixed.score(history, candidates, weights_zero)
    expected = full - base.similarities[candidates][:, [0]].toarray().reshape(-1) / 3.0
    np.testing.assert_allclose(suppressed, expected)
    bounded = fixed.score(history, candidates, weights)
    np.testing.assert_allclose(
        bounded, suppressed + rho * base.similarities[candidates][:, [0]].toarray().reshape(-1) / 3.0
    )
    # Contrast with the normalized scorer, where deleting player 0 divides by
    # the remaining mass and therefore raises the other normalized shares.
    normalized_deleted = base.score(history, candidates, weights_zero)
    assert not np.allclose(normalized_deleted, suppressed)


def test_fixed_denominator_uniform_scaling_is_not_invariant_for_normalized():
    """Normalized scores are invariant to uniform positive rescaling; the
    fixed-denominator variant intentionally is not (it is not scale-free)."""
    base = _model()
    fixed = FixedDenominatorItemKNN(base)
    history = np.array([0, 1, 2])
    candidates = np.array([1, 2, 3])
    np.testing.assert_allclose(
        base.score(history, candidates, np.full(3, 2.0)),
        base.score(history, candidates),
    )
    assert not np.allclose(
        fixed.score(history, candidates, np.full(3, 2.0)),
        fixed.score(history, candidates),
    )


def test_fixed_denominator_masked_and_batch_paths():
    base = _model()
    fixed = FixedDenominatorItemKNN(base)
    history = np.array([0, 1, 2])
    candidates = np.array([1, 2, 3])
    mask = np.array([False, True, True])
    # Masking divides by the full window size 3, not by the 2 retained.
    expected = (
        base.similarities[candidates][:, history[mask]].toarray().sum(axis=1) / 3.0
    )
    np.testing.assert_allclose(fixed.score_masked(history, candidates, mask), expected)
    weight_matrix = np.array([[1.0, 1.0, 1.0], [0.5, 1.0, 1.0]])
    batch = fixed.score_downweighted_batch(history, candidates, weight_matrix)
    np.testing.assert_allclose(
        batch[0], fixed.score(history, candidates, np.ones(3))
    )
    np.testing.assert_allclose(
        batch[1], fixed.score(history, candidates, np.array([0.5, 1.0, 1.0]))
    )


def test_random_attribution_tuple_seed_is_collision_resistant():
    """Integer-offset seeds collide for adjacent (seed, user) pairs; tuple
    entropy passed to SeedSequence does not."""
    a = random_attribution(8, seed=(42, 1, 1_000_000))
    b = random_attribution(8, seed=(43, 0, 1_000_000))
    # The old additive scheme mapped both pairs to 1_000_001 + 42.
    assert not np.allclose(a, b)
    # Determinism under identical tuple seeds.
    c = random_attribution(8, seed=(42, 1, 1_000_000))
    np.testing.assert_allclose(a, c)
    # Integer seeds remain supported for backward compatibility.
    np.testing.assert_allclose(random_attribution(4, seed=7), random_attribution(4, seed=7))
