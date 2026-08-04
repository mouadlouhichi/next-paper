from __future__ import annotations

import numpy as np

from cure_rec.game import FULL_MASK, exact_interactions, exact_shapley


def test_exact_shapley_efficiency_for_additive_game():
    # v(S) is the sum of active player coefficients.
    coefficients = [0.1, -0.2, 0.3, 0.0, 0.4, 0.05]
    improvements = {
        mask: sum(coefficient for index, coefficient in enumerate(coefficients) if mask & (1 << index))
        for mask in range(1 << len(coefficients))
    }
    values = exact_shapley(improvements)
    assert np.isclose(sum(values.values()), improvements[FULL_MASK])
    assert np.allclose(list(values.values()), coefficients)


def test_exact_interactions_are_zero_for_additive_game():
    coefficients = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    improvements = {
        mask: sum(coefficient for index, coefficient in enumerate(coefficients) if mask & (1 << index))
        for mask in range(1 << len(coefficients))
    }
    interactions = exact_interactions(improvements)
    assert all(np.isclose(value, 0.0) for value in interactions.values())


def test_pair_synergy_has_positive_interaction():
    # Players 0 and 1 receive an extra unit only together.
    improvements = {}
    for mask in range(64):
        improvements[mask] = 1.0 if (mask & 0b11) == 0b11 else 0.0
    interactions = exact_interactions(improvements)
    assert interactions[("repeat_cap", "explore_slot")] > 0.0
