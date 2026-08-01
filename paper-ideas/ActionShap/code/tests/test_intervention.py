"""Tests for feasible interventions (Definition 2).

The tests that matter most here are the ones about *sign*. A one-sided
intervention convention silently determines the sign of every effect when the
output sits near a bound, and every downstream metric that reads a sign then
reports an artifact of the convention. `test_saturated_output_*` pins that
failure mode so it cannot come back unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from actionshap.intervention import (
    EffectProfile, InterventionBudget, intervention_effects,
    intervention_profile, sweep_budgets,
)


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(200, 3))


# --------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------


def test_budget_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        InterventionBudget(np.array([1.0, -1.0]), scale="absolute")


def test_budget_from_std_scales_with_the_multiplier(X):
    one = InterventionBudget.from_std(X, 1.0)
    two = InterventionBudget.from_std(X, 2.0)
    assert np.allclose(two.values, 2 * one.values)
    assert np.allclose(one.values, X.std(axis=0, ddof=1))


def test_budget_length_must_match_the_matrix(X):
    with pytest.raises(ValueError, match="covers 2 factors"):
        intervention_effects(
            lambda Z: Z.sum(axis=1), X,
            InterventionBudget(np.ones(2), scale="absolute"),
        )


# --------------------------------------------------------------------------
# Direction
# --------------------------------------------------------------------------


def test_monotone_output_gives_opposite_signs_in_opposite_directions(X):
    """A budget of tau on a linear output must move it by exactly the coefficient
    times tau, upward, and the negative of that downward."""
    f = lambda Z: 2.0 * Z[:, 0] + Z[:, 1]
    budget = InterventionBudget(np.array([1.0, 1.0, 1.0]), scale="absolute")

    prof = intervention_profile(f, X, budget, clip_to_observed=False)

    assert prof.increase == pytest.approx([2.0, 1.0, 0.0], abs=1e-9)
    assert prof.decrease == pytest.approx([-2.0, -1.0, 0.0], abs=1e-9)


def test_best_takes_the_larger_magnitude_and_keeps_its_sign():
    prof = EffectProfile(
        increase=np.array([0.1, -0.5, 0.3]),
        decrease=np.array([-0.4, 0.2, -0.3]),
        baseline=0.5,
    )
    # index 2 ties in magnitude; the increase branch wins by convention.
    assert prof.best == pytest.approx([-0.4, -0.5, 0.3])
    assert list(prof.best_direction) == [-1, 1, 1]


def test_one_sided_modes_agree_with_the_profile(X):
    f = lambda Z: np.tanh(Z[:, 0]) + 0.5 * Z[:, 2]
    budget = InterventionBudget.from_std(X, 1.0)

    prof = intervention_profile(f, X, budget, clip_to_observed=False)
    up = intervention_effects(f, X, budget, clip_to_observed=False,
                              direction="increase")
    down = intervention_effects(f, X, budget, clip_to_observed=False,
                                direction="decrease")

    assert up == pytest.approx(prof.increase)
    assert down == pytest.approx(prof.decrease)
    assert intervention_effects(
        f, X, budget, clip_to_observed=False
    ) == pytest.approx(prof.best)


def test_unknown_direction_is_rejected(X):
    with pytest.raises(ValueError, match="unknown direction"):
        intervention_effects(
            lambda Z: Z[:, 0], X, InterventionBudget.from_std(X),
            direction="sideways",
        )


# --------------------------------------------------------------------------
# The saturation trap
# --------------------------------------------------------------------------


def _saturated(Z):
    """A near-certain probability, as an accurate surrogate produces.

    Any perturbation of a point at the top of a bounded output can only move
    it down, whichever way the perturbation goes.
    """
    return 1.0 / (1.0 + np.exp(-(8.0 - Z[:, 0] ** 2 - Z[:, 1] ** 2)))


def test_saturated_output_makes_every_one_sided_effect_negative(X):
    """The failure mode: sign(Delta) carries no information at a ceiling."""
    budget = InterventionBudget.from_std(X, 1.0)
    up = intervention_effects(_saturated, X, budget, clip_to_observed=False,
                              direction="increase")

    assert np.all(up[:2] < 0)


def test_saturated_output_is_not_rescued_by_measuring_both_directions(X):
    """Both directions still point down, so bidirectionality alone is not the
    fix -- the objective itself has to leave the bound. This is why
    `StaticPipeline.target_fn` exists alongside `membership_fn`.
    """
    budget = InterventionBudget.from_std(X, 1.0)
    prof = intervention_profile(_saturated, X, budget, clip_to_observed=False)

    assert np.all(prof.increase[:2] < 0)
    assert np.all(prof.decrease[:2] < 0)
    assert np.all(prof.best[:2] < 0)


def test_unsaturated_output_recovers_informative_signs(X):
    """The same functional far from the bound moves both ways."""
    f = lambda Z: 1.0 / (1.0 + np.exp(-(Z[:, 0] - Z[:, 1])))
    budget = InterventionBudget.from_std(X, 1.0)
    prof = intervention_profile(f, X, budget, clip_to_observed=False)

    assert prof.increase[0] > 0 and prof.increase[1] < 0
    assert prof.decrease[0] < 0 and prof.decrease[1] > 0


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


def test_zero_budget_means_no_effect_in_either_direction(X):
    budget = InterventionBudget(np.array([1.0, 0.0, 1.0]), scale="absolute")
    prof = intervention_profile(lambda Z: Z.sum(axis=1), X, budget,
                                clip_to_observed=False)
    assert prof.increase[1] == 0.0
    assert prof.decrease[1] == 0.0


def test_clipping_bounds_the_shift_to_the_observed_range(X):
    """An enormous budget cannot push a factor past anything the model saw, so
    the clipped effect must be smaller than the unclipped one."""
    f = lambda Z: Z[:, 0]
    budget = InterventionBudget(np.array([100.0, 0.0, 0.0]), scale="absolute")

    clipped = intervention_effects(f, X, budget, direction="increase",
                                   clip_to_observed=True)
    free = intervention_effects(f, X, budget, direction="increase",
                                clip_to_observed=False)

    assert clipped[0] < free[0]
    assert clipped[0] == pytest.approx(X[:, 0].max() - X[:, 0].mean())


def test_batching_does_not_change_the_answer(X):
    f = lambda Z: np.tanh(Z[:, 0])
    budget = InterventionBudget.from_std(X, 1.0)
    whole = intervention_effects(f, X, budget, clip_to_observed=False)
    batched = intervention_effects(f, X, budget, clip_to_observed=False,
                                   batch_size=17)
    assert whole == pytest.approx(batched)


def test_output_fn_returning_the_wrong_shape_is_caught(X):
    with pytest.raises(ValueError, match="expected"):
        intervention_effects(lambda Z: Z, X, InterventionBudget.from_std(X))


def test_sweep_covers_every_multiplier(X):
    out = sweep_budgets(lambda Z: Z[:, 0], X, (0.5, 1.0, 2.0),
                        clip_to_observed=False)
    assert set(out) == {0.5, 1.0, 2.0}
    # Linear output: the effect must scale with the budget.
    assert out[2.0][0] == pytest.approx(4 * out[0.5][0])
