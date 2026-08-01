"""Tests for the actionability metrics.

The properties asserted here are the ones the manuscript's claims rest on.
Where a test encodes a claim from the paper, the definition or proposition is
named, so that changing the behaviour forces a look at the text.
"""

from __future__ import annotations

import numpy as np
import pytest

from actionshap.metrics import (
    actionability_score,
    actionability_score_heldout,
    alignment,
    intervention_regret,
    stability,
    topk_intervention_precision,
)
from actionshap.rerank import eta_sweep, rerank


# --------------------------------------------------------------------------
# Definition 3: stability
# --------------------------------------------------------------------------


def test_identical_runs_are_perfectly_stable():
    runs = np.tile([1.0, -2.0, 0.5], (5, 1))
    assert np.allclose(stability(runs), 1.0)


def test_dispersion_exceeding_mean_floors_at_zero():
    # Mean ~0 with large spread: the factor carries no reliable signal.
    runs = np.array([[10.0], [-10.0], [10.0], [-10.0]])
    assert stability(runs)[0] == 0.0


def test_stability_is_bounded():
    rng = np.random.default_rng(0)
    s = stability(rng.normal(size=(8, 20)))
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_single_run_is_rejected():
    with pytest.raises(ValueError, match="R >= 2"):
        stability(np.array([[1.0, 2.0]]))


# --------------------------------------------------------------------------
# Definition 4 and Remark 1: the score and its held-out control
# --------------------------------------------------------------------------


def test_any_zero_factor_zeroes_the_score():
    """Each condition is necessary, so the product must vanish if any does."""
    m = np.array([0.0, 1.0, 1.0])
    d = np.array([5.0, 0.0, 5.0])
    s = np.array([1.0, 1.0, 0.0])
    assert np.allclose(actionability_score(m, d, s), 0.0)


def test_control_is_independent_of_modifiability():
    """Remark 1: AS^-m carries no construction-induced ordering."""
    d, s = np.array([2.0, -4.0]), np.array([0.9, 0.8])
    assert np.allclose(
        actionability_score_heldout(d, s), np.abs(d) * s
    )
    # An immutable factor still gets a non-zero control score.
    assert actionability_score(np.array([0.0]), np.array([4.0]), np.array([0.8]))[0] == 0.0
    assert actionability_score_heldout(np.array([4.0]), np.array([0.8]))[0] > 0.0


def test_score_uses_effect_magnitude_not_sign():
    up = actionability_score(np.array([1.0]), np.array([3.0]), np.array([1.0]))
    down = actionability_score(np.array([1.0]), np.array([-3.0]), np.array([1.0]))
    assert up == down


# --------------------------------------------------------------------------
# Definition 5 and Proposition 1: alignment
# --------------------------------------------------------------------------


def test_proposition_1_alignment_is_unity_under_local_linearity():
    """Under f(x+tau) = f(x) + sum_j w_j tau_j with common tau and m_j = 1,
    phi_j = w_j tau = Delta_j, so the orderings coincide exactly."""
    rng = np.random.default_rng(7)
    w = rng.normal(size=12)
    tau = 0.3
    phi = w * tau
    delta = w * tau
    assert alignment(phi, delta).spearman == pytest.approx(1.0)


def test_reversed_ordering_is_perfect_anticorrelation():
    phi = np.array([1.0, 2.0, 3.0, 4.0])
    delta = np.array([4.0, 3.0, 2.0, 1.0])
    assert alignment(phi, delta).spearman == pytest.approx(-1.0)


def test_restriction_changes_alignment_when_immutables_are_misranked():
    """The gap between unrestricted and restricted AIA is the H3 signature."""
    # Immutable factors carry the largest attributions but useless effects.
    phi = np.array([9.0, 8.0, 3.0, 2.0, 1.0])
    delta = np.array([0.1, 0.2, 3.0, 2.0, 1.0])
    m = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    unrestricted = alignment(phi, delta).spearman
    restricted = alignment(phi, delta, m, restrict_to_modifiable=True).spearman

    assert restricted == pytest.approx(1.0)
    assert restricted > unrestricted


def test_constant_vector_is_an_explicit_error_not_nan():
    with pytest.raises(ValueError, match="constant"):
        alignment(np.ones(5), np.arange(5.0))


def test_too_few_modifiables_to_correlate():
    with pytest.raises(ValueError, match="not meaningful"):
        alignment(
            np.arange(5.0), np.arange(5.0),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]), restrict_to_modifiable=True,
        )


# --------------------------------------------------------------------------
# Definition 6: top-k intervention precision
# --------------------------------------------------------------------------


def test_precision_requires_all_three_conditions():
    phi = np.array([4.0, 3.0, 2.0, 1.0])
    delta = np.array([
        5.0,    # modifiable, expected sign, large -> counts
        -5.0,   # modifiable, WRONG sign           -> excluded
        5.0,    # immutable                        -> excluded
        5.0,    # below delta threshold            -> excluded (see below)
    ])
    m = np.array([1.0, 1.0, 0.0, 1.0])
    direction = np.array([1.0, 1.0, 1.0, 1.0])

    assert topk_intervention_precision(
        phi, delta, m, k=3, delta=1.0, direction=direction
    ) == pytest.approx(1 / 3)

    # Raising the threshold above every effect drives precision to zero.
    assert topk_intervention_precision(
        phi, delta, m, k=3, delta=99.0, direction=direction
    ) == 0.0


def test_precision_without_a_direction_drops_the_sign_condition():
    """Omitting `direction` must relax the metric, never silently reintroduce
    a sign test against sign(phi) -- which measures a level, not a derivative.
    """
    phi = np.array([4.0, 3.0, 2.0, 1.0])
    delta = np.array([5.0, -5.0, 5.0, 5.0])
    m = np.array([1.0, 1.0, 0.0, 1.0])

    # The wrong-signed factor at index 1 now counts: modifiable and large.
    assert topk_intervention_precision(
        phi, delta, m, k=3, delta=1.0
    ) == pytest.approx(2 / 3)


def test_precision_tolerates_factors_with_no_established_direction():
    """A zero direction means undetermined, which cannot contradict an effect."""
    phi = np.array([3.0, 2.0, 1.0])
    delta = np.array([5.0, -5.0, 5.0])
    m = np.ones(3)

    assert topk_intervention_precision(
        phi, delta, m, k=2, delta=1.0, direction=np.array([1.0, 0.0, 1.0])
    ) == 1.0


def test_precision_rejects_a_misshaped_direction():
    with pytest.raises(ValueError, match="direction has shape"):
        topk_intervention_precision(
            np.ones(3), np.ones(3), np.ones(3), k=2, delta=0.0,
            direction=np.ones(2),
        )


def test_precision_penalizes_spending_topk_on_immutables():
    """The top-k set is over ALL factors, so wasted slots cost precision."""
    phi = np.array([9.0, 8.0, 1.0])
    delta = np.array([1.0, 1.0, 1.0])
    m = np.array([0.0, 0.0, 1.0])
    assert topk_intervention_precision(phi, delta, m, k=2, delta=0.0) == 0.0


def test_precision_k_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        topk_intervention_precision(np.ones(3), np.ones(3), np.ones(3), k=4, delta=0.0)


# --------------------------------------------------------------------------
# Definition 7: intervention regret
# --------------------------------------------------------------------------


def test_regret_is_zero_when_attribution_picks_the_best_feasible_action():
    phi = np.array([3.0, 2.0, 1.0])
    delta = np.array([9.0, 5.0, 1.0])
    m = np.ones(3)
    r = intervention_regret(phi, delta, m)
    assert r.regret == pytest.approx(0.0)
    assert r.is_optimal


def test_regret_ignores_immutable_factors_entirely():
    """An immutable factor with a huge effect is not the optimum: it is not
    an available action, so it cannot define forgone outcome."""
    phi = np.array([9.0, 1.0, 2.0])
    delta = np.array([100.0, 5.0, 3.0])
    m = np.array([0.0, 1.0, 1.0])

    r = intervention_regret(phi, delta, m)
    assert r.optimal_index == 1          # not index 0, despite Delta = 100
    assert r.chosen_index == 2           # largest |phi| among modifiables
    assert r.regret == pytest.approx(2.0)
    assert r.normalized == pytest.approx(2.0 / 5.0)


def test_regret_is_never_negative():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = rng.integers(3, 12)
        r = intervention_regret(rng.normal(size=n), rng.normal(size=n), np.ones(n))
        assert r.regret >= -1e-12


def test_candidate_mask_bounds_the_comparison():
    phi = np.array([1.0, 2.0, 3.0])
    delta = np.array([50.0, 5.0, 1.0])
    m = np.ones(3)

    # Excluding the true optimum from the candidate set changes the target.
    r = intervention_regret(phi, delta, m, candidate_mask=np.array([False, True, True]))
    assert r.optimal_index == 1
    assert r.optimal_effect == pytest.approx(5.0)


def test_no_eligible_factor_is_an_error():
    with pytest.raises(ValueError, match="no eligible"):
        intervention_regret(np.ones(3), np.ones(3), np.zeros(3))


# --------------------------------------------------------------------------
# Definition 8: reranking
# --------------------------------------------------------------------------


def test_rerank_endpoints_recover_each_ordering():
    phi = np.array([1.0, 5.0, 3.0])
    as_ = np.array([9.0, 1.0, 2.0])

    assert np.argmax(rerank(phi, as_, 0.0)) == np.argmax(np.abs(phi))
    assert np.argmax(rerank(phi, as_, 1.0)) == np.argmax(as_)


def test_rerank_rejects_eta_outside_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        rerank(np.ones(3), np.ones(3), 1.5)


def test_rerank_survives_an_all_immutable_dataset():
    """Every AS_j is zero when nothing is modifiable; this must not divide by zero."""
    out = rerank(np.array([1.0, 2.0]), np.array([0.0, 0.0]), 0.5)
    assert np.all(np.isfinite(out))


def test_eta_sweep_detects_the_switch_and_cuts_regret():
    # Attribution favours factor 0, but factor 1 is the better feasible action.
    phi = np.array([9.0, 1.0])
    delta = np.array([1.0, 8.0])
    m = np.array([1.0, 1.0])
    as_ = np.abs(delta) * m

    sweep = eta_sweep(phi, as_, delta, m, etas=(0.0, 1.0))
    assert not sweep[0].switched
    assert sweep[1].switched
    assert sweep[1].normalized_regret < sweep[0].normalized_regret


def test_eta_sweep_requires_a_zero_baseline():
    with pytest.raises(ValueError, match="must start at eta=0"):
        eta_sweep(np.ones(3), np.ones(3), np.ones(3), np.ones(3), etas=(0.5, 1.0))


# --------------------------------------------------------------------------
# Shape validation across the module
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, args",
    [
        (actionability_score, (np.ones(3), np.ones(4), np.ones(3))),
        (alignment, (np.ones(3), np.ones(5))),
        (intervention_regret, (np.ones(3), np.ones(3), np.ones(2))),
    ],
)
def test_mismatched_shapes_are_rejected(fn, args):
    with pytest.raises(ValueError, match="shape mismatch|mismatch"):
        fn(*args)
