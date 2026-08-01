"""Tests for the misalignment decomposition.

The load-bearing claims are Corollary 1 (efficiency, hence additivity and
order-independence) and the assertion in Section 3.7.2 that H1 and H2 together
restore exact local linearity. Both are tested here against synthetic models
whose structure is known by construction.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from actionshap.decomposition import (
    ALL_SWITCHES,
    PurifiedGA2M,
    decompose_misalignment,
    shapley_over_switches,
)


def _fit(X, y, interactions=3, seed=0):
    from interpret.glassbox import ExplainableBoostingRegressor

    ebm = ExplainableBoostingRegressor(
        interactions=interactions,
        max_bins=32,
        max_interaction_bins=32,   # shared grid; required by PurifiedGA2M
        outer_bags=2,
        random_state=seed,
        n_jobs=1,
    )
    ebm.fit(X, y)
    return ebm


@pytest.fixture(scope="module")
def curved_interacting():
    """A model with curvature AND interaction, so both switches have work to do."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(1500, 4))
    y = (
        2.0 * X[:, 0]                    # linear
        + 1.5 * X[:, 1] ** 2             # curvature   -> H1
        + 2.0 * X[:, 0] * X[:, 2]        # interaction -> H2
        + 0.5 * X[:, 3]
        + 0.05 * rng.normal(size=1500)
    )
    return X, y, PurifiedGA2M(_fit(X, y), X)


# --------------------------------------------------------------------------
# The Shapley game itself
# --------------------------------------------------------------------------


def test_shapley_is_efficient_on_random_games():
    """Efficiency is unconditional: it holds for ANY characteristic function."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = {
            frozenset(c): float(rng.normal())
            for r in range(4)
            for c in itertools.combinations(ALL_SWITCHES, r)
        }
        u[frozenset()] = 0.0
        psi = shapley_over_switches(u)
        assert sum(psi.values()) == pytest.approx(u[frozenset(ALL_SWITCHES)])


def test_shapley_gives_a_null_switch_nothing():
    """A switch that never changes alignment must receive a zero share."""
    u = {frozenset(): 0.0}
    for r in range(1, 4):
        for c in itertools.combinations(ALL_SWITCHES, r):
            # Value depends only on H1 and H2; H3 is a null player.
            u[frozenset(c)] = 1.0 * ("H1" in c) + 2.0 * ("H2" in c)
    psi = shapley_over_switches(u)
    assert psi["H3"] == pytest.approx(0.0)
    assert psi["H1"] == pytest.approx(1.0)
    assert psi["H2"] == pytest.approx(2.0)


def test_shapley_is_symmetric():
    """Interchangeable switches receive equal shares."""
    u = {frozenset(): 0.0}
    for r in range(1, 4):
        for c in itertools.combinations(ALL_SWITCHES, r):
            u[frozenset(c)] = float(len(c))  # depends only on cardinality
    psi = shapley_over_switches(u)
    assert psi["H1"] == pytest.approx(psi["H2"]) == pytest.approx(psi["H3"])


def test_missing_coalition_is_rejected():
    u = {frozenset(): 0.0, frozenset({"H1"}): 1.0}
    with pytest.raises(ValueError, match="missing coalitions"):
        shapley_over_switches(u)


# --------------------------------------------------------------------------
# The purified surrogate and its switches
# --------------------------------------------------------------------------


def test_surrogate_reproduces_the_ebm_it_purifies(curved_interacting):
    """Purification redistributes mass between terms; it must not change f."""
    X, _, g = curved_interacting
    assert np.allclose(g.predict(X), g.ebm.predict(X), atol=1e-8)


def test_pairs_are_pure_after_purification(curved_interacting):
    """Zero weighted marginals on both axes is what makes the split unique."""
    _, _, g = curved_interacting
    weights = {
        tuple(t): np.asarray(w, float)
        for t, w in zip(g.ebm.term_features_, g.ebm.bin_weights_)
        if len(t) == 2
    }
    assert g.pairs, "fixture should have produced at least one pair term"
    for key, tensor in g.pairs.items():
        w = weights[key]
        assert np.allclose((tensor * w).sum(axis=0), 0, atol=1e-6)
        assert np.allclose((tensor * w).sum(axis=1), 0, atol=1e-6)


def test_h1_and_h2_together_give_an_exactly_linear_model(curved_interacting):
    """Section 3.7.2: the grand coalition must satisfy Proposition 1's
    hypothesis, not merely approximate it."""
    X, _, g = curved_interacting
    f = g.output_fn(linearize_mains=True, drop_pairs=True)

    # Superposition holds exactly for an affine function.
    rng = np.random.default_rng(5)
    a, b = rng.uniform(-2, 2, size=(2, 1, X.shape[1]))
    lam = 0.37
    lhs = f(lam * a + (1 - lam) * b)
    rhs = lam * f(a) + (1 - lam) * f(b)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_h2_alone_keeps_curvature(curved_interacting):
    """Dropping pairs must NOT linearize: H1 and H2 target different failures."""
    X, _, g = curved_interacting
    f = g.output_fn(linearize_mains=False, drop_pairs=True)
    rng = np.random.default_rng(6)
    a, b = rng.uniform(-2, 2, size=(2, 1, X.shape[1]))
    lam = 0.37
    # The quadratic term in x1 survives, so superposition should fail.
    assert f(lam * a + (1 - lam) * b) != pytest.approx(lam * f(a) + (1 - lam) * f(b), abs=1e-6)


def test_h1_alone_keeps_interactions(curved_interacting):
    X, _, g = curved_interacting
    with_pairs = g.predict(X, linearize_mains=True, drop_pairs=False)
    without = g.predict(X, linearize_mains=True, drop_pairs=True)
    assert not np.allclose(with_pairs, without)


def test_refuses_a_model_with_split_binning():
    """Guard against the silent-corruption case the docstring warns about."""
    from interpret.glassbox import ExplainableBoostingRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = X[:, 0] + X[:, 1] * X[:, 2]
    ebm = ExplainableBoostingRegressor(
        interactions=2, max_bins=32, max_interaction_bins=8,
        outer_bags=2, random_state=0, n_jobs=1,
    )
    ebm.fit(X, y)
    with pytest.raises(ValueError, match="binning levels"):
        PurifiedGA2M(ebm, X)


def test_fidelity_is_high_on_a_well_specified_surrogate(curved_interacting):
    X, y, g = curved_interacting
    assert g.fidelity(X, y) > 0.95


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_decomposition_satisfies_corollary_1(curved_interacting):
    """The whole point: shares sum to the measured gap, exactly."""
    X, _, g = curved_interacting

    def attribute(f, Xa):
        # A cheap deterministic stand-in for a real attributor. The identity of
        # the attribution method does not affect Corollary 1, which is exactly
        # what makes efficiency a useful invariant to test against.
        #
        # Reduced per instance and THEN averaged in magnitude. Taking the
        # difference of means instead would return identically zero on a
        # linear model, since replacing a column by its mean leaves the mean
        # of an affine function unchanged -- which is a degeneracy of this
        # stand-in, not of mean-imputation attribution in general.
        base = f(Xa)
        out = np.empty(Xa.shape[1])
        for j in range(Xa.shape[1]):
            Xp = Xa.copy()
            Xp[:, j] = Xa[:, j].mean()
            out[j] = np.abs(base - f(Xp)).mean()
        return out

    m = np.array([1.0, 1.0, 1.0, 0.0])         # one immutable factor
    budgets = X.std(axis=0, ddof=1)

    shares = decompose_misalignment(
        g, X, attribute, m, budgets, n_matched_draws=25, seed=0
    )

    shares.check_efficiency()                   # raises on violation
    assert sum(shares.psi.values()) == pytest.approx(shares.closed_gap, abs=1e-9)
    assert set(shares.psi) == set(ALL_SWITCHES)
    assert len(shares.coalition_aia) == 8
    assert shares.dominant in ALL_SWITCHES


def test_decomposition_rejects_too_few_modifiable_factors(curved_interacting):
    X, _, g = curved_interacting
    with pytest.raises(ValueError, match="at least 3"):
        decompose_misalignment(
            g, X, lambda f, Xa: np.ones(Xa.shape[1]),
            np.array([1.0, 0.0, 0.0, 0.0]), X.std(axis=0),
        )


def test_decomposition_rejects_shape_mismatch(curved_interacting):
    X, _, g = curved_interacting
    with pytest.raises(ValueError, match="shape mismatch"):
        decompose_misalignment(
            g, X, lambda f, Xa: np.ones(Xa.shape[1]),
            np.ones(3), np.ones(4),
        )
