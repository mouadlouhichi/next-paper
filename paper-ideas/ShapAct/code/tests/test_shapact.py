"""ShapAct test suite (ShapAct Implementation Spec A.9).

Unit tests (fast, synthetic) plus integration tests against the real
pipeline results, mirroring the SignalShap test conventions.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from shapact.config import SOURCES
from shapact.fusion import znorm_matrix
from shapact.game import (efficiency_check, per_user_consistency_check,
                          shapley_from_values)


# --------------------------------------------------------------------------
# 6. Degenerate normalization
# --------------------------------------------------------------------------

def test_constant_row_znorm_is_zero():
    X = np.tile([3.0], (4, 1))
    z, deg = znorm_matrix(X)
    assert np.allclose(z, 0.0)
    assert deg.all()


def test_znorm_matches_manual():
    X = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 5.0]])
    z, deg = znorm_matrix(X)
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    assert np.allclose(z, (X - m) / s)
    assert not deg.any()


# --------------------------------------------------------------------------
# 1/2. Efficiency and per-user consistency on a synthetic game
# --------------------------------------------------------------------------

def _synthetic_values(n_users=100, seed=0):
    rng = np.random.default_rng(seed)
    per_user = {}
    v = {}
    for r in range(len(SOURCES) + 1):
        for comb in itertools.combinations(SOURCES, r):
            C = tuple(sorted(comb))
            base = rng.normal(size=n_users)
            val = base * (1.0 + 0.1 * len(C)) + 0.05 * len(C)
            per_user[C] = val
            v[C] = float(val.mean())
    v[()] = 0.0
    per_user[()] = np.zeros(n_users)
    return v, per_user


def test_efficiency_identity_synthetic():
    v, pu = _synthetic_values()
    phi, phi_u = shapley_from_values(v, pu, 100)
    assert efficiency_check(v, phi) < 1e-9


def test_per_user_consistency_synthetic():
    v, pu = _synthetic_values()
    phi, phi_u = shapley_from_values(v, pu, 100)
    assert per_user_consistency_check(phi, phi_u, 100) < 1e-9


# --------------------------------------------------------------------------
# 5. Symmetry
# --------------------------------------------------------------------------

def test_symmetric_players_get_equal_shapley():
    # Build a game where CF and CB are symmetric: v depends on |C ∩ {CF,CB}|
    v, pu = {}, {}
    n = 50
    for r in range(len(SOURCES) + 1):
        for comb in itertools.combinations(SOURCES, r):
            C = tuple(sorted(comb))
            n_sym = len([g for g in C if g in ("CF", "CB")])
            val = np.full(n, n_sym * 0.5 + 0.1 * len(C))
            v[C] = float(val.mean())
            pu[C] = val
    v[()] = 0.0
    pu[()] = np.zeros(n)
    phi, _ = shapley_from_values(v, pu, n)
    assert phi["CF"] == pytest.approx(phi["CB"], abs=1e-12)


# --------------------------------------------------------------------------
# 3. Null ranker calibration
# --------------------------------------------------------------------------

def test_null_ranker_analytic_calibration():
    from shapact.game import null_ranker_ndcg

    rng = np.random.default_rng(0)
    n_users, N = 20000, 200
    cand = np.tile(np.arange(N), (n_users, 1))
    rng.shuffle(cand, axis=1)
    test = pd.DataFrame({"user": np.arange(n_users),
                         "item": np.arange(n_users) % N})
    ndcg = null_ranker_ndcg(cand, test, seed=7)
    analytic = (1.0 / N) * sum(1.0 / np.log2(r + 2.0) for r in range(10))
    # std of the mean is ~0.1/sqrt(n_users); 0.003 is ~13 sigma headroom
    assert abs(ndcg.mean() - analytic) < 0.003


# --------------------------------------------------------------------------
# 9. Fidelity decomposition invariant (integration)
# --------------------------------------------------------------------------

def test_fidelity_decomposition_invariant(real_audit):
    for g in real_audit["fidelity"]:
        assert real_audit["fidelity"][g]["decomp_max"] < 1e-9


# --------------------------------------------------------------------------
# 10. Reflexivity aggregate identity (integration)
# --------------------------------------------------------------------------

def test_reflexivity_aggregate_identity(real_audit):
    for g_star, r in real_audit["reflexivity"].items():
        assert abs(r["aggregate"] - r["target"]) < 1e-9


# --------------------------------------------------------------------------
# 11. Synthetic decision test: redundant pair
# --------------------------------------------------------------------------

def _redundant_pair_audit_like():
    """Synthetic game with CF and CB perfectly redundant.

    v(C) = value of the union's unique contribution; a redundant sibling adds
    nothing to a coalition that already contains the other.
    """
    rng = np.random.default_rng(1)
    n = 200
    v, pu = {}, {}
    for r in range(3):
        for comb in itertools.combinations(SOURCES, r):
            C = tuple(sorted(comb))
            has = ("CF" in C) or ("CB" in C)
            val = np.full(n, (1.0 if has else 0.0) + 0.1 * len(C))
            v[C] = float(val.mean())
            pu[C] = val
    v[()] = 0.0
    pu[()] = np.zeros(n)
    phi, _ = shapley_from_values(v, pu, n)
    loo = {g: v[tuple(sorted(SOURCES))] - v[tuple(sorted(set(SOURCES) - {g}))]
           for g in SOURCES}
    sh_rule = min(SOURCES, key=lambda g: phi[g])
    loo_rule = min(SOURCES, key=lambda g: loo[g])
    assert sh_rule != loo_rule
    # realized loss of retiring the Shapley pick <= realized loss of LOO pick
    def realized(g):
        return v[tuple(sorted(SOURCES))] - v[tuple(sorted(set(SOURCES) - {g}))]
    assert realized(sh_rule) <= realized(loo_rule) + 1e-12
