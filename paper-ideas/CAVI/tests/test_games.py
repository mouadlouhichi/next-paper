"""Theory tests for the CAV allocation and its guarantees."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cavi.games import (Feasibility, CooperativeGame, RestrictedGame,
                        mean_variance_game, myerson_value, mc_myerson_value,
                        exact_shapley)
from cavi.allocation import compute_cav, verify_additivity_identity


def additive_value_fn(weights):
    def fn(S):
        return float(sum(weights[i] for i in S))
    return fn


def test_exact_shapley_matches_weights():
    w = np.array([1.0, 2.0, 3.0, 4.0])
    phi = exact_shapley(additive_value_fn(w), list(range(4)))
    np.testing.assert_allclose(phi, w, rtol=1e-9)
    # efficiency
    assert abs(phi.sum() - w.sum()) < 1e-9


def test_myerson_component_efficiency():
    # Two disconnected components: {0,1} and {2}. Feasibility has hyperedge {0,1}
    feas = Feasibility([[0, 1]])
    players = [0, 1, 2]
    w = np.array([1.0, 2.0, 3.0])
    game = CooperativeGame(players, additive_value_fn(w))
    phi = myerson_value(game, feas, players)
    # component efficiency: sum over {0,1} = u({0,1}) = 3; player 2 alone = u({2})=3
    assert abs(phi[0] + phi[1] - 3.0) < 1e-9
    assert abs(phi[2] - 3.0) < 1e-9


def test_myerson_null_player():
    feas = Feasibility([[0, 1, 2]])
    players = [0, 1, 2]
    def fn(S):
        return float(2.0 if (0 in S) and (1 in S) else 0.0)
    game = CooperativeGame(players, fn)
    phi = myerson_value(game, feas, players)
    # player 2 is a null player
    assert abs(phi[2]) < 1e-9


def test_additivity_identity_exact():
    # mean and variance games; CAV = Shapley(mean) - kappa*Shapley(var)
    players = list(range(4))
    feas = Feasibility([[0, 1, 2], [2, 3]])
    mean_fn = additive_value_fn(np.array([1.0, 0.5, 0.2, 0.8]))
    def var_fn(S):
        return float(0.1 + 0.05 * sum(i * i for i in S))
    kappa = 0.5
    diff, ok = verify_additivity_identity(mean_fn, var_fn, kappa, feas,
                                          players, M=None)
    assert ok, f"additivity identity violated, max diff {diff}"


def test_additivity_identity_mc():
    players = list(range(6))
    feas = Feasibility([[0, 1], [2, 3], [4, 5]])
    rng = np.random.default_rng(1)
    mean_fn = additive_value_fn(rng.random(6))
    def var_fn(S):
        return float(np.sum(np.asarray(S) * 0.1))
    diff, ok = verify_additivity_identity(mean_fn, var_fn, 0.7, feas,
                                          players, M=2000, seed=0)
    # MC tolerance looser
    assert abs(diff) < 1e-2, f"MC additivity diff {diff}"


def test_risk_sensitivity():
    # Higher-variance lever gets penalized more as kappa rises.
    players = [0, 1]
    feas = Feasibility([[0, 1]])
    def mean_fn(S):
        return float(1.0 if 0 in S else (0.5 if 1 in S else 0.0))
    # lever 0 has higher variance contribution
    def var_fn(S):
        return float(2.0 if 0 in S else (0.2 if 1 in S else 0.0))
    cav0 = compute_cav(mean_fn, var_fn, 0.0, feas, players, M=None)
    cav1 = compute_cav(mean_fn, var_fn, 1.0, feas, players, M=None)
    # lever 0 loses more value as kappa increases
    drop0 = cav0.cav[0] - cav1.cav[0]
    drop1 = cav0.cav[1] - cav1.cav[1]
    assert drop0 > drop1
