"""Tests for the recourse planner and OPE module."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cavi.allocation import CAV
from cavi.games import Feasibility
from cavi.recourse import MinimalActionPlanner, greedy_gap, check_submodular
from cavi.ope import (naive_bayes_propensity, ips_estimate, dr_estimate,
                      snips_estimate, effective_sample_size, discrepancy_gate)
from cavi.uncertainty import calibration_ece


def make_cav():
    players = [0, 1, 2]
    cav = CAV(phi_mean=np.array([2.0, 1.0, 0.5]),
              phi_var=np.array([0.1, 0.1, 0.1]),
              kappa=0.0, players=players)
    return cav


def test_greedy_plan_budget():
    cav = make_cav()
    costs = [1.0, 1.0, 1.0]
    planner = MinimalActionPlanner(cav, costs, budget=2.0)
    sel, cost = planner.greedy_plan()
    assert cost <= 2.0 + 1e-9
    assert len(sel) == 2  # picks top-2 by value-per-cost


def test_greedy_respects_uplift_target():
    cav = make_cav()
    costs = [1.0, 1.0, 1.0]
    planner = MinimalActionPlanner(cav, costs, budget=5.0)
    # uplift_fn: value-per-cost selects lever 0 first
    def uplift(S):
        return sum(cav.cav[i] for i in S)
    sel, cost = planner.greedy_plan(min_uplift=2.5, uplift_fn=uplift)
    assert 0 in sel  # highest value lever selected
    assert uplift(sel) >= 2.5 - 1e-9


def test_greedy_gap():
    cav = make_cav()
    costs = [1.0, 2.0, 3.0]
    def uplift(S):
        return sum(cav.cav[i] for i in S)
    gap = greedy_gap(cav, costs, uplift, min_uplift=2.9, budget=6.0, max_subset=3)
    assert gap["greedy_cost"] <= 6.0


def test_submodularity_check():
    def f(S):
        return float(sum(np.exp(-0.5 * i) for i in S))
    assert check_submodular([0, 1, 2, 3], f, samples=1000) is True


def test_naive_bayes_propensity_range():
    users = [0, 0, 1, 1, 2]
    items = [5, 5, 6, 7, 8]
    obs = [True, True, True, False, False]
    p = naive_bayes_propensity(users, items, obs)
    assert np.all((p > 0) & (p < 1))


def test_dr_is_doubly_robust_idea():
    # DR recovers a known mean when outcome model is correct even if propensity is off
    rng = np.random.default_rng(0)
    n = 2000
    propensity = np.full(n, 0.5)
    indicator = np.ones(n)
    rewards = rng.normal(3.0, 1.0, n)
    outcome_model = np.full(n, 3.0)  # correct outcome model
    dr = dr_estimate(rewards, propensity, outcome_model, indicator)
    assert abs(dr - 3.0) < 0.1


def test_effective_sample_size():
    propensity = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    indicator = np.ones(5)
    ess = effective_sample_size(propensity, indicator)
    assert ess == pytest.approx(5.0, abs=1e-6)


def test_discrepancy_gate():
    assert discrepancy_gate(2.0, 2.1, tolerance=0.5)["pass_gate"] is True
    assert discrepancy_gate(-1.0, -0.9, tolerance=0.5)["pass_gate"] is False
    assert discrepancy_gate(2.0, 10.0, tolerance=0.5)["pass_gate"] is False


def test_calibration_ece():
    rng = np.random.default_rng(0)
    pred = np.linspace(0.1, 0.9, 100)
    outcomes = (rng.random(100) < pred).astype(float)
    res = calibration_ece(pred, outcomes, n_bins=5)
    assert 0.0 <= res["ece"] <= 1.0
