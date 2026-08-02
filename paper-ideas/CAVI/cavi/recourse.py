"""
recourse.py — budget-constrained minimal-action planning.

Given per-lever CAV and per-lever cost c_i, select the *smallest-cost* feasible
coalition S* that reaches a target future-utility uplift under a budget B.

    S* = argmin_S sum_{i in S} c_i   s.t.  cost(S) <= B,  E[Delta_v](S) >= Delta*

Algorithm: Shapley-guided risk-adjusted greedy on CAV_i / c_i (value per unit
cost). When the marginal-gain function is monotone and submodular, this is a
(1-1/e)-approximation (proposition in §3.5.5); we expose a submodularity check
and an exhaustive reference for small lever sets to quantify the greedy gap.
"""
from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .allocation import CAV
from .games import Feasibility


class MinimalActionPlanner:
    def __init__(self, cav: CAV, costs: Sequence[float],
                 feasibility: Optional[Feasibility] = None,
                 budget: float = float("inf")):
        self.cav = cav
        self.costs = {p: float(c) for p, c in zip(cav.players, costs)}
        self.budget = budget
        self.feas = feasibility

    def value_per_cost(self, i: int) -> float:
        c = self.costs[i]
        if c <= 0:
            return float("inf") if self.cav.cav[i] > 0 else -float("inf")
        return float(self.cav.cav[i] / c)

    def greedy_plan(self, min_uplift: Optional[float] = None,
                    uplift_fn: Optional[Callable[[List[int]], float]] = None
                    ) -> Tuple[List[int], float]:
        """
        Greedy risk-adjusted selection. Returns (selected_levers, realized_cost).
        If uplift_fn is provided and min_uplift given, stops when reached;
        otherwise fills to budget.
        """
        selected: List[int] = []
        cost = 0.0
        pool = list(self.cav.players)
        while pool:
            # pick best feasible value-per-cost
            best = None; best_ratio = -float("inf")
            for i in pool:
                if cost + self.costs[i] <= self.budget:
                    r = self.value_per_cost(i)
                    if r > best_ratio:
                        best_ratio, best = r, i
            if best is None:
                break
            selected.append(best)
            cost += self.costs[best]
            pool.remove(best)
            if uplift_fn is not None and min_uplift is not None:
                if uplift_fn(selected) >= min_uplift:
                    break
        return selected, cost

    def exhaustive_reference(self, uplift_fn: Callable[[List[int]], float],
                             min_uplift: float, max_subset: int = 12):
        """
        For small lever sets, find the min-cost subset reaching min_uplift
        (budget-respecting). Returns (best_subset, best_cost) or (None, inf).
        """
        import itertools
        players = self.cav.players
        best = None; best_cost = float("inf")
        n = len(players)
        for r in range(1, min(n, max_subset) + 1):
            for combo in itertools.combinations(players, r):
                c = sum(self.costs[i] for i in combo)
                if c > self.budget or c >= best_cost:
                    continue
                if uplift_fn(list(combo)) >= min_uplift:
                    best = list(combo); best_cost = c
        return best, best_cost


def greedy_gap(cav: CAV, costs: Sequence[float], uplift_fn: Callable[[List[int]], float],
               min_uplift: float, budget: float, max_subset: int = 12) -> dict:
    """Quantify the greedy-vs-exhaustive gap for validation (honest bound)."""
    planner = MinimalActionPlanner(cav, costs, budget=budget)
    g, gcost = planner.greedy_plan(min_uplift=min_uplift, uplift_fn=uplift_fn)
    e, ecost = planner.exhaustive_reference(uplift_fn, min_uplift, max_subset)
    gap = None
    if gcost < float("inf") and ecost < float("inf") and gcost > 0:
        gap = (gcost - ecost) / gcost
    return {"greedy": g, "greedy_cost": gcost,
            "exhaustive": e, "exhaustive_cost": ecost,
            "relative_cost_gap": gap}


def check_submodular(ground_set: Sequence[int],
                     f: Callable[[Sequence[int]], float],
                     samples: int = 2000, seed: int = 0) -> bool:
    """
    Empirical check of submodularity of the marginal-gain f: for all sampled
    A subset B and x not in B, f(A u {x}) - f(A) >= f(B u {x}) - f(B).
    Returns True if no violation found on the samples.
    """
    rng = np.random.default_rng(seed)
    n = len(ground_set)
    gs = list(ground_set)
    for _ in range(samples):
        A = set(); B = set()
        for i in gs:
            r = rng.random()
            if r < 0.5:
                A.add(i)
            if r < 0.7:
                B.add(i)
        B = B | A
        cand = [i for i in gs if i not in B]
        if not cand:
            continue
        x = int(rng.choice(cand))
        margA = f(list(A | {x})) - f(list(A))
        margB = f(list(B | {x})) - f(list(B))
        if margA < margB - 1e-9:
            return False
    return True
