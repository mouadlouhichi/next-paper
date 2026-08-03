"""
game.py — FairShap cooperative game: fairness-aware coalition value and
preference-aware Shapley exposure attribution.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / x.sum()) / n)


def fairness_term(exposure, items):
    vals = [exposure.get(i, 0.0) for i in items]
    return 1.0 - gini(vals)


def diversity_term(item_sim, items):
    if len(items) < 2:
        return 0.0
    pairs = 0; total = 0.0
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            total += 1.0 - item_sim.get(items[a], {}).get(items[b], 0.0)
            pairs += 1
    return float(total / pairs) if pairs else 0.0


def _rank_quality(exposure, items):
    tot = sum(exposure.values()) or 1.0
    return float(sum(exposure.get(i, 0.0) for i in items) / (len(items) * tot))


def coalition_value(exposure, item_sim, items, alpha=0.6, beta=0.25, gamma=0.15):
    if not items:
        return 0.0
    nd = _rank_quality(exposure, items)
    dv = diversity_term(item_sim, items)
    fa = fairness_term(exposure, items)
    return alpha * nd + beta * dv + gamma * fa


def mc_shapley(value_fn, players, M=500, seed=0, base=None):
    rng = np.random.default_rng(seed)
    n = len(players)
    base = list(base) if base else []
    acc = np.zeros(n)
    for _ in range(M):
        perm = rng.permutation(n)
        running = list(base)
        prev = value_fn(running)
        for local_i in perm:
            running = running + [players[local_i]]
            v = value_fn(running)
            acc[local_i] += v - prev
            prev = v
    return acc / M


def exposure_shapley(exposure, item_sim, candidate_items, M=500, seed=0,
                     alpha=0.6, beta=0.25, gamma=0.15):
    players = list(candidate_items)
    if not players:
        return {}
    def v(S):
        return coalition_value(exposure, item_sim, S, alpha, beta, gamma)
    phi = mc_shapley(v, players, M=M, seed=seed)
    return {players[i]: float(phi[i]) for i in range(len(players))}


def exposure_efficiency_check(phi, exposure, items, tol=1e-4):
    total_phi = sum(phi.get(i, 0.0) for i in items)
    fair_grand = fairness_term(exposure, items)
    return abs(total_phi - fair_grand) <= tol
