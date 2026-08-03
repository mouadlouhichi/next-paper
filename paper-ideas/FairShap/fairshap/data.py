"""
data.py — MovieLens-1M loading, temporal split, and fairness metadata
(popularity tiers, provider groups, user activity groups).
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple

import numpy as np


def load_ml1m(ratings_path: str) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    """Load MovieLens-1M (rating >= 4 = positive interaction).
    Returns (users_items, items)."""
    rows = []
    with open(ratings_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) == 4:
                u, i, r, t = int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
                if r >= 4.0:
                    rows.append((u, i, t))
    users_items: Dict[int, List[Tuple[int, int]]] = {}
    for u, i, t in rows:
        users_items.setdefault(u, []).append((i, t))
    for u in users_items:
        users_items[u].sort(key=lambda x: x[1])
        users_items[u] = [i for i, _ in users_items[u]]
    n_items = max(i for u in users_items for i in users_items[u]) + 1
    return users_items, {i: str(i) for i in range(n_items)}


def temporal_split(seq: Sequence[int], n_test: int = 1):
    if len(seq) <= n_test:
        return list(seq), []
    return list(seq[:-n_test]), list(seq[-n_test:])


def popularity(users_items):
    pop = {}
    for u, its in users_items.items():
        for i in its:
            pop[i] = pop.get(i, 0) + 1
    return pop


def popularity_tier(pop, tail_frac=0.2, head_frac=0.2):
    vals = np.array([pop.get(i, 0) for i in range(max(pop) + 1)])
    head_thr = np.quantile(vals[vals > 0], 1 - head_frac) if (vals > 0).any() else 0
    tail_thr = np.quantile(vals[vals > 0], tail_frac) if (vals > 0).any() else 0
    tier = {}
    for i, v in enumerate(vals):
        if v > head_thr:
            tier[i] = 0
        elif v < tail_thr:
            tier[i] = 2
        else:
            tier[i] = 1
    return tier


def provider_of_items(users_items, pop):
    vals = sorted(pop.values())
    provider = {}
    for i in pop:
        q = sum(1 for v in vals if v <= pop[i]) / max(len(vals), 1)
        provider[i] = min(int(q * 10), 9)
    return provider


def user_activity_group(users_items):
    lens = sorted([(u, len(its)) for u, its in users_items.items()], key=lambda x: x[1])
    n = len(lens)
    group = {}
    for k, (u, _) in enumerate(lens):
        frac = k / max(n, 1)
        group[u] = 0 if frac < 0.33 else (1 if frac < 0.66 else 2)
    return group


def item_similarity(Q, topk=50):
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    sim = Qn @ Qn.T
    out = {}
    for i in range(Q.shape[0]):
        nbrs = np.argsort(-sim[i])[1:topk + 1]
        out[i] = {int(j): float(sim[i, j]) for j in nbrs}
    return out
