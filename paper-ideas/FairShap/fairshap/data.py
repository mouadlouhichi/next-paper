"""
data.py — MovieLens-1M loading, temporal split, and fairness metadata
(popularity tiers, provider groups, user activity groups).
"""
from __future__ import annotations
import os
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
    """
    Item -> top-k similar items (cosine). Memory-safe for large item sets:
    computes neighbors in chunks without materializing the full NxN matrix.
    """
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    n = Q.shape[0]
    out = {}
    chunk = 1000
    for s in range(0, n, chunk):
        block = Qn[s:s + chunk] @ Qn.T          # (chunk, n)
        for local_i in range(block.shape[0]):
            i = s + local_i
            row = block[local_i]
            nbrs = np.argsort(-row)[1:topk + 1]  # skip self (rank 0)
            out[i] = {int(j): float(row[j]) for j in nbrs}
    return out


# ---------------------------------------------------------------------------
# Multi-dataset loaders (ML-1M, Amazon-Book, Yelp2018)
# ---------------------------------------------------------------------------
def load_lightgcn_dataset(data_dir):
    """Load a canonical LightGCN split (train.txt / test.txt).
    Format per line: 'user item1 item2 ...'. Returns (train_dict, test_dict)
    as user -> list of item ids, with items remapped to 0..N-1."""
    def read(name):
        out = {}
        with open(name, encoding="utf-8", errors="replace") as f:
            for line in f:
                toks = line.strip().split()
                if not toks:
                    continue
                u = int(toks[0])
                out[u] = [int(x) for x in toks[1:]]
        return out
    tr = read(os.path.join(data_dir, "train.txt"))
    te = read(os.path.join(data_dir, "test.txt"))
    return tr, te


def load_dataset(name, ml1m_path=None):
    """
    Load any supported dataset into (users_items, n_items).
    users_items: user -> list of item ids (train interactions).
    Amazon-Book / Yelp2018: train split is the history, test split held out.
    ML-1M: rating >= 4 is a positive interaction.
    """
    if name == "ml-1m":
        return load_ml1m(ml1m_path)
    elif name in ("amazon-book", "yelp2018"):
        base = os.path.join(os.path.dirname(__file__), "..", "..", "CAVI", "data", name)
        tr, te = load_lightgcn_dataset(base)
        n_items = max(i for u in tr for i in tr[u]) + 1
        return tr, {i: str(i) for i in range(n_items)}
    else:
        raise ValueError(f"Unknown dataset {name}")
