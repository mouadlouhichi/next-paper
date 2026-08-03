from __future__ import annotations

from itertools import combinations
import math

import numpy as np
from scipy import sparse
from tqdm.auto import tqdm

from .metrics import topk
from .rerank import attribution_adjustment, zscore, sim_user_items


def _ndcg_single(scores: np.ndarray, target: int, train_items: np.ndarray, k: int = 20) -> float:
    s = scores.copy()
    s[train_items] = -np.inf
    recs = topk(s[None, :], k)[0]
    loc = np.where(recs == target)[0]
    if len(loc) == 0:
        return 0.0
    return float(1.0 / np.log2(int(loc[0]) + 2))


def _diversity(scores: np.ndarray, train_items: np.ndarray, item_vectors: sparse.csr_matrix, k: int = 20) -> float:
    s = scores.copy(); s[train_items] = -np.inf
    recs = topk(s[None, :], k)[0]
    if len(recs) < 2:
        return 0.0
    sim = (item_vectors[recs] @ item_vectors[recs].T).toarray()
    return float(np.mean(1.0 - sim[np.triu_indices(len(recs), 1)]))


def coalition_value(
    base_scores_u: np.ndarray,
    train_items: np.ndarray,
    coalition_idx: np.ndarray,
    val_target: int,
    item_vectors: sparse.csr_matrix,
    alpha: float = 0.70,
    beta: float = 0.30,
    lambda_pref: float = 0.20,
    lambda_attr_value: float = 0.10,
) -> float:
    if len(coalition_idx) == 0:
        raw_w = np.zeros(0, dtype=np.float32)
        coalition_items = train_items[:0]
        pref = 0.0
    else:
        coalition_items = train_items[coalition_idx]
        raw_w = np.ones(len(coalition_items), dtype=np.float32)
        pref_sims = sim_user_items(train_items, item_vectors)[coalition_idx]
        pref = float(np.sum(pref_sims))
    # The local prototype uses coalition membership as the deterministic mask effect.
    adj_all = np.zeros_like(base_scores_u, dtype=np.float32)
    if len(coalition_items):
        candidates = np.arange(base_scores_u.shape[0], dtype=np.int64)
        adj_all = attribution_adjustment(coalition_items, raw_w, item_vectors, candidates)
    scores = zscore(base_scores_u) + lambda_attr_value * zscore(adj_all)
    ndcg = _ndcg_single(scores, val_target, train_items, k=20)
    div = _diversity(scores, train_items, item_vectors, k=20)
    return alpha * ndcg + beta * div + lambda_pref * pref


def exact_shapley(base_scores_u, train_items, val_target, item_vectors, **kwargs) -> np.ndarray:
    n = len(train_items)
    phi = np.zeros(n, dtype=np.float32)
    players = np.arange(n)
    fact = math.factorial
    nfact = fact(n)
    cache: dict[tuple[int, ...], float] = {}
    def v(tup):
        tup = tuple(sorted(tup))
        if tup not in cache:
            cache[tup] = coalition_value(base_scores_u, train_items, np.array(tup, dtype=np.int64), val_target, item_vectors, **kwargs)
        return cache[tup]
    for p in players:
        others = [o for o in players if o != p]
        for r in range(n):
            for S in combinations(others, r):
                w = fact(r) * fact(n - r - 1) / nfact
                phi[p] += w * (v((*S, p)) - v(S))
    return phi


def permutation_shapley(base_scores_u, train_items, val_target, item_vectors, m: int = 128, seed: int = 42, **kwargs) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(train_items)
    phi = np.zeros(n, dtype=np.float64)
    cache: dict[tuple[int, ...], float] = {}
    def v(prefix):
        tup = tuple(sorted(prefix))
        if tup not in cache:
            cache[tup] = coalition_value(base_scores_u, train_items, np.array(tup, dtype=np.int64), val_target, item_vectors, **kwargs)
        return cache[tup]
    for _ in range(m):
        perm = rng.permutation(n)
        S: list[int] = []
        prev = v(S)
        for p in perm:
            S.append(int(p))
            cur = v(S)
            phi[p] += cur - prev
            prev = cur
    return (phi / m).astype(np.float32)


def compute_shapley_for_users(split, base_scores: np.ndarray, item_vectors: sparse.csr_matrix, max_users: int | None = None, m: int = 128, exact_threshold: int = 8, seed: int = 42, **kwargs) -> dict[int, np.ndarray]:
    train_csr = split.train_csr
    val_targets = split.val.sort_values("user").set_index("user").item.to_dict()
    users = np.arange(split.n_users)
    if max_users is not None:
        users = users[:max_users]
    out: dict[int, np.ndarray] = {}
    for u in tqdm(users, desc="Shapley users"):
        train_items = train_csr[u].indices
        if len(train_items) == 0 or u not in val_targets:
            out[int(u)] = np.zeros(len(train_items), dtype=np.float32)
            continue
        if len(train_items) <= exact_threshold:
            phi = exact_shapley(base_scores[u], train_items, int(val_targets[u]), item_vectors, **kwargs)
        else:
            phi = permutation_shapley(base_scores[u], train_items, int(val_targets[u]), item_vectors, m=m, seed=seed + int(u), **kwargs)
        out[int(u)] = phi
    return out
