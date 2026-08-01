"""The exact source-attribution game (SignalShap Definitions 1-3, Prop. 1, 3).

    v(C) = NDCG@10(fusion_C) - NDCG@10(pi)

with f_emptyset = pi so v(emptyset) = 0. The player set is the five signal
sources; the exact Shapley value is computed over all 2^5 = 32 coalitions.
Because v = (1/|U|) sum_u v_u, per-user Shapley values fall out of the same
coalition sweep at no extra cost (Prop. 3).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .config import SOURCES
from .fusion import ZCache, fit_fusion_from_pairs, fused_scores


def ndcg_at_k(rank_of_target: np.ndarray, k: int) -> np.ndarray:
    """Per-user NDCG@k with a single relevant item per user.

    rank_of_target[u] = position (1-based) of the user's held-out item in the
    ranked candidate list; 0 means not retrieved -> NDCG 0.
    """
    gains = np.zeros_like(rank_of_target, dtype=float)
    pos = (rank_of_target >= 1) & (rank_of_target <= k)
    gains[pos] = 1.0 / np.log2(rank_of_target[pos] + 1.0)
    return gains


def rank_of_target(scores: np.ndarray, cand: np.ndarray,
                   test: pd.DataFrame) -> np.ndarray:
    """1-based rank of each user's test item among their scored candidates.

    Vectorized: rank = 1 + number of candidates with a strictly higher score
    than the test item's score; 0 when the item was not retrieved.
    """
    n_users = cand.shape[0]
    test_items = test.set_index("user")["item"].to_dict()
    t = np.full(n_users, -1, dtype=np.int64)
    for u, it in test_items.items():
        t[u] = it
    mask = cand == t[:, None]
    found = mask.any(axis=1)
    pos = np.argmax(mask, axis=1)
    s_target = np.where(found, scores[np.arange(n_users), pos], np.inf)
    out = (scores > s_target[:, None]).sum(axis=1) + 1
    out[~found] = 0
    return out.astype(int)


def null_ranker_ndcg(cand: np.ndarray, test: pd.DataFrame,
                     seed: int = 42) -> np.ndarray:
    """Per-user NDCG@10 of a uniform-random ranker (fixed seed)."""
    rng = np.random.default_rng(seed)
    n_users = cand.shape[0]
    N = cand.shape[1]
    test_items = test.set_index("user")["item"].to_dict()
    out = np.zeros(n_users)
    for u in range(n_users):
        it = test_items.get(u)
        if it is None:
            continue
        if it not in cand[u]:
            continue
        pos = int(rng.integers(0, N))
        out[u] = 1.0 / np.log2(pos + 2.0) if pos < 10 else 0.0
    return out


def coalition_ndcg(theta: np.ndarray, zcache: ZCache, cand: np.ndarray,
                   test: pd.DataFrame, coalition: tuple, k: int = 10):
    """Per-user NDCG@10 under the fusion for `coalition`."""
    scores = fused_scores(theta, zcache, coalition)
    ranks = rank_of_target(scores, cand, test)
    return ndcg_at_k(ranks, k)


def coalition_values(zcache: ZCache, cand: np.ndarray, test: pd.DataFrame,
                     ds, cfg, seed: int = 42):
    """Evaluate v(C) for every coalition C subset of the five sources.

    Returns (v, per_user, thetas) where v[C] is a scalar, per_user[C] is the
    per-user NDCG vector, and thetas[C] the fitted fusion weights. The empty
    coalition is the null ranker (v = 0 by construction).
    """
    v = {}
    per_user = {}
    thetas = {}
    null_ndcg = null_ranker_ndcg(cand, test, seed=seed)
    pairs = zcache.pair_features(ds, cfg, seed=seed)
    coalitions = []
    for r in range(0, len(SOURCES) + 1):
        for comb in itertools.combinations(SOURCES, r):
            coalitions.append(tuple(sorted(comb)))
    for C in coalitions:
        if len(C) == 0:
            v[C] = 0.0
            per_user[C] = np.zeros(len(ds.users))
            thetas[C] = np.zeros(len(SOURCES))
            continue
        theta = fit_fusion_from_pairs(pairs, C, cfg)
        ndcg = coalition_ndcg(theta, zcache, cand, test, C)
        v[C] = float(ndcg.mean() - null_ndcg.mean())
        per_user[C] = ndcg - null_ndcg
        thetas[C] = theta
    return v, per_user, thetas


def shapley_from_values(v, per_user, n_users: int):
    """Exact Shapley values (global and per-user) from coalition values."""
    K = len(SOURCES)
    phi = {}
    phi_u = {g: np.zeros(n_users) for g in SOURCES}
    for idx, g in enumerate(SOURCES):
        acc = 0.0
        acc_u = np.zeros(n_users)
        others = [h for h in SOURCES if h != g]
        for r in range(K):
            for comb in itertools.combinations(others, r):
                S = comb
                w = (math.factorial(r) * math.factorial(K - r - 1)
                     / math.factorial(K))
                Sg = tuple(sorted(S + (g,)))
                marg = v[Sg] - v[tuple(sorted(S))]
                acc += w * marg
                acc_u += w * (per_user[Sg] - per_user[tuple(sorted(S))])
        phi[g] = acc
        phi_u[g] = acc_u
    return phi, phi_u


def efficiency_check(v, phi, tol: float = 1e-9) -> float:
    total = sum(phi[g] for g in SOURCES)
    return abs(total - v[tuple(sorted(SOURCES))])


def per_user_consistency_check(phi, phi_u, n_users: int,
                               tol: float = 1e-9) -> float:
    worst = 0.0
    for g in SOURCES:
        worst = max(worst, abs(phi_u[g].mean() - phi[g]))
    return worst
