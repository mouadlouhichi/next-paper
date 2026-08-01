"""The three-level counterfactual ladder (ShapAct Structure Sec. 3.3).

    L0 masked        withhold g's score column, fixed scorers + candidates
    L1 regenerated   candidates rebuilt from the remaining sources (same scorers)
    L2 never-built   g is never trained, never scored, never fused

For L2, the never-built world for source g is a four-player game whose grand
coalition is G\\{g}: candidates are regenerated from the four surviving
sources and every coalition evaluation uses that fixed candidate set.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import SOURCES
from .fusion import ZCache, build_candidates
from .game import coalition_ndcg, null_ranker_ndcg


def regenerated_world(ds, sources: dict, cfg, exclude: str | None,
                      seed: int = 42):
    """Build the candidate set + ZCache for a world without `exclude`."""
    avail = sorted(g for g in SOURCES if g != exclude)
    cand2, _ = build_candidates_restricted(ds, sources, cfg, avail, seed=seed)
    zc = ZCache(ds, {g: sources[g] for g in avail}, cand2, cfg)
    return cand2, zc, avail


def build_candidates_restricted(ds, sources: dict, cfg, avail, seed: int = 42):
    n_users = len(ds.users)
    n_items = len(ds.items)
    train_set = ds.train.groupby("user")["item"].apply(set).to_dict()

    top_lists = {}
    for g in avail:
        s = sources[g]
        rows = np.full((n_users, cfg.source_top_n), -1, dtype=np.int32)
        for u in range(n_users):
            excl = train_set.get(u, set())
            scores = s.score(np.full(n_items, u), np.arange(n_items))
            order = np.lexsort((np.arange(n_items), -scores))
            picked = [int(it) for it in order if it not in excl][:cfg.source_top_n]
            rows[u, : len(picked)] = picked
        top_lists[g] = rows

    # best cross-source rank truncation over the restricted source set
    cand = _best_rank_restricted(top_lists, cfg.candidates_n, n_users, avail)
    return cand, top_lists


def _best_rank_restricted(top_lists: dict, N: int, n_users: int, avail) -> np.ndarray:
    cand = np.full((n_users, N), -1, dtype=np.int32)
    for u in range(n_users):
        best = {}
        for g in avail:
            for r, it in enumerate(top_lists[g][u]):
                if it < 0:
                    break
                rk = r + 1
                cur = best.get(it)
                if cur is None or rk < cur:
                    best[it] = rk
        items = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))
        cand[u, : min(N, len(items))] = [k for k, _ in items[:N]]
    return cand


def evaluate_world(cand, zc, avail, ds, cfg, seed: int = 42):
    """Coalition values for every C subset of `avail` in a fixed world.

    Returns (v, per_user, thetas) keyed by tuple; the empty coalition is the
    null ranker on this candidate set (v=0 by construction).
    """
    from .fusion import fit_fusion_from_pairs

    v, per_user, thetas = {}, {}, {}
    null_ndcg = null_ranker_ndcg(cand, ds.test, seed=seed)
    pairs = zc.pair_features(ds, cfg, seed=seed)
    for r in range(len(avail) + 1):
        for comb in itertools.combinations(avail, r):
            C = tuple(sorted(comb))
            if len(C) == 0:
                v[C] = 0.0
                per_user[C] = np.zeros(len(ds.users))
                thetas[C] = np.zeros(len(SOURCES))
                continue
            theta = fit_fusion_from_pairs(pairs, C, cfg)
            ndcg = coalition_ndcg(theta, zc, cand, ds.test, C)
            v[C] = float(ndcg.mean() - null_ndcg.mean())
            per_user[C] = ndcg - null_ndcg
            thetas[C] = theta
    return v, per_user, thetas


def shapley_world(v, per_user, avail, n_users: int):
    from .game import shapley_from_values

    phi, phi_u = {}, {}
    K = len(avail)
    for idx, g in enumerate(avail):
        acc = 0.0
        acc_u = np.zeros(n_users)
        others = [h for h in avail if h != g]
        for r in range(K):
            for comb in itertools.combinations(others, r):
                import math
                w = (math.factorial(r) * math.factorial(K - r - 1)
                     / math.factorial(K))
                S = tuple(sorted(comb))
                Sg = tuple(sorted(S + (g,)))
                acc += w * (v[Sg] - v[S])
                acc_u += w * (per_user[Sg] - per_user[S])
        phi[g] = acc
        phi_u[g] = acc_u
    return phi, phi_u
