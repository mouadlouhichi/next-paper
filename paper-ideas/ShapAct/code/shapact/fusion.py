"""Candidate generation, normalization, and the fusion ranker.

Design follows SignalShap Implementation Spec A.4, A.6, A.7:

* candidates: per user, union of each source's top-N_g items excluding the
  user's train items, truncated to N=200 by round-robin across sources
* recall ceiling: fraction of users whose held-out test item is retrieved
* normalization: per-user per-source z-score over the candidate list, with
  a constant-column fallback to zeros (sigma=0 guard)
* fusion: pairwise logistic ranker (BPR-style) over the K normalized scores;
  base scorers trained once, coalitions realized by column masking + refit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import DatasetConfig, SOURCES

_SCORE_CHUNK = 200_000


def _score_chunked(source, u: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Score (u, i) pairs in chunks to bound memory for embedding sources."""
    out = np.empty(len(u))
    for lo in range(0, len(u), _SCORE_CHUNK):
        hi = min(lo + _SCORE_CHUNK, len(u))
        out[lo:hi] = source.score(u[lo:hi], i[lo:hi])
    return out


def build_candidates(ds, sources: dict, cfg: DatasetConfig, seed: int = 42):
    """Per-user candidate arrays (n_users, N) plus per-source top lists.

    Items already in the user's training history are excluded from every
    source's list; the held-out test item is eligible (it is not in train).
    Ties are broken deterministically by item id.

    Truncation to N is by *best cross-source rank*: each union item is scored
    by the best (minimum) rank it attains in any source's top list, and the
    N items with the best such rank are kept. This preserves the design
    intent of "no single source dominates the pool" (an item only enters
    through a source that ranks it highly) while avoiding the fixed
    ~40-items-per-source cap of a strict round-robin, which was measured to
    cut the retrieval ceiling from 0.80 to 0.50 on MovieLens-1M.
    """
    n_users = len(ds.users)
    n_items = len(ds.items)
    train_set = ds.train.groupby("user")["item"].apply(set).to_dict()

    top_lists = {}
    for g in SOURCES:
        s = sources[g]
        rows = np.full((n_users, cfg.source_top_n), -1, dtype=np.int32)
        for u in range(n_users):
            excl = train_set.get(u, set())
            scores = s.score(np.full(n_items, u), np.arange(n_items))
            order = np.lexsort((np.arange(n_items), -scores))
            picked = [int(it) for it in order if it not in excl][:cfg.source_top_n]
            rows[u, : len(picked)] = picked
        top_lists[g] = rows

    cand = _best_rank_truncate(top_lists, cfg.candidates_n, n_users)
    return cand, top_lists


def _best_rank_truncate(top_lists: dict, N: int, n_users: int) -> np.ndarray:
    """Keep the N union items with the best minimum rank across sources."""
    cand = np.full((n_users, N), -1, dtype=np.int32)
    for u in range(n_users):
        best = {}
        for gi, g in enumerate(SOURCES):
            L = top_lists[g][u]
            for r, it in enumerate(L):
                if it < 0:
                    break  # lists are prefix-filled
                rk = r + 1
                cur = best.get(it)
                if cur is None or rk < cur:
                    best[it] = rk
        items = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))
        cand[u, : min(N, len(items))] = [k for k, _ in items[:N]]
    return cand


def candidate_recall(cand: np.ndarray, test: pd.DataFrame) -> float:
    d = test.set_index("user")["item"].to_dict()
    hits = n = 0
    for u, it in d.items():
        if u < cand.shape[0]:
            n += 1
            hits += int(it in cand[u])
    return hits / max(n, 1)


def znorm_matrix(scores: np.ndarray, eps: float = 1e-12):
    mu = scores.mean(axis=1, keepdims=True)
    sigma = scores.std(axis=1, keepdims=True)
    degenerate = sigma < eps
    z = np.where(degenerate, 0.0, (scores - mu) / np.where(degenerate, 1.0, sigma))
    return z.astype(np.float64), degenerate.ravel()


class ZCache:
    """Per-user per-source z-values over a candidate list, with the list
    statistics needed to z-normalize out-of-list items under the same
    transform (used for fusion training pairs)."""

    def __init__(self, ds, sources: dict, cand: np.ndarray, cfg: DatasetConfig):
        n_users, N = cand.shape
        self.sources = sources
        self.sources_list = list(sources)
        self.z = {}
        self.degenerate = {}
        self.mu = {}
        self.sigma = {}
        self.cand = cand
        self.n_items = len(ds.items)
        # (user, item) -> position in the candidate list (or -1)
        self.pos_map = np.full((n_users, self.n_items), -1, dtype=np.int32)
        valid = cand >= 0
        uu, cc = np.where(valid)
        self.pos_map[uu, cand[uu, cc]] = cc

        for g in self.sources_list:
            raw = _score_chunked(
                sources[g], np.repeat(np.arange(n_users), N), cand.ravel()
            ).reshape(n_users, N)
            raw = np.where(cand < 0, np.nan, raw)
            filled = np.nan_to_num(raw, nan=0.0)
            z, deg = znorm_matrix(filled)
            self.z[g] = z
            self.degenerate[g] = deg
            # list stats (on non-missing entries)
            m = np.nanmean(np.where(cand < 0, np.nan, raw), axis=1)
            sd = np.nanstd(np.where(cand < 0, np.nan, raw), axis=1)
            self.mu[g] = np.nan_to_num(m, nan=0.0)
            self.sigma[g] = np.where(sd > 1e-12, sd, 1.0)

    def z_stack(self) -> np.ndarray:
        """(n_users, N, K) tensor of z-values aligned with sources_list."""
        return np.stack([self.z[g] for g in self.sources_list], axis=-1)

    def pair_features(self, ds, cfg: DatasetConfig, seed: int,
                      signal: str = "val"):
        """Vectorized (pos, neg) pair features for the fusion ranker.

        `signal` selects the positive evidence used to train the fusion:
          'val'   the user's held-out validation item (default). The
                  validation item is the user's second-to-last interaction,
                  the same distribution as the test target; training on it
                  avoids the memorization bias of train-item pairs, which we
                  measured to halve fused NDCG@10 on MovieLens-1M (0.037 vs
                  0.074). This is a disclosed instantiation decision.
          'train' BPR-style sampling of R training positives per user.
        Negatives are sampled from items outside the user's train, validation,
        and test interactions (no leakage into the fusion signal).

        The same sampled pairs are reused for every coalition refit, so
        coalition differences come only from the refit (column masking), per
        the SignalShap design.
        """
        rng = np.random.default_rng(seed)
        n_users = len(ds.users)
        R = cfg.fusion_pairs
        K = len(self.sources_list)
        item_set = np.arange(self.n_items)
        all_known = pd.concat(
            [ds.train[["user", "item"]], ds.val[["user", "item"]],
             ds.test[["user", "item"]]], ignore_index=True)
        neg_mem = all_known.groupby("user")["item"].apply(set).to_dict()

        i_pos = np.full((n_users, R), -1, dtype=np.int64)
        i_neg = np.full((n_users, R), -1, dtype=np.int64)
        n_pos = np.zeros(n_users, dtype=int)
        if signal == "val":
            val_items = ds.val.set_index("user")["item"].to_dict()
            for u in range(n_users):
                if u not in val_items:
                    continue
                pool = np.setdiff1d(item_set, np.fromiter(
                    neg_mem.get(u, set()), dtype=np.int64))
                if len(pool) == 0:
                    continue
                n_pos[u] = R
                i_pos[u, :] = val_items[u]
                i_neg[u, :] = rng.choice(pool, size=R, replace=False)
        else:
            train_pos = ds.train.groupby("user")["item"].apply(
                lambda s: np.asarray(s, dtype=np.int64)).to_dict()
            for u in range(n_users):
                pos = train_pos.get(u)
                if pos is None or len(pos) == 0:
                    continue
                pool = np.setdiff1d(item_set, np.fromiter(
                    neg_mem.get(u, set()), dtype=np.int64))
                if len(pool) == 0:
                    continue
                r = min(R, len(pos))
                n_pos[u] = r
                i_pos[u, :r] = rng.choice(pos, size=r, replace=False)
                i_neg[u, :r] = rng.choice(pool, size=r, replace=False)

        items = np.concatenate([i_pos, i_neg], axis=1)   # (n_users, 2R)
        users = np.repeat(np.arange(n_users), 2 * R)
        flat = items.ravel()
        Z = np.zeros((n_users, 2 * R, K))
        for gi, g in enumerate(self.sources_list):
            raw = _score_chunked(self.sources[g], users, flat).reshape(
                n_users, 2 * R)
            pos = np.where(items >= 0,
                           self.pos_map[np.arange(n_users)[:, None],
                                        np.maximum(items, 0)], -1)
            has = pos >= 0
            safe = np.where(has, pos, 0)
            cached = np.take_along_axis(self.z[g], safe, axis=1)
            outside = (raw - self.mu[g][:, None]) / self.sigma[g][:, None]
            Z[:, :, gi] = np.where(has, cached, outside)

        rows = []
        for u in range(n_users):
            r = n_pos[u]
            if r > 0:
                rows.append(Z[u, :r] - Z[u, R:R + r])
        D = np.concatenate(rows, axis=0) if rows else np.zeros((0, K))
        return {"D": D, "local": self.sources_list}


def fit_fusion_from_pairs(pairs: dict, coalition: tuple, cfg: DatasetConfig):
    """Refit the pairwise logistic ranker on a coalition of sources.

    `pairs` is the output of ZCache.pair_features (shared across coalitions);
    the coalition is realized by masking columns, per the SignalShap design.
    """
    local = pairs["local"]
    cols = [local.index(g) for g in coalition]
    D = pairs["D"][:, cols]
    X = np.concatenate([D, -D])
    y = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    clf = LogisticRegression(fit_intercept=False, C=cfg.fusion_c)
    clf.fit(X, y)
    theta = np.zeros(len(SOURCES))
    for j, g in enumerate(coalition):
        theta[SOURCES.index(g)] = clf.coef_[0, j]
    return theta


def fused_scores(theta: np.ndarray, zcache: ZCache,
                 coalition: tuple) -> np.ndarray:
    """Linear fused scores for all (user, candidate) pairs."""
    cols = [zcache.sources_list.index(g) for g in coalition]
    Zall = zcache.z_stack()
    return Zall[:, :, cols] @ theta[[SOURCES.index(g) for g in coalition]]
