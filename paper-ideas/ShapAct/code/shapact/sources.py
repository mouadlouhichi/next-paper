"""The five signal sources (SignalShap Implementation Spec A.5).

Each source is a standard, independently published method. All are fitted on
training data only, then score arbitrary (user, item) pairs.

    CF  BPR-MF (implicit feedback, 64 factors)         p_u . q_i
    CB  TF-IDF content similarity                       cos(profile_u, c_i)
    POP static popularity                               log(1 + count_i)
    REC time-decayed popularity (half-life tau)         log(1 + sum_t exp(-ln2 dt/tau))
    SEQ first-order Markov transitions                  log(1 + count(last_u -> i))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .config import DatasetConfig, SOURCES


class Source:
    name: str

    def score(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CFSource(Source):
    name = "CF"

    def __init__(self, cfg: DatasetConfig, train: pd.DataFrame,
                 n_users: int, n_items: int, seed: int = 42):
        rows = train["user"].to_numpy()
        cols = train["item"].to_numpy()
        data = np.ones(len(rows), dtype=np.float32)
        mat = sparse.csr_matrix((data, (rows, cols)),
                                shape=(n_users, n_items))
        import implicit
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=cfg.cf_factors, learning_rate=cfg.cf_lr,
            regularization=cfg.cf_reg, iterations=cfg.cf_iters,
            random_state=seed, num_threads=0,
        )
        model.fit(mat, show_progress=False)
        self.P = model.user_factors.astype(np.float64)   # (n_users, d)
        self.Q = model.item_factors.astype(np.float64)   # (n_items, d)

    def score(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", self.P[u], self.Q[i])


class CBSource(Source):
    name = "CB"

    def __init__(self, item_tfidf: np.ndarray, train: pd.DataFrame,
                 n_users: int):
        # user profile = mean TF-IDF of the user's training items
        rows = train["user"].to_numpy()
        cols = train["item"].to_numpy()
        mat = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_users, item_tfidf.shape[0]))
        nrm = np.asarray(mat.sum(axis=1)).ravel()
        nrm[nrm == 0] = 1.0
        profile = mat @ item_tfidf / nrm[:, None]
        profile = profile / (np.linalg.norm(profile, axis=1, keepdims=True) + 1e-12)
        items = item_tfidf / (
            np.linalg.norm(item_tfidf, axis=1, keepdims=True) + 1e-12)
        self.profile = profile
        self.items = items
        # full (n_users, n_items) similarity matrix: O(1) scoring per pair,
        # avoids materializing (batch, n_terms) gathers for large vocabularies
        self.S = (profile @ items.T).astype(np.float32)

    def score(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.S[u, i]


class POPSource(Source):
    name = "POP"

    def __init__(self, train: pd.DataFrame, n_items: int):
        cnt = np.bincount(train["item"].to_numpy(), minlength=n_items).astype(float)
        self.counts = np.log1p(cnt)

    def score(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.counts[i]


class RECSource(Source):
    name = "REC"

    def __init__(self, cfg: DatasetConfig, train: pd.DataFrame,
                 n_items: int):
        half = cfg.rec_half_life_days * 86400.0
        t_ref = train["timestamp"].max()
        dt = (t_ref - train["timestamp"].to_numpy()).astype(float)
        w = np.exp(-np.log(2.0) * dt / half)
        acc = np.zeros(n_items)
        np.add.at(acc, train["item"].to_numpy(), w)
        self.scores = np.log1p(acc)

    def score(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.scores[i]


class SEQSource(Source):
    name = "SEQ"

    def __init__(self, train: pd.DataFrame, n_items: int):
        tr = train.sort_values(["user", "timestamp"]).reset_index(drop=True)
        a = tr["item"].to_numpy()
        b = np.empty_like(a)
        b[:-1] = a[1:]
        b[-1] = -1
        same_user = tr["user"].to_numpy()[1:] == tr["user"].to_numpy()[:-1]
        # transitions a->b where the user stays the same
        trans = np.column_stack([a[:-1][same_user], b[:-1][same_user]])
        mat = np.zeros((n_items, n_items), dtype=np.float64)
        np.add.at(mat, (trans[:, 0], trans[:, 1]), 1.0)
        # last item per user
        last_u = tr.groupby("user")["item"].last()
        self.T = mat
        self.last_u = last_u

    def score(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        out = np.zeros(len(u))
        lu = self.last_u.to_numpy()[u]
        # entries whose transition target is valid: i >= 0 and last item known
        ok = (i >= 0) & (lu >= 0)
        if ok.any():
            out[ok] = self.T[lu[ok], i[ok]]
        return np.log1p(out)


def fit_sources(cfg: DatasetConfig, ds) -> dict[str, Source]:
    train = ds.train
    sources = {
        "CF": CFSource(cfg, train, len(ds.users), len(ds.items)),
        "CB": CBSource(ds.item_tfidf, train, len(ds.users)),
        "POP": POPSource(train, len(ds.items)),
        "REC": RECSource(cfg, train, len(ds.items)),
        "SEQ": SEQSource(train, len(ds.items)),
    }
    return sources


def score_pairs(sources: dict[str, Source], u: np.ndarray,
                i: np.ndarray) -> dict[str, np.ndarray]:
    return {g: sources[g].score(u, i) for g in SOURCES}
