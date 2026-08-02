"""
recommender.py — history-conditioned profile-aggregation recommender with BPR
item factors, plus the forward dynamics model.

The recommender is history-conditioned (score reads the retained profile at
inference), satisfying the ActionShap masking-sensitivity gate: masking a lever
genuinely moves v(S).
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Set

import numpy as np


def bpr_item_factors(ratings, users: Sequence[int], n_items: int, d: int = 32,
                     epochs: int = 6, triplets: int = 60000, lr: float = 0.05,
                     reg: float = 0.01, seed: int = 0, threshold: float = 4.0,
                     verbose: bool = False) -> np.ndarray:
    """BPR-MF item factors over the implicit matrix (rating >= threshold)."""
    rng = np.random.default_rng(seed)
    users = list(users)
    ui: dict = {}
    for u, i, r, t in ratings:
        if u in users and r >= threshold:
            ui.setdefault(u, set()).add(i)
    pos = np.array([(u, i) for u, its in ui.items() for i in its])
    uid = {u: k for k, u in enumerate(users)}
    P = rng.normal(0, 0.01, (len(users), d)).astype(np.float64)
    Q = rng.normal(0, 0.01, (n_items, d)).astype(np.float64)
    all_items = np.arange(n_items)
    for ep in range(epochs):
        idx = rng.choice(len(pos), size=triplets, replace=True)
        for uu, ii in zip(pos[idx, 0], pos[idx, 1]):
            jj = int(rng.choice(all_items))
            pu = P[uid[uu]]
            qi, qj = Q[ii], Q[jj]
            x = pu.dot(qi) - pu.dot(qj)
            sig = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
            P[uid[uu]] += lr * (sig * (qi - qj) - reg * pu)
            Q[ii] += lr * (sig * pu - reg * qi)
            Q[jj] += lr * (-sig * pu - reg * qj)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    return np.ascontiguousarray(Q)


class ProfileRecommender:
    """f_u^S(i) = mean_embedding(base U S) . Q_i over a fixed candidate set."""

    def __init__(self, Q: np.ndarray, candidates: Sequence[int], K: int = 20):
        self.Q = Q
        self.cand = list(candidates)
        self.K = K

    def profile(self, items: Sequence[int]) -> np.ndarray:
        if len(items) == 0:
            return np.zeros(self.Q.shape[1])
        return self.Q[np.asarray(items)].mean(axis=0)

    def scores(self, items: Sequence[int]) -> np.ndarray:
        return self.Q[self.cand] @ self.profile(items)

    def ranking(self, items: Sequence[int]) -> np.ndarray:
        return np.argsort(-self.scores(items), kind="stable")

    def future_util(self, items: Sequence[int],
                    future: Sequence[int]) -> float:
        """Normalized DCG@K of the candidate ranking vs the future set."""
        if len(future) == 0:
            return 0.0
        s = self.scores(items)
        order = np.argsort(-s, kind="stable")
        cand_set = set(self.cand)
        rel_ranks = [k + 1 for k, cp in enumerate(order[:self.K])
                     if self.cand[cp] in future]
        n_rel = sum(1 for it in future if it in cand_set)
        idcg = sum(1.0 / np.log2(j + 1) for j in range(1, n_rel + 1))
        if idcg == 0:
            return 0.0
        return float(sum(1.0 / np.log2(r + 1) for r in rel_ranks) / idcg)


class DynamicsModel:
    """
    Learned greedy/tempered next-item dynamics. p_next proportional to
    exp(profile . Q / temperature). Ensemble over seeds gives the variance game.
    """

    def __init__(self, rec: ProfileRecommender, temp: float = 1.0):
        self.rec = rec
        self.temp = temp

    def step(self, profile_items: Sequence[int], rng: np.random.Generator
             ) -> int:
        p = self.rec.profile(profile_items)
        logits = self.rec.Q @ p / self.temp
        prof_set = set(profile_items)
        logits[list(prof_set)] = -1e9
        logits = logits - logits.max()
        pr = np.exp(logits)
        pr = pr / pr.sum()
        return int(rng.choice(len(self.rec.Q), p=pr))

    def forward_value(self, base: Sequence[int], levers_active: Sequence[int],
                      future: Sequence[int], H: int, gamma: float = 0.9,
                      ensemble: int = 8, seed: int = 0
                      ) -> tuple[float, float]:
        """(mean, var) of discounted future utility over an ensemble of rollouts."""
        vals = []
        for e in range(ensemble):
            rng = np.random.default_rng(seed + e)
            prof = list(base) + list(levers_active)
            g = 0.0
            for tau in range(1, H + 1):
                nxt = self.step(prof, rng)
                prof.append(nxt)
                g += (gamma ** (tau - 1)) * self.rec.future_util(prof, future)
            vals.append(g)
        vals = np.array(vals)
        return float(vals.mean()), float(vals.var())
