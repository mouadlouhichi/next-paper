"""CPU-first recommender baselines for external-data analysis.

These models are intentionally separated from CURE-Sim policy control. They let the
notebook fetch a public interaction dataset, train documented baselines, evaluate
ranking logic, and produce reproducible analysis assets before invoking the causal
oracle benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeaveOneOutSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    user_count: int


@dataclass(frozen=True)
class RankingMetrics:
    model: str
    evaluated_users: int
    recall_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float


def chronological_leave_one_out(interactions: pd.DataFrame) -> LeaveOneOutSplit:
    """Create train/validation/test splits without fabricating timestamps."""
    required = {"user_id", "item_id", "response", "timestamp"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"Cannot split interactions; missing {sorted(missing)}")
    positives = interactions[interactions["response"] > 0].copy()
    if positives["timestamp"].isna().all():
        raise ValueError("Chronological evaluation needs timestamps; do not invent order for matrix ratings.")
    positives = positives.dropna(subset=["timestamp"]).sort_values(["user_id", "timestamp", "item_id"], kind="stable")
    grouped = positives.groupby("user_id", sort=False)
    eligible = grouped.filter(lambda frame: len(frame) >= 3)
    if eligible.empty:
        raise ValueError("Need at least three positive interactions per evaluated user.")
    test = eligible.groupby("user_id", sort=False).tail(1)
    remainder = eligible.drop(test.index)
    validation = remainder.groupby("user_id", sort=False).tail(1)
    train = remainder.drop(validation.index)
    return LeaveOneOutSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        user_count=int(test["user_id"].nunique()),
    )


class PopularityRecommender:
    name = "popularity"

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        positive = interactions[interactions["response"] > 0]
        self.item_scores = positive.groupby("item_id").size().astype(float).to_dict()
        self.all_items = np.sort(interactions["item_id"].unique())
        self.user_seen = {
            int(user): set(frame["item_id"].astype(int))
            for user, frame in interactions.groupby("user_id")
        }
        return self

    def score(self, user_id: int, items: np.ndarray) -> np.ndarray:
        return np.asarray([self.item_scores.get(int(item), 0.0) for item in items], dtype=float)

    def recommend(self, user_id: int, k: int) -> np.ndarray:
        candidates = np.asarray([item for item in self.all_items if int(item) not in self.user_seen.get(int(user_id), set())], dtype=int)
        scores = self.score(user_id, candidates)
        order = np.lexsort((candidates, -scores))
        return candidates[order[:k]]


class BPRMFRecommender:
    """Small NumPy BPR-MF implementation for notebook-scale baseline analysis.

    The implementation trades throughput for transparency. `max_updates` controls
    runtime directly and makes it suitable for an M4 laptop without a GPU. It is
    not presented as the final paper architecture.
    """

    name = "bpr_mf"

    def __init__(
        self,
        factors: int = 32,
        learning_rate: float = 0.05,
        regularization: float = 1e-4,
        max_updates: int = 50_000,
        seed: int = 42,
    ):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_updates = max_updates
        self.seed = seed

    def fit(self, interactions: pd.DataFrame) -> "BPRMFRecommender":
        positives = interactions[interactions["response"] > 0][["user_id", "item_id"]].drop_duplicates()
        self.user_ids = np.sort(interactions["user_id"].unique())
        self.item_ids = np.sort(interactions["item_id"].unique())
        self.user_to_index = {int(user): index for index, user in enumerate(self.user_ids)}
        self.item_to_index = {int(item): index for index, item in enumerate(self.item_ids)}
        pairs = np.asarray([
            (self.user_to_index[int(row.user_id)], self.item_to_index[int(row.item_id)])
            for row in positives.itertuples(index=False)
        ], dtype=int)
        self.user_seen = {index: set() for index in range(len(self.user_ids))}
        for user_index, item_index in pairs:
            self.user_seen[int(user_index)].add(int(item_index))
        rng = np.random.default_rng(self.seed)
        scale = 0.05
        self.user_factors = rng.normal(0.0, scale, size=(len(self.user_ids), self.factors))
        self.item_factors = rng.normal(0.0, scale, size=(len(self.item_ids), self.factors))
        if len(pairs) == 0:
            return self

        # A transparent online SGD loop avoids incorrect duplicate-index batch updates.
        updates = min(self.max_updates, max(len(pairs), 1) * 10)
        for _ in range(updates):
            user_index, positive_index = pairs[rng.integers(len(pairs))]
            if len(self.user_seen[int(user_index)]) >= len(self.item_ids):
                continue
            negative_index = rng.integers(len(self.item_ids))
            while negative_index in self.user_seen[int(user_index)]:
                negative_index = rng.integers(len(self.item_ids))
            user_vec = self.user_factors[user_index].copy()
            pos_vec = self.item_factors[positive_index].copy()
            neg_vec = self.item_factors[negative_index].copy()
            margin = float(user_vec @ (pos_vec - neg_vec))
            gradient = 1.0 / (1.0 + np.exp(np.clip(margin, -30.0, 30.0)))
            self.user_factors[user_index] += self.learning_rate * (gradient * (pos_vec - neg_vec) - self.regularization * user_vec)
            self.item_factors[positive_index] += self.learning_rate * (gradient * user_vec - self.regularization * pos_vec)
            self.item_factors[negative_index] += self.learning_rate * (-gradient * user_vec - self.regularization * neg_vec)
        return self

    def score(self, user_id: int, items: np.ndarray) -> np.ndarray:
        user_index = self.user_to_index.get(int(user_id))
        if user_index is None:
            return np.zeros(len(items), dtype=float)
        item_indices = np.asarray([self.item_to_index.get(int(item), -1) for item in items], dtype=int)
        scores = np.full(len(items), -np.inf, dtype=float)
        valid = item_indices >= 0
        scores[valid] = self.item_factors[item_indices[valid]] @ self.user_factors[user_index]
        return scores

    def recommend(self, user_id: int, k: int) -> np.ndarray:
        user_index = self.user_to_index.get(int(user_id))
        if user_index is None:
            return self.item_ids[:k]
        candidates = np.asarray([
            item for index, item in enumerate(self.item_ids)
            if index not in self.user_seen.get(int(user_index), set())
        ], dtype=int)
        scores = self.score(user_id, candidates)
        order = np.lexsort((candidates, -scores))
        return candidates[order[:k]]


def evaluate_leave_one_out(model, split: LeaveOneOutSplit, k: int = 10, max_users: int = 1_000) -> RankingMetrics:
    """Evaluate standard ranking metrics on a bounded, reproducible user subset."""
    test = split.test.sort_values("user_id", kind="stable").head(max_users)
    hits: list[float] = []
    ndcgs: list[float] = []
    for row in test.itertuples(index=False):
        ranking = model.recommend(int(row.user_id), k)
        positions = np.where(ranking == int(row.item_id))[0]
        if len(positions):
            rank = int(positions[0]) + 1
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
    return RankingMetrics(
        model=model.name,
        evaluated_users=len(test),
        recall_at_k=float(np.mean(hits)) if hits else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        hit_rate_at_k=float(np.mean(hits)) if hits else 0.0,
    )
