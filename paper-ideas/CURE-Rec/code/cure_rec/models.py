"""Shared-candidate recommender baselines and evaluation protocol."""

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
class EvaluationCase:
    user_id: int
    target_item: int
    candidate_items: np.ndarray


@dataclass(frozen=True)
class RankingMetrics:
    model: str
    evaluated_users: int
    candidate_coverage: float
    cold_test_items: int
    recall_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float


def chronological_leave_one_out(interactions: pd.DataFrame) -> LeaveOneOutSplit:
    """Create train/validation/test splits without fabricated timestamps."""
    required = {"user_id", "item_id", "response", "timestamp"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"Cannot split interactions; missing {sorted(missing)}")
    positives = interactions[interactions["response"] > 0].copy()
    if positives["timestamp"].isna().all():
        raise ValueError("Chronological evaluation needs timestamps; do not invent order for matrix ratings.")
    positives = positives.dropna(subset=["timestamp"]).sort_values(["user_id", "timestamp", "item_id"], kind="stable")
    eligible = positives.groupby("user_id", sort=False).filter(lambda frame: len(frame) >= 3)
    if eligible.empty:
        raise ValueError("Need at least three positive interactions per evaluated user.")
    test = eligible.groupby("user_id", sort=False).tail(1)
    remainder = eligible.drop(test.index)
    validation = remainder.groupby("user_id", sort=False).tail(1)
    train = remainder.drop(validation.index)
    return LeaveOneOutSplit(train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), int(test["user_id"].nunique()))


def build_shared_candidates(split: LeaveOneOutSplit, *, use_validation: bool = False, max_users: int = 1_000) -> tuple[list[EvaluationCase], int]:
    """Build identical full-catalog warm-item candidates for every baseline.

    Catalog is exactly the item set observed in training. Test/validation items not
    in this catalog are cold and are excluded from the warm-item ranking metric;
    their count is reported rather than silently removed.
    """
    holdout = split.validation if use_validation else split.test
    catalog = np.sort(split.train["item_id"].astype(int).unique())
    catalog_set = set(catalog.tolist())
    seen = {int(user): set(frame["item_id"].astype(int)) for user, frame in split.train.groupby("user_id")}
    cases: list[EvaluationCase] = []
    cold = 0
    for row in holdout.sort_values("user_id", kind="stable").itertuples(index=False):
        user, target = int(row.user_id), int(row.item_id)
        if target not in catalog_set:
            cold += 1
            continue
        candidates = catalog[~np.isin(catalog, list(seen.get(user, set())))]
        if target not in candidates:
            raise AssertionError("Held-out warm target was removed from its candidate set")
        cases.append(EvaluationCase(user, target, candidates))
        if len(cases) >= max_users:
            break
    return cases, cold


class PopularityRecommender:
    name = "popularity"

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        positive = interactions[interactions["response"] > 0]
        self.item_scores = positive.groupby("item_id").size().astype(float).to_dict()
        self.all_items = np.sort(interactions["item_id"].astype(int).unique())
        return self

    def score(self, user_id: int, items: np.ndarray) -> np.ndarray:
        return np.asarray([self.item_scores.get(int(item), 0.0) for item in items], dtype=float)


class BPRMFRecommender:
    """Vectorized NumPy BPR-MF fallback; PyTorch Adam model is preferred when available."""

    name = "bpr_mf_numpy"

    def __init__(self, factors: int = 64, learning_rate: float = 0.01, regularization: float = 1e-4, max_updates: int = 500_000, batch_size: int = 2_048, seed: int = 42):
        self.factors, self.learning_rate, self.regularization = factors, learning_rate, regularization
        self.max_updates, self.batch_size, self.seed = max_updates, batch_size, seed
        self.loss_history: list[dict[str, float]] = []
        self.validation_history: list[dict[str, float]] = []

    def fit(self, interactions: pd.DataFrame, validation: pd.DataFrame | None = None) -> "BPRMFRecommender":
        positives = interactions[interactions["response"] > 0][["user_id", "item_id"]].drop_duplicates()
        self.user_ids = np.sort(interactions["user_id"].astype(int).unique())
        self.item_ids = np.sort(interactions["item_id"].astype(int).unique())
        self.user_to_index = {u: k for k, u in enumerate(self.user_ids)}
        self.item_to_index = {i: k for k, i in enumerate(self.item_ids)}
        pairs = np.asarray([(self.user_to_index[int(r.user_id)], self.item_to_index[int(r.item_id)]) for r in positives.itertuples(index=False)], dtype=int)
        self.user_seen = {k: set() for k in range(len(self.user_ids))}
        for u, i in pairs: self.user_seen[int(u)].add(int(i))
        pairs = pairs[np.asarray([len(self.user_seen[int(u)]) < len(self.item_ids) for u, _ in pairs])]
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.05, (len(self.user_ids), self.factors))
        self.item_factors = rng.normal(0, 0.05, (len(self.item_ids), self.factors))
        if not len(pairs): return self
        updates = 0
        while updates < self.max_updates:
            n = min(self.batch_size, self.max_updates - updates)
            sample = pairs[rng.integers(len(pairs), size=n)]
            users, pos = sample[:, 0], sample[:, 1]
            neg = rng.integers(len(self.item_ids), size=n)
            invalid = np.asarray([j in self.user_seen[int(u)] for u, j in zip(users, neg)])
            while invalid.any():
                neg[invalid] = rng.integers(len(self.item_ids), size=int(invalid.sum()))
                invalid = np.asarray([j in self.user_seen[int(u)] for u, j in zip(users, neg)])
            pu, qi, qj = self.user_factors[users].copy(), self.item_factors[pos].copy(), self.item_factors[neg].copy()
            margin = np.einsum("ij,ij->i", pu, qi - qj)
            grad = 1 / (1 + np.exp(np.clip(margin, -30, 30)))
            np.add.at(self.user_factors, users, self.learning_rate * (grad[:, None] * (qi - qj) - self.regularization * pu))
            np.add.at(self.item_factors, pos, self.learning_rate * (grad[:, None] * pu - self.regularization * qi))
            np.add.at(self.item_factors, neg, self.learning_rate * (-grad[:, None] * pu - self.regularization * qj))
            updates += n
            self.loss_history.append({"updates": float(updates), "bpr_loss": float(np.mean(np.logaddexp(0, -margin)))})
        return self

    def score(self, user_id: int, items: np.ndarray) -> np.ndarray:
        u = self.user_to_index.get(int(user_id))
        if u is None: return np.zeros(len(items))
        idx = np.asarray([self.item_to_index.get(int(i), -1) for i in items])
        out = np.full(len(items), -np.inf)
        valid = idx >= 0
        out[valid] = self.item_factors[idx[valid]] @ self.user_factors[u]
        return out


def z_normalize(values: np.ndarray) -> np.ndarray:
    std = values.std()
    return np.zeros_like(values) if std < 1e-12 else (values - values.mean()) / std


class PopularityHybrid:
    """Validation-tuned blend of personalized and popularity scores."""

    def __init__(self, personalized, popularity: PopularityRecommender, alpha: float):
        self.personalized, self.popularity, self.alpha = personalized, popularity, alpha
        self.name = f"hybrid_alpha_{alpha:.2f}"

    def score(self, user_id: int, items: np.ndarray) -> np.ndarray:
        return self.alpha * z_normalize(self.personalized.score(user_id, items)) + (1 - self.alpha) * z_normalize(self.popularity.score(user_id, items))


def evaluate_cases(model, cases: list[EvaluationCase], cold_count: int, k: int = 10) -> RankingMetrics:
    # Sequential models can batch their user-history encodings once while still
    # receiving exactly the same candidate vector for every ranking case.
    prepare = getattr(model, "prepare_evaluation", None)
    finish = getattr(model, "finish_evaluation", None)
    if callable(prepare):
        prepare([case.user_id for case in cases])
    hits: list[float] = []; ndcgs: list[float] = []
    try:
        for case in cases:
            scores = model.score(case.user_id, case.candidate_items)
            order = np.lexsort((case.candidate_items, -scores))
            ranked = case.candidate_items[order[:k]]
            position = np.where(ranked == case.target_item)[0]
            if len(position):
                rank = int(position[0]) + 1; hits.append(1.0); ndcgs.append(1 / np.log2(rank + 1))
            else: hits.append(0.0); ndcgs.append(0.0)
    finally:
        if callable(finish):
            finish()
    total = len(cases) + cold_count
    return RankingMetrics(model.name, len(cases), len(cases) / total if total else 0.0, cold_count, float(np.mean(hits)) if hits else 0.0, float(np.mean(ndcgs)) if ndcgs else 0.0, float(np.mean(hits)) if hits else 0.0)


def evaluate_user_metrics(model, split: LeaveOneOutSplit, k: int = 10, max_users: int = 1_000, *, use_validation: bool = False) -> pd.DataFrame:
    """Export one warm-user row per model for paired external statistics."""
    cases, _ = build_shared_candidates(split, use_validation=use_validation, max_users=max_users)
    rows = []
    for case in cases:
        scores = model.score(case.user_id, case.candidate_items)
        order = np.lexsort((case.candidate_items, -scores))
        ranked = case.candidate_items[order[:k]]
        position = np.where(ranked == case.target_item)[0]
        hit = int(len(position) > 0)
        ndcg = float(1 / np.log2(int(position[0]) + 2)) if hit else 0.0
        rows.append({"user_id": case.user_id, "model": model.name, "hit": hit, "ndcg": ndcg})
    return pd.DataFrame(rows)


def evaluate_leave_one_out(model, split: LeaveOneOutSplit, k: int = 10, max_users: int = 1_000, *, use_validation: bool = False) -> RankingMetrics:
    cases, cold = build_shared_candidates(split, use_validation=use_validation, max_users=max_users)
    return evaluate_cases(model, cases, cold, k=k)
