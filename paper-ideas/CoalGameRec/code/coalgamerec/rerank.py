from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .utils import as_1d_float, stable_zscore


@dataclass(frozen=True)
class RerankConfig:
    """Configuration for the common post-hoc reranking operator."""

    lambda_attr: float = 0.10
    tau_att: float = 0.10
    eps: float = 1e-12


def zscore(x: np.ndarray) -> np.ndarray:
    """Backward-compatible alias used by the notebook and attribution module."""
    return stable_zscore(x)


def user_profile(train_items: np.ndarray, item_vectors: sparse.csr_matrix) -> sparse.csr_matrix:
    """Normalized mean of a user's train-only item vectors.

    Returns a 1 x n_users sparse row vector. The representation uses only the
    training graph, matching the leakage-control protocol.
    """
    if len(train_items) == 0:
        return sparse.csr_matrix((1, item_vectors.shape[1]), dtype=np.float32)
    h = item_vectors[train_items].mean(axis=0)
    h = sparse.csr_matrix(h, dtype=np.float32)
    norm = float(np.sqrt(h.multiply(h).sum()))
    return h if norm == 0.0 else h / norm


def sim_user_items(train_items: np.ndarray, item_vectors: sparse.csr_matrix) -> np.ndarray:
    """Cosine similarity between each historical item and the user's train profile."""
    if len(train_items) == 0:
        return np.zeros(0, dtype=np.float32)
    prof = user_profile(train_items, item_vectors)
    sims = prof.dot(item_vectors[train_items].T)
    return as_1d_float(sims)


def family_weights(
    family: str,
    train_items: np.ndarray,
    item_vectors: sparse.csr_matrix,
    item_degree: np.ndarray,
    shapley: np.ndarray | None = None,
    tau_att: float = 0.1,
) -> np.ndarray:
    """Executable family-specific historical-interaction weights.

    These correspond to the protocol definitions in Implementation_Spec §A.5a.
    """
    n = len(train_items)
    if family == "uniform":
        return np.ones(n, dtype=np.float32)
    if family == "additive-pref":
        return np.maximum(0.0, sim_user_items(train_items, item_vectors)).astype(np.float32)
    if family == "attention":
        s = sim_user_items(train_items, item_vectors) / float(tau_att)
        if n == 0:
            return s.astype(np.float32)
        s = s - np.max(s)
        e = np.exp(s)
        return (e / max(float(e.sum()), 1e-12)).astype(np.float32)
    if family == "heuristic-pop":
        vals = np.log1p(item_degree[train_items]).astype(np.float32)
        m = float(vals.max()) if n else 0.0
        return vals / m if m > 0 else np.zeros(n, dtype=np.float32)
    if family == "shapley-mc":
        if shapley is None:
            raise ValueError("shapley weights required for shapley-mc")
        if len(shapley) != n:
            raise ValueError(f"shapley length {len(shapley)} does not match user history length {n}")
        return shapley.astype(np.float32)
    raise ValueError(f"unknown family {family}")


def attribution_adjustment(
    train_items: np.ndarray,
    raw_weights: np.ndarray,
    item_vectors: sparse.csr_matrix,
    candidate_items: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute candidate-level kernel adjustment without dense item-item matrices."""
    raw_weights = np.asarray(raw_weights, dtype=np.float32)
    denom = float(np.sum(np.abs(raw_weights)) + eps)
    if len(train_items) == 0 or denom <= eps:
        return np.zeros(len(candidate_items), dtype=np.float32)
    # h = sum_j w_j x_j, candidate adjustment = x_i h / sum |w|.
    h = item_vectors[train_items].T.dot(raw_weights)
    vals = item_vectors[candidate_items].dot(h) / denom
    return as_1d_float(vals)


def rerank_user_scores(
    base_scores_u: np.ndarray,
    train_items: np.ndarray,
    raw_weights: np.ndarray,
    item_vectors: sparse.csr_matrix,
    lambda_attr: float = 0.10,
) -> np.ndarray:
    candidates = np.arange(base_scores_u.shape[0], dtype=np.int64)
    adj = attribution_adjustment(train_items, raw_weights, item_vectors, candidates)
    return stable_zscore(base_scores_u) + float(lambda_attr) * stable_zscore(adj)


def rerank_all(
    base_scores: np.ndarray,
    split,
    item_vectors: sparse.csr_matrix,
    family: str,
    shapley_by_user: dict[int, np.ndarray] | None = None,
    lambda_attr: float = 0.10,
    tau_att: float = 0.10,
) -> np.ndarray:
    train_csr = split.train_csr
    item_degree = np.asarray(train_csr.sum(axis=0)).ravel()
    out = np.empty_like(base_scores, dtype=np.float32)
    for u in range(split.n_users):
        items = train_csr[u].indices
        shap = None if shapley_by_user is None else shapley_by_user.get(u)
        if family == "shapley-mc" and shap is None:
            # In quick notebook runs, only a subset of users may have Shapley.
            shap = np.zeros(len(items), dtype=np.float32)
        w = family_weights(family, items, item_vectors, item_degree, shapley=shap, tau_att=tau_att)
        out[u] = rerank_user_scores(base_scores[u], items, w, item_vectors, lambda_attr=lambda_attr)
    return out
