from __future__ import annotations

import numpy as np
from scipy import sparse


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / (sd + 1e-12)).astype(np.float32)


def user_profile(train_items: np.ndarray, item_vectors: sparse.csr_matrix) -> sparse.csr_matrix:
    if len(train_items) == 0:
        return sparse.csr_matrix((1, item_vectors.shape[1]), dtype=np.float32)
    h = item_vectors[train_items].mean(axis=0)
    h = sparse.csr_matrix(h)
    norm = np.sqrt(h.multiply(h).sum())
    return h if norm == 0 else h / norm


def sim_user_items(train_items: np.ndarray, item_vectors: sparse.csr_matrix) -> np.ndarray:
    prof = user_profile(train_items, item_vectors)
    return np.asarray(prof @ item_vectors[train_items].T).ravel().astype(np.float32)


def family_weights(family: str, train_items: np.ndarray, item_vectors: sparse.csr_matrix, item_degree: np.ndarray, shapley: np.ndarray | None = None, tau_att: float = 0.1) -> np.ndarray:
    n = len(train_items)
    if family == "uniform":
        return np.ones(n, dtype=np.float32)
    if family == "additive-pref":
        return np.maximum(0.0, sim_user_items(train_items, item_vectors)).astype(np.float32)
    if family == "attention":
        s = sim_user_items(train_items, item_vectors) / tau_att
        s = s - np.max(s) if n else s
        e = np.exp(s)
        return (e / max(e.sum(), 1e-12)).astype(np.float32)
    if family == "heuristic-pop":
        vals = np.log1p(item_degree[train_items]).astype(np.float32)
        m = float(vals.max()) if n else 0.0
        return vals / m if m > 0 else np.zeros(n, dtype=np.float32)
    if family == "shapley-mc":
        if shapley is None:
            raise ValueError("shapley weights required for shapley-mc")
        return shapley.astype(np.float32)
    raise ValueError(f"unknown family {family}")


def attribution_adjustment(train_items: np.ndarray, raw_weights: np.ndarray, item_vectors: sparse.csr_matrix, candidate_items: np.ndarray) -> np.ndarray:
    denom = float(np.sum(np.abs(raw_weights)) + 1e-12)
    if len(train_items) == 0 or denom == 0:
        return np.zeros(len(candidate_items), dtype=np.float32)
    # h = sum_j w_j x_j, candidate adjustment = x_i h / sum |w|.
    # Use sparse @ dense operations to avoid constructing dense item-item kernels.
    h = item_vectors[train_items].T.dot(raw_weights.astype(np.float32))
    vals = item_vectors[candidate_items].dot(h) / denom
    return np.asarray(vals).ravel().astype(np.float32)


def rerank_user_scores(base_scores_u: np.ndarray, train_items: np.ndarray, raw_weights: np.ndarray, item_vectors: sparse.csr_matrix, lambda_attr: float = 0.10) -> np.ndarray:
    candidates = np.arange(base_scores_u.shape[0], dtype=np.int64)
    adj = attribution_adjustment(train_items, raw_weights, item_vectors, candidates)
    return zscore(base_scores_u) + lambda_attr * zscore(adj)


def rerank_all(base_scores: np.ndarray, split, item_vectors: sparse.csr_matrix, family: str, shapley_by_user: dict[int, np.ndarray] | None = None, lambda_attr: float = 0.10) -> np.ndarray:
    train_csr = split.train_csr
    item_degree = np.asarray(train_csr.sum(axis=0)).ravel()
    out = np.empty_like(base_scores, dtype=np.float32)
    for u in range(split.n_users):
        items = train_csr[u].indices
        shap = None if shapley_by_user is None else shapley_by_user.get(u)
        w = family_weights(family, items, item_vectors, item_degree, shapley=shap)
        out[u] = rerank_user_scores(base_scores[u], items, w, item_vectors, lambda_attr=lambda_attr)
    return out
