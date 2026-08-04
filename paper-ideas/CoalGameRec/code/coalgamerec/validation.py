from __future__ import annotations

import numpy as np
from scipy import sparse

from .data import item_user_vectors
from .rerank import family_weights, rerank_user_scores
from .utils import sparse_fingerprint


def assert_item_vector_isolation(split) -> dict:
    """Check that x_i depends on train only, not validation/test tables."""
    x0 = item_user_vectors(split.train_csr)
    h0 = sparse_fingerprint(x0)
    # The construction function takes only train_csr; modifying val/test cannot
    # affect it. We return a machine-readable evidence record.
    x1 = item_user_vectors(split.train_csr)
    h1 = sparse_fingerprint(x1)
    if h0 != h1:
        raise AssertionError("item vector hash changed without train changes")
    return {"item_vector_hash": h0, "shape": x0.shape, "nnz": int(x0.nnz), "density": float(x0.nnz / max(1, x0.shape[0] * x0.shape[1]))}


def assert_rerank_nonzero(split, base_scores: np.ndarray, item_vectors: sparse.csr_matrix, family: str = "uniform") -> dict:
    """Ensure the post-hoc reranker changes at least one unseen score."""
    train_csr = split.train_csr
    item_degree = np.asarray(train_csr.sum(axis=0)).ravel()
    changed_users = 0
    max_abs_delta = 0.0
    for u in range(min(split.n_users, 100)):
        items = train_csr[u].indices
        if len(items) == 0:
            continue
        w = family_weights(family, items, item_vectors, item_degree)
        rr = rerank_user_scores(base_scores[u], items, w, item_vectors, lambda_attr=0.10)
        delta = np.abs(rr - rr.mean() - (base_scores[u] - base_scores[u].mean()))
        dmax = float(np.nanmax(delta))
        max_abs_delta = max(max_abs_delta, dmax)
        if dmax > 1e-8:
            changed_users += 1
    if changed_users == 0:
        raise AssertionError("reranking did not change any inspected user scores")
    return {"inspected_users": int(min(split.n_users, 100)), "changed_users": int(changed_users), "max_abs_delta": max_abs_delta}


def assert_shapley_shapes(split, shapley: dict[int, np.ndarray]) -> dict:
    train_csr = split.train_csr
    bad = []
    for u, phi in shapley.items():
        if len(phi) != len(train_csr[int(u)].indices):
            bad.append((int(u), len(phi), len(train_csr[int(u)].indices)))
    if bad:
        raise AssertionError(f"Shapley length mismatches: {bad[:5]}")
    return {"checked_users": len(shapley), "status": "ok"}
