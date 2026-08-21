from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def mask_seen(scores: np.ndarray, train_csr: sparse.csr_matrix,
              exclude_by_user: dict[int, int] | None = None) -> np.ndarray:
    out = scores.copy()
    for u in range(out.shape[0]):
        out[u, train_csr[u].indices] = -np.inf
        if exclude_by_user is not None and u in exclude_by_user:
            out[u, int(exclude_by_user[u])] = -np.inf
    return out


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.shape[1]:
        return np.argsort(-scores, axis=1)
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


def per_user_hit_ndcg(scores: np.ndarray, target_items: np.ndarray, train_csr: sparse.csr_matrix, ks=(5, 10, 20),
                      exclude_by_user: dict[int, int] | None = None) -> dict:
    masked = mask_seen(scores, train_csr, exclude_by_user)
    res = {}
    maxk = max(ks)
    recs = topk(masked, maxk)
    for k in ks:
        top = recs[:, :k]
        hits = (top == target_items[:, None])
        hit = hits.any(axis=1).astype(np.float32)
        # one relevant item; DCG = 1/log2(rank+1) if hit else 0, IDCG=1
        ranks = np.argmax(hits, axis=1) + 1
        ndcg = np.where(hit > 0, 1.0 / np.log2(ranks + 1), 0.0).astype(np.float32)
        res[f"HitRate@{k}"] = hit
        res[f"NDCG@{k}"] = ndcg
    return res


def summarize_metric_dict(per_user: dict) -> dict:
    return {k: float(np.mean(v)) for k, v in per_user.items()}


def catalogue_coverage(recs: np.ndarray, n_items: int) -> float:
    return float(np.unique(recs).size / max(1, n_items))


def ild(recs: np.ndarray, item_vectors: sparse.csr_matrix) -> np.ndarray:
    vals = []
    X = item_vectors.tocsr()
    for row in recs:
        k = len(row)
        if k < 2:
            vals.append(0.0)
            continue
        V = X[row]
        sim = (V @ V.T).toarray()
        upper = sim[np.triu_indices(k, 1)]
        vals.append(float(np.mean(1.0 - upper)))
    return np.asarray(vals, dtype=np.float32)


def evaluate(scores: np.ndarray, split, item_vectors: sparse.csr_matrix, ks=(5, 10, 20),
             exclude_by_user: dict[int, int] | None = None) -> tuple[dict, dict]:
    train_csr = split.train_csr
    targets = split.test.sort_values("user").item.values.astype(np.int64)
    per_user = per_user_hit_ndcg(scores, targets, train_csr, ks=ks, exclude_by_user=exclude_by_user)
    summary = summarize_metric_dict(per_user)
    masked = mask_seen(scores, train_csr, exclude_by_user)
    recs20 = topk(masked, max(ks))
    summary[f"Coverage@{max(ks)}"] = catalogue_coverage(recs20, split.n_items)
    summary[f"ILD@{max(ks)}"] = float(np.mean(ild(recs20, item_vectors)))
    return summary, per_user
