from __future__ import annotations

import numpy as np
from scipy import sparse

from .metrics import per_user_hit_ndcg


def top_attributed_items(train_items: np.ndarray, weights: np.ndarray, fraction: float = 0.2, positive_only: bool = True) -> np.ndarray:
    if len(train_items) == 0:
        return train_items
    w = np.asarray(weights, dtype=np.float32)
    if positive_only:
        order = np.lexsort((np.arange(len(w)), -w))
    else:
        order = np.lexsort((np.arange(len(w)), -np.abs(w)))
    k = max(1, int(np.ceil(len(train_items) * fraction)))
    return train_items[order[:k]]


def deletion_comprehensiveness(
    base_scores: np.ndarray,
    split,
    attribution_by_user: dict[int, np.ndarray],
    fraction: float = 0.2,
    ks=(20,),
) -> dict[str, float]:
    """Remove top-attributed historical items from the seen mask and measure rank degradation proxy.

    This is a lightweight diagnostic for the prototype. In the HCCF adapter, this
    should be replaced by true masked-forward deletion scores.
    """
    train_csr = split.train_csr.copy().tolil()
    for u, weights in attribution_by_user.items():
        items = split.train_csr[int(u)].indices
        rem = top_attributed_items(items, weights, fraction=fraction)
        if len(rem):
            train_csr[int(u), rem] = 0
    train_csr = train_csr.tocsr()
    targets = split.test.sort_values("user").item.values.astype(np.int64)
    changed = per_user_hit_ndcg(base_scores, targets, train_csr, ks=ks)
    original = per_user_hit_ndcg(base_scores, targets, split.train_csr, ks=ks)
    return {f"DeletionDelta_{m}": float(np.mean(original[m] - changed[m])) for m in original}


def insertion_sufficiency(
    base_scores: np.ndarray,
    split,
    attribution_by_user: dict[int, np.ndarray],
    fraction: float = 0.2,
    ks=(20,),
) -> dict[str, float]:
    """Keep only top-attributed historical items in the seen mask and report metric proxy.

    This prototype diagnostic uses candidate masking as a cheap sufficiency proxy.
    The HCCF adapter should implement true masked-forward sufficiency.
    """
    train_csr = sparse.lil_matrix(split.train_csr.shape, dtype=split.train_csr.dtype)
    for u, weights in attribution_by_user.items():
        items = split.train_csr[int(u)].indices
        keep = top_attributed_items(items, weights, fraction=fraction)
        if len(keep):
            train_csr[int(u), keep] = 1
    train_csr = train_csr.tocsr()
    targets = split.test.sort_values("user").item.values.astype(np.int64)
    vals = per_user_hit_ndcg(base_scores, targets, train_csr, ks=ks)
    return {f"Insertion_{m}": float(np.mean(v)) for m, v in vals.items()}


def attribution_concentration(attribution_by_user: dict[int, np.ndarray]) -> dict[str, float]:
    ents = []
    top20_mass = []
    for w in attribution_by_user.values():
        a = np.abs(np.asarray(w, dtype=np.float64))
        total = a.sum()
        if total <= 0:
            continue
        p = a / total
        ents.append(float(-(p * np.log(p + 1e-12)).sum()))
        k = max(1, int(np.ceil(0.2 * len(a))))
        top20_mass.append(float(np.sort(a)[-k:].sum() / total))
    return {
        "AttributionEntropy_mean": float(np.mean(ents)) if ents else 0.0,
        "AttributionTop20Mass_mean": float(np.mean(top20_mass)) if top20_mass else 0.0,
    }
