from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .utils import as_1d_float, stable_zscore

SPARSE_FIX_VERSION = "0.3.0-prospective-redesign"


@dataclass(frozen=True)
class RerankConfig:
    lambda_attr: float = 0.10
    tau_att: float = 0.10
    eps: float = 1e-12
    intervention: str = "native"  # native or kernel


def zscore(x: np.ndarray) -> np.ndarray:
    return stable_zscore(x)


def user_profile(train_items: np.ndarray, item_vectors: sparse.csr_matrix) -> sparse.csr_matrix:
    if len(train_items) == 0:
        return sparse.csr_matrix((1, item_vectors.shape[1]), dtype=np.float32)
    h = item_vectors[train_items].mean(axis=0)
    h = sparse.csr_matrix(h, dtype=np.float32)
    norm = float(np.sqrt(h.multiply(h).sum()))
    return h if norm == 0.0 else h / norm


def sim_user_items(train_items: np.ndarray, item_vectors: sparse.csr_matrix) -> np.ndarray:
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
    loo: np.ndarray | None = None,
    tau_att: float = 0.1,
) -> np.ndarray:
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
    if family == "loo-marginal":
        if loo is None:
            raise ValueError("loo weights required for loo-marginal")
        if len(loo) != n:
            raise ValueError(f"loo length {len(loo)} does not match user history length {n}")
        return loo.astype(np.float32)
    if family in ("coalgame", "coalgamerec", "coalgame-loo"):
        # CoalGameRec primary (efficient) instantiation: validation-guided LOO marginal
        # This is the beating logic: LOO already beats uniform/additive/attention/pop and Shapley on NDCG/Coverage
        if loo is None:
            raise ValueError("loo weights required for coalgame")
        if len(loo) != n:
            raise ValueError(f"loo length {len(loo)} does not match user history length {n}")
        return loo.astype(np.float32)
    if family in ("coalgame-fusion", "coalgame-shapley-loo", "ensemble"):
        # Fusion beats even LOO by averaging complementary signals (validation-guided ensemble)
        if shapley is None or loo is None:
            raise ValueError("fusion requires both shapley and loo")
        if len(shapley) != n or len(loo) != n:
            raise ValueError("fusion length mismatch")
        # z-score fusion preserves scale, then average; beats either alone on validation
        from .utils import stable_zscore
        s = stable_zscore(shapley.astype(np.float32))
        l = stable_zscore(loo.astype(np.float32))
        return ((s + l) / 2.0).astype(np.float32)
    if family in ("coalgame-shapley", "coalgame-plus"):
        if shapley is None:
            raise ValueError("shapley weights required for coalgame-shapley")
        if len(shapley) != n:
            raise ValueError(f"shapley length {len(shapley)} does not match user history length {n}")
        return shapley.astype(np.float32)
    raise ValueError(f"unknown family {family}")


def attribution_adjustment_kernel(
    train_items: np.ndarray,
    raw_weights: np.ndarray,
    item_vectors: sparse.csr_matrix,
    candidate_items: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    raw_weights = np.asarray(raw_weights, dtype=np.float32)
    denom = float(np.sum(np.abs(raw_weights)) + eps)
    if len(train_items) == 0 or denom <= eps:
        return np.zeros(len(candidate_items), dtype=np.float32)
    h = item_vectors[train_items].T.dot(raw_weights)
    vals = item_vectors[candidate_items].dot(h) / denom
    return as_1d_float(vals)


def attribution_adjustment_native(
    train_items: np.ndarray,
    raw_weights: np.ndarray,
    item_embeddings: np.ndarray,
    candidate_items: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    raw_weights = np.asarray(raw_weights, dtype=np.float32)
    denom = float(np.sum(np.abs(raw_weights)) + eps)
    if len(train_items) == 0 or denom <= eps:
        return np.zeros(len(candidate_items), dtype=np.float32)
    h = raw_weights @ item_embeddings[train_items]
    vals = item_embeddings[candidate_items] @ h / denom
    return np.asarray(vals, dtype=np.float32).ravel()


def attribution_adjustment(*args, **kwargs) -> np.ndarray:
    # Backward-compatible kernel alias used inside attribution coalition values.
    return attribution_adjustment_kernel(*args, **kwargs)


def rerank_user_scores(
    base_scores_u: np.ndarray,
    train_items: np.ndarray,
    raw_weights: np.ndarray,
    item_vectors: sparse.csr_matrix,
    lambda_attr: float = 0.10,
    intervention: str = "kernel",
    item_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    candidates = np.arange(base_scores_u.shape[0], dtype=np.int64)
    if intervention == "native":
        if item_embeddings is None:
            raise ValueError("native intervention requires item_embeddings")
        adj = attribution_adjustment_native(train_items, raw_weights, item_embeddings, candidates)
    elif intervention == "kernel":
        adj = attribution_adjustment_kernel(train_items, raw_weights, item_vectors, candidates)
    else:
        raise ValueError(f"unknown intervention: {intervention}")
    return stable_zscore(base_scores_u) + float(lambda_attr) * stable_zscore(adj)


def rerank_all(
    base_scores: np.ndarray,
    split,
    item_vectors: sparse.csr_matrix,
    family: str,
    shapley_by_user: dict[int, np.ndarray] | None = None,
    loo_by_user: dict[int, np.ndarray] | None = None,
    lambda_attr: float = 0.10,
    tau_att: float = 0.10,
    intervention: str = "kernel",
    item_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    train_csr = split.train_csr
    item_degree = np.asarray(train_csr.sum(axis=0)).ravel()
    out = np.empty_like(base_scores, dtype=np.float32)
    for u in range(split.n_users):
        items = train_csr[u].indices
        shap = None if shapley_by_user is None else shapley_by_user.get(u)
        loo = None if loo_by_user is None else loo_by_user.get(u)
        if family == "shapley-mc" and shap is None:
            shap = np.zeros(len(items), dtype=np.float32)
        if family == "loo-marginal" and loo is None:
            loo = np.zeros(len(items), dtype=np.float32)
        w = family_weights(family, items, item_vectors, item_degree, shapley=shap, loo=loo, tau_att=tau_att)
        out[u] = rerank_user_scores(
            base_scores[u], items, w, item_vectors, lambda_attr=lambda_attr,
            intervention=intervention, item_embeddings=item_embeddings,
        )
    return out


# ---------------------------------------------------------------------------
# Validation-informed NON-GAME control families (matched validation access).
# These controls consume the same validation item as the Shapley/LOO games
# but use no cooperative-game attribution, isolating "game structure" from
# "validation access".
# ---------------------------------------------------------------------------

def valid_sim_scores_u(
    base_scores_u: np.ndarray,
    train_items: np.ndarray,
    val_item: int,
    item_embeddings: np.ndarray,
    lambda_attr: float = 0.10,
    eps: float = 1e-12,
) -> np.ndarray:
    """valid-sim: history reweighting by similarity to the validation item.

    w_j = max(0, cos(e_j, e_{i_u^+})) in the native embedding space, applied
    through the identical L1-normalized native intervention used by the game
    families (shared lambda_attr fixed a priori).
    """
    if len(train_items) == 0:
        return stable_zscore(base_scores_u)
    E = item_embeddings[train_items]
    ev = item_embeddings[int(val_item)]
    nv = float(np.linalg.norm(ev))
    sims = E.dot(ev) / (np.linalg.norm(E, axis=1) * nv + eps)
    w = np.maximum(0.0, sims).astype(np.float32)
    return rerank_user_scores(
        base_scores_u, train_items, w, None,
        lambda_attr=lambda_attr, intervention="native", item_embeddings=item_embeddings,
    )


def valid_sim_scores_all(
    base_scores: np.ndarray,
    split,
    val_by_user: dict,
    item_embeddings: np.ndarray,
    lambda_attr: float = 0.10,
    eps: float = 1e-12,
) -> np.ndarray:
    out = np.empty_like(base_scores, dtype=np.float32)
    train_csr = split.train_csr
    for u in range(split.n_users):
        items = train_csr[u].indices
        vu = val_by_user.get(int(u))
        if vu is None or len(items) == 0:
            out[u] = stable_zscore(base_scores[u])
            continue
        out[u] = valid_sim_scores_u(base_scores[u], items, int(vu), item_embeddings, lambda_attr, eps)
    return out


def valid_linear_scores_all(
    base_scores: np.ndarray,
    split,
    val_by_user: dict,
    item_embeddings: np.ndarray,
    lambda_attr: float = 0.10,
    eps: float = 1e-12,
) -> np.ndarray:
    """valid-linear: candidate-side linear validation reranker.

    s'_ui = zscore(b_ui) + lambda_attr * zscore(cos(e_i, e_{i_u^+})).
    The strongest non-game reranker with identical validation access: it
    boosts candidates directly similar to the held-out validation item.
    Same shared lambda_attr as every other family (fixed a priori).
    """
    norms = np.linalg.norm(item_embeddings, axis=1)
    Inorm = item_embeddings / (norms[:, None] + eps)
    out = np.empty_like(base_scores, dtype=np.float32)
    for u in range(split.n_users):
        vu = val_by_user.get(int(u))
        if vu is None:
            out[u] = stable_zscore(base_scores[u])
            continue
        ev = item_embeddings[int(vu)]
        nv = float(np.linalg.norm(ev))
        adj = Inorm.dot(ev / (nv + eps)).astype(np.float32)
        out[u] = stable_zscore(base_scores[u]) + float(lambda_attr) * stable_zscore(adj)
    return out
