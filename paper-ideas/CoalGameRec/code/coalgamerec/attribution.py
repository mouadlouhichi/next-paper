from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm.auto import tqdm

from .metrics import topk
from .rerank import attribution_adjustment, zscore, sim_user_items
from .utils import as_1d_float


@dataclass(frozen=True)
class ShapleyConfig:
    alpha: float = 1.0
    beta: float = 0.0
    lambda_pref: float = 0.0
    lambda_attr_value: float = 0.10
    value_mode: str = "pairwise_logsigmoid"  # pairwise_logsigmoid, ndcg_ild
    n_val_negatives: int = 100
    m: int = 128
    exact_threshold: int = 8
    seed: int = 42
    max_players_per_user: int | None = 24
    player_selection: str = "stratified"  # stratified, similarity, val_similarity, diverse, random, recent, first


_COALITION_KWARGS = {
    "alpha", "beta", "lambda_pref", "lambda_attr_value",
    "value_mode", "val_negatives",
}


def _coalition_kwargs(kwargs: dict) -> dict:
    """Keep only arguments accepted by coalition_value.

    Estimator-level options such as ``antithetic`` or ``m`` may be passed through
    the public compute function. Exact Shapley also calls coalition_value, so we
    must not forward estimator-only options into the value function.
    """
    return {k: v for k, v in kwargs.items() if k in _COALITION_KWARGS}


def _logsigmoid(x: np.ndarray) -> np.ndarray:
    # stable log(sigmoid(x)) = -softplus(-x)
    return -np.logaddexp(0.0, -x)


def _ndcg_single(scores: np.ndarray, target: int, train_items: np.ndarray, k: int = 20) -> float:
    s = scores.copy()
    s[train_items] = -np.inf
    recs = topk(s[None, :], k)[0]
    loc = np.where(recs == target)[0]
    if len(loc) == 0:
        return 0.0
    return float(1.0 / np.log2(int(loc[0]) + 2))


def _diversity(scores: np.ndarray, train_items: np.ndarray, item_vectors: sparse.csr_matrix, k: int = 20) -> float:
    s = scores.copy()
    s[train_items] = -np.inf
    recs = topk(s[None, :], k)[0]
    if len(recs) < 2:
        return 0.0
    sim = (item_vectors[recs] @ item_vectors[recs].T).toarray()
    return float(np.mean(1.0 - sim[np.triu_indices(len(recs), 1)]))


def _cos_to_item(train_items: np.ndarray, item_vectors: sparse.csr_matrix, item: int) -> np.ndarray:
    if len(train_items) == 0:
        return np.zeros(0, dtype=np.float32)
    return as_1d_float(item_vectors[train_items].dot(item_vectors[item].T))


def _diverse_greedy(train_items: np.ndarray, item_vectors: sparse.csr_matrix, k: int, forbidden: set[int] | None = None) -> list[int]:
    if forbidden is None:
        forbidden = set()
    candidates = [i for i in range(len(train_items)) if i not in forbidden]
    if not candidates or k <= 0:
        return []
    # Start with the item closest to the user's profile, then maximize minimum distance.
    sims_profile = sim_user_items(train_items, item_vectors)
    first = max(candidates, key=lambda idx: (sims_profile[idx], -idx))
    chosen = [first]
    while len(chosen) < k and len(chosen) < len(candidates):
        chosen_items = train_items[np.array(chosen, dtype=np.int64)]
        best = None
        best_score = -np.inf
        for idx in candidates:
            if idx in chosen:
                continue
            sim_to_chosen = as_1d_float(item_vectors[train_items[idx]].dot(item_vectors[chosen_items].T))
            min_dist = float(np.min(1.0 - sim_to_chosen)) if len(sim_to_chosen) else 0.0
            score = (min_dist, sims_profile[idx], -idx)
            if best is None or score > best[0]:
                best = (score, idx)
        if best is None:
            break
        chosen.append(best[1])
    return chosen[:k]


def select_players(
    train_items: np.ndarray,
    item_vectors: sparse.csr_matrix,
    max_players: int | None,
    strategy: str = "stratified",
    val_target: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Return indices into train_items selected for the Shapley/LOO game.

    The recommended laptop-feasible prospective design is `strategy='stratified'`:
    one third profile-similar interactions, one third validation-similar interactions,
    and one third preference-diverse interactions. This avoids the previous pure
    similarity preselection that advantaged the attention baseline.
    """
    n = len(train_items)
    if max_players is None or n <= int(max_players):
        return np.arange(n, dtype=np.int64)
    k = int(max_players)
    rng = np.random.default_rng(seed)
    if strategy == "first":
        return np.arange(k, dtype=np.int64)
    if strategy == "recent":
        # If train_items are supplied in temporal order this selects the latest k;
        # otherwise it is a deterministic suffix fallback.
        return np.arange(n - k, n, dtype=np.int64)
    if strategy == "random":
        return np.sort(rng.choice(n, size=k, replace=False).astype(np.int64))
    if strategy == "similarity":
        sims = sim_user_items(train_items, item_vectors)
        order = np.lexsort((np.arange(n), -sims))
        return np.sort(order[:k].astype(np.int64))
    if strategy == "val_similarity":
        if val_target is None:
            return select_players(train_items, item_vectors, k, "similarity", seed=seed)
        sims = _cos_to_item(train_items, item_vectors, val_target)
        order = np.lexsort((np.arange(n), -sims))
        return np.sort(order[:k].astype(np.int64))
    if strategy == "diverse":
        return np.sort(np.array(_diverse_greedy(train_items, item_vectors, k), dtype=np.int64))
    if strategy == "stratified":
        k1 = k // 3
        k2 = k // 3
        k3 = k - k1 - k2
        chosen: list[int] = []
        # profile-similar
        sims = sim_user_items(train_items, item_vectors)
        for idx in np.lexsort((np.arange(n), -sims)):
            if idx not in chosen:
                chosen.append(int(idx))
            if len(chosen) >= k1:
                break
        # validation-similar cheap screening
        if val_target is not None:
            vals = _cos_to_item(train_items, item_vectors, val_target)
            for idx in np.lexsort((np.arange(n), -vals)):
                if idx not in chosen:
                    chosen.append(int(idx))
                if len(chosen) >= k1 + k2:
                    break
        # preference-diverse fill
        forbidden = set(chosen)
        chosen.extend(_diverse_greedy(train_items, item_vectors, k3, forbidden=forbidden))
        # deterministic fallback if overlap left holes
        if len(chosen) < k:
            for idx in rng.permutation(n):
                if int(idx) not in chosen:
                    chosen.append(int(idx))
                if len(chosen) >= k:
                    break
        return np.sort(np.array(chosen[:k], dtype=np.int64))
    raise ValueError(f"unknown player_selection strategy: {strategy}")


def sample_validation_negatives(n_items: int, train_items: np.ndarray, val_target: int, n_negatives: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocked = set(map(int, train_items)) | {int(val_target)}
    pool = np.array([i for i in range(n_items) if i not in blocked], dtype=np.int64)
    if len(pool) == 0:
        return np.zeros(0, dtype=np.int64)
    size = min(int(n_negatives), len(pool))
    return np.sort(rng.choice(pool, size=size, replace=False).astype(np.int64))


def coalition_scores(base_scores_u, train_items, coalition_idx, item_vectors, lambda_attr_value):
    if len(coalition_idx) == 0:
        adj_all = np.zeros_like(base_scores_u, dtype=np.float32)
    else:
        coalition_items = train_items[coalition_idx]
        raw_w = np.ones(len(coalition_items), dtype=np.float32)
        candidates = np.arange(base_scores_u.shape[0], dtype=np.int64)
        adj_all = attribution_adjustment(coalition_items, raw_w, item_vectors, candidates)
    return zscore(base_scores_u) + float(lambda_attr_value) * zscore(adj_all)


def coalition_value(
    base_scores_u: np.ndarray,
    train_items: np.ndarray,
    coalition_idx: np.ndarray,
    val_target: int,
    item_vectors: sparse.csr_matrix,
    alpha: float = 1.0,
    beta: float = 0.0,
    lambda_pref: float = 0.0,
    lambda_attr_value: float = 0.10,
    pref_sims: np.ndarray | None = None,
    value_mode: str = "pairwise_logsigmoid",
    val_negatives: np.ndarray | None = None,
) -> float:
    """Prospective primary value: smooth validation pairwise utility.

    The default removes the additive similarity degeneracy (`lambda_pref=0`) and
    avoids discontinuous single-positive NDCG as the attribution utility. NDCG and
    HitRate remain final test metrics, not the primary coalition value.
    """
    scores = coalition_scores(base_scores_u, train_items, coalition_idx, item_vectors, lambda_attr_value)
    if value_mode == "pairwise_logsigmoid":
        if val_negatives is None or len(val_negatives) == 0:
            rank_util = 0.0
        else:
            diffs = scores[int(val_target)] - scores[val_negatives]
            rank_util = float(np.mean(_logsigmoid(diffs)))
        value = rank_util
    elif value_mode == "ndcg_ild":
        ndcg = _ndcg_single(scores, val_target, train_items, k=20)
        div = _diversity(scores, train_items, item_vectors, k=20)
        value = float(alpha * ndcg + beta * div)
    else:
        raise ValueError(f"unknown value_mode: {value_mode}")
    if lambda_pref:
        if pref_sims is None:
            pref_sims = sim_user_items(train_items, item_vectors)
        value += float(lambda_pref) * float(np.sum(pref_sims[coalition_idx])) if len(coalition_idx) else 0.0
    return float(value)


def exact_shapley(base_scores_u, train_items, val_target, item_vectors, **kwargs) -> np.ndarray:
    kwargs = _coalition_kwargs(kwargs)
    n = len(train_items)
    phi = np.zeros(n, dtype=np.float32)
    players = np.arange(n)
    fact = math.factorial
    nfact = fact(n)
    cache: dict[tuple[int, ...], float] = {}
    pref_sims = sim_user_items(train_items, item_vectors)

    def v(tup):
        tup = tuple(sorted(tup))
        if tup not in cache:
            cache[tup] = coalition_value(base_scores_u, train_items, np.array(tup, dtype=np.int64), val_target, item_vectors, pref_sims=pref_sims, **kwargs)
        return cache[tup]

    for p in players:
        others = [o for o in players if o != p]
        for r in range(n):
            for S in combinations(others, r):
                w = fact(r) * fact(n - r - 1) / nfact
                phi[p] += w * (v((*S, p)) - v(S))
    return phi


def permutation_shapley(base_scores_u, train_items, val_target, item_vectors, m: int = 128, seed: int = 42, antithetic: bool = True, **kwargs) -> np.ndarray:
    kwargs = _coalition_kwargs(kwargs)
    rng = np.random.default_rng(seed)
    n = len(train_items)
    phi = np.zeros(n, dtype=np.float64)
    cache: dict[tuple[int, ...], float] = {}
    pref_sims = sim_user_items(train_items, item_vectors)

    def v(prefix):
        tup = tuple(sorted(prefix))
        if tup not in cache:
            cache[tup] = coalition_value(base_scores_u, train_items, np.array(tup, dtype=np.int64), val_target, item_vectors, pref_sims=pref_sims, **kwargs)
        return cache[tup]

    draws = 0
    while draws < m:
        perm = rng.permutation(n)
        perms = [perm]
        if antithetic and draws + 1 < m:
            perms.append(perm[::-1])
        for pp in perms:
            S: list[int] = []
            prev = v(S)
            for p in pp:
                S.append(int(p))
                cur = v(S)
                phi[p] += cur - prev
                prev = cur
            draws += 1
    return (phi / max(1, m)).astype(np.float32)


def loo_marginal(base_scores_u, train_items, val_target, item_vectors, **kwargs) -> np.ndarray:
    kwargs = _coalition_kwargs(kwargs)
    n = len(train_items)
    full = np.arange(n, dtype=np.int64)
    pref_sims = sim_user_items(train_items, item_vectors)
    v_full = coalition_value(base_scores_u, train_items, full, val_target, item_vectors, pref_sims=pref_sims, **kwargs)
    out = np.zeros(n, dtype=np.float32)
    for p in range(n):
        without = np.delete(full, p)
        out[p] = v_full - coalition_value(base_scores_u, train_items, without, val_target, item_vectors, pref_sims=pref_sims, **kwargs)
    return out


def _load_checkpoint(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        return {}
    arr = np.load(path, allow_pickle=True)
    return {int(k): v.astype(np.float32) for k, v in arr.items()}


def _save_checkpoint(path: Path, out: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{str(k): v for k, v in out.items()})


def compute_attribution_for_users(
    split,
    base_scores: np.ndarray,
    item_vectors: sparse.csr_matrix,
    method: str = "shapley-mc",
    max_users: int | None = None,
    m: int = 128,
    exact_threshold: int = 8,
    seed: int = 42,
    max_players_per_user: int | None = None,
    player_selection: str = "stratified",
    checkpoint_path: str | Path | None = None,
    save_every: int = 25,
    n_val_negatives: int = 100,
    **kwargs,
) -> dict[int, np.ndarray]:
    train_csr = split.train_csr
    val_targets = split.val.sort_values("user").set_index("user").item.to_dict()
    users = np.arange(split.n_users)
    if max_users is not None:
        users = users[:max_users]
    out: dict[int, np.ndarray] = {}
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if ckpt is not None:
        out.update(_load_checkpoint(ckpt))
    pbar = tqdm(users, desc=f"{method} users")
    computed_since_save = 0
    for u in pbar:
        u_int = int(u)
        if u_int in out:
            continue
        train_items_full = train_csr[u_int].indices
        if len(train_items_full) == 0 or u_int not in val_targets:
            out[u_int] = np.zeros(len(train_items_full), dtype=np.float32)
            continue
        val_target = int(val_targets[u_int])
        selected_idx = select_players(train_items_full, item_vectors, max_players_per_user, strategy=player_selection, val_target=val_target, seed=seed + u_int)
        train_items = train_items_full[selected_idx]
        val_neg = sample_validation_negatives(base_scores.shape[1], train_items_full, val_target, n_val_negatives, seed + 100000 + u_int)
        local_kwargs = dict(kwargs)
        local_kwargs["val_negatives"] = val_neg
        if method == "loo-marginal":
            phi_selected = loo_marginal(base_scores[u_int], train_items, val_target, item_vectors, **local_kwargs)
        elif len(train_items) <= exact_threshold:
            phi_selected = exact_shapley(base_scores[u_int], train_items, val_target, item_vectors, **local_kwargs)
        else:
            phi_selected = permutation_shapley(base_scores[u_int], train_items, val_target, item_vectors, m=m, seed=seed + u_int, **local_kwargs)
        phi_full = np.zeros(len(train_items_full), dtype=np.float32)
        phi_full[selected_idx] = phi_selected
        out[u_int] = phi_full
        computed_since_save += 1
        pbar.set_postfix({"hist": len(train_items_full), "players": len(train_items), "done": len(out)})
        if ckpt is not None and computed_since_save >= save_every:
            _save_checkpoint(ckpt, out)
            computed_since_save = 0
    if ckpt is not None:
        _save_checkpoint(ckpt, out)
    return out


def compute_shapley_for_users(*args, **kwargs) -> dict[int, np.ndarray]:
    return compute_attribution_for_users(*args, method="shapley-mc", **kwargs)
