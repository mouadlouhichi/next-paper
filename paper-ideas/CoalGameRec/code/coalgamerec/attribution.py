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


@dataclass(frozen=True)
class ShapleyConfig:
    alpha: float = 0.70
    beta: float = 0.30
    lambda_pref: float = 0.20
    lambda_attr_value: float = 0.10
    m: int = 128
    exact_threshold: int = 8
    seed: int = 42
    max_players_per_user: int | None = None
    player_selection: str = "similarity"  # similarity, recent, first


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


def select_players(train_items: np.ndarray, item_vectors: sparse.csr_matrix, max_players: int | None, strategy: str = "similarity") -> np.ndarray:
    """Return indices into train_items selected for the Shapley game.

    Full-history interaction Shapley is prohibitively expensive on a laptop for
    users with long histories. This bounded-player option makes the estimator
    executable and auditable. Non-selected interactions receive zero Shapley
    weight in the current run artifact and the selection rule is recorded in the
    config. Use max_players=None for the unbounded protocol.
    """
    n = len(train_items)
    if max_players is None or n <= int(max_players):
        return np.arange(n, dtype=np.int64)
    k = int(max_players)
    if strategy == "first":
        return np.arange(k, dtype=np.int64)
    if strategy == "recent":
        return np.arange(n - k, n, dtype=np.int64)
    if strategy == "similarity":
        sims = sim_user_items(train_items, item_vectors)
        # stable tie-breaking by original position
        order = np.lexsort((np.arange(n), -sims))
        return np.sort(order[:k].astype(np.int64))
    raise ValueError(f"unknown player_selection strategy: {strategy}")


def coalition_value(
    base_scores_u: np.ndarray,
    train_items: np.ndarray,
    coalition_idx: np.ndarray,
    val_target: int,
    item_vectors: sparse.csr_matrix,
    alpha: float = 0.70,
    beta: float = 0.30,
    lambda_pref: float = 0.20,
    lambda_attr_value: float = 0.10,
    pref_sims: np.ndarray | None = None,
) -> float:
    """Preference-aware per-user coalition value v_pref,u(S_u).

    Relevance is the validation target; the test target is not used here. The
    expensive user preference similarities are precomputed once per user when
    possible and passed through pref_sims.
    """
    if pref_sims is None:
        pref_sims = sim_user_items(train_items, item_vectors)
    if len(coalition_idx) == 0:
        coalition_items = train_items[:0]
        raw_w = np.zeros(0, dtype=np.float32)
        pref = 0.0
    else:
        coalition_items = train_items[coalition_idx]
        raw_w = np.ones(len(coalition_items), dtype=np.float32)
        pref = float(np.sum(pref_sims[coalition_idx]))

    if len(coalition_items):
        candidates = np.arange(base_scores_u.shape[0], dtype=np.int64)
        adj_all = attribution_adjustment(coalition_items, raw_w, item_vectors, candidates)
    else:
        adj_all = np.zeros_like(base_scores_u, dtype=np.float32)

    scores = zscore(base_scores_u) + float(lambda_attr_value) * zscore(adj_all)
    ndcg = _ndcg_single(scores, val_target, train_items, k=20)
    div = _diversity(scores, train_items, item_vectors, k=20)
    return float(alpha * ndcg + beta * div + lambda_pref * pref)


def exact_shapley(base_scores_u, train_items, val_target, item_vectors, **kwargs) -> np.ndarray:
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


def permutation_shapley(base_scores_u, train_items, val_target, item_vectors, m: int = 128, seed: int = 42, **kwargs) -> np.ndarray:
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

    for _ in range(m):
        perm = rng.permutation(n)
        S: list[int] = []
        prev = v(S)
        for p in perm:
            S.append(int(p))
            cur = v(S)
            phi[p] += cur - prev
            prev = cur
    return (phi / max(1, m)).astype(np.float32)


def _load_checkpoint(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        return {}
    arr = np.load(path, allow_pickle=True)
    return {int(k): v.astype(np.float32) for k, v in arr.items()}


def _save_checkpoint(path: Path, out: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{str(k): v for k, v in out.items()})


def compute_shapley_for_users(
    split,
    base_scores: np.ndarray,
    item_vectors: sparse.csr_matrix,
    max_users: int | None = None,
    m: int = 128,
    exact_threshold: int = 8,
    seed: int = 42,
    max_players_per_user: int | None = None,
    player_selection: str = "similarity",
    checkpoint_path: str | Path | None = None,
    save_every: int = 25,
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
    pbar = tqdm(users, desc="Shapley users")
    computed_since_save = 0
    for u in pbar:
        u_int = int(u)
        if u_int in out:
            continue
        train_items_full = train_csr[u_int].indices
        if len(train_items_full) == 0 or u_int not in val_targets:
            out[u_int] = np.zeros(len(train_items_full), dtype=np.float32)
            continue
        selected_idx = select_players(train_items_full, item_vectors, max_players_per_user, strategy=player_selection)
        train_items = train_items_full[selected_idx]
        if len(train_items) <= exact_threshold:
            phi_selected = exact_shapley(base_scores[u_int], train_items, int(val_targets[u_int]), item_vectors, **kwargs)
        else:
            phi_selected = permutation_shapley(base_scores[u_int], train_items, int(val_targets[u_int]), item_vectors, m=m, seed=seed + u_int, **kwargs)
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
