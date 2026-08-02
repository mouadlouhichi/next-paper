"""Fixed candidate retrieval for ActionShap."""

from __future__ import annotations

import numpy as np


def fixed_candidates(
    model,
    histories: dict[int, np.ndarray],
    test_items: dict[int, int],
    n_items: int,
    k: int = 200,
) -> tuple[dict[int, np.ndarray], float]:
    """Retrieve top-k candidates once and return candidate recall.

    The held-out test item is never inserted artificially. This makes recall a
    real diagnostic and ensures all coalition evaluations use the same set.
    Training-history items are excluded, but validation/test items remain
    eligible for retrieval.
    """
    if k < 1 or k > n_items:
        raise ValueError("k must be between 1 and n_items")
    all_items = np.arange(n_items, dtype=int)
    result: dict[int, np.ndarray] = {}
    hits = []
    for u in sorted(test_items):
        history = np.asarray(histories[u], dtype=int)
        scores = model.score(history, all_items)
        forbidden = set(history.tolist())
        allowed = np.array([i not in forbidden for i in all_items], dtype=bool)
        candidates = all_items[allowed]
        values = scores[allowed]
        order = np.lexsort((candidates, -values))[:k]
        result[u] = candidates[order]
        hits.append(int(test_items[u] in set(result[u].tolist())))
    recall = float(np.mean(hits)) if hits else 0.0
    return result, recall


def fixed_evaluation_sets(
    histories: dict[int, np.ndarray],
    test_items: dict[int, int],
    n_items: int,
    size: int = 200,
    seed: int = 0,
) -> tuple[dict[int, np.ndarray], float]:
    """Build fixed sampled evaluation sets with the target always included.

    This is deliberately not retrieval. Negatives are sampled uniformly from
    the user's unseen catalogue and the held-out target is inserted by design.
    Therefore target coverage is exactly one, while the paper must describe the
    result as sampled-ranking evaluation rather than full-catalog retrieval.
    """
    if size < 2 or size > n_items:
        raise ValueError("size must be at least two and no larger than n_items")
    rng = np.random.default_rng(seed)
    all_items = np.arange(n_items, dtype=int)
    result: dict[int, np.ndarray] = {}
    for u in sorted(test_items):
        history = set(np.asarray(histories[u], dtype=int).tolist())
        target = int(test_items[u])
        eligible = all_items[~np.isin(all_items, np.fromiter(history | {target}, dtype=int))]
        if eligible.size < size - 1:
            raise ValueError(f"user {u} has fewer than {size - 1} eligible negatives")
        negatives = rng.choice(eligible, size=size - 1, replace=False)
        result[u] = np.sort(np.concatenate(([target], negatives)))
    return result, 1.0
