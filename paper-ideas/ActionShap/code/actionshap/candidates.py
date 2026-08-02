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
