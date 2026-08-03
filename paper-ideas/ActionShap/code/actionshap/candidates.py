"""Leakage-safe fixed evaluation sets and deterministic ranking tie-breaks."""

from __future__ import annotations

import numpy as np


def global_item_priorities(n_items: int, seed: int = 42) -> np.ndarray:
    """One seeded catalogue-wide tie-break reused for every user and method."""
    if n_items < 1:
        raise ValueError("n_items must be positive")
    permutation = np.random.default_rng(seed).permutation(n_items)
    priorities = np.empty(n_items, dtype=np.int64)
    priorities[permutation] = np.arange(n_items, dtype=np.int64)
    return priorities


def tie_break_for_candidates(
    candidate_items: np.ndarray, priorities: np.ndarray
) -> np.ndarray:
    candidates = np.asarray(candidate_items, dtype=int)
    priority = np.asarray(priorities, dtype=np.int64)
    if candidates.ndim != 1 or priority.ndim != 1:
        raise ValueError("candidate_items and priorities must be one-dimensional")
    if np.any((candidates < 0) | (candidates >= priority.size)):
        raise ValueError("candidate item outside priority catalogue")
    return priority[candidates]


def _excluded_mask(
    n_items: int,
    observed_items: np.ndarray,
    target_item: int,
) -> np.ndarray:
    observed = np.unique(np.asarray(observed_items, dtype=int))
    if np.any((observed < 0) | (observed >= n_items)) or not 0 <= target_item < n_items:
        raise ValueError("observed or target item outside catalogue")
    excluded = np.zeros(n_items, dtype=bool)
    excluded[observed] = True
    # The temporal target must remain evaluable even in a dataset with repeated
    # item interactions.  Such users should also be counted and disclosed.
    excluded[int(target_item)] = True
    return excluded


def fixed_candidates(
    model,
    histories: dict[int, np.ndarray],
    test_items: dict[int, int],
    n_items: int,
    k: int = 200,
) -> tuple[dict[int, np.ndarray], float]:
    """Retrieve top-k unseen candidates once and return target recall.

    This retrieval diagnostic never inserts the test target.  ``histories`` must
    contain every item observed before test, not merely the attribution window.
    """
    if k < 1 or k > n_items:
        raise ValueError("k must be between 1 and n_items")
    all_items = np.arange(n_items, dtype=int)
    result: dict[int, np.ndarray] = {}
    hits: list[int] = []
    for user in sorted(test_items):
        observed = np.unique(np.asarray(histories[user], dtype=int))
        scores = model.score(observed, all_items)
        allowed = ~np.isin(all_items, observed)
        candidates = all_items[allowed]
        values = np.asarray(scores)[allowed]
        take = min(k, candidates.size)
        order = np.lexsort((candidates, -values))[:take]
        result[user] = candidates[order]
        hits.append(int(int(test_items[user]) in set(result[user].tolist())))
    return result, float(np.mean(hits)) if hits else 0.0


def fixed_evaluation_sets(
    histories: dict[int, np.ndarray],
    test_items: dict[int, int],
    n_items: int,
    size: int = 200,
    seed: int = 42,
) -> tuple[dict[int, np.ndarray], float]:
    """Build target-plus-unseen-negative sampled ranking sets.

    ``histories`` must contain the complete set of items observed before test,
    including the validation event.  The target is included exactly once by
    design; consequently the returned target coverage is exactly one and must
    never be described as retrieval recall.
    """
    if size < 2 or size > n_items:
        raise ValueError("size must be at least two and no larger than n_items")
    rng = np.random.default_rng(seed)
    all_items = np.arange(n_items, dtype=int)
    result: dict[int, np.ndarray] = {}
    for user in sorted(test_items):
        target = int(test_items[user])
        excluded = _excluded_mask(n_items, histories[user], target)
        eligible = all_items[~excluded]
        if eligible.size < size - 1:
            raise ValueError(f"user {user} has fewer than {size - 1} unseen negatives")
        negatives = rng.choice(eligible, size=size - 1, replace=False)
        result[user] = np.sort(np.concatenate(([target], negatives))).astype(int)
    return result, 1.0


def full_unseen_evaluation_sets(
    histories: dict[int, np.ndarray],
    test_items: dict[int, int],
    n_items: int,
) -> tuple[dict[int, np.ndarray], float]:
    """Build the true full unseen catalogue plus the temporal target."""
    all_items = np.arange(n_items, dtype=int)
    result: dict[int, np.ndarray] = {}
    for user in sorted(test_items):
        target = int(test_items[user])
        observed = np.unique(np.asarray(histories[user], dtype=int))
        allowed = ~np.isin(all_items, observed)
        allowed[target] = True
        result[user] = all_items[allowed]
    return result, 1.0
