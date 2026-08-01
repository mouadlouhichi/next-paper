"""Ranking metrics used for validation and reporting (DyHuCoG Sec. 3.7)."""

from __future__ import annotations

import numpy as np


def ndcg_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    gains = np.zeros_like(ranks, dtype=float)
    pos = (ranks >= 1) & (ranks <= k)
    gains[pos] = 1.0 / np.log2(ranks[pos] + 1.0)
    return gains


def recall_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    return (ranks >= 1).astype(float) * (ranks <= k).astype(float)


def mrr(ranks: np.ndarray) -> np.ndarray:
    out = np.zeros_like(ranks, dtype=float)
    out[ranks >= 1] = 1.0 / ranks[ranks >= 1]
    return out


def coverage(lists: np.ndarray, n_items: int) -> float:
    return float(len(np.unique(lists[lists >= 0])) / n_items)
