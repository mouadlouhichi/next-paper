"""
data.py — MovieLens-1M loading and temporal, leakage-safe splitting.
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple

import numpy as np


def load_ratings(path: str) -> List[Tuple[int, int, float, int]]:
    """Return list of (user, item, rating, timestamp)."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                u, i, r, t = parts
                rows.append((int(u), int(i), float(r), int(t)))
    return rows


def load_items(path: str) -> Dict[int, str]:
    """Return dict item_id -> genres (pipe-separated)."""
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                d[int(parts[0])] = parts[2]
    return d


def build_user_sequences(ratings: List[Tuple[int, int, float, int]]
                         ) -> Dict[int, List[Tuple[int, int]]]:
    """Sort each user's (item, ts) pairs by time."""
    by_user: Dict[int, List[Tuple[int, int]]] = {}
    for u, i, r, t in ratings:
        by_user.setdefault(u, []).append((i, t))
    for u in by_user:
        by_user[u].sort(key=lambda x: x[1])
    return by_user


def temporal_split(seq: List[Tuple[int, int]], nmax: int, h_fut: int
                   ) -> Tuple[List[int], List[int], List[int]]:
    """
    Leakage-safe temporal split for a user's item sequence:
      base      = interactions before the lever window
      window    = the nmax interactions immediately before the future window
      future    = the h_fut held-out interactions (targets)
    Returns (base, window, future) as item-id lists.
    """
    if len(seq) < nmax + h_fut + 1:
        return [], [], []
    items = [i for i, _ in seq]
    future = items[-h_fut:]
    window = items[-(h_fut + nmax): -h_fut]
    base = items[: -(h_fut + nmax)]
    return base, window, future


def dominant_genre(items: Dict[int, str], history: List[int]) -> str:
    from collections import Counter
    c = Counter()
    for it in history:
        for g in (items.get(it, "") or "").split("|"):
            if g:
                c[g] += 1
    return c.most_common(1)[0][0] if c else ""
