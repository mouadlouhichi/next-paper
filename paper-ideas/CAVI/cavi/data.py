"""
data.py — MovieLens-1M loading and temporal, leakage-safe splitting.
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple

import numpy as np


def _open_lines(path: str) -> List[str]:
    """
    Read a text file robustly across encodings. MovieLens-1M's `movies.dat`
    contains Latin-1/Windows-1252 accented characters (e.g. 0xe9 = 'é') in some
    movie titles, which crash a naive UTF-8 read. We try UTF-8 first, then
    fall back to Latin-1 (which never fails and preserves the byte values).
    As a final guarantee we decode raw bytes with errors='replace', which can
    never raise UnicodeDecodeError for any input file.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    # last-resort: read as bytes and decode leniently (never raises)
    with open(path, "rb") as f:
        data = f.read()
    return data.decode("utf-8", errors="replace").splitlines(keepends=True)


def load_ratings(path: str) -> List[Tuple[int, int, float, int]]:
    """Return list of (user, item, rating, timestamp)."""
    rows = []
    for line in _open_lines(path):
        parts = line.strip().split("\t")
        if len(parts) == 4:
            u, i, r, t = parts
            rows.append((int(u), int(i), float(r), int(t)))
    return rows


def load_items(path: str) -> Dict[int, str]:
    """Return dict item_id -> genres (pipe-separated)."""
    d = {}
    for line in _open_lines(path):
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


# ---------------------------------------------------------------------------
# LightGCN-format loaders (Amazon-Book, Yelp2018, Gowalla, ...)
# ---------------------------------------------------------------------------
# Canonical split format used by LightGCN / HCCF / HPCF / DyHuCoG:
#   train.txt  : each line  "user item1 item2 ..."   (space-separated; user id
#                is the first token, then the remapped item ids the user
#                interacted with in the TRAIN set)
#   test.txt   : same format for the held-out interactions
#   user_list.txt / item_list.txt : "org_id remap_id" (with a header line)
#
# NOTE: these canonical splits carry NO timestamps and NO item metadata (the
# proposal's SignalShap §4.1 notes this explicitly). Item ids are already
# remapped to 0..N-1, so there is no genre/metadata to build levers from. The
# proposal recommends rebuilding Amazon-Book from the raw Amazon Reviews 2018
# corpus to recover timestamps/metadata; these loaders support the *canonical*
# split that is actually shared and validated in this repo, and document the
# limitation.

def load_lightgcn_split(data_dir: str, prefix: str
                        ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Load a LightGCN-format train/test split.
    Returns (train_dict, test_dict): user id -> list of item ids.
    """
    def _read(name):
        out: Dict[int, List[int]] = {}
        for line in _open_lines(os.path.join(data_dir, name)):
            toks = line.strip().split()
            if not toks:
                continue
            u = int(toks[0])
            out[u] = [int(x) for x in toks[1:]]
        return out
    # standard names: train.txt / test.txt inside the dataset directory
    return _read("train.txt"), _read("test.txt")


def load_remap_lists(data_dir: str, prefix: str
                     ) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Load user_list.txt / item_list.txt ("org_id remap_id" with a header).
    Returns (user_org_by_remap, item_org_by_remap): remapped id -> original id.
    """
    def _read(name, has_header=True):
        out: Dict[int, str] = {}
        lines = _open_lines(os.path.join(data_dir, name))
        start = 1 if (has_header and lines and not lines[0].strip()[0].isdigit()) else 0
        for line in lines[start:]:
            toks = line.strip().split()
            if len(toks) >= 2:
                out[int(toks[1])] = toks[0]
        return out
    return _read("user_list.txt"), _read("item_list.txt")


def lightgcn_stats(train_dict: Dict[int, List[int]],
                   test_dict: Dict[int, List[int]]) -> Dict[str, int]:
    """Basic dataset statistics for a LightGCN split."""
    n_users = max(list(train_dict.keys()) + list(test_dict.keys())) + 1
    all_items = set()
    n_inter = 0
    for u, its in train_dict.items():
        all_items.update(its)
        n_inter += len(its)
    for u, its in test_dict.items():
        all_items.update(its)
        n_inter += len(its)
    n_items = max(all_items) + 1 if all_items else 0
    density = n_inter / (n_users * n_items) if n_users * n_items else 0.0
    return {"users": n_users, "items": n_items, "interactions": n_inter,
            "density": density}


def build_user_seq_from_split(train_dict: Dict[int, List[int]],
                              test_dict: Dict[int, List[int]]
                              ) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build a per-user ordered item sequence from a LightGCN split (for the CAVI
    temporal base/window/future decomposition). Train items are treated as the
    seen history (in the order given), and the test items as the future target
    set. Since the canonical split has no timestamps, the train order is used
    verbatim as the within-user ordering.
    """
    seqs: Dict[int, List[Tuple[int, int]]] = {}
    for u, its in train_dict.items():
        seqs[u] = [(i, k) for k, i in enumerate(its)]  # (item, synthetic ts=order)
    return seqs


def future_from_test(test_dict: Dict[int, List[int]], u: int) -> List[int]:
    """Future (held-out target) items for a user from the test split."""
    return list(test_dict.get(u, []))


def item_popularity(train_dict: Dict[int, List[int]]) -> Dict[int, int]:
    """Item -> interaction-count popularity from the train split."""
    pop: Dict[int, int] = {}
    for u, its in train_dict.items():
        for i in its:
            pop[i] = pop.get(i, 0) + 1
    return pop


def movability_from_popularity(pop: Dict[int, int],
                               history: List[int],
                               threshold_ratio: float = 0.1) -> List[bool]:
    """
    Metadata-free feasibility: mark an item as *immovable* (anchor) if it is in
    the top `threshold_ratio` most-popular items in `history`. This is a
    stand-in for the genre-anchor feasibility used on ML-1M, for datasets that
    carry no genre/metadata (Amazon-Book, Yelp2018 canonical splits).
    Returns a boolean mask aligned with `history`.
    """
    if not history:
        return [True] * len(history)
    pops = [pop.get(i, 0) for i in history]
    if max(pops) == 0:
        return [True] * len(history)
    thr = threshold_ratio * max(pops)
    return [p >= thr for p in pops]  # True = anchor = immovable
