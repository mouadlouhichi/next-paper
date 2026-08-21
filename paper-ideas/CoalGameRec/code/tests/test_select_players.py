#!/usr/bin/env python3
"""Unit tests for Algorithm 1 (select_players), including the short-history
edge cases demanded by round-9 review item 15."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coalgamerec.attribution import select_players  # noqa: E402


def make_vectors(n_items: int, dim: int = 8, seed: int = 0) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    return sparse.random(n_items, dim, density=0.7, random_state=rng).tocsr()


def test_short_histories_return_full_history():
    iv = make_vectors(50)
    for n in (1, 2, 3, 5, 24):
        items = np.arange(n)
        sel = select_players(items, iv, 24, strategy="stratified", val_target=40, seed=7)
        assert len(sel) == n, (n, sel)
        assert set(sel.tolist()) == set(items.tolist())


def test_long_history_selects_k_deterministically():
    iv = make_vectors(200)
    items = np.arange(100)
    a = select_players(items, iv, 24, strategy="stratified", val_target=150, seed=7)
    b = select_players(items, iv, 24, strategy="stratified", val_target=150, seed=7)
    assert len(a) == 24
    assert np.array_equal(a, b)  # deterministic
    assert a.max() < 100


def test_boundary_25_items():
    iv = make_vectors(200)
    items = np.arange(25)
    sel = select_players(items, iv, 24, strategy="stratified", val_target=150, seed=7)
    assert len(sel) == 24  # selection activates strictly when |H_u| > k


def test_no_val_target_falls_back():
    iv = make_vectors(200)
    items = np.arange(100)
    sel = select_players(items, iv, 24, strategy="stratified", val_target=None, seed=7)
    assert len(sel) == 24


if __name__ == "__main__":
    test_short_histories_return_full_history()
    test_long_history_selects_k_deterministically()
    test_boundary_25_items()
    test_no_val_target_falls_back()
    print("ALL select_players tests passed")
