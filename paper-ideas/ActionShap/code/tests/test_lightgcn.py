"""Smoke tests for the LightGCN-style model (review-6 competitive model)."""
import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from actionshap.lightgcn import fit_lightgcn


def _block_histories(n_users=240, n_items=60, blocks=3, seed=0):
    rng = np.random.default_rng(seed)
    histories = {}
    for u in range(n_users):
        b = u % blocks
        pool = np.arange(b * 20, (b + 1) * 20)
        k = int(rng.integers(5, 12))
        histories[u] = rng.choice(pool, size=k, replace=False)
    return histories, n_items


def test_fit_and_interface():
    histories, n_items = _block_histories()
    model = fit_lightgcn(histories, n_items, dim=8, layers=1, epochs=3, lr=0.05)
    u = next(iter(histories))
    hist = histories[u]
    cands = np.arange(n_items)
    full = model.score(hist, cands)
    masked = model.score_masked(hist, cands, np.ones(len(hist), dtype=bool))
    assert full.shape == (n_items,)
    assert np.allclose(full, masked, atol=1e-10)
    half = model.score_downweighted(hist, cands, np.full(len(hist), 0.5))
    assert half.shape == (n_items,)
    empty = model.score_masked(hist, cands, np.zeros(len(hist), dtype=bool))
    assert np.allclose(empty, 0.0)
    batch = model.score_downweighted_batch(hist, cands[:5], np.ones((2, len(hist))))
    assert batch.shape == (2, 5)


def test_learns_block_structure():
    histories, n_items = _block_histories(n_users=480, seed=1)
    model = fit_lightgcn(histories, n_items, dim=16, layers=2, epochs=25, lr=0.05)
    rng = np.random.default_rng(7)
    hits = 0
    total = 0
    for u in list(histories)[:200]:
        h = histories[u]
        target = int(h[-1])
        hist = h[:-1]
        if len(hist) < 2:
            continue
        scores = model.score(hist, np.arange(n_items))
        rank = int((scores > scores[target]).sum()) + 1
        hits += rank <= 10
        total += 1
    # in-block held-out target: chance HR@10 is ~10/60; require a clear lift
    assert hits / total > 0.25, f"HR@10 too low: {hits/total:.3f}"
