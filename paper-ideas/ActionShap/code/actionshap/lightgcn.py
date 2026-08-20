"""LightGCN-style graph recommender with an inference-time history weighting
interface (review 6, critical item: competitive neural/graph model).

Item embeddings are trained on the user--item interaction graph with
LightGCN's symmetric-normalized propagation (K layers, layer-combination
weight 1/(K+1)) and a BPR objective; the propagation is a linear operator,
so forward and backward passes are exact sparse matrix products.

The user representation is recomputed *at scoring time* as the weighted mean
of the propagated item embeddings of the retained history, so masking and
bounded downweighting of historical interactions change scores without
retraining - the interface required by the ActionShap protocol. Graph
construction and scoring truncate histories to the most recent ``max_len``
interactions, consistent with the protocol's player window.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import sparse as sp

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    sp = None
    _HAS_SCIPY = False


def has_scipy() -> bool:
    return _HAS_SCIPY


def _normalized_item_operator(histories, n_items: int, max_len: int):
    """Symmetric-normalized item-side LightGCN operator P (sparse CSR).

    P = D_i^{-1/2} A_{iu} D_u^{-1} A_{ui} D_i^{-1/2}, where A counts each
    (user, item) edge once over the truncated histories. Applying P once is
    one two-hop (item -> user -> item) propagation step.
    """
    rows, cols = [], []
    for u, h in histories.items():
        items = np.unique(np.asarray(h, dtype=int)[-max_len:])
        if items.size == 0:
            continue
        rows.append(np.full(items.size, u))
        cols.append(items)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    n_users = rows.max() + 1 if rows.size else 1
    A_ui = sp.csr_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(n_users, n_items)
    )
    deg_u = np.asarray(A_ui.sum(axis=1)).ravel()
    deg_i = np.asarray(A_ui.sum(axis=0)).ravel()
    inv_u = np.where(deg_u > 0, 1.0 / deg_u, 0.0)
    inv_sqrt_i = np.where(deg_i > 0, 1.0 / np.sqrt(deg_i), 0.0)
    # P = S_i @ A_iu @ D_u^{-1} @ A_ui @ S_i  (S_i = diag(inv_sqrt_i))
    M = A_ui.T.multiply(inv_u).tocsr()  # items x users, col-scaled by inv_u
    P = (M @ A_ui).tocsr()
    P = sp.diags(inv_sqrt_i) @ P @ sp.diags(inv_sqrt_i)
    return P.tocsr()


def _propagate(E: np.ndarray, P, layers: int) -> np.ndarray:
    """Layer-combined LightGCN forward pass: (1/(K+1)) sum_k P^k E."""
    acc = E.copy()
    cur = E
    for _ in range(layers):
        cur = P @ cur
        acc = acc + cur
    return acc / (layers + 1)


class LightGCNAdapter:
    """Numpy-facing scorer with the ActionShap weighting interface."""

    def __init__(self, item_emb: np.ndarray, max_len: int):
        self.item_emb = np.asarray(item_emb, dtype=float)
        self.max_len = max_len

    def _user_vector(self, history_items: np.ndarray, weights: np.ndarray):
        hist = np.asarray(history_items, dtype=int)[-self.max_len:]
        w = np.asarray(weights, dtype=float)[-self.max_len:]
        if hist.size == 0:
            return np.zeros(self.item_emb.shape[1])
        wsum = float(w.sum())
        if wsum <= 0.0:
            return np.zeros(self.item_emb.shape[1])
        return (self.item_emb[hist] * w[:, None]).sum(axis=0) / wsum

    def score(self, history_items, candidate_items, weights=None):
        if weights is None:
            weights = np.ones(len(history_items))
        u = self._user_vector(history_items, weights)
        return self.item_emb[np.asarray(candidate_items, dtype=int)] @ u

    def score_masked(self, history_items, candidate_items, mask):
        w = np.where(np.asarray(mask, dtype=bool), 1.0, 0.0)
        return self.score(history_items, candidate_items, w)

    def score_downweighted(self, history_items, candidate_items, weights):
        return self.score(history_items, candidate_items, weights)

    def score_downweighted_batch(self, history_items, candidate_items, weight_matrix):
        return np.vstack(
            [
                self.score(history_items, candidate_items, row)
                for row in np.asarray(weight_matrix)
            ]
        )


def fit_lightgcn(
    histories: dict[int, np.ndarray],
    n_items: int,
    *,
    dim: int = 64,
    layers: int = 2,
    epochs: int = 15,
    lr: float = 0.01,
    reg: float = 1e-4,
    max_len: int = 20,
    seed: int = 0,
    batch_users: int = 1024,
    verbose: bool = False,
) -> LightGCNAdapter:
    """Train LightGCN-style item embeddings with BPR; return a scoring adapter.

    The gradient of the layer-combined propagation is the same linear
    operator applied to the loss gradient at the combined embeddings, so the
    backward pass reuses ``_propagate`` with the gradient input.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for fit_lightgcn")
    rng = np.random.default_rng(seed)
    P = _normalized_item_operator(histories, n_items, max_len)
    E = rng.normal(0.0, 0.1, size=(n_items, dim))
    user_list = []
    hist_arrays = {}
    for u, h in histories.items():
        items = np.unique(np.asarray(h, dtype=int)[-max_len:])
        if items.size >= 2:
            user_list.append(u)
            hist_arrays[u] = items
    user_arr = np.asarray(user_list)
    for epoch in range(epochs):
        rng.shuffle(user_arr)
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, user_arr.size, batch_users):
            batch = user_arr[start : start + batch_users]
            hists = [hist_arrays[u] for u in batch]
            max_h = max(h.size for h in hists)
            ids = np.zeros((len(batch), max_h), dtype=int)
            hmask = np.zeros((len(batch), max_h), dtype=bool)
            for b, h in enumerate(hists):
                ids[b, : h.size] = h
                hmask[b, : h.size] = True
            Ef = _propagate(E, P, layers)
            counts = hmask.sum(axis=1, keepdims=True)
            U = (Ef[ids] * hmask[:, :, None]).sum(axis=1) / counts  # user vecs
            pos_idx = (rng.random(len(batch)) * counts.ravel()).astype(int)
            pos = ids[np.arange(len(batch)), pos_idx]
            neg = rng.integers(0, n_items, size=len(batch))
            seen = [set(h) for h in hists]
            for b in range(len(batch)):
                while neg[b] in seen[b]:
                    neg[b] = rng.integers(0, n_items)
            x = ((Ef[pos] - Ef[neg]) * U).sum(axis=1)
            g = 1.0 / (1.0 + np.exp(np.clip(x, -40.0, 40.0)))
            epoch_loss += float(-np.log(g + 1e-12).mean())
            n_steps += 1
            # gradients at the combined embeddings
            grad_Ef = np.zeros_like(E)
            np.add.at(grad_Ef, pos, -g[:, None] * U)
            np.add.at(grad_Ef, neg, g[:, None] * U)
            grad_U = g[:, None] * (Ef[pos] - Ef[neg]) / counts
            np.add.at(grad_Ef, ids[hmask], np.repeat(grad_U, counts.ravel(), axis=0))
            # L2 on base embeddings of involved items
            involved = np.unique(np.concatenate([pos, neg, ids[hmask]]))
            grad_Ef[involved] += reg * E[involved]
            # exact backprop through the linear propagation
            grad_E = _propagate(grad_Ef, P, layers)
            E -= lr * grad_E
        if verbose:
            print(f"[lightgcn] epoch {epoch + 1}/{epochs} loss={epoch_loss / max(n_steps, 1):.4f}",
                  flush=True)
    Ef_final = _propagate(E, P, layers)
    return LightGCNAdapter(Ef_final, max_len)
