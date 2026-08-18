"""Minimal SASRec-style sequential recommender with an inference-time history
weighting interface (review 3, critical 3).

The user representation is computed *at scoring time* as a history-weighted
sum of position-encoded item embeddings passed through one transformer block,
so masking/bounded downweighting of historical interactions changes scores
without retraining - the interface required by the ActionShap protocol.
Training uses the same leave-one-out BPR-style objective as the profile model.
"""
from __future__ import annotations

import numpy as np

try:  # torch is an optional dependency of the core package
    import torch
    from torch import nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    _HAS_TORCH = False


def has_torch() -> bool:
    return _HAS_TORCH


if _HAS_TORCH:

    class WeightedSASRec(nn.Module):
        def __init__(self, n_items: int, dim: int = 64, heads: int = 1,
                     max_len: int = 64, dropout: float = 0.1):
            super().__init__()
            self.dim = dim
            self.max_len = max_len
            self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
            self.pos_emb = nn.Embedding(max_len, dim)
            layer = nn.TransformerEncoderLayer(
                dim, heads, dim_feedforward=dim * 4, dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, 1)
            self.dropout = nn.Dropout(dropout)

        def _encode(self, items: "torch.Tensor", weights: "torch.Tensor"):
            # items: (B, L) int ids (0 pad), weights: (B, L) floats in [0, 1]
            L = items.size(1)
            pos = torch.arange(L, device=items.device).unsqueeze(0)
            x = self.item_emb(items) + self.pos_emb(pos)
            pad_mask = items == 0
            x = self.dropout(self.encoder(x, src_key_padding_mask=pad_mask))
            wsum = weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
            user = (x * weights.unsqueeze(-1)).sum(dim=1) / wsum
            return user

        def score(self, items: "torch.Tensor", weights: "torch.Tensor",
                  candidates: "torch.Tensor") -> "torch.Tensor":
            user = self._encode(items, weights)
            return user @ self.item_emb(candidates).T


def fit_sasrec(
    histories: dict[int, np.ndarray],
    n_items: int,
    *,
    dim: int = 64,
    epochs: int = 10,
    lr: float = 3e-4,
    max_len: int = 20,
    seed: int = 0,
    batch_users: int = 256,
):
    """Leave-one-out BPR-style training; returns a numpy scoring adapter."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is required for fit_sasrec")
    torch.manual_seed(seed)
    model = WeightedSASRec(n_items, dim=dim, max_len=max_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    users = list(histories)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(users)
        for start in range(0, len(users), batch_users):
            batch = users[start:start + batch_users]
            ids, weights, pos, neg = [], [], [], []
            for u in batch:
                items = np.unique(histories[u])
                if items.size < 2:
                    continue
                p = int(rng.integers(items.size))
                ctx = np.delete(items, p)
                tail = ctx[-max_len:]
                pad = max_len - tail.size
                ids.append(np.concatenate([np.zeros(pad, int), tail]) + 0)
                w = np.ones(max_len)
                w[:pad] = 0.0
                weights.append(w)
                pos.append(int(items[p]))
                n_ = int(rng.integers(0, n_items))
                while n_ in set(items):
                    n_ = int(rng.integers(0, n_items))
                neg.append(n_)
            if not ids:
                continue
            items_t = torch.from_numpy(np.vstack(ids) + 1)  # 0 = pad
            w_t = torch.from_numpy(np.vstack(weights)).float()
            pos_t = torch.from_numpy(np.array(pos) + 1)
            neg_t = torch.from_numpy(np.array(neg) + 1)
            user = model._encode(items_t, w_t)
            loss = -torch.nn.functional.logsigmoid(
                (user * model.item_emb(pos_t)).sum(-1)
                - (user * model.item_emb(neg_t)).sum(-1)
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return SASRecAdapter(model, max_len)


class SASRecAdapter:
    """Numpy-facing scorer with the ActionShap weighting interface."""

    def __init__(self, model, max_len: int):
        self.model = model.eval()
        self.max_len = max_len

    def _tensors(self, history_items: np.ndarray, weights: np.ndarray):
        hist = np.asarray(history_items, dtype=int)
        tail = hist[-self.max_len:]
        w = np.asarray(weights, dtype=float)[-self.max_len:]
        pad = self.max_len - tail.size
        ids = np.concatenate([np.zeros(pad, int), tail]) + 1
        wv = np.concatenate([np.zeros(pad), w])
        return (torch.from_numpy(ids).unsqueeze(0),
                torch.from_numpy(wv).unsqueeze(0).float())

    def score(self, history_items, candidate_items, weights=None):
        with torch.no_grad():
            if weights is None:
                weights = np.ones(len(history_items))
            ids, wv = self._tensors(history_items, weights)
            cand = torch.from_numpy(np.asarray(candidate_items, dtype=int) + 1)
            return self.model.score(ids, wv, cand).numpy().reshape(-1)

    def score_masked(self, history_items, candidate_items, mask):
        keep = np.asarray(mask, dtype=bool)
        w = np.ones(len(history_items))
        w[~keep] = 0.0
        return self.score(history_items, candidate_items, w)

    def score_downweighted(self, history_items, candidate_items, weights):
        return self.score(history_items, candidate_items, weights)

    def score_downweighted_batch(self, history_items, candidate_items, weight_matrix):
        return np.vstack([
            self.score(history_items, candidate_items, row)
            for row in np.asarray(weight_matrix)
        ])
