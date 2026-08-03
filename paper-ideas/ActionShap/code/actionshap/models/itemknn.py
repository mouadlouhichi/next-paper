"""Primary history-conditioned item-neighbourhood model for ActionShap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ItemKNNModel:
    """Frozen sparse item-item cosine similarities.

    Candidate scores are weighted means of similarities to retained history
    items.  The model therefore supports masking and bounded downweighting at
    inference without retraining, while providing an architecture distinct from
    the latent profile-aggregation model.
    """

    similarities: sparse.csr_matrix

    def __post_init__(self) -> None:
        matrix = sparse.csr_matrix(self.similarities, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1]
        ):
            raise ValueError("similarities must be a non-empty square sparse matrix")
        if matrix.nnz and (
            not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0)
        ):
            raise ValueError("similarities must be finite and non-negative")
        object.__setattr__(self, "similarities", matrix)

    @property
    def n_items(self) -> int:
        return int(self.similarities.shape[0])

    def score(
        self,
        history_items: np.ndarray,
        candidate_items: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        history = np.asarray(history_items, dtype=int)
        candidates = np.asarray(candidate_items, dtype=int)
        if history.ndim != 1 or candidates.ndim != 1:
            raise ValueError("history_items and candidate_items must be 1-D")
        if np.any((history < 0) | (history >= self.n_items)) or np.any(
            (candidates < 0) | (candidates >= self.n_items)
        ):
            raise ValueError("item ID outside similarity matrix")
        if weights is None:
            w = np.ones(history.size, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != history.shape or not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError(
                    "weights must be finite, non-negative, and history-aligned"
                )
        total = float(w.sum())
        if history.size == 0 or total == 0.0:
            return np.zeros(candidates.size, dtype=float)
        block = self.similarities[candidates][:, history]
        return np.asarray(block @ w).reshape(-1) / total

    def score_masked(
        self,
        history_items: np.ndarray,
        candidate_items: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        history = np.asarray(history_items, dtype=int)
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != history.shape:
            raise ValueError("mask must align with history_items")
        return self.score(history[keep], candidate_items)

    def score_downweighted(
        self,
        history_items: np.ndarray,
        candidate_items: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        return self.score(history_items, candidate_items, weights)

    def score_downweighted_batch(
        self,
        history_items: np.ndarray,
        candidate_items: np.ndarray,
        weight_matrix: np.ndarray,
    ) -> np.ndarray:
        """Score many intervention profiles against one cached similarity block."""
        history = np.asarray(history_items, dtype=int)
        candidates = np.asarray(candidate_items, dtype=int)
        weights = np.asarray(weight_matrix, dtype=float)
        if weights.ndim != 2 or weights.shape[1] != history.size:
            raise ValueError(
                "weight_matrix must have one column per history interaction"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weight_matrix must be finite and non-negative")
        block = self.similarities[candidates][:, history].toarray()
        scores = weights @ block.T
        denominators = weights.sum(axis=1)
        nonempty = denominators > 0
        scores[nonempty] /= denominators[nonempty, None]
        scores[~nonempty] = 0.0
        return scores


def _retain_top_k_rows(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Keep at most the largest ``k`` non-zero values in every CSR row."""
    if k < 1:
        raise ValueError("k must be positive")
    matrix = matrix.tocsr()
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_cols = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        if row_values.size > k:
            selected = np.argpartition(row_values, -k)[-k:]
            selected = selected[np.argsort(-row_values[selected], kind="stable")]
        else:
            selected = np.argsort(-row_values, kind="stable")
        rows.extend([row] * selected.size)
        cols.extend(row_cols[selected].tolist())
        values.extend(row_values[selected].tolist())
    return sparse.csr_matrix((values, (rows, cols)), shape=matrix.shape)


def fit_item_knn(
    histories: dict[int, np.ndarray],
    n_items: int,
    neighbours: int = 200,
) -> ItemKNNModel:
    """Fit cosine item-item similarities from complete training histories."""
    if n_items < 1 or neighbours < 1:
        raise ValueError("n_items and neighbours must be positive")
    user_ids = sorted(histories)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    for row, user in enumerate(user_ids):
        items = np.unique(np.asarray(histories[user], dtype=int))
        if np.any((items < 0) | (items >= n_items)):
            raise ValueError("history item outside declared catalogue")
        row_parts.append(np.full(items.size, row, dtype=int))
        col_parts.append(items)
    if not row_parts or sum(part.size for part in row_parts) == 0:
        raise ValueError("at least one training interaction is required")
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    interactions = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)),
        shape=(len(user_ids), n_items),
    )
    counts = np.asarray(interactions.sum(axis=0)).reshape(-1)
    inverse = np.zeros(n_items, dtype=float)
    nonzero = counts > 0
    inverse[nonzero] = 1.0 / np.sqrt(counts[nonzero])
    cooccurrence = (interactions.T @ interactions).tocsr().astype(float)
    cooccurrence.setdiag(0.0)
    cooccurrence.eliminate_zeros()
    cosine = sparse.diags(inverse) @ cooccurrence @ sparse.diags(inverse)
    cosine = _retain_top_k_rows(cosine.tocsr(), min(neighbours, n_items))
    return ItemKNNModel(cosine)
