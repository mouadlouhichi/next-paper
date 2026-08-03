"""History-conditioned profile aggregation recommender for ActionShap.

This model is intentionally inference-time history conditioned. Item embeddings
are frozen after training; changing the retained interaction profile changes the
user vector without retraining. It is the latent architecture-robustness model
specified in ``ActionShap_Recommendation_Spec.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProfileAggregationModel:
    """Frozen item embeddings with a profile computed from retained history.

    Parameters
    ----------
    item_embeddings:
        Array with shape ``(n_items, dimension)``.
    history_weights:
        Optional positive weights aligned with a history passed to ``score``.
        If omitted, every retained interaction has weight one.
    """

    item_embeddings: np.ndarray

    def __post_init__(self) -> None:
        e = np.asarray(self.item_embeddings, dtype=float)
        if e.ndim != 2 or e.shape[0] == 0 or e.shape[1] == 0:
            raise ValueError("item_embeddings must be a non-empty 2-D array")
        object.__setattr__(self, "item_embeddings", e)

    @property
    def n_items(self) -> int:
        return int(self.item_embeddings.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.item_embeddings.shape[1])

    def profile(
        self,
        history_items: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the normalized profile; the empty profile is zero."""
        ids = np.asarray(history_items, dtype=int)
        if ids.ndim != 1:
            raise ValueError("history_items must be 1-D")
        if np.any((ids < 0) | (ids >= self.n_items)):
            raise ValueError("history item id is outside item_embeddings")
        if weights is None:
            w = np.ones(ids.size, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != ids.shape:
                raise ValueError("weights must have the same shape as history_items")
            if np.any(w < 0) or not np.all(np.isfinite(w)):
                raise ValueError("weights must be finite and non-negative")
        denom = float(w.sum())
        if denom == 0.0 or ids.size == 0:
            return np.zeros(self.dimension, dtype=float)
        return (self.item_embeddings[ids] * w[:, None]).sum(axis=0) / denom

    def score(
        self,
        history_items: np.ndarray,
        candidate_items: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Score candidates from the retained interaction history."""
        candidates = np.asarray(candidate_items, dtype=int)
        if candidates.ndim != 1:
            raise ValueError("candidate_items must be 1-D")
        if np.any((candidates < 0) | (candidates >= self.n_items)):
            raise ValueError("candidate item id is outside item_embeddings")
        return self.item_embeddings[candidates] @ self.profile(history_items, weights)

    def score_masked(
        self,
        history_items: np.ndarray,
        candidate_items: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Score candidates after masking interaction players."""
        mask = np.asarray(mask, dtype=bool)
        ids = np.asarray(history_items, dtype=int)
        if mask.shape != ids.shape:
            raise ValueError("mask must have the same shape as history_items")
        return self.score(ids[mask], candidate_items)

    def score_downweighted(
        self,
        history_items: np.ndarray,
        candidate_items: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Score candidates after bounded interaction downweighting."""
        return self.score(history_items, candidate_items, weights)

    def score_downweighted_batch(
        self,
        history_items: np.ndarray,
        candidate_items: np.ndarray,
        weight_matrix: np.ndarray,
    ) -> np.ndarray:
        """Score many intervention weight vectors in one matrix operation."""
        history = np.asarray(history_items, dtype=int)
        candidates = np.asarray(candidate_items, dtype=int)
        weights = np.asarray(weight_matrix, dtype=float)
        if weights.ndim != 2 or weights.shape[1] != history.size:
            raise ValueError(
                "weight_matrix must have one column per history interaction"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weight_matrix must be finite and non-negative")
        denominators = weights.sum(axis=1)
        profiles = weights @ self.item_embeddings[history]
        nonempty = denominators > 0
        profiles[nonempty] /= denominators[nonempty, None]
        profiles[~nonempty] = 0.0
        return profiles @ self.item_embeddings[candidates].T


def fit_item_embeddings(
    histories: dict[int, np.ndarray],
    n_items: int,
    dimension: int = 64,
    epochs: int = 10,
    samples_per_user: int = 1,
    learning_rate: float = 0.03,
    regularization: float = 1e-4,
    seed: int = 0,
) -> ProfileAggregationModel:
    """Fit item embeddings with a leave-one-out BPR-style objective.

    For each sampled positive item, the temporary user profile is formed from
    the *other* items in that user's complete training history.  Excluding the
    positive from its own context prevents the trivial self-similarity shortcut
    present in the pilot implementation.  Only item embeddings are retained;
    the deployed user representation is always recomputed from the history
    supplied at inference time.
    """
    if n_items < 1 or dimension < 1 or epochs < 1 or samples_per_user < 1:
        raise ValueError(
            "n_items, dimension, epochs, and samples_per_user must be positive"
        )
    if learning_rate <= 0 or regularization < 0:
        raise ValueError(
            "learning_rate must be positive and regularization non-negative"
        )
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 0.05, size=(n_items, dimension))
    users = [
        (u, np.unique(np.asarray(items, dtype=int)))
        for u, items in sorted(histories.items())
        if 2 <= len(np.unique(items)) < n_items
    ]
    if not users:
        raise ValueError(
            "at least one user with two distinct training items is required"
        )
    for _ in range(epochs):
        rng.shuffle(users)
        for _, items in users:
            for _sample in range(samples_per_user):
                positive_position = int(rng.integers(items.size))
                pos = int(items[positive_position])
                context = np.delete(items, positive_position)
                neg = int(rng.integers(0, n_items))
                while neg in items:
                    neg = int(rng.integers(0, n_items))

                # Copy every vector used in the gradient so updates cannot leak
                # into another term through NumPy's indexed assignment semantics.
                context_vectors = q[context].copy()
                pos_vector = q[pos].copy()
                neg_vector = q[neg].copy()
                profile = context_vectors.mean(axis=0)
                difference = pos_vector - neg_vector
                margin = float(profile @ difference)
                bpr_gradient = 1.0 / (1.0 + np.exp(np.clip(margin, -35.0, 35.0)))

                q[pos] = pos_vector + learning_rate * (
                    bpr_gradient * profile - regularization * pos_vector
                )
                q[neg] = neg_vector + learning_rate * (
                    -bpr_gradient * profile - regularization * neg_vector
                )
                q[context] = context_vectors + learning_rate * (
                    bpr_gradient * difference / context.size
                    - regularization * context_vectors
                )
    return ProfileAggregationModel(q)
