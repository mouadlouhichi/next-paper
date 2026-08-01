"""Actionability-guided reranking (Definition 8).

    phi~_j = (1 - eta) * |phi_j| / max_k |phi_k| + eta * AS_j / max_k AS_k

Both terms are max-normalized before mixing because attribution magnitudes
and Actionability Scores are in unrelated units; without normalization eta
would not interpolate between them in any interpretable way.

Evaluated by the intervention switch rate and by realized regret reduction,
so that the remedy is reported in the same currency as the diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import alignment, intervention_regret

__all__ = ["rerank", "RerankOutcome", "eta_sweep"]


def rerank(attribution: np.ndarray, action_score: np.ndarray, eta: float) -> np.ndarray:
    """Definition 8. Blend attribution magnitude with actionability."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    a = np.abs(np.asarray(attribution, float))
    s = np.asarray(action_score, float)
    if a.shape != s.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {s.shape}")

    return (1.0 - eta) * _max_normalize(a) + eta * _max_normalize(s)


def _max_normalize(v: np.ndarray) -> np.ndarray:
    # An all-zero vector arises legitimately: every factor immutable makes
    # every AS_j zero. Returning zeros keeps rerank() a pure function of the
    # other term rather than raising on a valid input.
    peak = np.max(np.abs(v))
    return v / peak if peak > 0 else np.zeros_like(v)


@dataclass(frozen=True)
class RerankOutcome:
    """One point on the eta sweep reported in Table 14."""

    eta: float
    switched: bool
    chosen_index: int
    normalized_regret: float
    aia: float


def eta_sweep(
    attribution: np.ndarray,
    action_score: np.ndarray,
    effect: np.ndarray,
    modifiability: np.ndarray,
    etas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    candidate_mask: np.ndarray | None = None,
) -> list[RerankOutcome]:
    """Trace switch rate, regret, and alignment as eta varies.

    The baseline for "switched" is the eta=0 choice, which by construction is
    the unmodified attribution's pick.
    """
    if not etas or etas[0] != 0.0:
        raise ValueError("the sweep must start at eta=0 to define the baseline")

    baseline_idx: int | None = None
    out: list[RerankOutcome] = []

    for eta in etas:
        blended = rerank(attribution, action_score, eta)
        reg = intervention_regret(blended, effect, modifiability, candidate_mask)
        if baseline_idx is None:
            baseline_idx = reg.chosen_index

        # Alignment is recomputed against the blended ordering to expose the
        # trade-off: pushing eta up should cut regret, and may cost AIA.
        try:
            aia = alignment(blended, effect).spearman
        except ValueError:
            aia = float("nan")

        out.append(
            RerankOutcome(
                eta=float(eta),
                switched=reg.chosen_index != baseline_idx,
                chosen_index=reg.chosen_index,
                normalized_regret=reg.normalized,
                aia=aia,
            )
        )
    return out
