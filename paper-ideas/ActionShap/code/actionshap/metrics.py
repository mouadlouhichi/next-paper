"""Actionability metrics.

Implements Definitions 3-7 of the manuscript:

    Def. 3  stability          s_j
    Def. 4  Actionability Score AS_j = m_j * |Delta_j| * s_j
    Eq.  8  held-out control    AS^-m_j = |Delta_j| * s_j
    Def. 5  alignment           AIA = rho(|phi|, |Delta|)
    Def. 6  top-k precision     P@k
    Def. 7  intervention regret Reg

Every function here is pure: it takes attribution and intervention vectors
and returns numbers. Producing those vectors is the job of `attribution`
and `intervention`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "stability",
    "actionability_score",
    "actionability_score_heldout",
    "alignment",
    "AlignmentResult",
    "topk_intervention_precision",
    "intervention_regret",
    "RegretResult",
]

# Guards the stability ratio when a mean attribution is ~0. Deliberately
# small: a factor whose mean attribution is far below this is genuinely
# unstable and should score near 0, not be rescued by the epsilon.
_STABILITY_EPS = 1e-12


def stability(repeated: np.ndarray, eps: float = _STABILITY_EPS) -> np.ndarray:
    """Definition 3. Stability of each factor across repeated attribution runs.

    Parameters
    ----------
    repeated
        Array of shape ``(R, n_factors)``: one attribution vector per run.
    eps
        Added to the denominator to keep the ratio finite.

    Returns
    -------
    Array of shape ``(n_factors,)`` in [0, 1]. A factor whose dispersion
    across runs exceeds its mean magnitude scores 0.
    """
    repeated = np.asarray(repeated, dtype=float)
    if repeated.ndim != 2:
        raise ValueError(f"expected (R, n_factors), got shape {repeated.shape}")
    if repeated.shape[0] < 2:
        raise ValueError(
            f"stability needs R >= 2 runs, got R={repeated.shape[0]}. "
            "A single run carries no dispersion information."
        )

    # ddof=1: these R runs are a sample of the seed distribution, not the
    # population. With R=5 the difference from ddof=0 is ~12%, which matters
    # because s_j feeds multiplicatively into AS_j.
    sd = repeated.std(axis=0, ddof=1)
    mean_mag = np.abs(repeated.mean(axis=0))
    return np.clip(1.0 - sd / (mean_mag + eps), 0.0, 1.0)


def actionability_score(
    modifiability: np.ndarray,
    effect: np.ndarray,
    stability_: np.ndarray,
) -> np.ndarray:
    """Definition 4. ``AS_j = m_j * |Delta_j| * s_j``.

    The product form is a modelling choice, justified in the manuscript by
    each factor being individually necessary. `sensitivity.py` tests it
    against additive and min-based alternatives.
    """
    m, d, s = _broadcast_check(modifiability, effect, stability_)
    return m * np.abs(d) * s


def actionability_score_heldout(
    effect: np.ndarray,
    stability_: np.ndarray,
) -> np.ndarray:
    """Equation 8. Modifiability-held-out control ``AS^-m_j = |Delta_j| * s_j``.

    Reported alongside every Actionability Score. Because ``m_j = 0`` forces
    ``AS_j = 0``, any claim about how AS ranks unmodifiable factors is true by
    construction; this control carries no such tautology (Remark 1).
    """
    d, s = np.asarray(effect, float), np.asarray(stability_, float)
    if d.shape != s.shape:
        raise ValueError(f"shape mismatch: effect {d.shape}, stability {s.shape}")
    return np.abs(d) * s


@dataclass(frozen=True)
class AlignmentResult:
    """Definition 5, reported both unrestricted and restricted to modifiables."""

    spearman: float
    spearman_p: float
    kendall: float
    kendall_p: float
    n_factors: int

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"AIA(rho={self.spearman:+.3f}, tau={self.kendall:+.3f}, "
            f"n={self.n_factors})"
        )


def alignment(
    attribution: np.ndarray,
    effect: np.ndarray,
    modifiability: np.ndarray | None = None,
    restrict_to_modifiable: bool = False,
) -> AlignmentResult:
    """Definition 5. Rank agreement between |attribution| and |effect|.

    Spearman is the primary statistic; Kendall's tau is reported for
    robustness. Set ``restrict_to_modifiable`` to compute over
    ``{j : m_j > 0}`` only -- the difference between the two is the visible
    signature of the infeasibility mechanism (H3).
    """
    a = np.abs(np.asarray(attribution, float))
    d = np.abs(np.asarray(effect, float))
    if a.shape != d.shape:
        raise ValueError(f"shape mismatch: attribution {a.shape}, effect {d.shape}")

    if restrict_to_modifiable:
        if modifiability is None:
            raise ValueError("restrict_to_modifiable=True requires modifiability")
        keep = np.asarray(modifiability, float) > 0
        if keep.sum() < 3:
            raise ValueError(
                f"only {keep.sum()} modifiable factors; rank correlation is "
                "not meaningful below 3"
            )
        a, d = a[keep], d[keep]

    # Constant input makes rho undefined and scipy returns nan with a warning.
    # Surface it as an explicit error rather than letting nan propagate into
    # a results table.
    if np.ptp(a) == 0 or np.ptp(d) == 0:
        raise ValueError(
            "attribution or effect vector is constant; rank correlation undefined"
        )

    rho, rho_p = stats.spearmanr(a, d)
    tau, tau_p = stats.kendalltau(a, d)
    return AlignmentResult(float(rho), float(rho_p), float(tau), float(tau_p), a.size)


def topk_intervention_precision(
    attribution: np.ndarray,
    effect: np.ndarray,
    modifiability: np.ndarray,
    k: int,
    delta: float,
    direction: np.ndarray | None = None,
) -> float:
    """Definition 6. Fraction of the top-k attributed factors that are usable.

    A factor counts only if it is modifiable, its realized effect clears the
    practical magnitude threshold ``delta``, and -- when a direction is
    supplied -- that effect points the way the attribution implied.

    Note the top-k set is taken over ALL factors, not just modifiable ones:
    the metric is meant to penalize a method for spending its top-k budget
    on factors nobody can act on.

    Parameters
    ----------
    direction
        Per-factor expected sign of the effect, from
        `attribution.feature_direction`. Pass ``None`` to drop the sign
        condition entirely.

        This is a parameter rather than ``sign(attribution)`` because the two
        are different quantities. A global attribution reduced to magnitudes
        has no sign at all, and even a signed one describes a factor's current
        *level* rather than the effect of *increasing* it -- so comparing
        ``sign(phi_j)`` against ``sign(Delta_j)`` compares a level against a
        derivative and can be systematically wrong. Supply a real directional
        estimate, or omit the condition and say so.
    """
    a, d, m = _broadcast_check(attribution, effect, modifiability)
    if not 1 <= k <= a.size:
        raise ValueError(f"k={k} out of range for {a.size} factors")
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")

    # Descending by |phi|. argsort is ascending, so negate. Ties break by
    # index, which is arbitrary but deterministic.
    topk = np.argsort(-np.abs(a))[:k]

    usable = (m[topk] > 0) & (np.abs(d[topk]) >= delta)

    if direction is not None:
        dirn = np.asarray(direction, float)
        if dirn.shape != a.shape:
            raise ValueError(f"direction has shape {dirn.shape}, expected {a.shape}")
        # A factor with no established direction (0) cannot be contradicted.
        agrees = (dirn[topk] == 0) | (np.sign(d[topk]) == dirn[topk])
        usable &= agrees

    return float(usable.sum()) / k


@dataclass(frozen=True)
class RegretResult:
    """Definition 7, with the two factor identities kept for the case studies."""

    regret: float
    normalized: float
    chosen_index: int
    optimal_index: int
    chosen_effect: float
    optimal_effect: float

    @property
    def is_optimal(self) -> bool:
        return self.chosen_index == self.optimal_index


def intervention_regret(
    attribution: np.ndarray,
    effect: np.ndarray,
    modifiability: np.ndarray,
    candidate_mask: np.ndarray | None = None,
) -> RegretResult:
    """Definition 7. Outcome forgone by acting on the attribution's pick.

    ``candidate_mask`` restricts the comparison to a pre-declared candidate
    set. This is mandatory in the dynamic regime, where an exhaustive sweep
    over players is infeasible and regret would otherwise silently depend on
    which method is being evaluated.
    """
    a, d, m = _broadcast_check(attribution, effect, modifiability)

    eligible = m > 0
    if candidate_mask is not None:
        eligible &= np.asarray(candidate_mask, bool)
    if not eligible.any():
        raise ValueError("no eligible factors: regret is undefined")

    # Restrict first, then argmax within the restricted view, then map the
    # index back. Masking with -inf would also work but breaks if a caller
    # passes an all-zero attribution.
    idx = np.flatnonzero(eligible)
    chosen = idx[np.argmax(np.abs(a[idx]))]
    optimal = idx[np.argmax(np.abs(d[idx]))]

    best, got = abs(float(d[optimal])), abs(float(d[chosen]))
    regret = best - got
    return RegretResult(
        regret=regret,
        # best == 0 means no feasible intervention moves the output at all;
        # every choice is equally (un)regrettable, so normalized regret is 0.
        normalized=regret / best if best > 0 else 0.0,
        chosen_index=int(chosen),
        optimal_index=int(optimal),
        chosen_effect=got,
        optimal_effect=best,
    )


def _broadcast_check(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Coerce to float arrays and require identical 1-D shapes."""
    out = tuple(np.asarray(x, dtype=float) for x in arrays)
    shapes = {x.shape for x in out}
    if len(shapes) != 1:
        raise ValueError(f"shape mismatch across inputs: {[x.shape for x in out]}")
    if out[0].ndim != 1:
        raise ValueError(f"expected 1-D vectors, got {out[0].ndim}-D")
    return out
