"""Feasible interventions (Definition 2).

    Delta_j = E_x[ f(x | do(x_j <- x_j + tau_j)) - f(x) ]

The distinction from faithfulness lives entirely in ``tau_j``. A faithfulness
proxy sets ``x_j`` to a baseline, which is an unconstrained move and often an
out-of-distribution one. Here the move is bounded by what the decision-maker
could actually bring about, so ``tau_j`` must come from a declared budget --
never from a baseline, and never from the data range alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

import numpy as np

__all__ = [
    "InterventionBudget",
    "EffectProfile",
    "OutputFn",
    "intervention_effects",
    "intervention_profile",
    "sweep_budgets",
]

# A model output functional: takes a design matrix, returns one scalar per row.
# For the static regime this is the surrogate's cluster-membership
# probability; for the dynamic regime it is a ranking-quality scalar.
OutputFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class InterventionBudget:
    """Per-factor feasible perturbation budgets, in the units of the data.

    ``scale`` records how the budget was derived so that the appendix can
    report it. ``sd`` means the budget is a multiple of the within-domain
    standard deviation; ``absolute`` means a domain-specified achievable
    adjustment in natural units; ``relative`` means a fraction of the
    factor's current value, used for policy-style percentage reductions.
    """

    values: np.ndarray
    scale: Literal["sd", "absolute", "relative"]
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float)
        if v.ndim != 1:
            raise ValueError(f"budgets must be 1-D, got {v.ndim}-D")
        if np.any(v < 0):
            raise ValueError(
                "budgets must be non-negative; direction is handled by the "
                "sign of the measured effect, not by the budget"
            )
        object.__setattr__(self, "values", v)

    @classmethod
    def from_std(
        cls, X: np.ndarray, multiplier: float = 1.0
    ) -> "InterventionBudget":
        """One (or ``multiplier``) within-domain standard deviation per factor."""
        return cls(np.asarray(X, float).std(axis=0, ddof=1) * multiplier,
                   scale="sd", multiplier=multiplier)

    @classmethod
    def from_absolute(cls, values: Sequence[float]) -> "InterventionBudget":
        return cls(np.asarray(values, float), scale="absolute")

    def __len__(self) -> int:
        return self.values.size


@dataclass(frozen=True)
class EffectProfile:
    """Effects of moving each factor up, down, and whichever way works better."""

    increase: np.ndarray
    decrease: np.ndarray
    baseline: float

    @property
    def best(self) -> np.ndarray:
        """Signed effect of the more impactful of the two feasible directions."""
        up_wins = np.abs(self.increase) >= np.abs(self.decrease)
        return np.where(up_wins, self.increase, self.decrease)

    @property
    def best_direction(self) -> np.ndarray:
        """``+1`` where increasing the factor moves the output more, else ``-1``."""
        return np.where(np.abs(self.increase) >= np.abs(self.decrease), 1, -1)

    def as_dict(self) -> dict:
        return {
            "increase": self.increase.tolist(),
            "decrease": self.decrease.tolist(),
            "best": self.best.tolist(),
            "best_direction": self.best_direction.tolist(),
            "baseline": self.baseline,
        }


def intervention_profile(
    f: OutputFn,
    X: np.ndarray,
    budget: InterventionBudget,
    *,
    clip_to_observed: bool = True,
    batch_size: int | None = None,
) -> EffectProfile:
    """Definition 2 evaluated in both feasible directions.

    Evaluating only ``+tau_j`` makes the sign of every effect an artifact of
    that choice rather than a property of the factor. It matters most when the
    baseline output is near a bound: if the model is already almost certain,
    every one-sided shift can only push the output down, and ``sign(Delta_j)``
    carries no information at all. A decision-maker holding a budget can move
    a factor either way, so both ways are measured.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D design matrix, got shape {X.shape}")
    n_factors = X.shape[1]
    if len(budget) != n_factors:
        raise ValueError(
            f"budget covers {len(budget)} factors but X has {n_factors} columns"
        )

    baseline = np.asarray(f(X), dtype=float)
    if baseline.shape != (X.shape[0],):
        raise ValueError(
            f"output fn returned shape {baseline.shape}, expected ({X.shape[0]},)"
        )
    base_mean = float(baseline.mean())

    lo, hi = X.min(axis=0), X.max(axis=0)
    up = np.zeros(n_factors)
    down = np.zeros(n_factors)

    for j in range(n_factors):
        tau = budget.values[j]
        if tau == 0:
            # A zero budget is a factor that cannot be moved at all. Leaving
            # both directions at 0.0 is correct and skips two model calls.
            continue

        for sign, target in ((+1.0, up), (-1.0, down)):
            Xj = X.copy()
            shifted = Xj[:, j] + sign * tau
            if clip_to_observed:
                shifted = np.clip(shifted, lo[j], hi[j])
            Xj[:, j] = shifted
            target[j] = _apply(f, Xj, batch_size).mean() - base_mean

    return EffectProfile(increase=up, decrease=down, baseline=base_mean)


def intervention_effects(
    f: OutputFn,
    X: np.ndarray,
    budget: InterventionBudget,
    *,
    signed: bool = True,
    clip_to_observed: bool = True,
    batch_size: int | None = None,
    direction: Literal["increase", "decrease", "best"] = "best",
) -> np.ndarray:
    """Definition 2. Mean output change under a feasible shift of each factor.

    Parameters
    ----------
    signed
        Keep the sign of the mean change. ``|Delta_j|`` is taken by the metrics
        that need magnitude only.
    clip_to_observed
        Clip each shifted column back into the range observed in ``X``. This
        stops a nominally feasible budget from pushing a factor beyond
        anything the model ever saw, which would reintroduce exactly the
        out-of-distribution evaluation that Definition 2 exists to avoid.
    direction
        Which feasible move to report. ``best`` takes whichever of ``+tau_j``
        and ``-tau_j`` moves the output further, which is the only choice that
        does not let a one-sided convention determine the sign; see
        `intervention_profile`. The one-sided modes cost half as many model
        calls and are what a genuinely one-directional factor needs.

    Returns
    -------
    Array of shape ``(n_factors,)``.
    """
    if direction not in ("increase", "decrease", "best"):
        raise ValueError(f"unknown direction {direction!r}")

    if direction == "best":
        effects = intervention_profile(
            f, X, budget,
            clip_to_observed=clip_to_observed, batch_size=batch_size,
        ).best
        return effects if signed else np.abs(effects)

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D design matrix, got shape {X.shape}")
    n_factors = X.shape[1]
    if len(budget) != n_factors:
        raise ValueError(
            f"budget covers {len(budget)} factors but X has {n_factors} columns"
        )

    baseline = np.asarray(f(X), dtype=float)
    if baseline.shape != (X.shape[0],):
        raise ValueError(
            f"output fn returned shape {baseline.shape}, expected ({X.shape[0]},)"
        )
    base_mean = baseline.mean()
    step = 1.0 if direction == "increase" else -1.0

    lo, hi = X.min(axis=0), X.max(axis=0)
    effects = np.zeros(n_factors, dtype=float)

    for j in range(n_factors):
        tau = budget.values[j]
        if tau == 0:
            continue

        Xj = X.copy()
        shifted = Xj[:, j] + step * tau
        if clip_to_observed:
            shifted = np.clip(shifted, lo[j], hi[j])
        Xj[:, j] = shifted

        effects[j] = _apply(f, Xj, batch_size).mean() - base_mean

    return effects if signed else np.abs(effects)


def sweep_budgets(
    f: OutputFn,
    X: np.ndarray,
    multipliers: Sequence[float] = (0.5, 1.0, 1.5),
    **kwargs,
) -> dict[float, np.ndarray]:
    """Effects at several budget multiples, for the sensitivity grid (Sec. 4.4)."""
    return {
        float(m): intervention_effects(f, X, InterventionBudget.from_std(X, m), **kwargs)
        for m in multipliers
    }


def _apply(f: OutputFn, X: np.ndarray, batch_size: int | None) -> np.ndarray:
    if batch_size is None or X.shape[0] <= batch_size:
        return np.asarray(f(X), dtype=float)
    return np.concatenate(
        [np.asarray(f(X[i : i + batch_size]), float)
         for i in range(0, X.shape[0], batch_size)]
    )
