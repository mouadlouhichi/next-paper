"""Significance testing for method comparisons (Section 4.7.3, Appendix A2).

Paired tests over per-factor differences in the static regime and per-user
differences in the dynamic regime, with Holm-Bonferroni correction across the
family of pairwise comparisons and Wilcoxon signed-rank as a distribution-free
check. Cohen's d_z is reported because a corrected p-value alone says nothing
about whether a difference is large enough to matter.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "ClusteredPairedComparison",
    "PairedComparison",
    "compare_methods",
    "holm_bonferroni",
    "paired_user_seed_comparison",
]


@dataclass(frozen=True)
class PairedComparison:
    """One pairwise method comparison, before and after correction."""

    method_a: str
    method_b: str
    mean_difference: float
    t_statistic: float
    p_value: float
    p_corrected: float
    cohens_dz: float
    wilcoxon_p: float
    n_pairs: int
    shapiro_p: float

    @property
    def significant(self) -> bool:
        return self.p_corrected < 0.05

    @property
    def normality_suspect(self) -> bool:
        """When true, prefer the Wilcoxon p-value in the write-up."""
        return self.shapiro_p < 0.05

    @property
    def effect_label(self) -> str:
        d = abs(self.cohens_dz)
        return (
            "negligible"
            if d < 0.2
            else "small"
            if d < 0.5
            else "medium"
            if d < 0.8
            else "large"
        )


def holm_bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down correction.

    Returns corrected p-values in the input order. Enforces monotonicity so a
    corrected value is never smaller than one earlier in the sorted sequence,
    and clips at 1.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")

    order = np.argsort(p)
    n = p.size
    adjusted = np.empty(n, dtype=float)

    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def compare_methods(
    per_unit_scores: Mapping[str, np.ndarray],
    *,
    alternative: str = "two-sided",
) -> list[PairedComparison]:
    """All pairwise paired comparisons across methods, Holm-corrected.

    Parameters
    ----------
    per_unit_scores
        Method name -> vector of per-unit scores. A "unit" is a factor in the
        static regime and a user in the dynamic regime. All vectors must be the
        same length and aligned elementwise: entry *i* of every vector must
        refer to the same unit, or the pairing is meaningless.
    """
    names = sorted(per_unit_scores)
    if len(names) < 2:
        raise ValueError("need at least two methods to compare")

    lengths = {len(np.asarray(v)) for v in per_unit_scores.values()}
    if len(lengths) != 1:
        raise ValueError(f"all methods need equally many units, got lengths {lengths}")
    n = lengths.pop()
    if n < 3:
        raise ValueError(f"need at least 3 paired units, got {n}")

    raw: list[tuple[str, str, float, float, float, float, float, float]] = []
    for a, b in itertools.combinations(names, 2):
        x = np.asarray(per_unit_scores[a], float)
        y = np.asarray(per_unit_scores[b], float)
        d = x - y

        if np.allclose(d, 0):
            # Identical vectors: the tests are undefined but the answer is
            # unambiguous, so short-circuit rather than emit nan.
            raw.append((a, b, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0))
            continue
        if np.ptp(d) == 0.0:
            # A constant non-zero paired effect has zero standard error. Avoid
            # asking SciPy to divide by zero (and emitting precision warnings).
            direction = float(np.sign(d.mean()))
            raw.append(
                (
                    a,
                    b,
                    float(d.mean()),
                    direction * np.inf,
                    0.0,
                    direction * np.inf,
                    0.0,
                    float("nan"),
                )
            )
            continue

        t_stat, p = stats.ttest_rel(x, y, alternative=alternative)
        sd = d.std(ddof=1)
        if sd > 0:
            dz = float(d.mean() / sd)
        else:
            # A constant non-zero difference is a perfectly consistent effect,
            # so d_z diverges. Reporting 0.0 here would state the opposite of
            # the truth. Only an all-zero difference is genuinely no effect,
            # and that case is short-circuited above.
            dz = float(np.sign(d.mean()) * np.inf)

        try:
            _, wp = stats.wilcoxon(x, y, alternative=alternative)
        except ValueError:
            wp = float("nan")

        # Shapiro needs n >= 3 and is unreliable past a few thousand points,
        # where it flags trivial departures. Report nan rather than a
        # misleading verdict.
        sp = float(stats.shapiro(d).pvalue) if 3 <= n <= 5000 else float("nan")

        raw.append((a, b, float(d.mean()), float(t_stat), float(p), dz, float(wp), sp))

    corrected = holm_bonferroni([r[4] for r in raw])

    return [
        PairedComparison(
            method_a=a,
            method_b=b,
            mean_difference=md,
            t_statistic=t,
            p_value=p,
            p_corrected=float(pc),
            cohens_dz=dz,
            wilcoxon_p=wp,
            n_pairs=n,
            shapiro_p=sp,
        )
        for (a, b, md, t, p, dz, wp, sp), pc in zip(raw, corrected)
    ]


@dataclass(frozen=True)
class ClusteredPairedComparison:
    """Paired comparison after averaging repeated seeds within each user."""

    mean_difference: float
    median_difference: float
    ci95_low: float
    ci95_high: float
    permutation_p: float
    sign_test_p: float
    cohens_dz: float
    n_users: int
    wins: int
    losses: int
    ties: int
    bootstrap_draws: int
    permutation_draws: int


def paired_user_seed_comparison(
    method_a: np.ndarray,
    method_b: np.ndarray,
    *,
    bootstrap_draws: int = 10_000,
    permutation_draws: int = 10_000,
    seed: int = 42,
) -> ClusteredPairedComparison:
    """Compare methods without treating repeated model seeds as new users.

    Inputs may be ``(users, seeds)`` matrices or one-dimensional user vectors.
    Finite seed observations are averaged within each user first.  Bootstrap
    resampling and sign flips are then performed over distinct users, the actual
    inferential unit of the recommendation experiment.
    """
    a = np.asarray(method_a, dtype=float)
    b = np.asarray(method_b, dtype=float)
    if a.shape != b.shape or a.ndim not in (1, 2):
        raise ValueError("method arrays must have the same 1-D or 2-D shape")
    if bootstrap_draws < 1 or permutation_draws < 1:
        raise ValueError("resampling draw counts must be positive")
    if a.ndim == 2:

        def finite_row_mean(values: np.ndarray) -> np.ndarray:
            finite = np.isfinite(values)
            counts = finite.sum(axis=1)
            sums = np.where(finite, values, 0.0).sum(axis=1)
            means = np.full(values.shape[0], np.nan, dtype=float)
            np.divide(sums, counts, out=means, where=counts > 0)
            return means

        a = finite_row_mean(a)
        b = finite_row_mean(b)
    valid = np.isfinite(a) & np.isfinite(b)
    differences = a[valid] - b[valid]
    if differences.size < 3:
        raise ValueError("at least three distinct paired users are required")
    rng = np.random.default_rng(seed)
    # Resample in bounded chunks so an all-user MovieLens or sparse-dataset run
    # does not allocate a draws-by-users matrix hundreds of megabytes large.
    bootstrap_means = np.empty(bootstrap_draws, dtype=float)
    chunk = 256
    for start in range(0, bootstrap_draws, chunk):
        stop = min(start + chunk, bootstrap_draws)
        indices = rng.integers(
            0, differences.size, size=(stop - start, differences.size)
        )
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    exceedances = 0
    observed_mean = abs(differences.mean())
    for start in range(0, permutation_draws, chunk):
        stop = min(start + chunk, permutation_draws)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(stop - start, differences.size))
        permuted = (signs * differences).mean(axis=1)
        exceedances += int(np.count_nonzero(np.abs(permuted) >= observed_mean))
    permutation_p = (exceedances + 1) / (permutation_draws + 1)
    positive = int(np.count_nonzero(differences > 0))
    negative = int(np.count_nonzero(differences < 0))
    ties = int(differences.size - positive - negative)
    non_ties = positive + negative
    sign_p = (
        1.0
        if non_ties == 0
        else float(stats.binomtest(min(positive, negative), non_ties, 0.5).pvalue)
    )
    standard_deviation = differences.std(ddof=1)
    if np.allclose(differences, 0.0):
        dz = 0.0
    elif standard_deviation > 0:
        dz = float(differences.mean() / standard_deviation)
    else:
        dz = float(np.sign(differences.mean()) * np.inf)
    return ClusteredPairedComparison(
        mean_difference=float(differences.mean()),
        median_difference=float(np.median(differences)),
        ci95_low=float(np.quantile(bootstrap_means, 0.025)),
        ci95_high=float(np.quantile(bootstrap_means, 0.975)),
        permutation_p=float(permutation_p),
        sign_test_p=sign_p,
        cohens_dz=dz,
        n_users=int(differences.size),
        wins=positive,
        losses=negative,
        ties=ties,
        bootstrap_draws=int(bootstrap_draws),
        permutation_draws=int(permutation_draws),
    )
