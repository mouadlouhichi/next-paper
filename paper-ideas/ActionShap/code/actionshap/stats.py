"""Significance testing for method comparisons (Section 4.7.3, Appendix A2).

Paired tests over per-factor differences in the static regime and per-user
differences in the dynamic regime, with Holm-Bonferroni correction across the
family of pairwise comparisons and Wilcoxon signed-rank as a distribution-free
check. Cohen's d_z is reported because a corrected p-value alone says nothing
about whether a difference is large enough to matter.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

__all__ = ["PairedComparison", "compare_methods", "holm_bonferroni"]


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
        return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


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
            method_a=a, method_b=b, mean_difference=md, t_statistic=t,
            p_value=p, p_corrected=float(pc), cohens_dz=dz,
            wilcoxon_p=wp, n_pairs=n, shapiro_p=sp,
        )
        for (a, b, md, t, p, dz, wp, sp), pc in zip(raw, corrected)
    ]
