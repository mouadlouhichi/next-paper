"""Decomposition of the misalignment gap (Section 3.7, Corollary 1).

Proposition 1 fails in exactly three ways, and each gets a switch:

    H1  curvature      nonlinear main effects  -> replace each by its linear part
    H2  interaction    cross terms             -> drop the pairwise terms
    H3  infeasibility  budget and modifiability-> common budget, modifiables only

H1 and H2 are *model* switches derived from a functional-ANOVA decomposition
of a single fitted surrogate; H3 is an *evaluation* switch. Together H1 and H2
leave ``intercept + sum_j (a_j + b_j x_j)``, which is exactly locally linear,
so the grand coalition satisfies Proposition 1's hypotheses on the surrogate.

The three switches are then treated as players in a second cooperative game,

    u(Q) = AIA(Q) - AIA({}),      psi_c = Shapley allocation of u,

so the shares sum to the measured gap by efficiency rather than by post-hoc
normalization (Corollary 1). Note the symbols: this game uses ``u`` and
``psi``, never ``v`` and ``phi``, which belong to the attribution game.

IDENTIFIABILITY. Under dependent inputs the main/interaction split is not
unique and a main effect can absorb interaction signal. Both switches are
therefore derived as projections of ONE purified fit: purification (Lengerich
et al. 2020, via ``interpret.utils.purify``) moves interaction mass into main
effects until the pair tensors have zero weighted marginals, which makes the
split unique. Fitting an interaction-free and an interaction-bearing model
independently would reintroduce exactly the absorption this prevents.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import factorial
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from .metrics import alignment

__all__ = [
    "Switch",
    "ALL_SWITCHES",
    "PurifiedGA2M",
    "MechanismShares",
    "decompose_misalignment",
    "shapley_over_switches",
]

Switch = Literal["H1", "H2", "H3"]
ALL_SWITCHES: tuple[Switch, ...] = ("H1", "H2", "H3")


# ==========================================================================
# The purified surrogate and its switched projections
# ==========================================================================


class PurifiedGA2M:
    """A GA2M whose main/interaction split is identifiable, plus its switches.

    Fit with ``max_interaction_bins == max_bins`` so that a feature is binned
    identically whether it appears alone or in a pair. Without that, the
    main-effect impurities that purification returns live on a different grid
    from the model's own main effects and cannot be folded in without an
    interpolation step that would itself blur the split.
    """

    def __init__(self, ebm, X_ref: np.ndarray):
        self.ebm = ebm
        self.X_ref = np.asarray(X_ref, dtype=float)
        self.n_features = self.X_ref.shape[1]

        for f, levels in enumerate(ebm.bins_):
            if len(levels) != 1:
                raise ValueError(
                    f"feature {f} has {len(levels)} binning levels. Refit with "
                    "max_interaction_bins == max_bins so mains and pairs share "
                    "a grid; see this class's docstring."
                )
        self.cuts: list[np.ndarray] = [np.asarray(b[0], float) for b in ebm.bins_]

        self._purify()
        self._fit_linear_parts()

    # -- construction -----------------------------------------------------

    def _purify(self) -> None:
        """Split the fit into intercept, pure mains, and pure pairs."""
        from interpret.utils import purify

        self.intercept = float(np.ravel(self.ebm.intercept_)[0])
        self.mains: dict[int, np.ndarray] = {}
        self.pairs: dict[tuple[int, int], np.ndarray] = {}

        for term, scores in zip(self.ebm.term_features_, self.ebm.term_scores_):
            if len(term) == 1:
                j = term[0]
                self.mains[j] = self.mains.get(j, 0.0) + np.asarray(scores, float)
            elif len(term) != 2:
                raise ValueError(
                    f"term {term} has order {len(term)}. Only mains and pairs are "
                    "supported; higher-order structure would fall outside the "
                    "H1/H2 split and must be disclosed as residual instead."
                )

        for i, term in enumerate(self.ebm.term_features_):
            if len(term) != 2:
                continue
            j, k = term
            pure, impurities, icept = purify(
                np.asarray(self.ebm.term_scores_[i], float),
                np.asarray(self.ebm.bin_weights_[i], float),
            )
            self.pairs[(j, k)] = np.asarray(pure, float)
            self.intercept += float(np.ravel(icept)[0])

            # Fold the extracted main-effect impurities back into the mains.
            # This is the step that makes the split unique: after it, every
            # pair tensor has zero weighted marginals and carries only
            # variance no subset of features could explain.
            for axes, vec in impurities:
                ax = axes[0] if isinstance(axes, tuple) else axes
                target = (j, k)[ax]
                vec = np.asarray(vec, float)
                if target not in self.mains:
                    self.mains[target] = np.zeros_like(vec)
                self.mains[target] = self.mains[target] + vec

        for j in range(self.n_features):
            self.mains.setdefault(j, np.zeros(len(self.cuts[j]) + 3))

    def _fit_linear_parts(self) -> None:
        """L2 projection of each purified main effect onto a line.

        Fitting against the reference sample rather than against bin centres
        weights the projection by the empirical distribution automatically,
        which is the right inner product for a functional-ANOVA projection.
        """
        idx = self._bin_indices(self.X_ref)
        self.linear: dict[int, tuple[float, float]] = {}
        for j in range(self.n_features):
            g = self.mains[j][idx[:, j]]
            x = self.X_ref[:, j]
            if np.ptp(x) == 0:
                self.linear[j] = (float(g.mean()), 0.0)
                continue
            b, a = np.polyfit(x, g, deg=1)
            self.linear[j] = (float(a), float(b))

    def _bin_indices(self, X: np.ndarray) -> np.ndarray:
        """Map values to EBM bin indices.

        Index 0 is reserved for missing and the final index for unknown, so
        real bins start at 1 -- hence the ``+ 1``.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty(X.shape, dtype=np.intp)
        for j in range(self.n_features):
            out[:, j] = np.searchsorted(self.cuts[j], X[:, j], side="right") + 1
        return out

    # -- the switched models ----------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        *,
        linearize_mains: bool = False,
        drop_pairs: bool = False,
    ) -> np.ndarray:
        """Evaluate the surrogate under a chosen combination of switches.

        ``linearize_mains`` is H1 and ``drop_pairs`` is H2. With both set the
        result is exactly affine in ``X``, which is what makes the grand
        coalition satisfy Proposition 1 rather than merely approximate it.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.full(X.shape[0], self.intercept, dtype=float)

        if linearize_mains:
            # Evaluated continuously, not through bins: a binned line is a
            # staircase, and Proposition 1 needs genuine linearity.
            for j in range(self.n_features):
                a, b = self.linear[j]
                out += a + b * X[:, j]
        else:
            idx = self._bin_indices(X)
            for j in range(self.n_features):
                out += self.mains[j][idx[:, j]]

        if not drop_pairs and self.pairs:
            idx = self._bin_indices(X)
            for (j, k), tensor in self.pairs.items():
                out += tensor[idx[:, j], idx[:, k]]

        return out

    def output_fn(self, **switches) -> Callable[[np.ndarray], np.ndarray]:
        """A plain callable, for handing to `intervention.intervention_effects`."""
        return lambda X: self.predict(X, **switches)

    # -- diagnostics -------------------------------------------------------

    def fidelity(self, X: np.ndarray, y: np.ndarray) -> float:
        """R^2 of the purified surrogate against the model it stands in for.

        Reported in Table 8. A large residual in Table 15 is only interpretable
        alongside this number: high fidelity with a large residual implicates a
        mechanism outside H1-H3, low fidelity merely implicates the surrogate.
        """
        y = np.asarray(y, float)
        pred = self.predict(X)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def concurvity(self, X: np.ndarray | None = None) -> np.ndarray:
        """Per-feature R^2 of predicting x_j from the other features.

        High values mean the main/interaction split is unstable no matter how
        carefully it is computed, because the features carry overlapping
        information. Section 3.7.3 commits to reporting this.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_score

        X = self.X_ref if X is None else np.asarray(X, float)
        out = np.empty(self.n_features)
        for j in range(self.n_features):
            others = np.delete(X, j, axis=1)
            if others.shape[1] == 0:
                out[j] = 0.0
                continue
            scores = cross_val_score(
                HistGradientBoostingRegressor(max_iter=100, random_state=0),
                others, X[:, j], cv=3, scoring="r2",
            )
            out[j] = max(0.0, float(scores.mean()))
        return out

    def shape_shift(self, ebm_no_interactions) -> np.ndarray:
        """Displacement of each main effect when pairs are present vs absent.

        Large displacement means interaction mass is being absorbed into main
        effects and the H1/H2 split should not be trusted. Normalized by the
        weighted RMS of the interaction-free shape so it is unit-free.
        """
        idx = self._bin_indices(self.X_ref)
        out = np.empty(self.n_features)

        solo: dict[int, np.ndarray] = {}
        for term, scores in zip(
            ebm_no_interactions.term_features_, ebm_no_interactions.term_scores_
        ):
            if len(term) == 1:
                solo[term[0]] = np.asarray(scores, float)

        for j in range(self.n_features):
            if j not in solo:
                out[j] = np.nan
                continue
            a = self.mains[j][idx[:, j]]
            b = solo[j][idx[:, j]]
            # Shape functions are identified only up to an additive constant,
            # so centre both before comparing.
            a, b = a - a.mean(), b - b.mean()
            denom = np.sqrt((b**2).mean())
            out[j] = float(np.sqrt(((a - b) ** 2).mean()) / denom) if denom > 0 else np.nan
        return out


# ==========================================================================
# The decomposition game
# ==========================================================================


def shapley_over_switches(u: dict[frozenset[Switch], float]) -> dict[Switch, float]:
    """Exact Shapley values for the three-player switch game.

    With |Sigma| = 3 there are only 8 coalitions, so this enumerates rather
    than sampling: the allocation is exact and the efficiency check in
    `decompose_misalignment` is meaningful to machine precision.
    """
    players = list(ALL_SWITCHES)
    n = len(players)
    missing = [frozenset(c) for r in range(n + 1)
               for c in itertools.combinations(players, r)
               if frozenset(c) not in u]
    if missing:
        raise ValueError(f"u is missing coalitions: {[set(m) for m in missing]}")

    psi: dict[Switch, float] = {}
    for c in players:
        rest = [p for p in players if p != c]
        total = 0.0
        for r in range(len(rest) + 1):
            for combo in itertools.combinations(rest, r):
                Q = frozenset(combo)
                weight = factorial(len(Q)) * factorial(n - len(Q) - 1) / factorial(n)
                total += weight * (u[Q | {c}] - u[Q])
        psi[c] = total
    return psi


@dataclass(frozen=True)
class MechanismShares:
    """Table 15: the three allocations, the measured gap, and the residual."""

    psi: dict[Switch, float]
    aia_baseline: float
    aia_grand: float
    coalition_aia: dict[frozenset[Switch], float]
    residual: float
    cardinality_matched: bool

    @property
    def closed_gap(self) -> float:
        return self.aia_grand - self.aia_baseline

    @property
    def dominant(self) -> Switch:
        return max(self.psi, key=lambda k: self.psi[k])

    def check_efficiency(self, atol: float = 1e-9) -> None:
        """Corollary 1 must hold to machine precision, or something is wrong."""
        total = sum(self.psi.values())
        if not np.isclose(total, self.closed_gap, atol=atol):
            raise AssertionError(
                f"Corollary 1 violated: shares sum to {total:.12f} but the "
                f"measured gap is {self.closed_gap:.12f}. Efficiency is "
                "unconditional, so this indicates a bug in u(Q), not a "
                "modelling problem."
            )


def decompose_misalignment(
    surrogate: PurifiedGA2M,
    X: np.ndarray,
    attribute: Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray], np.ndarray],
    modifiability: np.ndarray,
    budgets: np.ndarray,
    *,
    common_budget: float | None = None,
    cardinality_matched: bool = True,
    n_matched_draws: int = 200,
    seed: int = 42,
) -> MechanismShares:
    """Allocate the misalignment gap across H1, H2, and H3.

    Parameters
    ----------
    attribute
        ``(output_fn, X) -> phi``. Both phi AND the intervention effects are
        recomputed on every switched model: reusing the baseline attribution
        would measure the wrong thing, since the switches change the model.
    common_budget
        The single budget H3 imposes. Defaults to the mean of ``budgets``.
    cardinality_matched
        H3 shrinks the factor set, and a rank correlation over fewer items is
        systematically different regardless of feasibility. When enabled, the
        H3 coalitions are corrected by the expected shift from evaluating on a
        random subset of the same size, so the allocation reflects feasibility
        rather than set size.

    Returns
    -------
    `MechanismShares`, with efficiency already verified.
    """
    from .intervention import InterventionBudget, intervention_effects

    X = np.asarray(X, float)
    m = np.asarray(modifiability, float)
    budgets = np.asarray(budgets, float)
    rng = np.random.default_rng(seed)

    if not (X.shape[1] == m.size == budgets.size):
        raise ValueError(
            f"shape mismatch: X has {X.shape[1]} columns, modifiability {m.size}, "
            f"budgets {budgets.size}"
        )
    free = m >= 1.0
    if free.sum() < 3:
        raise ValueError(
            f"H3 restricts to fully modifiable factors (m_j = 1) and only "
            f"{free.sum()} qualify; rank correlation needs at least 3."
        )
    if common_budget is None:
        common_budget = float(budgets.mean())

    coalition_aia: dict[frozenset[Switch], float] = {}

    for r in range(len(ALL_SWITCHES) + 1):
        for combo in itertools.combinations(ALL_SWITCHES, r):
            Q = frozenset(combo)
            f = surrogate.output_fn(
                linearize_mains="H1" in Q, drop_pairs="H2" in Q
            )

            tau = np.full(X.shape[1], common_budget) if "H3" in Q else budgets
            delta = intervention_effects(f, X, InterventionBudget(tau, scale="absolute"))
            phi = np.asarray(attribute(f, X), float)

            if "H3" in Q:
                a = alignment(phi[free], delta[free]).spearman
                if cardinality_matched:
                    a -= _size_effect(phi, delta, int(free.sum()), n_matched_draws, rng)
            else:
                a = alignment(phi, delta).spearman

            coalition_aia[Q] = float(a)

    baseline = coalition_aia[frozenset()]
    u = {Q: v - baseline for Q, v in coalition_aia.items()}
    psi = shapley_over_switches(u)
    grand = coalition_aia[frozenset(ALL_SWITCHES)]

    shares = MechanismShares(
        psi=psi,
        aia_baseline=baseline,
        aia_grand=grand,
        coalition_aia=coalition_aia,
        # Reported separately and NOT attributed: ambiguous between surrogate
        # error, interaction past the pairwise ceiling, and a mechanism
        # outside H1-H3. Section 3.7.4 adjudicates it against fidelity.
        residual=1.0 - grand,
        cardinality_matched=cardinality_matched,
    )
    shares.check_efficiency()
    return shares


def _size_effect(
    phi: np.ndarray,
    delta: np.ndarray,
    size: int,
    n_draws: int,
    rng: np.random.Generator,
) -> float:
    """Expected alignment shift from evaluating on a smaller random subset.

    Subtracted from the H3 coalitions so that the infeasibility share is not
    inflated by the mere fact that fewer factors are being ranked.
    """
    full = alignment(phi, delta).spearman
    n = phi.size
    if size >= n:
        return 0.0

    vals = []
    for _ in range(n_draws):
        pick = rng.choice(n, size=size, replace=False)
        try:
            vals.append(alignment(phi[pick], delta[pick]).spearman)
        except ValueError:
            continue  # constant subset; skip rather than bias with a zero
    return float(np.mean(vals)) - full if vals else 0.0
