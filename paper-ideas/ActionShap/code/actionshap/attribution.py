"""Uniform interface over the importance signals compared in Table 7.

Every signal reduces to the same contract: given a model and a design matrix,
return one real number per factor. Wrapping them this way is what lets the
manuscript report a single alignment metric across methods that are otherwise
not comparable -- and it is also what makes the DyHuCoG comparison in Section
4.6.2 possible, since the Shapley weighting and the attention gate become two
implementations of one interface over the same trained model.

Signals are run ``R`` times under different seeds; the dispersion across runs
is what `metrics.stability` consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from .metrics import stability

__all__ = [
    "Attributor",
    "AttributionRuns",
    "run_repeated",
    "TreeShapAttributor",
    "KernelShapAttributor",
    "LimeAttributor",
    "PermutationAttributor",
    "RandomAttributor",
    "feature_direction",
]


@runtime_checkable
class Attributor(Protocol):
    """One importance signal."""

    name: str

    def attribute(self, X: np.ndarray, seed: int) -> np.ndarray:
        """Return a per-factor importance vector of shape ``(n_factors,)``."""
        ...


@dataclass(frozen=True)
class AttributionRuns:
    """``R`` repetitions of one signal, plus the summaries derived from them."""

    name: str
    repeated: np.ndarray  # (R, n_factors)
    seeds: tuple[int, ...]

    @property
    def phi(self) -> np.ndarray:
        """Mean attribution across runs -- the vector used for ranking."""
        return self.repeated.mean(axis=0)

    @property
    def s(self) -> np.ndarray:
        """Per-factor stability (Definition 3)."""
        return stability(self.repeated)

    @property
    def n_factors(self) -> int:
        return self.repeated.shape[1]


def run_repeated(
    attributor: Attributor,
    X: np.ndarray,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
) -> AttributionRuns:
    """Run one signal under each seed.

    Deterministic methods (TreeSHAP, permutation with a fixed order) will
    return identical vectors across seeds and therefore score stability 1.0.
    That is the correct answer, not a bug: they are perfectly stable.
    """
    if len(seeds) < 2:
        raise ValueError("need >= 2 seeds so that stability is defined")

    runs = []
    for seed in seeds:
        v = np.asarray(attributor.attribute(X, seed), dtype=float)
        if v.ndim != 1 or v.size != X.shape[1]:
            raise ValueError(
                f"{attributor.name} returned shape {v.shape}, "
                f"expected ({X.shape[1]},)"
            )
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{attributor.name} returned non-finite values at seed {seed}")
        runs.append(v)

    return AttributionRuns(attributor.name, np.vstack(runs), tuple(seeds))


# --------------------------------------------------------------------------
# Concrete signals. Each collapses a per-instance attribution matrix to one
# vector by mean absolute value over instances: the manuscript's metrics are
# global (one ordering per dataset), not per-instance.
# --------------------------------------------------------------------------


def _global_reduce(per_instance: np.ndarray) -> np.ndarray:
    """(n_instances, n_factors) -> (n_factors,) by mean |value|.

    Mean absolute value rather than mean: positive and negative
    per-instance attributions would otherwise cancel and report a genuinely
    important factor as unimportant.
    """
    return np.abs(np.asarray(per_instance, float)).mean(axis=0)


@dataclass
class TreeShapAttributor:
    """TreeSHAP over a tree ensemble (static regime).

    Interventional perturbation integrates against a background sample, which
    is the same estimand Definition 2 targets; ``tree_path_dependent`` would
    condition on the tree structure instead and is not comparable.

    Cost scales with ``n_background * n_explain``, so both are subsampled. The
    background is redrawn per seed, which is what makes stability meaningful
    for a method that is otherwise deterministic: it measures sensitivity to
    the reference distribution rather than to a random number generator.
    """

    model: object
    name: str = "TreeSHAP"
    check_additivity: bool = False
    # Cost is O(n_background * n_explain). These values keep a five-seed run
    # near a minute while leaving the global ordering stable to ~0.95; raise
    # both for a final run and report what was used.
    n_background: int = 100
    n_explain: int = 500
    reference_labels: np.ndarray | None = None
    target_class: int | None = None

    def attribute(self, X: np.ndarray, seed: int) -> np.ndarray:
        return _global_reduce(self.per_instance(X, seed)[0])

    def per_instance(self, X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Signed per-instance attributions, and the row indices they cover.

        Which class the attributions are taken for has to match the output
        function the effects were measured on, or the comparison is between two
        different games:

        * ``target_class`` -- one fixed cluster, to pair with
          `StaticPipeline.target_fn`.
        * ``reference_labels`` -- each row's own cluster, to pair with
          `StaticPipeline.membership_fn`.
        * neither -- averaged over classes. For a probability output that
          average is near zero by construction, so it is usable for magnitudes
          but too noisy to read a direction from.
        """
        import shap

        rng = np.random.default_rng(seed)
        bg = X[rng.choice(X.shape[0], min(self.n_background, X.shape[0]), replace=False)]
        rows = rng.choice(X.shape[0], min(self.n_explain, X.shape[0]), replace=False)

        explainer = shap.TreeExplainer(
            self.model, data=bg, feature_perturbation="interventional"
        )
        vals = explainer.shap_values(X[rows], check_additivity=self.check_additivity)

        if self.target_class is not None or self.reference_labels is not None:
            stacked = np.stack(vals, axis=-1) if isinstance(vals, list) \
                else np.asarray(vals, float)
            if stacked.ndim == 3:
                if self.target_class is not None:
                    return stacked[:, :, self.target_class], rows
                labels = np.asarray(self.reference_labels)[rows]
                return stacked[np.arange(len(rows)), :, labels], rows

        return _collapse_classes(vals, signed=True), rows


@dataclass
class KernelShapAttributor:
    """Model-agnostic KernelSHAP.

    The dominant cost in the dynamic regime: every coalition sample triggers
    a full model evaluation. ``n_background`` and ``nsamples`` are the two
    knobs that decide whether a run takes minutes or hours.
    """

    predict: Callable[[np.ndarray], np.ndarray]
    name: str = "KernelSHAP"
    n_background: int = 100
    nsamples: int | str = "auto"

    def attribute(self, X: np.ndarray, seed: int) -> np.ndarray:
        import shap

        rng = np.random.default_rng(seed)
        bg = shap.kmeans(X, min(self.n_background, X.shape[0]))
        idx = rng.choice(X.shape[0], size=min(200, X.shape[0]), replace=False)
        explainer = shap.KernelExplainer(self.predict, bg, seed=seed)
        vals = explainer.shap_values(X[idx], nsamples=self.nsamples, silent=True)
        return _global_reduce(_collapse_classes(vals))


@dataclass
class LimeAttributor:
    """LIME with a stochastic neighbourhood.

    Expected to be penalized by Definition 3: the neighbourhood is resampled
    per seed, so the attribution moves between runs.
    """

    predict: Callable[[np.ndarray], np.ndarray]
    feature_names: Sequence[str]
    name: str = "LIME"
    num_samples: int = 5000
    n_instances: int = 200

    def attribute(self, X: np.ndarray, seed: int) -> np.ndarray:
        from lime.lime_tabular import LimeTabularExplainer

        rng = np.random.default_rng(seed)
        explainer = LimeTabularExplainer(
            X,
            feature_names=list(self.feature_names),
            mode="classification",
            discretize_continuous=False,
            random_state=seed,
        )
        idx = rng.choice(X.shape[0], size=min(self.n_instances, X.shape[0]),
                         replace=False)
        acc = np.zeros(X.shape[1])
        for i in idx:
            exp = explainer.explain_instance(
                X[i], self.predict, num_features=X.shape[1],
                num_samples=self.num_samples,
            )
            for fid, w in exp.as_map()[exp.available_labels()[0]]:
                acc[fid] += abs(w)
        return acc / len(idx)


@dataclass
class PermutationAttributor:
    """Removal-based importance with no game-theoretic guarantee."""

    predict: Callable[[np.ndarray], np.ndarray]
    score: Callable[[np.ndarray, np.ndarray], float]
    y: np.ndarray
    name: str = "Permutation"
    n_repeats: int = 10

    def attribute(self, X: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        base = self.score(self.y, self.predict(X))
        out = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            drops = []
            for _ in range(self.n_repeats):
                Xp = X.copy()
                rng.shuffle(Xp[:, j])
                drops.append(base - self.score(self.y, self.predict(Xp)))
            out[j] = float(np.mean(drops))
        return out


@dataclass
class RandomAttributor:
    """Uniform random importance.

    Not in Table 7 of the manuscript, but every alignment table needs a floor:
    without one, a middling AIA has no reference point and a reviewer will
    ask what value would have arisen by chance.
    """

    n_factors: int
    name: str = "Random"

    def attribute(self, X: np.ndarray, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).random(self.n_factors)


def _collapse_classes(vals, signed: bool = False) -> np.ndarray:
    """Normalize the several shapes SHAP returns into (n_instances, n_factors).

    Depending on version and model type, shap_values gives a list over
    classes, a 3-D array with classes last, or a plain 2-D array. Multiclass
    output is collapsed over classes, which treats cluster membership as one
    importance question rather than one per cluster.

    ``signed=True`` averages the raw values and preserves direction, which
    `feature_direction` needs. The default takes magnitudes, since positive and
    negative contributions across classes would otherwise cancel and report an
    important feature as unimportant.
    """
    if isinstance(vals, list):
        stacked = np.stack(vals, axis=-1)
    else:
        stacked = np.asarray(vals, dtype=float)
        if stacked.ndim == 2:
            return stacked
        if stacked.ndim != 3:
            raise ValueError(f"unexpected SHAP output with shape {stacked.shape}")
    return stacked.mean(axis=-1) if signed else np.abs(stacked).mean(axis=-1)


def feature_direction(
    per_instance: np.ndarray, X_rows: np.ndarray
) -> np.ndarray:
    """Which way each factor pushes the output as its value increases.

    Returns a sign per factor, from the correlation between a factor's value
    and its own attribution across instances.

    This exists because a SHAP value and an intervention effect are not the
    same kind of quantity. A SHAP value says how much a factor's *current
    level* contributes; ``Delta_j`` says what happens when that level is
    *increased*. Comparing their raw signs, as a literal reading of Definition
    6 would, compares a level against a derivative. The correlation below is a
    derivative-like summary and is the quantity that can legitimately be
    checked against ``sign(Delta_j)``.
    """
    per_instance = np.asarray(per_instance, float)
    X_rows = np.asarray(X_rows, float)
    if per_instance.shape != X_rows.shape:
        raise ValueError(
            f"shape mismatch: attributions {per_instance.shape}, "
            f"values {X_rows.shape}"
        )

    out = np.zeros(per_instance.shape[1])
    for j in range(per_instance.shape[1]):
        x, phi = X_rows[:, j], per_instance[:, j]
        if np.ptp(x) == 0 or np.ptp(phi) == 0:
            continue  # no variation: direction is undefined, leave at 0
        out[j] = np.sign(np.corrcoef(x, phi)[0, 1])
    return out
