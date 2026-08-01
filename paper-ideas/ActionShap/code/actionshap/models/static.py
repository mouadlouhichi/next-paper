"""The static-regime pipeline: standardize, cluster, fit a surrogate, attribute.

Rebuilt from the two published clustering papers, which released no code. Every
choice the papers leave open is marked ``IMPLEMENTER CHOICE`` and defaults to
the value recorded in ``docs/clustering_spec.md``, so the provenance of each
number stays auditable.

Two points where the published description and the published algorithm
disagree, resolved here in favour of the algorithm and the results text:

  * PCA is computed and then never consumed. Clustering runs on the full
    standardized matrix; the projection is for plots only.
  * The surrogate is trained on ORIGINAL, unscaled features, so attributions
    are natively in original units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["StaticPipeline", "ClusterQuality"]


@dataclass(frozen=True)
class ClusterQuality:
    """Partition quality, for benchmarking against the published values."""

    k: int
    silhouette: float
    davies_bouldin: float

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"k={self.k}: silhouette={self.silhouette:.3f}, "
            f"DB={self.davies_bouldin:.3f}"
        )


@dataclass
class StaticPipeline:
    """Clustering plus a surrogate that makes the partition attributable.

    Parameters
    ----------
    k
        Number of clusters. The papers use 3 for both datasets, though for
        wine k=2 scores better on both reported metrics; `sweep_k` reproduces
        the comparison.
    n_pca_components
        Kept only so the visualization projection is available. Changing it
        cannot affect any result.
    silhouette_sample
        Silhouette is O(n^2) and the air-quality set has 383,585 rows, so it is
        estimated on a subsample above this size. The papers do not say whether
        they subsampled, which is one reason their values may not reproduce
        exactly.
    """

    k: int = 3
    n_pca_components: int = 2
    random_state: int = 42          # IMPLEMENTER CHOICE: no seed is ever stated
    num_leaves: int = 31            # stated: "default kept"
    n_estimators: int = 100         # stated: "default kept"
    test_size: float = 0.2          # IMPLEMENTER CHOICE: no split is stated
    silhouette_sample: int = 20_000

    scaler_: object = field(init=False, default=None)
    pca_: object = field(init=False, default=None)
    kmeans_: object = field(init=False, default=None)
    surrogate_: object = field(init=False, default=None)
    labels_: np.ndarray = field(init=False, default=None)
    projection_: np.ndarray = field(init=False, default=None)
    fidelity_: dict = field(init=False, default_factory=dict)

    def fit(self, X: np.ndarray) -> "StaticPipeline":
        from lightgbm import LGBMClassifier
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=float)

        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)

        self.pca_ = PCA(
            n_components=self.n_pca_components, random_state=self.random_state
        ).fit(X_scaled)
        self.projection_ = self.pca_.transform(X_scaled)

        # On X_scaled, NOT on the projection. See the module docstring.
        self.kmeans_ = KMeans(
            n_clusters=self.k,
            init="k-means++",       # IMPLEMENTER CHOICE
            n_init=10,              # IMPLEMENTER CHOICE
            max_iter=300,           # IMPLEMENTER CHOICE
            tol=1e-4,               # IMPLEMENTER CHOICE
            random_state=self.random_state,
        ).fit(X_scaled)
        self.labels_ = self.kmeans_.labels_

        # Surrogate on ORIGINAL features. Trees are scale-invariant so this is
        # numerically inconsequential, but it matches the published algorithm
        # and keeps attributions in interpretable units.
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, self.labels_,
            test_size=self.test_size,
            stratify=self.labels_,
            random_state=self.random_state,
        )
        self.surrogate_ = LGBMClassifier(
            objective="multiclass",
            num_class=self.k,
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,               # determinism, and avoids a sandbox issue
            verbose=-1,
        ).fit(X_tr, y_tr)

        # The papers report a single F1 without saying which averaging mode,
        # so report all three and let the manuscript name the one it quotes.
        pred = self.surrogate_.predict(X_te)
        self.fidelity_ = {
            f"f1_{avg}": float(f1_score(y_te, pred, average=avg))
            for avg in ("macro", "micro", "weighted")
        }
        self.fidelity_["accuracy"] = float((pred == y_te).mean())
        return self

    # -- the characteristic function --------------------------------------

    def membership_probability(self, X: np.ndarray) -> np.ndarray:
        """P(cluster assigned to each row | features), re-deriving assignments.

        Suitable as a regression target on unperturbed data. NOT suitable as
        the output function for interventions -- use `membership_fn` for that,
        because re-deriving the assignment lets the reference cluster move
        under the intervention and mixes two different effects together.
        """
        self._check_fitted()
        X = np.atleast_2d(X)
        proba = self.surrogate_.predict_proba(X)
        assigned = self.kmeans_.predict(self.scaler_.transform(X))
        return proba[np.arange(len(assigned)), assigned]

    def membership_fn(self, X_reference: np.ndarray):
        """v(S) for the attribution game, against a frozen reference cluster.

        Returns ``f(X) -> P(c_i | x_i)`` where ``c_i`` is the cluster row i
        occupied in ``X_reference``. Interventions then answer one question --
        does this change push the row further into its own regime or out of it
        -- and a positive effect unambiguously means membership strengthened.

        The returned function is bound to the reference rows positionally, so
        it must be called with matrices of the same row count and ordering.
        """
        self._check_fitted()
        X_reference = np.atleast_2d(np.asarray(X_reference, float))
        reference = self.kmeans_.predict(self.scaler_.transform(X_reference))
        rows = np.arange(len(reference))
        surrogate = self.surrogate_

        def f(X: np.ndarray) -> np.ndarray:
            X = np.atleast_2d(np.asarray(X, float))
            if X.shape[0] != reference.size:
                raise ValueError(
                    f"membership_fn is bound to {reference.size} reference rows "
                    f"but was called with {X.shape[0]}; rebuild it for this batch"
                )
            return surrogate.predict_proba(X)[rows, reference]

        return f

    def target_fn(self, target: int):
        """v(S) against one designated cluster: ``f(X) -> P(target | x)``.

        Preferred over `membership_fn` whenever a cluster can be named as the
        desirable one, because it is the only version of the game that gives
        interventions somewhere to go. ``P(own cluster | x)`` sits at ~0.99 for
        an accurate surrogate, so every feasible perturbation drives it down
        and ``sign(Delta_j)`` degenerates to -1 for every factor. ``P(target)``
        starts near the cluster's share of the data and can move either way, so
        a sign carries information and Definition 6 becomes testable.

        Unlike `membership_fn` this is not bound to a reference row set and can
        be called with any batch.
        """
        self._check_fitted()
        if not 0 <= target < self.k:
            raise ValueError(f"target {target} outside 0..{self.k - 1}")
        surrogate = self.surrogate_

        def f(X: np.ndarray) -> np.ndarray:
            return surrogate.predict_proba(np.atleast_2d(X))[:, target]

        return f

    def rank_clusters_by(self, descriptor: np.ndarray) -> list[tuple[int, float]]:
        """Clusters ordered by the mean of a held-out descriptor, best first.

        Used to pick the intervention target without letting the descriptor
        into the clustering: wine ``quality`` names the desirable regime but
        never enters ``X``.
        """
        self._check_fitted()
        descriptor = np.asarray(descriptor, float)
        if descriptor.size != self.labels_.size:
            raise ValueError(
                f"descriptor has {descriptor.size} values but the fit covered "
                f"{self.labels_.size} rows"
            )
        means = [
            (c, float(descriptor[self.labels_ == c].mean())) for c in range(self.k)
        ]
        return sorted(means, key=lambda t: -t[1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.surrogate_.predict_proba(np.atleast_2d(X))

    # -- quality ----------------------------------------------------------

    def quality(self, X: np.ndarray) -> ClusterQuality:
        """Silhouette and Davies-Bouldin, computed in the clustering space."""
        from sklearn.metrics import davies_bouldin_score, silhouette_score

        self._check_fitted()
        X_scaled = self.scaler_.transform(np.asarray(X, float))
        n = X_scaled.shape[0]

        if n > self.silhouette_sample:
            rng = np.random.default_rng(self.random_state)
            pick = rng.choice(n, size=self.silhouette_sample, replace=False)
            sil = silhouette_score(X_scaled[pick], self.labels_[pick])
        else:
            sil = silhouette_score(X_scaled, self.labels_)

        return ClusterQuality(
            k=self.k,
            silhouette=float(sil),
            davies_bouldin=float(davies_bouldin_score(X_scaled, self.labels_)),
        )

    def _check_fitted(self) -> None:
        if self.surrogate_ is None:
            raise RuntimeError("call fit() first")


def sweep_k(X: np.ndarray, ks=(2, 3, 4, 5), **kwargs) -> list[ClusterQuality]:
    """Partition quality across k, reproducing the papers' selection table.

    Worth running before trusting k=3: for wine, k=2 scores better on both
    reported metrics, so the published choice does not follow from the
    published criteria.
    """
    return [StaticPipeline(k=k, **kwargs).fit(X).quality(X) for k in ks]
