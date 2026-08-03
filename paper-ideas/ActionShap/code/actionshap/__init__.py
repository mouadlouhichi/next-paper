"""ActionShap: intervention-grounded evaluation for recommendation explanations.

The public package surface is recommendation-only. Legacy cross-domain modules
remain importable by their explicit module paths for provenance, but they are
not re-exported here and are not part of the canonical paper pipeline.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .models import (
    ItemKNNModel,
    ProfileAggregationModel,
    fit_item_embeddings,
    fit_item_knn,
)
from .recommendation import (
    UserGame,
    exhaustive_oracle,
    joint_attribution_score,
    mc_shapley,
    ndcg_at_k,
    profile_utility,
    select_downweight_action,
    select_joint_action,
    target_margin_utility,
)

__all__ = [
    "ItemKNNModel",
    "ProfileAggregationModel",
    "UserGame",
    "__version__",
    "exhaustive_oracle",
    "fit_item_embeddings",
    "fit_item_knn",
    "joint_attribution_score",
    "mc_shapley",
    "ndcg_at_k",
    "profile_utility",
    "select_downweight_action",
    "select_joint_action",
    "target_margin_utility",
]
