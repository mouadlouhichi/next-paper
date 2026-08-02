"""ActionShap: intervention-grounded evaluation of recommendation explanations.

The package contains two layers during the migration from the old proposal:

* ``recommendation`` and ``models.profile`` implement the revised
  recommendation-only specification: history-conditioned profiles, fixed
  candidate utilities, Monte Carlo Shapley values, and joint-action selection.
* The older feature-attribution, clustering, and modifiability modules remain
  available as legacy code until the recommendation pipeline is complete; they
  are not part of the revised paper's primary experiment.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .metrics import (
    actionability_score,
    actionability_score_heldout,
    alignment,
    intervention_regret,
    stability,
    topk_intervention_precision,
)
from .intervention import InterventionBudget, intervention_effects
from .modifiability import ModifiabilityTable, load_modifiability
from .rerank import eta_sweep, rerank
from .models.profile import ProfileAggregationModel, fit_item_embeddings
from .recommendation import (
    UserGame,
    exhaustive_oracle,
    joint_attribution_score,
    mc_shapley,
    ndcg_at_k,
    profile_utility,
    select_joint_action,
)

__all__ = [
    "__version__",
    "stability",
    "actionability_score",
    "actionability_score_heldout",
    "alignment",
    "topk_intervention_precision",
    "intervention_regret",
    "InterventionBudget",
    "intervention_effects",
    "ModifiabilityTable",
    "load_modifiability",
    "rerank",
    "eta_sweep",
    "ProfileAggregationModel",
    "fit_item_embeddings",
    "UserGame",
    "ndcg_at_k",
    "profile_utility",
    "mc_shapley",
    "joint_attribution_score",
    "select_joint_action",
    "exhaustive_oracle",
]
