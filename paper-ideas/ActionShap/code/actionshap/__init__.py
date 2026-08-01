"""ActionShap: intervention-grounded evaluation of cooperative attribution.

Reference implementation for the manuscript of the same name. The public
surface mirrors the manuscript's definitions so that code and paper can be
read against each other:

    Definition 1  modifiability          -> `modifiability`
    Definition 2  intervention effect    -> `intervention`
    Definition 3  stability              -> `metrics.stability`
    Definition 4  Actionability Score    -> `metrics.actionability_score`
    Definition 5  alignment (AIA)        -> `metrics.alignment`
    Definition 6  top-k precision        -> `metrics.topk_intervention_precision`
    Definition 7  intervention regret    -> `metrics.intervention_regret`
    Definition 8  A-Shapley reranking    -> `rerank.rerank`
    Corollary 1   mechanism decomposition-> `decomposition`
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
]
