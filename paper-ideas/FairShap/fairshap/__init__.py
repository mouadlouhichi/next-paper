"""FairShap — Shapley-Value-Guided Two-Sided Fairness and Popularity Debiasing
in Hypergraph Recommendation.

Augments a hypergraph recommender with a preference-aware Shapley estimator of
each item's marginal contribution to a fairness-aware utility (ranking quality +
diversity + exposure equality), then uses these attributions to correct exposure
during propagation and re-ranking.
"""
from .metrics import (ndcg_at_k, recall_at_k, exposure_gini, arp, gini,
                      intra_list_diversity, catalogue_coverage,
                      consumer_ndcg_gap, compute_all)
from .game import (coalition_value, exposure_shapley, mc_shapley,
                   exposure_efficiency_check)
from .rerank import fair_rerank, deterministic_rerank, calibrated_rerank
from .model import HypergraphGNN, train_hypergraph, train_hypergraph_with_fair_loss
from .pipeline import FairShapPipeline, recommend, recommend_scores

__all__ = [
    "ndcg_at_k", "recall_at_k", "exposure_gini", "arp", "gini",
    "intra_list_diversity", "catalogue_coverage", "consumer_ndcg_gap", "compute_all",
    "coalition_value", "exposure_shapley", "mc_shapley", "exposure_efficiency_check",
    "fair_rerank", "deterministic_rerank", "calibrated_rerank",
    "HypergraphGNN", "train_hypergraph", "train_hypergraph_with_fair_loss",
    "FairShapPipeline", "recommend", "recommend_scores",
]
__version__ = "0.1.0"
