"""CAVI — Cooperative Action-Value Intelligence: a forward-looking, uncertainty-
adjusted, feasibility-restricted cooperative-game framework for actionable
recommendation.

Core object: Cooperative Action Values (CAV) = the Myerson-restricted Shapley
value of the forward mean-variance certainty-equivalent game.

    CAV_i = Shapley_F(u_t)_i,   u_t(S) = E[V_t(S)] - kappa*Var[V_t(S)]

by additivity = Shapley_F(mean)_i - kappa * Shapley_F(var)_i.
"""
from .allocation import CAV, compute_cav, verify_additivity_identity
from .games import (Feasibility, CooperativeGame, RestrictedGame,
                    mean_variance_game, mc_myerson_value, myerson_value)
from .recourse import (MinimalActionPlanner, check_submodular, greedy_gap)
from .recommender import ProfileRecommender, DynamicsModel, bpr_item_factors

__all__ = [
    "CAV", "compute_cav", "verify_additivity_identity",
    "Feasibility", "CooperativeGame", "RestrictedGame",
    "mean_variance_game", "mc_myerson_value", "myerson_value",
    "MinimalActionPlanner", "check_submodular", "greedy_gap",
    "ProfileRecommender", "DynamicsModel", "bpr_item_factors",
]
__version__ = "0.1.0"
