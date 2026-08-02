"""
allocation.py — CAV allocation, additivity theorem, and component-Shapley.

Key identity (Theorem 1 corollary, §3.5.4 Step 3): because the Shapley
operator is additive in the characteristic function and the restriction is
linear (u^F = v^F - kappa*(v^sigma2)^F component-wise), we have

    CAV_i = Shapley_F(u_t)_i
          = Shapley_F(v_t)_i - kappa * Shapley_F(v^sigma2_t)_i
          = phi^mu_i - kappa * phi^sigma2_i

This module computes phi^mu (mean Shapley) and phi^sigma2 (variance Shapley)
over the restricted game, plus the combined CAV, and exposes a verification
function that checks the identity to machine precision.
"""
from __future__ import annotations
from typing import Callable, List, Optional, Sequence

import numpy as np

from .games import (CooperativeGame, Feasibility, RestrictedGame,
                    mc_myerson_value, myerson_value)


def component_shapley(value_fn: Callable[[Sequence[int]], float],
                      feas: Feasibility,
                      players: Sequence[int],
                      M: Optional[int] = None, seed: int = 0) -> np.ndarray:
    """
    Shapley value of the *restricted game* v^F for a single characteristic
    function v. This is Shapley_F(v).
    """
    players = list(players)
    game = CooperativeGame(list(players), value_fn)
    n = len(players)
    if M is None and n <= 14:
        return myerson_value(game, feas, players)
    if M is None:
        M = 1000
    return mc_myerson_value(game, feas, players, M=M, seed=seed)


class CAV:
    """
    Container for the mean, variance, and risk-adjusted CAV allocations.
    """

    def __init__(self, phi_mean: np.ndarray, phi_var: np.ndarray,
                 kappa: float, players: Sequence[int]):
        self.phi_mean = np.asarray(phi_mean)
        self.phi_var = np.asarray(phi_var)
        self.kappa = kappa
        self.players = list(players)
        self.cav = self.phi_mean - kappa * self.phi_var

    def ordered(self, key="cav", descending=True):
        idx = np.argsort(self._get(key))[::-1 if descending else 1]
        return [(self.players[i], float(self._get(key)[i])) for i in idx]

    def _get(self, key):
        return {"cav": self.cav, "mean": self.phi_mean,
                "var": self.phi_var}[key]


def compute_cav(mean_fn, var_fn, kappa, feas: Feasibility, players,
                M: Optional[int] = None, seed: int = 0) -> CAV:
    """
    Compute the full CAV decomposition (mean Shapley, variance Shapley, CAV).
    """
    players = list(players)
    phi_mean = component_shapley(mean_fn, feas, players, M=M, seed=seed)
    phi_var = component_shapley(var_fn, feas, players, M=M, seed=seed + 1)
    return CAV(phi_mean, phi_var, kappa, players)


def verify_additivity_identity(mean_fn, var_fn, kappa, feas, players,
                               M=None, seed=0, rtol=1e-6, atol=1e-9):
    """
    Check that CAV computed two ways agree:
      (a) Myerson value of the combined game u_t = mean - kappa*var
      (b) Shapley_F(mean) - kappa * Shapley_F(var)
    Returns (max_abs_diff, pass_bool).
    """
    from .games import mean_variance_game

    players = list(players)
    combined_game = mean_variance_game(mean_fn, var_fn, kappa, players)
    n = len(players)
    if M is None and n <= 14:
        cav_combined = myerson_value(combined_game, feas, players)
    else:
        M = M or 1000
        cav_combined = mc_myerson_value(combined_game, feas, players,
                                        M=M, seed=seed)
    cav_split = compute_cav(mean_fn, var_fn, kappa, feas, players,
                            M=M, seed=seed)
    diff = float(np.max(np.abs(cav_combined - cav_split.cav)))
    return diff, diff <= atol + rtol * float(np.max(np.abs(cav_combined)))
