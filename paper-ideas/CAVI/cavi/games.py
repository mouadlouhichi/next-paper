"""
games.py — Cooperative-game definitions for the CAVI framework.

Implements the *forward mean-variance certainty-equivalent game* from the
proposal (§3.5.4):

    u_t(S) = E[V_t(S)] - kappa * Var[V_t(S)] = v_t(S) - kappa * v^sigma2_t(S)

and the *Myerson restricted game* over a feasibility structure F:

    u^F_t(S) = sum_{C in comp_F(S)} u_t(C)

where comp_F(S) are the maximal connected components of S in the hypergraph F.

Players are "actionable levers"; a coalition S is a set of levers to intervene
on together (do(S)). This module is intentionally dependency-light (numpy only)
so the theory can be validated standalone.
"""
from __future__ import annotations
import itertools
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Feasibility structure (hypergraph connectivity)
# ---------------------------------------------------------------------------
class Feasibility:
    """
    A hypergraph F over a player set {0,...,n-1}. Two players are connected if
    they appear together in some hyperedge (or transitively through a chain of
    hyperedges). This defines the standard hyperedge-chain notion of
    connectivity used by the Myerson (1977) / van den Nouweland-Borm-Tijs
    (1992) restricted-game characterization.
    """

    def __init__(self, hyperedges: Sequence[Sequence[int]]):
        self.n_players: Optional[int] = None
        self.hyperedges: List[FrozenSet[int]] = []
        seen: Set[int] = set()
        for e in hyperedges:
            fe = frozenset(int(x) for x in e)
            if len(fe) >= 2:
                self.hyperedges.append(fe)
                seen.update(fe)
        if seen:
            self.n_players = max(seen) + 1
        # adjacency over players via shared hyperedges
        self._adj: Dict[int, Set[int]] = {}
        for e in self.hyperedges:
            for a in e:
                self._adj.setdefault(a, set()).update(e - {a})

    def neighbors(self, i: int) -> Set[int]:
        return self._adj.get(i, set())

    def _component_of(self, start: int, members: Set[int]) -> Set[int]:
        """BFS component of `start` within `members`."""
        comp: Set[int] = {start}
        frontier: List[int] = [start]
        while frontier:
            cur = frontier.pop()
            for nb in self.neighbors(cur):
                if nb in members and nb not in comp:
                    comp.add(nb)
                    frontier.append(nb)
        return comp

    def components(self, S: Sequence[int]) -> List[List[int]]:
        """Maximal connected components of the coalition S in F."""
        members: Set[int] = set(int(x) for x in S)
        comps: List[List[int]] = []
        unvisited = set(members)
        while unvisited:
            start = unvisited.pop()
            comp = self._component_of(start, members)
            comps.append(sorted(comp))
            unvisited -= comp
        return comps

    def complete(self, n_players: int) -> "Feasibility":
        """Full-connectivity feasibility (every coalition is one component)."""
        if n_players < 2:
            return Feasibility([])
        # a single hyperedge over all players makes every subset connected
        return Feasibility([tuple(range(n_players))])


# ---------------------------------------------------------------------------
# Cooperative game
# ---------------------------------------------------------------------------
class CooperativeGame:
    """
    A transferable-utility game (N, u) with a characteristic function u(S).
    Evaluated lazily through a callable to support expensive forward rollouts.
    """

    def __init__(self, players: Sequence[int], value: Callable[[Sequence[int]], float]):
        self.players: List[int] = list(players)
        self.value = value  # u(S) -> float

    def __call__(self, S: Sequence[int]) -> float:
        return float(self.value(list(S)))


# ---------------------------------------------------------------------------
# Myerson restricted game
# ---------------------------------------------------------------------------
class RestrictedGame:
    """
    The Myerson restricted game u^F(S) = sum_{C in comp_F(S)} u(C).

    Args:
        game:   the underlying TU game (value function over coalitions)
        feas:   the feasibility structure
        verbose_value: optional cache/diagnostic wrapper (unused here)
    """

    def __init__(self, game: CooperativeGame, feas: Feasibility):
        self.game = game
        self.feas = feas

    def __call__(self, S: Sequence[int]) -> float:
        comps = self.feas.components(S)
        return float(sum(self.game(c) for c in comps))


# ---------------------------------------------------------------------------
# Exact Shapley / Myerson value
# ---------------------------------------------------------------------------
def _coalition_weights(n: int) -> np.ndarray:
    """|S|!(n-|S|-1)!/n! for |S|=0..n-1."""
    logfac = np.zeros(n + 1)
    for k in range(1, n + 1):
        logfac[k] = logfac[k - 1] + np.log(k)
    w = np.zeros(n)
    for k in range(n):
        w[k] = np.exp(logfac[k] + logfac[n - k - 1] - logfac[n])
    return w


def exact_shapley(value_fn: Callable[[Sequence[int]], float],
                  players: Sequence[int]) -> np.ndarray:
    """
    Exact Shapley value of a value function over a player set. Players are
    indices 0..len(players)-1. value_fn(S) evaluated on a list of player indices.
    """
    n = len(players)
    w = _coalition_weights(n)
    phi = np.zeros(n)
    for S_mask in range(1 << n):
        S = [i for i in range(n) if (S_mask >> i) & 1]
        k = len(S)
        for i in range(n):
            if (S_mask >> i) & 1:
                continue
            S_plus = S + [i]
            phi[i] += w[k] * (value_fn(S_plus) - value_fn(S))
    return phi


def myerson_value(game: CooperativeGame, feas: Feasibility,
                  players: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Exact Myerson value = Shapley value of the restricted game u^F.

    This is the allocation CAV = Shapley_F(u_t) from the proposal.
    Only exact for small player sets (n <= ~15); use mc_myerson_value for larger.
    """
    if players is None:
        players = list(range(len(game.players)))
    players = list(players)
    rg = RestrictedGame(game, feas)
    n = len(players)
    # map global player ids to local 0..n-1
    idx = {p: k for k, p in enumerate(players)}

    def local_value(S_local: Sequence[int]) -> float:
        global_S = [players[i] for i in S_local]
        return rg(global_S)

    return exact_shapley(local_value, list(range(n)))


# ---------------------------------------------------------------------------
# Monte-Carlo Myerson value (scalable)
# ---------------------------------------------------------------------------
def mc_myerson_value(game: CooperativeGame, feas: Feasibility,
                     players: Optional[Sequence[int]] = None,
                     M: int = 1000, seed: int = 0) -> np.ndarray:
    """
    Monte-Carlo estimate of the Myerson value via permutation prefix-walks on
    the restricted game. For each permutation, we add players one at a time and
    accumulate the marginal of the *restricted* value. Efficiency holds per
    permutation (telescoping), so the estimator is unbiased for the exact
    Myerson value.
    """
    if players is None:
        players = list(range(len(game.players)))
    players = list(players)
    rg = RestrictedGame(game, feas)
    n = len(players)
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    for _ in range(M):
        perm = rng.permutation(n)
        running: List[int] = []
        prev = 0.0
        for pos, local_i in enumerate(perm):
            running.append(local_i)
            S_global = [players[i] for i in running]
            v = rg(S_global)
            acc[local_i] += v - prev
            prev = v
    return acc / M


# ---------------------------------------------------------------------------
# Mean-variance certainty-equivalent game
# ---------------------------------------------------------------------------
def mean_variance_game(mean_fn: Callable[[Sequence[int]], float],
                       var_fn: Callable[[Sequence[int]], float],
                       kappa: float,
                       players: Sequence[int]) -> CooperativeGame:
    """
    u_t(S) = E[V_t(S)] - kappa * Var[V_t(S)] = mean_fn(S) - kappa*var_fn(S).
    """
    def u(S):
        return float(mean_fn(list(S))) - kappa * float(var_fn(list(S)))
    return CooperativeGame(list(players), u)


# ---------------------------------------------------------------------------
# CAV = Myerson value of the certainty-equivalent game (the headline object)
# ---------------------------------------------------------------------------
def cav_allocation(mean_fn, var_fn, kappa, feas: Feasibility,
                   players, M=None, seed=0) -> np.ndarray:
    """
    Cooperative Action Values:

        CAV_i = Shapley_F(u_t)_i,   u_t = mean - kappa*var

    which by additivity of Shapley equals
        Shapley_F(mean)_i - kappa * Shapley_F(var)_i
    (verified in the allocation module; this function computes the Myerson
    value of the combined game directly).
    """
    game = mean_variance_game(mean_fn, var_fn, kappa, players)
    n = len(players)
    if M is None:
        # exact if feasible (small); otherwise default MC
        if n <= 14:
            return myerson_value(game, feas, players)
        M = 1000
    return mc_myerson_value(game, feas, players, M=M, seed=seed)
