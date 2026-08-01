"""The construction-level audit: fidelity, order validity, reflexivity.

    P_g  predicted retirement effect  = v(G) - v(G\\{g})          (L0 masked)
    R_g  realized retirement effect   = v_nb(G\\{g}) - v(G)       (L2 never-built)
    F_g  fidelity gap                 = P_g - R_g = v_nb(G\\{g}) - v(G\\{g})

Prop. 1 (intervention-fidelity decomposition) is verified numerically as an
invariant: F_g computed both ways must agree to machine precision.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import SOURCES
from .counterfactuals import (evaluate_world, regenerated_world,
                              shapley_world)
from .game import shapley_from_values


class AuditResult:
    def __init__(self, ds, v, per_user, phi, phi_u, v_reg, v_nb, per_user_nb,
                 phi_nb, phi_u_nb, cand, cand_nb, sources, thetas=None):
        self.ds = ds
        self.v = v                    # L0 coalition values (grand candidates)
        self.per_user = per_user
        self.phi = phi                # exact source Shapley (global)
        self.phi_u = phi_u
        self.v_reg = v_reg            # L1: regenerated-candidate values
        self.v_nb = v_nb              # L2: never-built values per excluded g
        self.per_user_nb = per_user_nb
        self.phi_nb = phi_nb
        self.phi_u_nb = phi_u_nb
        self.cand = cand
        self.cand_nb = cand_nb
        self.sources = sources
        self.thetas = thetas          # L0 fusion weights per coalition

    # ---- RQ1: fidelity -------------------------------------------------
    # Convention (Structure doc Sec. 3.4 / Prop. 1, corrected):
    #   P_g = v(G) - v(G\\{g})            masked marginal (predicted loss)
    #   R_g = v(G) - v_nb(G\\{g})         realized loss of never building g
    #                                      (positive = retirement harms)
    #   F_g = P_g - R_g = v_nb(G\\{g}) - v(G\\{g})   (fidelity gap)
    # F > 0 means the attribution overstates the harm of removing g (the
    # system adapts to its absence); F < 0 means removal costs more than the
    # current-system marginal suggests.

    def predicted_effect(self, g):
        G = tuple(sorted(SOURCES))
        return self.v[G] - self.v[tuple(sorted(set(G) - {g}))]

    def realized_effect(self, g):
        G = tuple(sorted(SOURCES))
        return self.v[G] - self.v_nb[g][tuple(sorted(set(SOURCES) - {g}))]

    def fidelity_gap(self, g):
        P = self.predicted_effect(g)
        R = self.realized_effect(g)
        G = tuple(sorted(SOURCES))
        alt = self.v_nb[g][tuple(sorted(set(SOURCES) - {g}))] - self.v[
            tuple(sorted(set(SOURCES) - {g}))]
        return P - R, alt

    # ---- RQ2: order validity -------------------------------------------
    def realized_effects(self):
        return {g: self.realized_effect(g) for g in SOURCES}

    def order_stats(self):
        from scipy import stats as sst

        phis = np.array([self.phi[g] for g in SOURCES])
        Rs = np.array([self.realized_effect(g) for g in SOURCES])
        tau, _ = sst.kendalltau(phis, Rs)
        order_phi = np.argsort(phis)
        order_R = np.argsort(Rs)
        top1 = order_phi[0] == order_R[0]
        top2 = set(order_phi[:2]) == set(order_R[:2])
        return {"kendall_tau": float(tau), "top1_agree": bool(top1),
                "top2_agree": bool(top2),
                "order_phi": [SOURCES[i] for i in order_phi],
                "order_R": [SOURCES[i] for i in order_R]}

    def co_monotonicity_violations(self):
        """Pairs where P_g >= P_h but F_g < F_h (Prop. 2 violation)."""
        out = []
        for g in SOURCES:
            for h in SOURCES:
                if g == h:
                    continue
                Pg, Ph = self.predicted_effect(g), self.predicted_effect(h)
                Fg, _ = self.fidelity_gap(g)
                Fh, _ = self.fidelity_gap(h)
                if Pg >= Ph and Fg < Fh:
                    out.append((g, h))
        return out

    # ---- RQ3: reflexivity ----------------------------------------------
    def reflexivity(self, g_star):
        surv = [g for g in SOURCES if g != g_star]
        phi_nb = self.phi_nb[g_star]
        phi_old = {g: self.phi[g] for g in surv}
        denom = np.mean([abs(phi_old[g]) for g in surv])
        rho = np.mean([abs(phi_nb[g] - phi_old[g]) for g in surv]) / max(denom, 1e-12)
        agg = sum(phi_nb[g] for g in surv)
        G = tuple(sorted(SOURCES))
        target = self.v[G] - self.realized_effect(g_star)
        return {"rho": float(rho), "aggregate": float(agg),
                "target": float(target), "survivors": surv}
