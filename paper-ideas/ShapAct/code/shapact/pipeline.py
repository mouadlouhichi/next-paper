"""End-to-end ShapAct pipeline for one dataset.

Stages (ShapAct Implementation Spec B.7):
  1. data -> sources -> candidates -> ZCache
  2. L0 exact game (32 coalitions) + efficiency/per-user checks
  3. L1 regenerated-candidate values
  4. L2 never-built worlds (5 x four-player games)
  5. audit quantities (P, R, F), order statistics, reflexivity, decisions
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np

from .audit import AuditResult
from .config import SOURCES
from .counterfactuals import (evaluate_world, regenerated_world,
                              shapley_world)
from .data import load_dataset
from .decisions import (random_expected, realized_mean,
                        realized_ndcg_per_user, rule_recommendations)
from .fusion import ZCache, build_candidates, candidate_recall
from .game import (coalition_values, efficiency_check,
                   per_user_consistency_check, shapley_from_values)
from .sources import fit_sources


def run_dataset(cfg, seed: int = 42, verbose: bool = True):
    t0 = time.time()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    ds = load_dataset(cfg)
    log(f"[{cfg.name}] dataset: {ds.stats}")

    sources = fit_sources(cfg, ds)
    log(f"[{cfg.name}] sources fitted")

    cand, top_lists = build_candidates(ds, sources, cfg, seed=seed)
    recall = candidate_recall(cand, ds.test)
    log(f"[{cfg.name}] candidate recall@200 = {recall:.4f}")

    zc = ZCache(ds, sources, cand, cfg)
    deg_rates = {g: float(zc.degenerate[g].mean()) for g in SOURCES}
    log(f"[{cfg.name}] degenerate-normalization rates: {deg_rates}")

    # ---- L0 exact game -------------------------------------------------
    v, per_user, thetas = coalition_values(zc, cand, ds.test, ds, cfg, seed=seed)
    phi, phi_u = shapley_from_values(v, per_user, len(ds.users))
    eff = efficiency_check(v, phi)
    cons = per_user_consistency_check(phi, phi_u, len(ds.users))
    log(f"[{cfg.name}] efficiency gap = {eff:.2e}, per-user consistency = {cons:.2e}")
    assert eff < 1e-6, f"efficiency check failed: {eff}"
    assert cons < 1e-6, f"per-user consistency check failed: {cons}"

    # ---- L1 regenerated (candidate-set effect) -------------------------
    v_reg = {}
    for g in SOURCES:
        cand_r, zc_r, avail = regenerated_world(ds, sources, cfg, exclude=g,
                                                seed=seed)
        vr, _, _ = evaluate_world(cand_r, zc_r, avail, ds, cfg, seed=seed)
        v_reg[g] = vr
        if g == SOURCES[0]:
            log(f"[{cfg.name}] L1 regenerated for {g}: "
                f"grand value = {vr[tuple(avail)]:.4f}")

    # ---- L2 never-built worlds -----------------------------------------
    v_nb, per_user_nb, phi_nb, phi_u_nb, cand_nb = {}, {}, {}, {}, {}
    for g in SOURCES:
        cand_r, zc_r, avail = regenerated_world(ds, sources, cfg, exclude=g,
                                                seed=seed)
        vr, pur, _ = evaluate_world(cand_r, zc_r, avail, ds, cfg, seed=seed)
        pr, pru = shapley_world(vr, pur, avail, len(ds.users))
        v_nb[g] = vr
        per_user_nb[g] = pur
        phi_nb[g] = pr
        phi_u_nb[g] = pru
        cand_nb[g] = cand_r
        log(f"[{cfg.name}] L2 never-built {g}: "
            f"uplift = {vr[tuple(avail)]:.4f}")

    audit = AuditResult(ds, v, per_user, phi, phi_u, v_reg, v_nb,
                        per_user_nb, phi_nb, phi_u_nb, cand, cand_nb,
                        sources, thetas=thetas)
    audit.recall = recall
    audit.degenerate_rates = deg_rates
    audit.cfg = cfg
    audit.seed = seed
    audit.timing = time.time() - t0
    return audit


def validation_stats(audit) -> dict:
    """Numbers for the external-validity table (compare with published
    Q1-paper baselines on the same datasets) and internal checks.

    * candidate-set NDCG@10 of each single source and the fusion (our
      protocol: 500 candidates, temporal leave-one-out)
    * full-catalog NDCG@20 of each single source and the fusion (train items
      masked) for comparison with published baselines
    * null ranker calibration (analytic 0.0227 x recall)
    """
    import numpy as np

    from .game import (coalition_ndcg, ndcg_at_k, null_ranker_ndcg,
                       rank_of_target)

    ds = audit.ds
    zc = ZCache(ds, audit.sources, audit.cand, audit.cfg)
    n_users = len(ds.users)
    test_items = ds.test.set_index("user")["item"].to_dict()
    train_set = ds.train.groupby("user")["item"].apply(set).to_dict()
    G = tuple(sorted(SOURCES))

    # candidate-set single-source NDCG@10 (raw source scores)
    from .fusion import _score_chunked

    src_ndcg10 = {}
    for g in SOURCES:
        flat_u = np.repeat(np.arange(n_users), audit.cand.shape[1])
        raw = _score_chunked(audit.sources[g], flat_u,
                             audit.cand.ravel()).reshape(audit.cand.shape)
        raw = np.where(audit.cand < 0, -np.inf, raw)
        ranks = rank_of_target(raw, audit.cand, ds.test)
        src_ndcg10[g] = float(ndcg_at_k(ranks, 10).mean())

    theta_G = audit.thetas[G]
    ndcg10_f = float(coalition_ndcg(theta_G, zc, audit.cand, ds.test, G).mean())
    ndcg20_f = float(coalition_ndcg(theta_G, zc, audit.cand, ds.test, G,
                                    k=20).mean())
    null10 = float(null_ranker_ndcg(audit.cand, ds.test,
                                    seed=audit.seed).mean())

    # full-catalog NDCG@20 (train items masked)
    def full_ndcg20(score_fn):
        gains = np.zeros(n_users)
        for u in range(n_users):
            it = test_items.get(u)
            if it is None:
                continue
            scores = score_fn(u)
            m = np.full(len(scores), True)
            m[list(train_set.get(u, set()))] = False
            scores = np.where(m, scores, -np.inf)
            r = 1 + int(np.sum(scores > scores[it]))
            if r <= 20:
                gains[u] = 1.0 / np.log2(r + 1)
        return float(gains.mean())

    fc = {}
    for g in SOURCES:
        fc[g] = full_ndcg20(
            lambda u, g=g: audit.sources[g].score(
                np.full(len(ds.items), u), np.arange(len(ds.items))))
    fc["FUSION"] = full_ndcg20(
        lambda u: (np.column_stack([
            (audit.sources[g].score(np.full(len(ds.items), u),
                                    np.arange(len(ds.items)))
             - zc.mu[g][u]) / zc.sigma[g][u] for g in SOURCES])
            @ theta_G[[SOURCES.index(g) for g in G]]))

    return {
        "null_ndcg10": null10,
        "fusion_ndcg10": ndcg10_f,
        "fusion_ndcg20": ndcg20_f,
        "single_source_ndcg10": src_ndcg10,
        "full_catalog_ndcg20": fc,
        "recall": audit.recall,
    }


def audit_summary(audit) -> dict:
    """All headline numbers of the audit as a JSON-serializable dict."""
    cfg = audit.cfg
    G = tuple(sorted(SOURCES))
    out = {
        "dataset": cfg.name,
        "seed": audit.seed,
        "stats": audit.ds.stats,
        "recall": audit.recall,
        "degenerate_rates": audit.degenerate_rates,
        "v_grand": float(audit.v[G]),
        "phi": {g: float(audit.phi[g]) for g in SOURCES},
        "phi_share": {g: float(audit.phi[g] / max(sum(audit.phi.values()), 1e-12))
                      for g in SOURCES},
        "predicted": {g: float(audit.predicted_effect(g)) for g in SOURCES},
        "realized": {g: float(audit.realized_effect(g)) for g in SOURCES},
        "fidelity": {},
        "order": audit.order_stats(),
        "co_monotonicity_violations": audit.co_monotonicity_violations(),
        "reflexivity": {},
        "decisions": {},
        "validation": validation_stats(audit),
        "timing_s": audit.timing,
    }
    for g in SOURCES:
        F, F_alt = audit.fidelity_gap(g)
        out["fidelity"][g] = {"F": float(F), "F_alt": float(F_alt),
                              "decomp_max": float(abs(F - F_alt))}

    g_star = min(SOURCES, key=lambda g: audit.phi[g])
    out["g_star"] = g_star
    for g in SOURCES:
        out["reflexivity"][g] = audit.reflexivity(g)

    # decision rules
    zc = ZCache(audit.ds, audit.sources, audit.cand, audit.cfg)
    recs = rule_recommendations(audit, zc)
    per_user = {}
    for rule, g in recs.items():
        per_user[rule] = realized_ndcg_per_user(audit, g)
    per_user["Random"] = np.mean(
        [realized_ndcg_per_user(audit, g) for g in SOURCES], axis=0)
    out["decisions"]["recommendations"] = recs
    out["decisions"]["realized_mean"] = {
        rule: float(np.mean(per_user[rule])) for rule in per_user}
    out["decisions"]["random_expected"] = float(random_expected(audit))
    out["decisions"]["realized_per_retirement"] = {
        g: float(realized_mean(audit, g)) for g in SOURCES}
    out["decisions"]["per_user"] = {
        rule: per_user[rule].tolist() for rule in per_user}
    return out
