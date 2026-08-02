#!/usr/bin/env python3
"""
run_paperA_divergence.py — Paper A: pin down the forward value-function
operationalization and measure the *full variance-game Shapley*.

Resolves two open items from the CAVI README:
  1. The divergence magnitude between backward and forward orderings was
     configuration-sensitive (gate ~0.09 vs full experiment ~0.74). This
     script computes, on the SAME users and lever space, the divergence under
     several *matched* operationalizations so the effect of each choice is
     isolated rather than confounded.
  2. The variance channel was only ever measured with a singleton proxy
     (which gave rho~1.0). This computes the FULL Shapley of the variance game
     and tests whether risk-adjustment reorders levers when done properly.

Metrics per (config, user):
  - rho(back_shapley, fwd_mean_shapley)     : forward divergence
  - rho(fwd_mean, fwd_var)                  : does variance track mean?
  - rho(CAV@k=0, CAV@k>0)                    : does risk-adjustment reorder?
  - frac users reordered by risk

Configs swept (matched on everything else):
  A. horizon H in {1,3}                       : forward lookahead
  B. backward basis: immediate-next vs full-future
"""
import os
import sys
import json
import time
import argparse
import itertools

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cavi.data import (load_ratings, load_items, build_user_sequences,
                       temporal_split, dominant_genre)
from cavi.recommender import bpr_item_factors, ProfileRecommender, DynamicsModel
from cavi.games import Feasibility, CooperativeGame
from cavi.allocation import compute_cav, component_shapley, verify_additivity_identity

D = 32
K = 20
N_CAND = 150
H_FUT = 4
KAPPA = 0.5
ENS = 6
GAMMA = 0.9
DTY = 1.0


def spearman(a, b):
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def build_user(seq, nmax, base_len_min, items):
    base, window, future = temporal_split(seq, nmax, H_FUT)
    if not window or not base or len(base) < base_len_min or len(window) < 2:
        return None
    return base, window, future


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "gate", "data"))
    p.add_argument("--users", type=int, default=20)
    p.add_argument("--nmax", type=int, default=8)
    p.add_argument("--perm", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    ratings = load_ratings(os.path.join(args.data, "ml1m_ratings.dat"))
    items = load_items(os.path.join(args.data, "ml1m_items.dat"))
    seqs = build_user_sequences(ratings)
    n_items = max(items.keys()) + 1
    allu = [u for u, s in seqs.items() if len(s) >= args.nmax + H_FUT + 8]
    rng.shuffle(allu)
    train_users = allu[: max(len(allu) // 2, 1)][:1200]
    eval_users = allu[len(allu) // 2: len(allu) // 2 + args.users]

    Q = bpr_item_factors(ratings, train_users, n_items, D, epochs=12, triplets=150000,
                         seed=args.seed)
    print(f"[data] {len(eval_users)} eval users, BPR {Q.shape}")

    # configs: (label, horizon, backward_uses_full_future)
    configs = [("H1-fullfut", 1, True), ("H3-fullfut", 3, True),
               ("H3-nextonly", 3, False)]

    agg = {lab: {"rho_bf": [], "rho_mean_var": [], "rho_risk_reorder": [],
                 "frac_reordered": []} for lab, _, _ in configs}
    # value-signal diagnostics (is the forward value function degenerate?)
    signal = {"mean_range": [], "mean_range_gt": 0}

    n_ok = 0
    for u in eval_users:
        parsed = build_user(seqs[u], args.nmax, 8, items)
        if parsed is None:
            continue
        base, window, future = parsed
        rec0 = ProfileRecommender(Q, [])
        full = base + window
        scores = Q @ rec0.profile(full)
        cand = list(np.argsort(-scores)[:N_CAND]); cs = set(cand)
        for it in future + window:
            if it not in cs:
                cand.append(it); cs.add(it)
        rec = ProfileRecommender(Q, cand, K=K)
        dom = dominant_genre(items, base + window)
        movable = [k for k, it in enumerate(window)
                   if dom not in (items.get(it, "") or "")]
        if len(movable) < 2:
            continue
        lever_items = [window[k] for k in movable]
        feas = Feasibility([movable])
        dyn = DynamicsModel(rec, temp=DTY)

        n_ok += 1
        for lab, H, full_fut in configs:
            fut_for_back = future if full_fut else future[:1]

            def back_fn(S):
                ai = [lever_items[i] for i in S]
                return rec.future_util(base + ai, fut_for_back)

            def mean_fn(S, seed_off=0):
                ai = [lever_items[i] for i in S]
                m, _ = dyn.forward_value(base, ai, future, H, gamma=GAMMA,
                                         ensemble=ENS, seed=args.seed + seed_off)
                return m

            def var_fn(S):
                ai = [lever_items[i] for i in S]
                _, v = dyn.forward_value(base, ai, future, H, gamma=GAMMA,
                                         ensemble=ENS, seed=args.seed + 100)
                return v

            # value-signal diagnostic (coalition-level spread of forward mean)
            if lab == "H3-fullfut":
                singleton = [mean_fn([i]) for i in range(len(movable))]
                signal["mean_range"].append(float(np.max(singleton) - np.min(singleton)))

            phi_back = component_shapley(back_fn, feas, list(range(len(movable))),
                                         M=args.perm, seed=args.seed)
            phi_fwd_mean = component_shapley(mean_fn, feas, list(range(len(movable))),
                                             M=args.perm, seed=args.seed)
            phi_fwd_var = component_shapley(var_fn, feas, list(range(len(movable))),
                                             M=args.perm, seed=args.seed + 1)
            cav_k0 = phi_fwd_mean - 0.0 * phi_fwd_var
            cav_k = phi_fwd_mean - KAPPA * phi_fwd_var

            agg[lab]["rho_bf"].append(spearman(np.abs(phi_back), np.abs(phi_fwd_mean)))
            agg[lab]["rho_mean_var"].append(spearman(np.abs(phi_fwd_mean), np.abs(phi_fwd_var)))
            agg[lab]["rho_risk_reorder"].append(spearman(np.abs(cav_k0), np.abs(cav_k)))
            # reordered = top-1 or ranking changes materially
            top_k0 = set(np.argsort(-np.abs(cav_k0))[:2])
            top_k = set(np.argsort(-np.abs(cav_k))[:2])
            agg[lab]["frac_reordered"].append(0.0 if top_k0 == top_k else 1.0)

    # value-signal aggregate
    mr = np.array(signal["mean_range"])
    sig_frac = float(np.mean(mr > 1e-3)) if len(mr) else float("nan")

    print("=" * 78)
    print("PAPER A — divergence & full variance-game Shapley (matched configs)")
    print("=" * 78)
    print(f"users evaluated: {n_ok}")
    print(f"forward value signal: mean coalition-range = {np.nanmean(mr):.3f}, "
          f"frac users with signal>1e-3 = {sig_frac:.3f}\n")
    hdr = f"{'config':<14} {'rho(B,F)':>10} {'rho(mean,var)':>14} {'rho(risk-reorder)':>18} {'frac-reorder':>13}"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for lab, _, _ in configs:
        rb = np.nanmean(agg[lab]["rho_bf"]); rv = np.nanmean(agg[lab]["rho_mean_var"])
        rr = np.nanmean(agg[lab]["rho_risk_reorder"]); fr = np.nanmean(agg[lab]["frac_reordered"])
        summary[lab] = {"rho_bf": float(rb), "rho_mean_var": float(rv),
                        "rho_risk_reorder": float(rr), "frac_reordered": float(fr)}
        print(f"{lab:<14} {rb:>10.3f} {rv:>14.3f} {rr:>18.3f} {fr:>13.3f}")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "paperA_divergence.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "signal": {"mean_range": float(np.nanmean(mr)),
                              "frac_signal": sig_frac}},
                  f, indent=2, default=str)
    print(f"\nsaved -> {out}  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
