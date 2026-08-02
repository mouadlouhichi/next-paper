#!/usr/bin/env python3
"""
run_cross_dataset.py — run the CAVI pipeline on any supported dataset.

Datasets (canonical LightGCN split, in data/<name>/):
  amazon-book   : train.txt/test.txt/user_list.txt/item_list.txt  (no metadata)
  yelp2018      : same
  gowalla       : same (if present)
  lastfm        : same (if present)
  ml-1m         : tab-separated ratings/items in gate/data/ (has genres)

Because the LightGCN canonical splits carry NO timestamps and NO item metadata
(proposal SignalShap §4.1), this script:
  - builds per-user sequences from the train split (order = given),
  - uses the test split as the future/target set,
  - uses *popularity-based* feasibility (top-decile-popular items are anchors /
    immovable) instead of the genre-anchor used for ML-1M.

Everything else is identical to the ML-1M experiment: history-conditioned BPR
recommender, forward mean/variance games, Myerson CAV allocation with exact
additivity verification, minimal-action recourse, and OPE.
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cavi.data import (load_lightgcn_split, lightgcn_stats, load_remap_lists,
                       build_user_seq_from_split, future_from_test,
                       item_popularity, movability_from_popularity)
from cavi.recommender import bpr_item_factors, ProfileRecommender, DynamicsModel
from cavi.games import Feasibility
from cavi.allocation import compute_cav, component_shapley, verify_additivity_identity
from cavi.recourse import MinimalActionPlanner
from cavi.ope import dr_estimate, effective_sample_size, discrepancy_gate

D = 32
K = 20
N_CAND = 150
HORIZON = 3
GAMMA = 0.9
KAPPA = 0.5
ENS = 6
DTY = 1.0


def spearman(a, b):
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["amazon-book", "yelp2018", "gowalla",
                                         "lastfm", "ml-1m"], required=True)
    p.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    p.add_argument("--users", type=int, default=30)
    p.add_argument("--nmax", type=int, default=8)
    p.add_argument("--train-users", type=int, default=1200)
    p.add_argument("--budget", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def build_per_user(seqs, test_dict, pop, u, nmax, h_fut=1):
    """
    For a LightGCN split, form (base, window, future):
      base   = all train items before the last `nmax`
      window = the last `nmax` train items  (the actionable levers)
      future = the test items (held-out targets)
    h_fut is the number of test items used as targets (cap).
    """
    its = [i for i, _ in seqs.get(u, [])]
    if len(its) < nmax + 2:
        return None
    window = its[-nmax:]
    base = its[:-nmax]
    future = future_from_test(test_dict, u)[:h_fut]
    if not future or not base or len(window) < 2:
        return None
    return base, window, future


def main():
    args = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    if args.dataset == "ml-1m":
        from cavi.data import (load_ratings, load_items, build_user_sequences,
                               temporal_split, dominant_genre)
        data_dir = os.path.join(os.path.dirname(__file__), "..", "gate", "data")
        ratings = load_ratings(os.path.join(data_dir, "ml1m_ratings.dat"))
        items = load_items(os.path.join(data_dir, "ml1m_items.dat"))
        seqs = build_user_sequences(ratings)
        n_items = max(items.keys()) + 1
        tr_items = None
        def build(u):
            base, window, future = temporal_split(seqs[u], args.nmax, 4)
            if not base or len(window) < 2 or not future:
                return None
            return base, window, future
        def movability(window, base):
            dom = dominant_genre(items, base + window)
            return [dom not in (items.get(it, "") or "") for it in window]
    else:
        data_dir = os.path.join(args.data_root, args.dataset)
        train_dict, test_dict = load_lightgcn_split(data_dir, args.dataset)
        stats = lightgcn_stats(train_dict, test_dict)
        seqs = build_user_seq_from_split(train_dict, test_dict)
        n_items = stats["items"]
        pop = item_popularity(train_dict)
        tr_items = None
        def build(u):
            return build_per_user(seqs, test_dict, pop, u, args.nmax, h_fut=4)
        def movability(window, base):
            return [not m for m in movability_from_popularity(pop, window)]

    allu = [u for u, s in seqs.items() if s]
    rng.shuffle(allu)
    # split train/eval users for the recommender (avoid leakage)
    train_users = allu[: min(len(allu) // 2, args.train_users)]
    eval_users = allu[len(allu) // 2: len(allu) // 2 + args.users]

    # unified BPR interaction list: (u, i, rating, ts). ML-1M uses real ratings
    # with threshold; LightGCN datasets are implicit (rating=1.0, threshold=0).
    if args.dataset == "ml-1m":
        train_dict = None
        ratings_list = [(u, i, r, t) for u, i, r, t in ratings
                        if u in set(train_users)]
        thr = 4.0
    else:
        ratings_list = []
        for u in train_users:
            for i in train_dict.get(u, []):
                ratings_list.append((u, i, 1.0, 0))
        thr = 0.0

    print(f"[{args.dataset}] {stats if args.dataset != 'ml-1m' else ''}")
    Q = bpr_item_factors(ratings_list, train_users, n_items, D,
                         epochs=12, triplets=150000, seed=args.seed, threshold=thr)
    print(f"[model] BPR item factors {Q.shape}")

    per_user = []
    rho_bf = []
    for u in eval_users:
        parsed = build(u)
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
        mov = movability(window, base)
        movable = [k for k, ok in enumerate(mov) if ok]
        if len(movable) < 2:
            continue
        lever_items = [window[k] for k in movable]
        feas = Feasibility([movable])
        dyn = DynamicsModel(rec, temp=DTY)

        def mean_fn(S):
            ai = [lever_items[i] for i in S]
            m, _ = dyn.forward_value(base, ai, future, HORIZON, gamma=GAMMA,
                                     ensemble=ENS, seed=args.seed)
            return m
        def var_fn(S):
            ai = [lever_items[i] for i in S]
            _, v = dyn.forward_value(base, ai, future, HORIZON, gamma=GAMMA,
                                     ensemble=ENS, seed=args.seed + 100)
            return v
        def back_fn(S):
            ai = [lever_items[i] for i in S]
            return rec.future_util(base + ai, future)

        phi_back = component_shapley(back_fn, feas, list(range(len(movable))),
                                     M=60, seed=args.seed)
        cav = compute_cav(mean_fn, var_fn, KAPPA, feas, list(range(len(movable))),
                          M=None, seed=args.seed)
        diff, ok = verify_additivity_identity(mean_fn, var_fn, KAPPA, feas,
                                              list(range(len(movable))),
                                              M=None, seed=args.seed)

        rho = spearman(np.abs(phi_back), np.abs(cav.phi_mean))
        rho_bf.append(rho)

        costs = [1.0 + 0.1 * k for k in range(len(movable))]
        planner = MinimalActionPlanner(cav, costs, budget=args.budget)
        def uplift(S):
            return sum(cav.cav[i] for i in S)
        sel, sel_cost = planner.greedy_plan(min_uplift=None, uplift_fn=uplift)
        naive_lift = uplift(sel) - uplift([])

        per_user.append({
            "user": u, "n_levers": len(movable), "rho_bf": rho,
            "additivity_ok": bool(ok), "additivity_diff": diff,
            "plan": sel, "plan_cost": sel_cost, "naive_lift": naive_lift,
        })

    rho_bf = np.array([r for r in rho_bf if not np.isnan(r)])
    report = {
        "dataset": args.dataset,
        "n_users": len(per_user),
        "mean_rho_back_fwd": float(np.mean(rho_bf)) if len(rho_bf) else None,
        "frac_rho_lt_0.6": float(np.mean(rho_bf < 0.6)) if len(rho_bf) else None,
        "additivity_all_ok": all(p["additivity_ok"] for p in per_user),
        "mean_plan_size": float(np.mean([len(p["plan"]) for p in per_user])),
        "mean_naive_lift": float(np.mean([p["naive_lift"] for p in per_user])),
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"cross_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "report": report, "per_user": per_user},
                  f, indent=2, default=str)

    print("=" * 70)
    print(f"CAVI CROSS-DATASET — {args.dataset}")
    print("=" * 70)
    print(f"users evaluated            : {report['n_users']}")
    print(f"mean rho(back,forward)     : {report['mean_rho_back_fwd']}")
    print(f"frac rho<0.6               : {report['frac_rho_lt_0.6']}")
    print(f"additivity identity all OK : {report['additivity_all_ok']}")
    print(f"mean plan size             : {report['mean_plan_size']:.2f}")
    print(f"mean naive fwd lift        : {report['mean_naive_lift']:.4f}")
    print(f"saved -> {out}  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
