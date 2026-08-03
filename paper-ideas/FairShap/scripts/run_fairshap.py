#!/usr/bin/env python3
"""run_fairshap.py — full FairShap experiment on MovieLens-1M."""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairshap.data import (load_ml1m, temporal_split, popularity, popularity_tier,
                           provider_of_items, user_activity_group, item_similarity)
from fairshap.metrics import compute_all
from fairshap.model import train_hypergraph, train_hypergraph_with_fair_loss
from fairshap.rerank import fair_rerank, deterministic_rerank, calibrated_rerank
from fairshap.pipeline import recommend, recommend_scores
from fairshap.game import exposure_shapley

D = 32
K = 20
CAND = 100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "..",
                                                  "CAVI", "gate", "data", "ml1m_ratings.dat"))
    p.add_argument("--users", type=int, default=120)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--m", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    users_items, _ = load_ml1m(args.data)
    n_items = max(i for u in users_items for i in users_items[u]) + 1
    pop = popularity(users_items)
    tier = popularity_tier(pop)
    provider = provider_of_items(users_items, pop)
    ugroup = user_activity_group(users_items)
    allu = [u for u, its in users_items.items() if len(its) >= 5]
    print(f"[data] {len(allu)} users, {n_items} items")

    rng = np.random.default_rng(args.seeds[0])
    rng.shuffle(allu)
    train_users = allu[: max(len(allu) // 2, 1)]
    eval_users = allu[len(allu) // 2: len(allu) // 2 + args.users]
    train_ui = {u: users_items[u] for u in train_users}

    print("[model] training hypergraph recommender ...")
    Q = train_hypergraph(train_ui, n_items, max(allu) + 1, dim=D,
                         epochs=args.epochs, seed=args.seeds[0], verbose=True)
    print("[model] training fairness-regularized variant ...")
    Q_fair = train_hypergraph_with_fair_loss(train_ui, n_items, max(allu) + 1,
                                             dim=D, epochs=args.epochs,
                                             seed=args.seeds[0], lam_fair=0.2,
                                             popularity=pop, verbose=True)
    item_sim = item_similarity(Q)
    print("[model] embeddings done")

    eval_relevant, eval_profile, eval_cand = {}, {}, {}
    plain_lists, plain_lists_fair = {}, {}
    exposure = {}

    for u in eval_users:
        tr, te = temporal_split(users_items[u], 1)
        if not tr:
            continue
        eval_relevant[u] = te
        eval_profile[u] = tr
        cand = list(te)
        known = set(tr) | set(te)
        ri = np.random.default_rng(u)
        while len(cand) < CAND:
            it = int(ri.integers(0, n_items))
            if it in known or it in cand:
                continue
            cand.append(it)
        eval_cand[u] = cand
        plain_lists[u] = recommend(Q, tr, cand, K)
        plain_lists_fair[u] = recommend(Q_fair, tr, cand, K)

    for u, lst in plain_lists.items():
        for i, it in enumerate(lst[:K]):
            exposure[it] = exposure.get(it, 0.0) + 1.0 / np.log2(i + 2)

    universe = sorted(set([it for u in eval_cand for it in eval_cand[u]]))
    print(f"[game] exposure-Shapley over {len(universe)} items (M={args.m}) ...")
    phi_fair = {}
    chunk = 300
    for s in range(0, len(universe), chunk):
        sub = universe[s:s + chunk]
        phi_fair.update(exposure_shapley(exposure, item_sim, sub, M=args.m,
                                         seed=args.seeds[0]))
    print("[game] done")

    methods = {}
    methods["plain"] = plain_lists
    methods["fair_regularized"] = plain_lists_fair
    inv_pop_lists = {}
    for u, tr in eval_profile.items():
        scores = recommend_scores(Q, tr, eval_cand[u])
        boosted = {it: s * (1.0 / (1.0 + np.log1p(pop.get(it, 0)))) for it, s in scores.items()}
        inv_pop_lists[u] = sorted(boosted, key=lambda it: -boosted[it])[:K]
    methods["inv_pop"] = inv_pop_lists
    cal_lists = {}
    dist = {0: 0.2, 1: 0.3, 2: 0.5}
    for u, tr in eval_profile.items():
        scores = recommend_scores(Q, tr, eval_cand[u])
        cal_lists[u] = calibrated_rerank(scores, pop, tier, dist)[:K]
    methods["calibrated"] = cal_lists
    det_lists = {}
    for u, tr in eval_profile.items():
        scores = recommend_scores(Q, tr, eval_cand[u])
        det_lists[u] = deterministic_rerank(scores)[:K]
    methods["dpp"] = det_lists
    for gamma in args.gammas:
        fs_lists = {}
        for u, tr in eval_profile.items():
            scores = recommend_scores(Q, tr, eval_cand[u])
            fs_lists[u] = fair_rerank(scores, phi_fair, gamma)[:K]
        methods[f"fairshap_g{gamma}"] = fs_lists

    results = {}
    for name, lists in methods.items():
        met = compute_all(lists, eval_relevant, item_sim, pop, provider, ugroup,
                          n_items, K)
        results[name] = met
        print(f"\n[{name}]")
        print(f"  NDCG={met['ndcg']:.4f} Recall={met['recall']:.4f} "
              f"Gini={met['gini']:.4f} ARP={met['arp']:.1f} "
              f"LT={met['long_tail']:.4f} ILD={met['ild']:.4f} "
              f"consGap={met['consumer_gap']:.4f}")

    out = {"methods": results, "config": vars(args),
           "n_items": n_items, "n_eval_users": len(eval_users)}
    path = os.path.join(os.path.dirname(__file__), "..", "results", "fairshap_ml1m.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nsaved -> {path}  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
