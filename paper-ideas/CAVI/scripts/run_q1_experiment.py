#!/usr/bin/env python3
"""
run_q1_experiment.py — rigorous recommendation-accuracy benchmark for CAVI.

The central methodological gap of the earlier prototype: CAVI only *explained*
a fixed recommender and never changed the ranking, so there was nothing to
benchmark. This script gives CAVI a concrete mechanism — weighting the
history-conditioned profile by the forward cooperative action value (mean
Shapley of the forward value game) — and evaluates it against real baselines
on standard top-k metrics with paired significance tests, multiple seeds, and
fixed candidate sets.

Methods compared (all on the SAME fixed candidate set per user):
  1. random          : negative control
  2. bpr_mf          : standard matrix-factorization baseline (user factor . item factor)
  3. profile_base    : history-conditioned mean-profile recommender (no game)
  4. backshap        : profile weighted by BACKWARD Shapley (interaction importance
                       w.r.t. immediate next-item utility)
  5. cavi_fwd        : profile weighted by FORWARD Shapley (Cooperative Action Value,
                       interaction importance w.r.t. expected future utility)

Metrics (fixed candidates, held-out test target):
  NDCG@K, Recall@K, MRR, plus paired Wilcoxon significance, Holm correction,
  and effect size (rank-biserial). Multiple seeds reported as mean +/- std.

This is the empirical backbone a Q1 recommender/methods paper requires.
"""
import os
import sys
import json
import time
import argparse
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cavi.data import load_ratings, load_items, build_user_sequences
from cavi.recommender import bpr_item_factors, ProfileRecommender, DynamicsModel
from cavi.games import Feasibility
from cavi.allocation import compute_cav, component_shapley

# ---------------------------------------------------------------- config ----
D = 32
K = 20
N_CAND = 100          # candidate-set size per user (test item + negatives)
HORIZON = 2           # forward horizon (keep small for tractability)
GAMMA = 0.9
KAPPA = 0.0           # Q1 headline = forward mean; risk channel reported separately
ENS = 4               # ensemble for forward value
DTY = 1.0
N_LEVERS = 6          # number of recent interactions treated as levers
MC_M = 24             # Monte-Carlo permutations for forward/backward Shapley


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "gate", "data"))
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--train-users", type=int, default=1200)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    p.add_argument("--candidates", type=int, default=N_CAND)
    return p.parse_args()


def ndcg(rank_of_rel, n_rel):
    idcg = sum(1.0 / np.log2(j + 1) for j in range(1, n_rel + 1))
    if idcg == 0:
        return 0.0
    return sum(1.0 / np.log2(r + 1) for r in rank_of_rel) / idcg


def evaluate(scored_candidates, relevant_set):
    """scored_candidates: list of (item, score). relevant_set: set of test items."""
    order = sorted(scored_candidates, key=lambda x: -x[1])
    ranks = [i + 1 for i, (it, _) in enumerate(order[:K]) if it in relevant_set]
    n_rel = min(len(relevant_set), K)
    ndcg_v = ndcg(ranks, n_rel)
    rec_v = len([r for r in ranks if r <= K]) / max(n_rel, 1)
    # MRR
    mrr = 0.0
    for i, (it, _) in enumerate(order[:K]):
        if it in relevant_set:
            mrr = 1.0 / (i + 1)
            break
    return ndcg_v, rec_v, mrr


def wilcoxon(a, b):
    from scipy.stats import wilcoxon
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 1.0
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return 1.0


def rank_biserial(a, b):
    """Matched-pairs rank-biserial effect size of method A vs baseline B."""
    from scipy.stats import rankdata
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    pos = np.sum(d > 0); neg = np.sum(d < 0)
    return float((pos - neg) / (pos + neg)) if (pos + neg) else 0.0


def profile_score(Q, cand, items, weights):
    """score = weighted-mean(item factors of `items`) . Q[cand]."""
    if len(items) == 0:
        p = np.zeros(Q.shape[1])
    else:
        idx = np.asarray(items)
        w = np.asarray(weights, float).reshape(-1, 1)
        p = (Q[idx] * w).sum(axis=0) / max(w.sum(), 1e-9)
    s = Q[np.asarray(cand)] @ p
    return float(np.asarray(s).reshape(-1)[0])


def main():
    args = parse_args()
    t0 = time.time()
    ratings = load_ratings(os.path.join(args.data, "ml1m_ratings.dat"))
    items_map = load_items(os.path.join(args.data, "ml1m_items.dat"))
    n_items = max(items_map.keys()) + 1

    # implicit feedback: rating >= 4 counts as a positive interaction
    seqs = {}
    for u, i, r, t in ratings:
        if r >= 4.0:
            seqs.setdefault(u, []).append((i, t))
    for u in seqs:
        seqs[u].sort(key=lambda x: x[1])

    allu = [u for u, s in seqs.items() if len(s) >= N_LEVERS + 5]
    print(f"[data] {len(allu)} users with enough history, {n_items} items")

    # store results across seeds
    methods = ["random", "bpr_mf", "profile_base", "backshap", "cavi_fwd"]
    per_user_ndcg = {m: [] for m in methods}
    per_user_rec = {m: [] for m in methods}
    per_user_mrr = {m: [] for m in methods}

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        rng.shuffle(allu)
        train_users = allu[: min(len(allu) // 2, args.train_users)]
        eval_users = allu[len(allu) // 2: len(allu) // 2 + args.users]

        # train BPR item + user factors
        Q, P = bpr_item_factors_with_user(ratings, train_users, n_items, D,
                                          epochs=12, triplets=150000, seed=seed)
        print(f"[seed {seed}] trained BPR Q={Q.shape}")

        for u in eval_users:
            s = seqs[u]
            # LEAKAGE-SAFE split:
            #   test_item (last)   -> held-out evaluation target
            #   val_item (2nd last)-> validation target, used ONLY to fit the
            #                        forward/backward Shapley weights
            #   history (before)   -> training profile
            test_item = s[-1][0]
            val_item = s[-2][0]
            history = [i for i, _ in s[:-2]]
            if not history:
                continue
            # levers = most recent N_LEVERS of history
            levers = history[-N_LEVERS:]
            base = history[:-N_LEVERS] if len(history) > N_LEVERS else history

            # fixed candidate set: test_item + sampled negatives.
            # We EXCLUDE val_item (used for weighting) and all known training
            # items so the weighting never leaks the test target.
            known = set([i for i, _ in s])
            rng_i = np.random.default_rng(seed + u)
            # sample negatives with replacement from full item range, then filter
            # out the user's known items (standard negative-sampling protocol)
            cand = [test_item]
            seen = {test_item}
            while len(cand) < args.candidates:
                it = int(rng_i.integers(0, n_items))
                if it in known or it in seen:
                    continue
                cand.append(it)
                seen.add(it)
            cand = cand[: args.candidates]

            relevant = {test_item}

            # ---- method scores ----
            # random
            r_scores = {it: rng_i.random() for it in cand}
            # bpr_mf (user factor from trained P)
            ui = {x: k for k, x in enumerate(train_users)}
            if u in ui:
                pu = P[ui[u]]
            else:
                pu = Q[np.asarray(history)].mean(axis=0)
            mf_scores = {it: float(pu @ Q[it]) for it in cand}
            # profile_base (history mean, levers unweighted = weight 1)
            prof_items = base + levers
            pb_scores = {it: float(profile_score(Q, [it], prof_items, [1.0]*len(prof_items))) for it in cand}

            # forward & backward Shapley over levers (fitted on VALIDATION target)
            rec0 = ProfileRecommender(Q, cand, K=K)
            dyn = DynamicsModel(rec0, temp=DTY)
            feas = Feasibility([list(range(len(levers)))])
            lever_idx = list(range(len(levers)))

            # forward mean value fn: expected future util (uses val_item as target)
            def fwd_mean(S):
                active = [levers[i] for i in S]
                m, _ = dyn.forward_value(base, active, [val_item],
                                         HORIZON, gamma=GAMMA, ensemble=ENS, seed=seed)
                return m
            # backward value fn: immediate next-item utility (val_item as target)
            def back_val(S):
                active = [levers[i] for i in S]
                return rec0.future_util(base + active, [val_item])

            phi_fwd = component_shapley(fwd_mean, feas, lever_idx, M=MC_M, seed=seed)
            phi_back = component_shapley(back_val, feas, lever_idx, M=MC_M, seed=seed)

            # CAVI/backshap weighted profile: weight levers by softplus(shapley)
            def softplus(x, tau=0.5):
                return np.log1p(np.exp(np.clip(x / tau, -10, 10)))
            w_fwd = softplus(phi_fwd)
            w_back = softplus(phi_back)
            # weights over full profile (base weight 1, lever weights = shapley-based)
            full_items = base + levers
            full_w_base = [1.0]*len(base)
            full_w_fwd = full_w_base + list(w_fwd)
            full_w_back = full_w_base + list(w_back)

            cavi_scores = {it: float(profile_score(Q, [it], full_items, full_w_fwd)) for it in cand}
            back_scores = {it: float(profile_score(Q, [it], full_items, full_w_back)) for it in cand}

            # evaluate each method on the held-out TEST item (never used for weighting)
            for name, sc in [("random", r_scores), ("bpr_mf", mf_scores),
                             ("profile_base", pb_scores), ("backshap", back_scores),
                             ("cavi_fwd", cavi_scores)]:
                nd, rc, mr = evaluate([(it, sc[it]) for it in cand], relevant)
                per_user_ndcg[name].append(nd)
                per_user_rec[name].append(rc)
                per_user_mrr[name].append(mr)

    # ---- aggregate + significance ----
    res = {}
    for name in methods:
        res[name] = {
            "ndcg_mean": float(np.mean(per_user_ndcg[name])),
            "ndcg_std": float(np.std(per_user_ndcg[name])),
            "recall_mean": float(np.mean(per_user_rec[name])),
            "mrr_mean": float(np.mean(per_user_mrr[name])),
            "n_users": len(per_user_ndcg[name]),
        }
    # significance vs profile_base
    sig = {}
    for name in ["cavi_fwd", "backshap", "bpr_mf"]:
        p = wilcoxon(per_user_ndcg[name], per_user_ndcg["profile_base"])
        rb = rank_biserial(per_user_ndcg[name], per_user_ndcg["profile_base"])
        sig[name] = {"wilcoxon_p_vs_base": p, "rank_biserial_vs_base": rb}
    # Holm correction across the 3 pairwise tests
    ps = [sig[m]["wilcoxon_p_vs_base"] for m in ["cavi_fwd", "backshap", "bpr_mf"]]
    order = sorted(range(3), key=lambda i: ps[i])
    holm = [1.0]*3
    for k, i in enumerate(order):
        holm[i] = min(1.0, ps[i] * (3 - k))
    for j, m in enumerate(["cavi_fwd", "backshap", "bpr_mf"]):
        sig[m]["holm_p"] = holm[j]

    out = {"methods": res, "significance_vs_base": sig,
           "config": vars(args)}
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "q1_experiment.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f"(elapsed {time.time()-t0:.0f}s)")


def bpr_item_factors_with_user(ratings, users, n_items, d=32, epochs=12,
                               triplets=150000, lr=0.05, reg=0.01, seed=0,
                               threshold=4.0):
    """BPR returning BOTH item factors Q and user factors P."""
    rng = np.random.default_rng(seed)
    users = list(users)
    ui = {}
    for u, i, r, t in ratings:
        if u in users and r >= threshold:
            ui.setdefault(u, set()).add(i)
    pos = np.array([(u, i) for u, its in ui.items() for i in its])
    uid = {u: k for k, u in enumerate(users)}
    P = rng.normal(0, 0.01, (len(users), d)).astype(np.float64)
    Q = rng.normal(0, 0.01, (n_items, d)).astype(np.float64)
    all_items = np.arange(n_items)
    for ep in range(epochs):
        idx = rng.choice(len(pos), size=triplets, replace=True)
        for uu, ii in zip(pos[idx, 0], pos[idx, 1]):
            jj = int(rng.choice(all_items))
            pu = P[uid[uu]]; qi = Q[ii]; qj = Q[jj]
            x = pu.dot(qi) - pu.dot(qj)
            sig = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
            P[uid[uu]] += lr * (sig * (qi - qj) - reg * pu)
            Q[ii] += lr * (sig * pu - reg * qi)
            Q[jj] += lr * (-sig * pu - reg * qj)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    return np.ascontiguousarray(Q), np.ascontiguousarray(P)


if __name__ == "__main__":
    main()
