#!/usr/bin/env python3
"""
run_q1_v3_action.py — CAVI actionable-recourse benchmark (the paper's core claim).

Research question: given a user's history, WHICH single interaction should the
user act on (amplify) to most improve their FUTURE recommendation quality?

  - Forward CAV  = Shapley of the forward value over the user's recent
                   interactions, FIT on a validation window.
  - The "action" = amplifying the top-CAV interaction (reweight its item factor
                   by a factor > 1), representing the user engaging more with it.
  - Evaluated on a HELD-OUT future window (never used to fit the weights).

This is the "actionable intelligence" claim: the forward game ranks interactions
by their future value, which backward-Shapley and random do not. Metrics:
  - mean realized future-utility lift (CAV action vs backward vs random vs none)
  - intervention success rate (fraction where the CAV action improves future util)
  - paired significance (Wilcoxon + Holm) and effect size.

Leakage-safe: weights fit on validation items, evaluated on a separate held-out
future window, on fixed candidate sets, multiple seeds.
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cavi.data import load_ratings
from cavi.games import Feasibility
from cavi.allocation import component_shapley

D = 32
K = 20
TEMP = 1.0
LAMBDA_DIV = 0.3
N_LEVERS = 6
N_VAL = 2        # validation window length (fit weights)
N_FUT = 3        # held-out future window length (evaluate)
MC_M = 32
NEG_CAND = 40
AMPLIFY = 2.0    # amplification factor for the chosen action


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "gate", "data"))
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--train-users", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    p.add_argument("--candidates", type=int, default=100)
    return p.parse_args()


def bpr_item_factors(ratings, users, n_items, d=D, epochs=12, triplets=150000,
                     lr=0.05, reg=0.01, seed=0, threshold=4.0):
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


def softplus(x, tau=0.5):
    return np.log1p(np.exp(np.clip(x / tau, -10, 10)))


def sigmoid(x, T=1.0):
    return 1.0 / (1.0 + np.exp(-np.clip(x / T, -30, 30)))


def profile_score(Q, cand, items, weights):
    if len(items) == 0:
        p = np.zeros(Q.shape[1])
    else:
        idx = np.asarray(items)
        w = np.asarray(weights, float).reshape(-1, 1)
        p = np.sum(Q[idx] * w, axis=0) / max(w.sum(), 1e-9)
    s = Q[np.asarray(cand)] @ p
    return np.asarray(s).reshape(-1)


def future_util(Q, p, future, negs, K=K):
    """NDCG@K of profile p's ranking over (future + negs) candidates."""
    cand = list(future) + list(negs)
    scores = np.array([float(p @ Q[it]) for it in cand])
    order = np.argsort(-scores, kind="stable")
    cand_set = set(cand)
    rel_ranks = [i + 1 for i, cp in enumerate(order[:K]) if cand[cp] in set(future)]
    n_rel = min(len(future), K)
    idcg = sum(1.0 / np.log2(j + 1) for j in range(1, n_rel + 1))
    if idcg == 0:
        return 0.0
    return sum(1.0 / np.log2(r + 1) for r in rel_ranks) / idcg


def ndcg(ranks, n_rel):
    idcg = sum(1.0 / np.log2(j + 1) for j in range(1, n_rel + 1))
    if idcg == 0:
        return 0.0
    return sum(1.0 / np.log2(r + 1) for r in ranks) / idcg


def evaluate(Q, p, cand, relevant, K=K):
    scores = np.array([float(p @ Q[it]) for it in cand])
    order = np.argsort(-scores, kind="stable")
    ranks = [i + 1 for i, cp in enumerate(order[:K]) if cand[cp] in relevant]
    n_rel = min(len(relevant), K)
    nd = ndcg(ranks, n_rel)
    rc = len([r for r in ranks if r <= K]) / max(n_rel, 1)
    mrr = 0.0
    for i, cp in enumerate(order[:K]):
        if cand[cp] in relevant:
            mrr = 1.0 / (i + 1); break
    return nd, rc, mrr


def wilcoxon(a, b):
    from scipy.stats import wilcoxon
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b; d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 1.0
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return 1.0


def rank_biserial(a, b):
    from scipy.stats import rankdata
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b; d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 0.0
    pos = np.sum(d > 0); neg = np.sum(d < 0)
    return float((pos - neg) / (pos + neg)) if (pos + neg) else 0.0


def main():
    args = parse_args()
    t0 = time.time()
    ratings = load_ratings(os.path.join(args.data, "ml1m_ratings.dat"))
    n_items = max(i for _, i, _, _ in ratings) + 1

    seqs = {}
    for u, i, r, t in ratings:
        if r >= 4.0:
            seqs.setdefault(u, []).append((i, t))
    for u in seqs:
        seqs[u].sort(key=lambda x: x[1])
    allu = [u for u, s in seqs.items() if len(s) >= N_LEVERS + N_VAL + N_FUT + 1]
    print(f"[data] {len(allu)} users, {n_items} items")

    # metrics per action policy
    policies = ["none", "random", "backshap", "cavi_fwd"]
    lift = {p: [] for p in policies}          # realized future-util lift vs none
    success = {p: [] for p in policies}       # binary: did action help?
    fututil = {p: [] for p in policies}

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        rng.shuffle(allu)
        train_users = allu[: min(len(allu) // 2, args.train_users)]
        eval_users = allu[len(allu) // 2: len(allu) // 2 + args.users]
        Q, P = bpr_item_factors(ratings, train_users, n_items, seed=seed)
        print(f"[seed {seed}] BPR {Q.shape}")

        for u in eval_users:
            s = seqs[u]
            items = [i for i, _ in s]
            future = items[-N_FUT:]          # HELD-OUT evaluation window
            val = items[-(N_FUT + N_VAL): -N_FUT]  # validation window (fit weights)
            hist = items[: -(N_FUT + N_VAL)]
            if len(hist) < N_LEVERS:
                continue
            levers = hist[-N_LEVERS:]
            base = hist[:-N_LEVERS] if len(hist) > N_LEVERS else hist

            known = set(items)
            rng_i = np.random.default_rng(seed + u)
            # negatives for value function + evaluation candidates
            negs = []
            while len(negs) < NEG_CAND:
                it = int(rng_i.integers(0, n_items))
                if it in known or it in negs:
                    continue
                negs.append(it)

            base_sum = np.sum(Q[np.asarray(base)], axis=0)
            lever_arr = np.asarray(levers)

            # ---- fit forward CAV on VALIDATION window ----
            feas = Feasibility([list(range(len(levers)))])
            idx = list(range(len(levers)))

            def fwd(S):
                active = [levers[i] for i in S]
                p = base_sum.copy()
                for i in active:
                    p = p + Q[i]
                # smooth forward value vs validation items
                acc = 0.0
                for f in val:
                    pos_s = float(p @ Q[f])
                    neg_s = [float(p @ Q[ng]) for ng in negs]
                    acc += sigmoid(pos_s - float(np.mean(neg_s)), TEMP)
                return acc / max(len(val), 1)

            def back(S):
                active = [levers[i] for i in S]
                p = base_sum.copy()
                for i in active:
                    p = p + Q[i]
                return float(p @ Q[val[-1]])

            phi_fwd = component_shapley(fwd, feas, idx, M=MC_M, seed=seed)
            phi_back = component_shapley(back, feas, idx, M=MC_M, seed=seed)

            # ---- status quo profile and its future utility ----
            p0 = base_sum + np.sum(Q[lever_arr], axis=0)
            u0 = future_util(Q, p0, future, negs)

            # ---- define actions: amplify one lever ----
            top_fwd = int(np.argmax(phi_fwd))
            top_back = int(np.argmax(phi_back))
            top_rnd = int(rng_i.integers(0, len(levers)))

            actions = {"none": None, "random": top_rnd, "backshap": top_back,
                       "cavi_fwd": top_fwd}
            for pol, act in actions.items():
                if act is None:
                    p_act = p0
                else:
                    # amplify the chosen lever's item factor
                    p_act = p0 + (AMPLIFY - 1.0) * Q[levers[act]]
                u_act = future_util(Q, p_act, future, negs)
                fututil[pol].append(u_act)
                if pol != "none":
                    lift[pol].append(u_act - u0)
                    success[pol].append(1.0 if u_act > u0 + 1e-9 else 0.0)

    # ---- aggregate ----
    res = {}
    for p in policies:
        res[p] = {"mean_future_util": float(np.mean(fututil[p])),
                  "n_users": len(fututil[p])}
        if p != "none":
            res[p]["mean_lift_vs_none"] = float(np.mean(lift[p]))
            res[p]["success_rate"] = float(np.mean(success[p]))

    sig = {}
    for p in ["cavi_fwd", "backshap", "random"]:
        sig[p] = {"wilcoxon_p_vs_random": wilcoxon(lift[p], lift["random"]),
                  "rank_biserial_vs_random": rank_biserial(lift[p], lift["random"]),
                  "wilcoxon_lift_vs_zero": wilcoxon(lift[p], np.zeros(len(lift[p])))}
    # Holm across the 3 policy comparisons vs random
    ps = [sig[p]["wilcoxon_p_vs_random"] for p in ["cavi_fwd", "backshap", "random"]]
    order = sorted(range(3), key=lambda i: ps[i])
    for k, i in enumerate(order):
        p = ["cavi_fwd", "backshap", "random"][i]
        sig[p]["holm_p_vs_random"] = min(1.0, ps[i] * (3 - k))

    out = {"methods": res, "significance": sig, "config": vars(args)}
    path = os.path.join(os.path.dirname(__file__), "..", "results", "q1_v3_action.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f"(elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
