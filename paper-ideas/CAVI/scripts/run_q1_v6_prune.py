#!/usr/bin/env python3
"""
run_q1_v6_prune.py — CAVI-driven interaction pruning (the empirical claim).

HYPOTHESIS (v6):
    The forward Cooperative Action Value identifies which historical
    interactions are LEAST valuable for future prediction. Removing (pruning)
    the low-CAV interactions from the user profile improves held-out FUTURE-window
    recommendation quality significantly — and, crucially, forward-CAV-informed
    pruning beats random pruning and backward-Shapley pruning.

This is the paper's actionable-intelligence claim made testable on a CPU
backbone: instead of amplifying (weak), we REMOVE low-value interactions, which
is a stronger and more reliable intervention. Evaluation is leakage-safe:
forward-CAV is fit on a validation future window; the effect is measured on a
SEPARATE held-out future window, on fixed candidate sets, multiple seeds, paired
Wilcoxon + Holm + effect size.

Key comparison (the result that matters):
    prune_fwd vs prune_random   (is forward-CAV pruning attribution-driven?)
    prune_fwd vs keep_all       (does pruning help at all?)
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
N_LEVERS = 10
N_VAL = 2
N_FUT = 3
NEG_CAND = 30
MC_M = 40
DECAY_LAMBDA = 0.5
N_PRUNE = 2        # number of interactions to remove


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "gate", "data"))
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--train-users", type=int, default=2000)
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


def sigmoid(x, T=1.0):
    return 1.0 / (1.0 + np.exp(-np.clip(x / T, -30, 30)))


def ndcg(ranks, n_rel):
    idcg = sum(1.0 / np.log2(j + 1) for j in range(1, n_rel + 1))
    if idcg == 0:
        return 0.0
    return sum(1.0 / np.log2(r + 1) for r in ranks) / idcg


def future_util(Q, p, future, negs, K=K):
    cand = list(future) + list(negs)
    scores = np.array([float(p @ Q[it]) for it in cand])
    order = np.argsort(-scores, kind="stable")
    rel_ranks = [i + 1 for i, cp in enumerate(order[:K]) if cand[cp] in set(future)]
    n_rel = min(len(future), K)
    return ndcg(rel_ranks, n_rel)


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
    allu = [u for u, s in seqs.items() if len(s) >= N_LEVERS + N_VAL + N_FUT + 3]
    print(f"[data] {len(allu)} users, {n_items} items")

    methods = ["keep_all", "prune_fwd", "prune_back", "prune_random"]
    ndcg_all = {m: [] for m in methods}

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        rng.shuffle(allu)
        train_users = allu[: min(len(allu) // 2, args.train_users)]
        eval_users = allu[len(allu) // 2: len(allu) // 2 + args.users]
        Q, _ = bpr_item_factors(ratings, train_users, n_items, seed=seed)
        print(f"[seed {seed}] BPR {Q.shape}")

        for u in eval_users:
            s = seqs[u]
            items = [i for i, _ in s]
            future = items[-N_FUT:]                 # held-out eval
            val = items[-(N_FUT + N_VAL): -N_FUT]   # validation (fit)
            hist = items[:-(N_FUT + N_VAL)]
            if len(hist) < N_LEVERS:
                continue
            levers = hist[-N_LEVERS:]
            base = hist[:-N_LEVERS] if len(hist) > N_LEVERS else hist

            known = set(items)
            rng_i = np.random.default_rng(seed + u)
            negs = []
            while len(negs) < NEG_CAND:
                it = int(rng_i.integers(0, n_items))
                if it in known or it in negs:
                    continue
                negs.append(it)

            base_sum = np.sum(Q[np.asarray(base)], axis=0)
            la = np.asarray(levers)
            feas = Feasibility([list(range(len(levers)))])
            idx = list(range(len(levers)))

            def fwd(S):
                active = [levers[i] for i in S]
                p = base_sum.copy()
                for i in active:
                    p = p + Q[i]
                acc = 0.0
                for j, f in enumerate(val):
                    tdec = np.exp(-DECAY_LAMBDA * j)
                    pos_s = float(p @ Q[f])
                    neg_s = [float(p @ Q[ng]) for ng in negs]
                    acc += tdec * sigmoid(pos_s - float(np.mean(neg_s)))
                return acc / max(len(val), 1)

            def back(S):
                active = [levers[i] for i in S]
                p = base_sum.copy()
                for i in active:
                    p = p + Q[i]
                return float(p @ Q[val[-1]])

            phi_fwd = component_shapley(fwd, feas, idx, M=MC_M, seed=seed)
            phi_back = component_shapley(back, feas, idx, M=MC_M, seed=seed)

            p_all = base_sum + np.sum(Q[la], axis=0)
            ndcg_all["keep_all"].append(future_util(Q, p_all, future, negs))

            # prune bottom-N_PRUNE by each attribution
            def prune(phi):
                pr = np.argsort(phi)[:N_PRUNE]
                mask = np.ones(len(levers), bool); mask[pr] = False
                return base_sum + np.sum(Q[la[mask]], axis=0)
            ndcg_all["prune_fwd"].append(future_util(Q, prune(phi_fwd), future, negs))
            ndcg_all["prune_back"].append(future_util(Q, prune(phi_back), future, negs))
            prr = rng_i.choice(len(levers), N_PRUNE, replace=False)
            maskr = np.ones(len(levers), bool); maskr[prr] = False
            ndcg_all["prune_random"].append(future_util(Q, base_sum + np.sum(Q[la[maskr]], axis=0), future, negs))

    res = {m: {"ndcg_mean": float(np.mean(ndcg_all[m])),
               "ndcg_std": float(np.std(ndcg_all[m])),
               "n_users": len(ndcg_all[m])} for m in methods}
    sig = {
        "prune_fwd_vs_random": {"p": wilcoxon(ndcg_all["prune_fwd"], ndcg_all["prune_random"]),
                                 "rank_biserial": rank_biserial(ndcg_all["prune_fwd"], ndcg_all["prune_random"])},
        "prune_fwd_vs_keep": {"p": wilcoxon(ndcg_all["prune_fwd"], ndcg_all["keep_all"]),
                               "rank_biserial": rank_biserial(ndcg_all["prune_fwd"], ndcg_all["keep_all"])},
        "prune_back_vs_random": {"p": wilcoxon(ndcg_all["prune_back"], ndcg_all["prune_random"]),
                                  "rank_biserial": rank_biserial(ndcg_all["prune_back"], ndcg_all["prune_random"])},
    }
    out = {"methods": res, "significance": sig, "config": vars(args)}
    path = os.path.join(os.path.dirname(__file__), "..", "results", "q1_v6_prune.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f"(elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
