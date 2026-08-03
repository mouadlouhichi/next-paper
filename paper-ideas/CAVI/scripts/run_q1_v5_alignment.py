#!/usr/bin/env python3
"""
run_q1_v5_alignment.py — the paper's core claim, tested directly.

NEW HYPOTHESIS (v5):
    Forward Cooperative Action Values (CAV) predict the realized effect of an
    intervention on FUTURE utility better than backward Shapley values (or
    random). I.e., the forward game is a better *actionable-intelligence*
    signal: the interaction the forward game ranks highest is the one whose
    amplification most improves the user's future recommendations.

This is the "Attribution-Intervention Alignment" (AIA) idea from the proposal:
  - Per user, compute forward-CAV and backward-Shapley over recent interactions
    (fit on a VALIDATION future window).
  - For EACH lever, measure the REALIZED future-utility gain of amplifying it
    (on a separate HELD-OUT future window).
  - Test whether the forward-CAV ranking correlates with the realized-gain
    ranking better than backward-Shapley or random.

This directly measures the actionable-intelligence claim, and does not require
reweighting to beat baselines on raw accuracy (which the weak backbone cannot do).
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
N_LEVERS = 8
N_VAL = 2
N_FUT = 3
NEG_CAND = 30
MC_M = 40
AMPLIFY = 3.0
DECAY_LAMBDA = 0.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "gate", "data"))
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--train-users", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
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


def profile(Q, base_sum, levers, weights):
    return base_sum + np.sum(Q[np.asarray(levers)] * np.asarray(weights, float)[:, None], axis=0)


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


def spearman(a, b):
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


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

    # per-user alignment (Spearman between predicted attribution and realized
    # intervention gain) for each attribution method
    aia_fwd = []; aia_back = []; aia_random = []

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
            val = items[-(N_FUT + N_VAL): -N_FUT]   # validation (fit weights)
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
            feas = Feasibility([list(range(len(levers)))])
            idx = list(range(len(levers)))

            # ---- attribution (fit on VALIDATION window) ----
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

            # ---- realized intervention gain per lever (on HELD-OUT future) ----
            p0 = base_sum + np.sum(Q[np.asarray(levers)], axis=0)
            u0 = future_util(Q, p0, future, negs)
            gain = []
            for li in range(len(levers)):
                p_act = p0 + (AMPLIFY - 1.0) * Q[levers[li]]
                gain.append(future_util(Q, p_act, future, negs) - u0)
            gain = np.array(gain)

            # alignment: does predicted attribution rank match realized gain rank?
            aia_fwd.append(spearman(np.abs(phi_fwd), np.abs(gain)))
            aia_back.append(spearman(np.abs(phi_back), np.abs(gain)))
            aia_random.append(spearman(np.abs(rng_i.random(len(levers))), np.abs(gain)))

    def _clean(x):
        return np.array([v for v in x if not np.isnan(v)])

    af, ab, ar = _clean(aia_fwd), _clean(aia_back), _clean(aia_random)
    res = {
        "forward_CAV": {"mean_aia": float(np.mean(af)), "n_users": len(af)},
        "backward_Shapley": {"mean_aia": float(np.mean(ab)), "n_users": len(ab)},
        "random": {"mean_aia": float(np.mean(ar)), "n_users": len(ar)},
    }
    from scipy.stats import wilcoxon, spearmanr
    def wtest(x, y):
        d = np.asarray(x) - np.asarray(y)
        d = d[np.abs(d) > 1e-12]
        if len(d) == 0:
            return 1.0
        try:
            return float(wilcoxon(d).pvalue)
        except ValueError:
            return 1.0
    sig = {
        "fwd_vs_back": wtest(af, ab),
        "fwd_vs_random": wtest(af, ar),
        "back_vs_random": wtest(ab, ar),
    }
    out = {"methods": res, "significance": sig, "config": vars(args)}
    path = os.path.join(os.path.dirname(__file__), "..", "results", "q1_v5_alignment.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f"(elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
