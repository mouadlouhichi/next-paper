#!/usr/bin/env python3
"""
run_q1_v4.py — CAVI that LEARNS temporal importance from data.

NEW HYPOTHESIS (v4, replacing v1-v3):
    A forward-looking cooperative game, played over a user's recent
    interactions, learns a *temporal/recency weighting* from data: recent
    interactions receive high forward Cooperative Action Value (CAV) because
    they contribute most to FUTURE-window prediction. Applying these learned
    weights to the profile improves future-item recommendation significantly
    over uniform weighting and over backward-Shapley weighting.

Why this is different from v1-v3:
  - v1/v2/v3 amplified ONE interaction -> a weak action on a weak backbone.
  - v4 reweights the WHOLE recent profile by forward CAV, where the value
    function is explicitly FORWARD-looking (future window + time decay), so the
    forward game genuinely discriminates recent vs stale interactions.
  - We also TEST the mechanism: (a) does forward CAV correlate with recency?
    (b) does forward-weighted beat uniform / backward / a fixed time-decay?

Evaluation is leakage-safe (weights fit on a validation future window, tested on
a separate held-out future window), on fixed candidates, multiple seeds, paired
Wilcoxon + Holm + effect size.
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
N_LEVERS = 12          # recent interactions treated as levers (reweightable)
N_VAL = 3              # validation future window (fit weights)
N_FUT = 3              # held-out future window (evaluate)
NEG_CAND = 40
MC_M = 40              # Monte-Carlo Shapley permutations
DECAY_LAMBDA = 0.5     # time-decay for the value function (recent > old)
GAMMA_W = 0.8          # max extra weight for top interaction


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


def sigmoid(x, T=1.0):
    return 1.0 / (1.0 + np.exp(-np.clip(x / T, -30, 30)))


def future_value(Q, base_sum, levers, active_idx, future, negs, decay_lambda):
    """
    FORWARD value: how well the profile (base + active levers) ranks the
    FUTURE window items over negatives, with TIME DECAY on the future items
    (nearer future counts more). Positive and genuinely depends on which levers
    are active -> non-degenerate forward game.
    """
    p = base_sum.copy()
    for i in active_idx:
        p = p + Q[levers[i]]
    acc = 0.0
    for j, f in enumerate(future):
        tdec = np.exp(-decay_lambda * j)          # nearer future weighted more
        pos_s = float(p @ Q[f])
        neg_s = [float(p @ Q[ng]) for ng in negs]
        acc += tdec * sigmoid(pos_s - float(np.mean(neg_s)))
    return acc / max(len(future), 1)


def ndcg(ranks, n_rel):
    idcg = sum(1.0 / np.log2(j + 1) for j in range(1, n_rel + 1))
    if idcg == 0:
        return 0.0
    return sum(1.0 / np.log2(r + 1) for r in ranks) / idcg


def eval_ranking(Q, p, cand, future):
    """NDCG@K of profile p's scores over cand vs the future window (held-out)."""
    scores = np.array([float(p @ Q[it]) for it in cand])
    order = np.argsort(-scores, kind="stable")
    fut_set = set(future)
    rel_ranks = [i + 1 for i, cp in enumerate(order[:K]) if cand[cp] in fut_set]
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

    methods = ["uniform", "backward", "cavi_fwd", "timedecay", "random"]
    ndcg_all = {m: [] for m in methods}
    # mechanism test: correlation between forward-CAV and recency
    recency_corr = []

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
            future = items[-N_FUT:]                 # HELD-OUT eval window
            val = items[-(N_FUT + N_VAL): -N_FUT]   # validation window (fit)
            hist = items[:-(N_FUT + N_VAL)]
            if len(hist) < N_LEVERS:
                continue
            levers = hist[-N_LEVERS:]
            base = hist[:-N_LEVERS] if len(hist) > N_LEVERS else hist

            known = set(items)
            rng_i = np.random.default_rng(seed + u)
            # candidate set = future items + negatives (for ranking eval)
            cand = list(future)
            while len(cand) < args.candidates:
                it = int(rng_i.integers(0, n_items))
                if it in known or it in cand:
                    continue
                cand.append(it)
            negs = [it for it in cand if it not in set(future)][:NEG_CAND]

            base_sum = np.sum(Q[np.asarray(base)], axis=0)
            lever_arr = np.asarray(levers)
            feas = Feasibility([list(range(len(levers)))])
            idx = list(range(len(levers)))

            # ---- forward Shapley (fit on VALIDATION window, time-decayed) ----
            def fwd(S):
                return future_value(Q, base_sum, levers, S, val, negs, DECAY_LAMBDA)
            def back(S):
                active = [levers[i] for i in S]
                p = base_sum.copy()
                for i in active:
                    p = p + Q[i]
                return float(p @ Q[val[-1]])   # immediate next-item alignment

            phi_fwd = component_shapley(fwd, feas, idx, M=MC_M, seed=seed)
            phi_back = component_shapley(back, feas, idx, M=MC_M, seed=seed)

            # ---- profiles (full recent window reweighted) ----
            p_uniform = base_sum + np.sum(Q[lever_arr], axis=0)
            # CAV/backward weights: 1 + gamma*normalized_phi  (can downweight stale)
            def reweight(phi):
                ph = np.asarray(phi, float)
                rng_n = np.max(np.abs(ph)) + 1e-9
                return 1.0 + GAMMA_W * ph / rng_n
            w_fwd = reweight(phi_fwd)
            w_back = reweight(phi_back)
            p_cavi = base_sum + np.sum(Q[lever_arr] * w_fwd[:, None], axis=0)
            p_back = base_sum + np.sum(Q[lever_arr] * w_back[:, None], axis=0)
            # fixed time-decay baseline: weight_i = exp(-decay*(len-1-i))
            w_td = np.exp(-DECAY_LAMBDA * np.arange(len(levers))[::-1])
            p_td = base_sum + np.sum(Q[lever_arr] * w_td[:, None], axis=0)

            # mechanism: does forward-CAV order track recency (higher=more recent)?
            # lever 0 oldest ... lever N-1 most recent; higher phi for recent?
            recency_corr.append(float(np.corrcoef(np.arange(len(levers)), phi_fwd)[0, 1]))

            # ---- evaluate on HELD-OUT future window ----
            ndcg_all["uniform"].append(eval_ranking(Q, p_uniform, cand, future))
            ndcg_all["backward"].append(eval_ranking(Q, p_back, cand, future))
            ndcg_all["cavi_fwd"].append(eval_ranking(Q, p_cavi, cand, future))
            ndcg_all["timedecay"].append(eval_ranking(Q, p_td, cand, future))
            # random
            rsc = {it: rng_i.random() for it in cand}
            order = sorted(cand, key=lambda it: -rsc[it])
            rel_ranks = [i + 1 for i, it in enumerate(order[:K]) if it in set(future)]
            ndcg_all["random"].append(ndcg(rel_ranks, min(len(future), K)))

    res = {}
    for m in methods:
        res[m] = {"ndcg_mean": float(np.mean(ndcg_all[m])),
                  "ndcg_std": float(np.std(ndcg_all[m])),
                  "n_users": len(ndcg_all[m])}
    sig = {}
    for m in ["cavi_fwd", "backward", "timedecay"]:
        sig[m] = {"wilcoxon_p_vs_uniform": wilcoxon(ndcg_all[m], ndcg_all["uniform"]),
                  "rank_biserial_vs_uniform": rank_biserial(ndcg_all[m], ndcg_all["uniform"])}
    ps = [sig[m]["wilcoxon_p_vs_uniform"] for m in ["cavi_fwd", "backward", "timedecay"]]
    order = sorted(range(3), key=lambda i: ps[i])
    for k, i in enumerate(order):
        m = ["cavi_fwd", "backward", "timedecay"][i]
        sig[m]["holm_p"] = min(1.0, ps[i] * (3 - k))

    out = {"methods": res, "significance_vs_uniform": sig,
           "mechanism": {"mean_recency_corr": float(np.nanmean(recency_corr)),
                          "frac_positive_recency_corr": float(np.mean(np.array(recency_corr) > 0))},
           "config": vars(args)}
    path = os.path.join(os.path.dirname(__file__), "..", "results", "q1_v4_temporal.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f"(elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
