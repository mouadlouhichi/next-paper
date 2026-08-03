#!/usr/bin/env python3
"""
run_q1_v2.py — CAVI with a non-degenerate cooperative interaction-weighting
mechanism, benchmarked rigorously for a Q1 methods paper.

Why v2 (the honest fix):
  v1's forward/backward Shapley collapsed to exactly zero because the profile was
  a MEAN over history: removing one interaction shifted the profile by ~1/N, so
  every marginal contribution was ~0 and CAVI reduced to uniform weighting.
  v2 makes the mechanism genuinely discriminating:

  1. WEIGHTED-SUM profile. The user profile is
        p(S) = base_sum + sum_{i in S} q_i
     (unnormalized). Removing interaction i subtracts q_i -- a real, measurable
     change -- so each interaction's marginal contribution to the value is real.

  2. SMOOTH, MULTI-OBJECTIVE forward value. Instead of hard NDCG (a step
     function that is 0 for most coalitions), we use a smooth logistic reward on
     ranking future items, combined with a diversity term:
        v(S) = sum_f sigma((p(S) . q_f - mu_neg(S)) / T)  +  lambda * div(S)
     This is positive for every coalition and genuinely depends on which levers
     are active, so the cooperative game (and hence CAVI) is non-degenerate.

  3. COOPERATIVE ACTION VALUE = Shapley of the forward value over interactions.
     Interactions that push the profile toward the future items get high credit.

  4. CAVI REWEIGHTING: reweight interactions by softplus(shapley) so high-value
     interactions contribute more to the profile.

Evaluation is leakage-safe (weights fit on validation items, test item held out),
on fixed candidate sets, multiple seeds, paired Wilcoxon + Holm correction.
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

# ---------------------------------------------------------------- config ----
D = 32
K = 20
HORIZON = 2
GAMMA = 0.9
TEMP = 1.0
LAMBDA_DIV = 0.3
N_LEVERS = 6
MC_M = 32
NEG_CAND = 40   # negatives evaluated in the value function


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


def forward_value(Q, base_sum, levers, active_idx, future, negs, H, gamma=GAMMA,
                  temp=TEMP, lam=LAMBDA_DIV, seed=0):
    """
    Smooth multi-objective forward value of activating `active_idx` levers.
    p = base_sum + sum(active lever item factors).
    reward = sum over future items of logistic(margin vs negs) + diversity.
    Returns a positive scalar that genuinely varies with which levers are active.
    """
    rng = np.random.default_rng(seed)
    p = base_sum.copy()
    for i in active_idx:
        p = p + Q[levers[i]]
    # accuracy term: how well does p rank future items over negatives
    acc = 0.0
    for f in future:
        pos_s = float(p @ Q[f])
        neg_s = [float(p @ Q[ng]) for ng in negs]
        margin = pos_s - float(np.mean(neg_s))
        acc += sigmoid(margin, temp)
    acc /= max(len(future), 1)
    # diversity term: spread of top-k recommended (items whose score is highest)
    scores = {it: float(p @ Q[it]) for it in future + list(negs)}
    top = sorted(scores, key=lambda it: -scores[it])[:min(K, len(scores))]
    if len(top) >= 2:
        div = 0.0; cnt = 0
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                ca = 1.0 - float(np.dot(Q[top[a]], Q[top[b]]))  # cosine-ish distance
                div += ca; cnt += 1
        div = div / max(cnt, 1)
    else:
        div = 0.0
    return acc + lam * div


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
    allu = [u for u, s in seqs.items() if len(s) >= N_LEVERS + 5]
    print(f"[data] {len(allu)} users, {n_items} items")

    methods = ["random", "profile_base", "backshap", "cavi_fwd", "bpr_mf"]
    ndcg_all = {m: [] for m in methods}
    rec_all = {m: [] for m in methods}
    mrr_all = {m: [] for m in methods}

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        rng.shuffle(allu)
        train_users = allu[: min(len(allu) // 2, args.train_users)]
        eval_users = allu[len(allu) // 2: len(allu) // 2 + args.users]
        Q, P = bpr_item_factors(ratings, train_users, n_items, seed=seed)
        print(f"[seed {seed}] BPR {Q.shape}")

        for u in eval_users:
            s = seqs[u]
            test_item = s[-1][0]
            val_item = s[-2][0]
            future = [val_item]           # fit weights on validation item
            history = [i for i, _ in s[:-2]]
            if not history:
                continue
            levers = history[-N_LEVERS:]
            base = history[:-N_LEVERS] if len(history) > N_LEVERS else history

            # fixed candidate set (test + negatives)
            known = set([i for i, _ in s])
            cand = [test_item]
            rng_i = np.random.default_rng(seed + u)
            while len(cand) < args.candidates:
                it = int(rng_i.integers(0, n_items))
                if it in known or it in cand:
                    continue
                cand.append(it)
            # negatives for the value function (exclude known + test)
            negs = []
            while len(negs) < NEG_CAND:
                it = int(rng_i.integers(0, n_items))
                if it in known or it in cand or it in negs:
                    continue
                negs.append(it)

            base_sum = np.sum(Q[np.asarray(base)], axis=0)
            lever_arr = np.asarray(levers)
            full_sum = base_sum + np.sum(Q[lever_arr], axis=0)

            # ---- Shapley of forward value over levers (fit on validation) ----
            feas = Feasibility([list(range(len(levers)))])
            idx = list(range(len(levers)))

            def fwd(S):
                return forward_value(Q, base_sum, levers, S, future, negs, HORIZON,
                                     seed=seed)

            def back(S):
                # backward: immediate alignment of active profile with val item
                p = base_sum.copy()
                for i in S:
                    p = p + Q[levers[i]]
                return float(p @ Q[val_item])

            phi_fwd = component_shapley(fwd, feas, idx, M=MC_M, seed=seed)
            phi_back = component_shapley(back, feas, idx, M=MC_M, seed=seed)

            # ---- CAVI reweighting ----
            w_fwd = softplus(phi_fwd)
            w_back = softplus(phi_back)
            # profiles
            p_base = base_sum + np.sum(Q[lever_arr], axis=0)                 # weight 1 each
            p_cavi = base_sum + np.sum(Q[lever_arr] * w_fwd[:, None], axis=0)  # forward-weighted
            p_back = base_sum + np.sum(Q[lever_arr] * w_back[:, None], axis=0)  # back-weighted

            # BPR user factor
            ui = {x: k for k, x in enumerate(train_users)}
            pu = P[ui[u]] if u in ui else np.mean(Q[np.asarray(history)], axis=0)

            # ---- evaluate each method on the held-out TEST item ----
            relevant = {test_item}
            for name, p_vec in [("profile_base", p_base), ("backshap", p_back),
                                ("cavi_fwd", p_cavi), ("bpr_mf", pu)]:
                nd, rc, mr = evaluate(Q, p_vec, cand, relevant)
                ndcg_all[name].append(nd); rec_all[name].append(rc); mrr_all[name].append(mr)
            # random
            rsc = {it: rng_i.random() for it in cand}
            nd, rc, mr = _eval_random(rsc, cand, relevant)
            ndcg_all["random"].append(nd); rec_all["random"].append(rc); mrr_all["random"].append(mr)

    res = {}
    for m in methods:
        res[m] = {"ndcg_mean": float(np.mean(ndcg_all[m])),
                  "ndcg_std": float(np.std(ndcg_all[m])),
                  "recall_mean": float(np.mean(rec_all[m])),
                  "mrr_mean": float(np.mean(mrr_all[m])),
                  "n_users": len(ndcg_all[m])}
    sig = {}
    for m in ["cavi_fwd", "backshap", "bpr_mf"]:
        sig[m] = {"wilcoxon_p_vs_base": wilcoxon(ndcg_all[m], ndcg_all["profile_base"]),
                  "rank_biserial_vs_base": rank_biserial(ndcg_all[m], ndcg_all["profile_base"])}
    ps = [sig[m]["wilcoxon_p_vs_base"] for m in ["cavi_fwd", "backshap", "bpr_mf"]]
    order = sorted(range(3), key=lambda i: ps[i])
    for k, i in enumerate(order):
        m = ["cavi_fwd", "backshap", "bpr_mf"][i]
        sig[m]["holm_p"] = min(1.0, ps[i] * (3 - k))

    out = {"methods": res, "significance_vs_base": sig, "config": vars(args)}
    path = os.path.join(os.path.dirname(__file__), "..", "results", "q1_v2_experiment.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f"(elapsed {time.time()-t0:.0f}s)")


def _eval_random(rsc, cand, relevant):
    order = sorted(cand, key=lambda it: -rsc[it])
    ranks = [i + 1 for i, it in enumerate(order[:K]) if it in relevant]
    n_rel = min(len(relevant), K)
    nd = ndcg(ranks, n_rel)
    rc = len([r for r in ranks if r <= K]) / max(n_rel, 1)
    mrr = 0.0
    for i, it in enumerate(order[:K]):
        if it in relevant:
            mrr = 1.0 / (i + 1); break
    return nd, rc, mrr


if __name__ == "__main__":
    main()
