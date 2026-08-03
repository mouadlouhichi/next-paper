#!/usr/bin/env python3
"""
run_q1_v7_lightgcn.py — CAVI on a LightGCN backbone (the Q1 empirical path).

The CPU-BPR backbone could not produce significant forward-CAV gains because
interaction weighting barely moved rankings. A LightGCN encoder produces highly
discriminative item embeddings, so which interactions are pruned/reweighted in a
profile genuinely matters. This is the backbone your thesis already uses
(DyHuCoG is a hypergraph variant).

Experiment (v7):
  - Train LightGCN item embeddings Q on the training user-item graph.
  - History-conditioned profile = weighted aggregate of Q[active interactions].
  - Forward-CAV over recent interactions (fit on a validation future window),
    backward-Shapley, and random.
  - Intervention = prune the lowest-value interactions.
  - Evaluate held-out future-window NDCG, with paired significance.

Key result to look for: does forward-CAV-informed pruning beat random pruning
significantly (which the BPR backbone could not do)?
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cavi.data import load_ratings
from cavi.games import Feasibility
from cavi.allocation import component_shapley
from cavi.lightgcn import train_lightgcn

D = 32
K = 20
N_LEVERS = 10
N_VAL = 2
N_FUT = 3
NEG_CAND = 30
MC_M = 40
DECAY_LAMBDA = 0.5
N_PRUNE = 2
DEVICE = "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "gate", "data"))
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--train-users", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[7])
    p.add_argument("--epochs", type=int, default=25)
    return p.parse_args()


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

        # --- train LightGCN on training user-item graph ---
        uid_map = {u: k for k, u in enumerate(train_users)}
        edges_u = []; edges_i = []
        for u in train_users:
            for i, _ in seqs[u]:
                edges_u.append(uid_map[u]); edges_i.append(i)
        edges_u = np.array(edges_u); edges_i = np.array(edges_i)
        n_train_users = len(train_users)
        print(f"[seed {seed}] training LightGCN on {n_train_users} users, {len(edges_u)} edges")
        Q = train_lightgcn(edges_u, edges_i, n_train_users, n_items, dim=D,
                           n_layers=2, epochs=args.epochs, lr=0.01, batch_size=4096,
                           seed=seed, device=DEVICE, verbose=True)
        print(f"[seed {seed}] LightGCN Q={Q.shape}")

        for u in eval_users:
            s = seqs[u]
            items = [i for i, _ in s]
            future = items[-N_FUT:]
            val = items[-(N_FUT + N_VAL): -N_FUT]
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
        "prune_fwd_vs_random": wilcoxon(ndcg_all["prune_fwd"], ndcg_all["prune_random"]),
        "prune_fwd_vs_keep": wilcoxon(ndcg_all["prune_fwd"], ndcg_all["keep_all"]),
        "prune_back_vs_random": wilcoxon(ndcg_all["prune_back"], ndcg_all["prune_random"]),
    }
    out = {"methods": res, "significance_p": sig, "config": vars(args)}
    path = os.path.join(os.path.dirname(__file__), "..", "results", "q1_v7_lightgcn.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f"(elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
