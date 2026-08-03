"""
metrics.py — accuracy, diversity, and two-sided fairness metrics for FairShap.
"""
from __future__ import annotations
from typing import Dict, List, Sequence

import numpy as np


def ndcg_at_k(ranked, relevant, k=20):
    rel = set(relevant)
    ranks = [i + 1 for i, it in enumerate(ranked[:k]) if it in rel]
    n_rel = min(len(rel), k)
    idcg = sum(1.0 / np.log2(j + 1) for j in range(1, n_rel + 1))
    if idcg == 0:
        return 0.0
    return float(sum(1.0 / np.log2(r + 1) for r in ranks) / idcg)


def recall_at_k(ranked, relevant, k=20):
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for it in ranked[:k] if it in rel)
    return float(hits / min(len(rel), k))


def mrr(ranked, relevant, k=20):
    rel = set(relevant)
    for i, it in enumerate(ranked[:k]):
        if it in rel:
            return 1.0 / (i + 1)
    return 0.0


def intra_list_diversity(ranked, item_sim, k=20):
    top = ranked[:k]
    if len(top) < 2:
        return 0.0
    pairs = 0; total = 0.0
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            sim = item_sim.get(top[a], {}).get(top[b], 0.0)
            total += 1.0 - sim
            pairs += 1
    return float(total / pairs) if pairs else 0.0


def catalogue_coverage(all_lists, n_items):
    seen = set()
    for lst in all_lists:
        seen.update(lst)
    return float(len(seen) / n_items) if n_items else 0.0


def exposure_counts(ranked_lists, k=20):
    exp = {}
    for lst in ranked_lists:
        for i, it in enumerate(lst[:k]):
            exp[it] = exp.get(it, 0.0) + 1.0 / np.log2(i + 2)
    return exp


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / x.sum()) / n)


def exposure_gini(ranked_lists, k=20):
    exp = exposure_counts(ranked_lists, k)
    return gini(list(exp.values())) if exp else 0.0


def arp(ranked_lists, popularity, k=20):
    tot = 0.0; cnt = 0
    for lst in ranked_lists:
        for it in lst[:k]:
            tot += popularity.get(it, 0); cnt += 1
    return float(tot / cnt) if cnt else 0.0


def long_tail_exposure(ranked_lists, popularity, tail_frac=0.2, k=20):
    if not popularity:
        return 0.0
    thr = np.quantile(list(popularity.values()), tail_frac)
    exp = exposure_counts(ranked_lists, k)
    tot = sum(exp.values())
    if tot == 0:
        return 0.0
    tail = sum(v for it, v in exp.items() if popularity.get(it, 0) <= thr)
    return float(tail / tot)


def provider_disparity(ranked_lists, provider_of, k=20):
    exp = exposure_counts(ranked_lists, k)
    prov = {}
    for it, e in exp.items():
        p = provider_of.get(it, 0)
        prov[p] = prov.get(p, 0.0) + e
    if not prov:
        return 0.0
    vals = np.array(list(prov.values()))
    return float(vals.std() / (vals.mean() + 1e-9))


def consumer_ndcg_gap(user_lists, user_relevant, user_group, k=20):
    group_ndcg = {}
    for u, lst in user_lists.items():
        g = user_group.get(u, 0)
        group_ndcg.setdefault(g, []).append(ndcg_at_k(lst, user_relevant.get(u, []), k))
    means = [float(np.mean(v)) for v in group_ndcg.values() if v]
    if len(means) < 2:
        return 0.0
    return float(np.max(means) - np.min(means))


def compute_all(user_lists, user_relevant, item_sim, popularity, provider_of,
                user_group, n_items, k=20):
    lists = list(user_lists.values())
    rels = [user_relevant.get(u, []) for u in user_lists]
    nd = float(np.mean([ndcg_at_k(l, r, k) for l, r in zip(lists, rels)]))
    rc = float(np.mean([recall_at_k(l, r, k) for l, r in zip(lists, rels)]))
    ild = float(np.mean([intra_list_diversity(l, item_sim, k) for l in lists]))
    cov = catalogue_coverage(lists, n_items)
    gin = exposure_gini(lists, k)
    arpv = arp(lists, popularity, k)
    tail = long_tail_exposure(lists, popularity, k=k)
    disp = provider_disparity(lists, provider_of, k)
    gap = consumer_ndcg_gap(user_lists, user_relevant, user_group, k)
    return {
        "ndcg": nd, "recall": rc,
        "mrr": float(np.mean([mrr(l, r, k) for l, r in zip(lists, rels)])),
        "ild": ild, "coverage": cov,
        "gini": gin, "arp": arpv, "long_tail": tail, "provider_disparity": disp,
        "consumer_gap": gap,
    }
