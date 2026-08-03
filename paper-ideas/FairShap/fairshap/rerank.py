"""
rerank.py — Shapley-guided fair re-ranking (FairShap Sec. 4.5).
"""
from __future__ import annotations
from typing import Dict, Sequence

import numpy as np


def fair_rerank(candidate_scores, fair_phi, gamma=0.5):
    m = max(fair_phi.values()) if fair_phi else 1.0
    blended = {}
    for it, y in candidate_scores.items():
        phi_n = fair_phi.get(it, 0.0) / m if m else 0.0
        blended[it] = (1 - gamma) * y + gamma * phi_n
    return sorted(blended, key=lambda it: -blended[it])


def deterministic_rerank(candidate_scores):
    cand = list(candidate_scores)
    return sorted(cand, key=lambda it: -candidate_scores[it])


def calibrated_rerank(candidate_scores, popularity, item_group, target_dist):
    selected = []
    remaining = dict(candidate_scores)
    counts = {}
    while remaining:
        best_it = None; best_score = -1e18
        for it in remaining:
            g = item_group.get(it, 0)
            cur = counts.get(g, 0)
            gap = target_dist.get(g, 0.0) - (cur + 1) / max(len(selected) + 1, 1)
            sc = remaining[it] + 1.0 * gap
            if sc > best_score:
                best_score = sc; best_it = it
        if best_it is None:
            break
        selected.append(best_it)
        counts[item_group.get(best_it, 0)] = counts.get(item_group.get(best_it, 0), 0) + 1
        del remaining[best_it]
    return selected
