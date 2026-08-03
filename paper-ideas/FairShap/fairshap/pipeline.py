"""
pipeline.py — end-to-end FairShap pipeline.
"""
from __future__ import annotations
from typing import Dict, Sequence

import numpy as np

from . import game as G
from . import rerank as RR


def recommend(Q, profile_items, candidates, k=20):
    if not profile_items:
        return candidates[:k]
    p = Q[np.asarray(profile_items)].mean(axis=0)
    scores = Q[np.asarray(candidates)] @ p
    order = np.argsort(-scores, kind="stable")
    return [candidates[i] for i in order[:k]]


def recommend_scores(Q, profile_items, candidates):
    p = Q[np.asarray(profile_items)].mean(axis=0) if profile_items else np.zeros(Q.shape[1])
    s = Q[np.asarray(candidates)] @ p
    return {int(candidates[i]): float(s[i]) for i in range(len(candidates))}


class FairShapPipeline:
    def __init__(self, Q, exposure, item_sim, popularity, gamma=0.5,
                 m=200, seed=0):
        self.Q = Q
        self.exposure = exposure
        self.item_sim = item_sim
        self.popularity = popularity
        self.gamma = gamma
        self.m = m
        self.seed = seed
        self.phi_fair = {}

    def compute_exposure_attribution(self, candidate_items):
        self.phi_fair = G.exposure_shapley(self.exposure, self.item_sim,
                                           list(candidate_items), M=self.m,
                                           seed=self.seed)
        return self.phi_fair

    def rerank(self, user_profile, candidates, k=20):
        scores = recommend_scores(self.Q, user_profile, candidates)
        if not self.phi_fair:
            self.compute_exposure_attribution(candidates)
        ranked = RR.fair_rerank(scores, self.phi_fair, self.gamma)
        return ranked[:k]

    def recommend_plain(self, user_profile, candidates, k=20):
        return recommend(self.Q, user_profile, candidates, k)
