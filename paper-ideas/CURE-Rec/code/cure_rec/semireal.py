"""Semi-real integration: a learned ranker as the CURE-Sim base policy.

Reviewer concern (round 2): the causal game is evaluated over a transparent
hand-crafted base policy while the external MovieLens models are audited
separately, so the two evaluation paths never meet. This module bridges them in
a self-contained, disclosed way:

1. run the simulator under the default base policy with interaction logging on,
   producing an exposure/click log (user, item, response, timestamp);
2. fit a BPR-MF ranker on that logged feedback;
3. deploy the trained ranker as the base policy and re-run the exact
   intervention game on top of it via the ``policy_factory`` hook.

This is *semi-real*: the base policy is a learned recommender rather than a
hand-designed scorer, which is the deployment configuration the reviewer asks
about. It is still simulator-conditional evidence---the feedback loop is the
disclosed CURE-Sim mechanism---and does not create external causal claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cure_rec.config import Settings
from cure_rec.models import BPRMFRecommender
from cure_rec.policies import HistoryAwarePolicy
from cure_rec.simulator import CureSim


def collect_interaction_log(settings: Settings, scenario=None, *, seed: int | None = None) -> pd.DataFrame:
    """Roll the simulator under the default base policy and return a click log."""
    scenario = scenario if scenario is not None else settings.scenarios[0]
    seeded = settings.model_copy(deep=True)
    if seed is not None:
        seeded.run.seed = int(seed)
    simulator = CureSim(seeded, scenario, log_interactions=True)
    base = HistoryAwarePolicy(simulator, seeded.policy)

    def policy_fn(state, user_id, rng):
        return base.recommend(state, user_id, rng)

    simulator.rollout(policy_fn)
    frame = pd.DataFrame(simulator.interaction_log)
    if frame.empty:
        raise RuntimeError("Interaction log is empty; check simulator configuration")
    return frame[["user_id", "item_id", "response", "timestamp", "position"]]


@dataclass
class LearnedBPRPolicy:
    """Base policy that ranks the catalogue with a BPR-MF model trained on logs."""

    simulator: CureSim
    config: object
    model: BPRMFRecommender

    def rank_items(self, state, user_id: int) -> np.ndarray:
        candidates = self.simulator.available_items(user_id)
        scores = self.model.score(user_id, candidates)
        # Deterministic, machine-stable ordering; ties break by ascending item id.
        order = np.lexsort((candidates, -scores))
        return candidates[order]

    def recommend(self, state, user_id: int, rng):
        ranked = self.rank_items(state, user_id)
        slate = ranked[: self.simulator.settings.simulator.slate_size].tolist()
        return slate, {"base_policy": "learned_bpr", "ranked_pool_size": int(len(ranked)), "stats": {}}


def fit_logged_bpr(log: pd.DataFrame, *, factors: int = 32, max_updates: int = 60_000, seed: int = 42) -> BPRMFRecommender:
    """Fit BPR-MF on the simulator click log using only positive feedback."""
    positives = log[log["response"] > 0].copy()
    if positives.empty:
        raise RuntimeError("No positive feedback in the interaction log; cannot fit BPR")
    model = BPRMFRecommender(factors=factors, max_updates=max_updates, seed=seed)
    model.fit(positives)
    return model


def learned_policy_factory(model: BPRMFRecommender):
    """Return a policy factory compatible with ``run_exact_game(policy_factory=...)``."""

    def factory(simulator: CureSim, settings: Settings) -> LearnedBPRPolicy:
        return LearnedBPRPolicy(simulator=simulator, config=settings.policy, model=model)

    return factory
