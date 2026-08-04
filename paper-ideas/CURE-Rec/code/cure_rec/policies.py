"""Base recommendation policies used by the first CPU-first CURE-Rec milestone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cure_rec.config import PolicyConfig
from cure_rec.simulator import CureSim, PlatformState


@dataclass
class HistoryAwarePolicy:
    """A transparent base ranker using public profile similarity and popularity.

    It deliberately does not access hidden interests. Its score is documented and
    deterministic, making it a safe first policy for the CURE-Sim oracle tests.
    """

    simulator: CureSim
    config: PolicyConfig

    def rank_items(self, state: PlatformState, user_id: int) -> np.ndarray:
        candidates = self.simulator.available_items(user_id)
        profile_score = self.simulator.catalog.features[candidates] @ state.public_profiles[user_id]
        popularity_score = np.log1p(state.item_popularity[candidates])
        score = self.config.profile_weight * profile_score + self.config.popularity_weight * popularity_score
        # Stable ordering makes ties reproducible on every machine.
        return candidates[np.lexsort((candidates, -score))]

    def recommend(self, state: PlatformState, user_id: int, rng: np.random.Generator) -> tuple[list[int], dict]:
        ranked = self.rank_items(state, user_id)
        slate = ranked[: self.simulator.settings.simulator.slate_size].tolist()
        return slate, {"base_policy": "history_aware", "ranked_pool_size": int(len(ranked)), "stats": {}}
