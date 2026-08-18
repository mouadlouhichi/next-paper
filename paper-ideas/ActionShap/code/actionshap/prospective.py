"""Prospective (non-target-conditioned) audit (review 3, issue 1 / critical 1).

Instead of the held-out future target, the explained item is the recommender's
own top-1 recommendation under the full retained profile. The same bounded
intervention machinery then applies unchanged; the resulting AIA measures
whether attributions predict intervention effects on *generated*
recommendations, without any future label.
"""
from __future__ import annotations

import numpy as np

from .recommendation import UserGame


def prospective_target(model, game_like_players, catalogue_candidates: np.ndarray) -> int:
    """Top-1 item recommended from the current profile, excluding history items."""
    scores = model.score(game_like_players, catalogue_candidates)
    history = set(int(p) for p in np.asarray(game_like_players))
    order = np.argsort(-scores, kind="stable")
    for idx in order:
        if int(catalogue_candidates[idx]) not in history:
            return int(catalogue_candidates[idx])
    raise ValueError("no candidate outside history")


def build_prospective_game(
    model, players: np.ndarray, candidates: np.ndarray, tie_break: np.ndarray
) -> UserGame:
    target = prospective_target(model, players, candidates)
    return UserGame(
        players=np.asarray(players, dtype=int),
        candidate_items=np.asarray(candidates, dtype=int),
        target_item=target,
        tie_break=np.asarray(tie_break, dtype=np.int64),
    )
