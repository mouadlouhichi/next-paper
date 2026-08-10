"""Advanced reviewer ablations over an existing exact CURE-Rec game table."""
from __future__ import annotations
from math import factorial
import numpy as np
from cure_rec.config import INTERVENTION_NAMES, Settings
from cure_rec.game import ALL_MASKS, GameResult
from cure_rec.planner import _constraints_for_mask


def _values(game: GameResult, objective: str) -> dict[int, float]:
    if objective == "maximin":
        return game.robust_improvements or {m:min(s.values[m].improvement for s in game.scenario_games.values()) for m in ALL_MASKS}
    if objective == "mean":
        return {m:float(np.mean([s.values[m].improvement for s in game.scenario_games.values()])) for m in ALL_MASKS}
    raise ValueError(objective)


def select_objective(game: GameResult, settings: Settings, *, objective: str="maximin", constraint_mode: str="hard", penalty: float=1.0) -> dict:
    """Compare maximin/mean and hard/penalty selection on identical game values."""
    values=_values(game, objective)
    candidates=[]
    for mask in ALL_MASKS:
        feasible, margins=_constraints_for_mask(game, mask, settings)
        score=values[mask]
        if constraint_mode == "penalty":
            violations=max(0., margins["cost"]-settings.constraints.budget)
            violations+=max(0., settings.constraints.min_relevance_delta-margins["relevance_delta_lower"])
            violations+=max(0., margins["provider_disparity_upper"]-settings.constraints.max_provider_disparity)
            violations+=max(0., margins["fatigue_upper"]-settings.constraints.max_fatigue)
            score -= penalty*violations
        elif constraint_mode == "hard" and not feasible:
            continue
        candidates.append((score,mask,feasible,margins))
    if not candidates:
        return {"mask":0,"objective":objective,"constraint_mode":constraint_mode,"score":float("nan"),"feasible":False}
    score,mask,feasible,margins=max(candidates,key=lambda x:(x[0],-x[1]))
    return {"mask":mask,"objective":objective,"constraint_mode":constraint_mode,"score":score,"feasible":feasible,**margins}


def sampled_shapley(values: dict[int,float], permutations: int, seed: int=42) -> dict[str,float]:
    """Permutation estimator for exact-vs-sampled Shapley fidelity experiments."""
    rng=np.random.default_rng(seed); out=np.zeros(len(INTERVENTION_NAMES))
    for _ in range(permutations):
        order=rng.permutation(len(INTERVENTION_NAMES)); mask=0
        for i in order:
            nxt=mask|(1<<int(i)); out[int(i)] += values[nxt]-values[mask]; mask=nxt
    return {name:float(out[i]/permutations) for i,name in enumerate(INTERVENTION_NAMES)}
