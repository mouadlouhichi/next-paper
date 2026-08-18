"""Protocol-component ablations (review 3, issue 11).

Each ablation re-derives the method's joint action under a modified protocol
rule and reports the realized effect under the *unmodified* environment, so
the contribution of each component is measured on the same currency as the
headline results. Components ablated:

* ``abstention``      - forced action (no no-action option);
* ``signed``          - magnitude-only selection (|phi|) with downweighting;
* ``additive``        - additive selection vs. interaction-aware selection
                        using realized pair effects;
* ``antithetic``      - handled at attribution time (orchestrator flag);
* ``competitor_set``  - fixed Top-L (computed once at the full profile) vs.
                        dynamic per-coalition Top-L (orchestrator flag).
"""
from __future__ import annotations

import numpy as np


def select_forced(attribution: np.ndarray, budget: int) -> tuple[int, ...]:
    """No abstention: always act on the largest positive predicted benefits,
    falling back to the least harmful players when none are positive."""
    benefit = -np.asarray(attribution, dtype=float)
    order = np.argsort(-benefit, kind="stable")
    positive = [int(i) for i in order if benefit[i] > 0][:budget]
    if positive:
        return tuple(positive)
    return tuple(int(i) for i in order[:budget])


def select_magnitude(attribution: np.ndarray, budget: int) -> tuple[int, ...]:
    order = np.argsort(-np.abs(np.asarray(attribution, dtype=float)), kind="stable")
    return tuple(int(i) for i in order[:budget])


def select_interaction_aware(
    singleton_benefit: np.ndarray,
    pair_benefit: dict[tuple[int, int], float],
    budget: int = 2,
) -> tuple[int, ...]:
    """Best joint action using realized pair effects (interaction-aware)."""
    best: tuple[float, tuple[int, ...]] = (0.0, ())
    n = singleton_benefit.size
    for p in range(n):
        cand = (float(singleton_benefit[p]), (p,))
        if cand[0] > best[0]:
            best = cand
    if budget >= 2:
        for (p, q), val in pair_benefit.items():
            if val > best[0]:
                best = (float(val), (p, q))
    return best[1]


def ablation_report(
    default_action: tuple[int, ...],
    realized: dict[tuple[int, ...], float],
    variants: dict[str, tuple[int, ...]],
) -> dict[str, dict[str, float]]:
    base = realized.get(default_action, 0.0)
    out = {"default": {"realized_effect": float(base)}}
    for name, action in variants.items():
        eff = realized.get(action, 0.0)
        out[name] = {
            "realized_effect": float(eff),
            "delta_vs_default": float(eff - base),
            "action_changed": action != default_action,
        }
    return out
