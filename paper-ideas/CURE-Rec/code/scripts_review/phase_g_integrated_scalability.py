"""Integrated scalability experiment (reviewer Phase 9, item 5).

The archived 6/8/10-player benchmark was arithmetic only: the additional
operators were not integrated into the policy semantics. This driver integrates
them. The four extended operators (session_length_cap, freshness_quota,
provider_cooldown, category_coverage_quota) are implemented as real slate
transformations in ``cure_rec.interventions``; here we enumerate the exact game
over 6, 8, or 10 players, with full rollouts, and report:

- wall-clock time and peak memory of one exact game;
- exact Shapley values and the efficiency gap;
- exact-vs-sampled Shapley fidelity (permutation budgets 32..2048): MAE, sign
  agreement, rank correlation;
- the maximin feasible selection and its regret against the best frozen mask.

Claim scope: disclosed CURE-Sim mechanisms; simulator-conditional scalability
evidence, not external causal evidence.

Usage: python scripts_review/phase_g_integrated_scalability.py {6|8|10} [seed]
"""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import UTC, datetime
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cure_rec.config import EXTRA_PLAYER_NAMES, INTERVENTION_NAMES, load_settings  # noqa: E402
from cure_rec.interventions import Coalition, transform_slate  # noqa: E402
from cure_rec.policies import HistoryAwarePolicy  # noqa: E402
from cure_rec.simulator import CureSim  # noqa: E402

ASSETS = ROOT / "results" / "reviewer_phase_assets" / "integrated_scalability"
CONFIG = ROOT / "configs" / "curesim_full.yaml"
EXTRA_COSTS = {
    "session_length_cap": 0.07,
    "freshness_quota": 0.09,
    "provider_cooldown": 0.08,
    "category_coverage_quota": 0.06,
}
SAMPLE_BUDGETS = (32, 128, 512, 2048)


def player_library(n: int) -> tuple[str, ...]:
    if n == 6:
        return tuple(INTERVENTION_NAMES)
    if n in (8, 10):
        return tuple(INTERVENTION_NAMES) + tuple(EXTRA_PLAYER_NAMES[: n - 6])
    raise ValueError("player library size must be 6, 8, or 10")


def exact_shapley_n(improvements: dict[int, float], names: tuple[str, ...]) -> dict[str, float]:
    n = len(names)
    values: dict[str, float] = {}
    for player_index, player_name in enumerate(names):
        contribution = 0.0
        bit = 1 << player_index
        others = [i for i in range(n) if i != player_index]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                mask = sum(1 << i for i in subset)
                contribution += weight * (improvements[mask | bit] - improvements[mask])
        values[player_name] = float(contribution)
    return values


def sampled_shapley_n(improvements: dict[int, float], names: tuple[str, ...], budget: int, seed: int) -> dict[str, float]:
    n = len(names)
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for _ in range(budget):
        order = rng.permutation(n)
        mask = 0
        for i in order:
            nxt = mask | (1 << int(i))
            out[int(i)] += improvements[nxt] - improvements[mask]
            mask = nxt
    return {name: float(out[i] / budget) for i, name in enumerate(names)}


def rollout_value(settings, scenario, names: tuple[str, ...], mask: int, baseline_utility: float, baseline_relevance: float):
    simulator = CureSim(settings, scenario)
    base_policy = HistoryAwarePolicy(simulator, settings.policy)
    coalition = Coalition.from_mask(mask, names)

    def policy_fn(state, user_id, rng):
        result = transform_slate(base_policy, state, user_id, coalition, settings.interventions, rng)
        return result.slate, result.manifest

    summary = simulator.rollout(policy_fn)
    cost = coalition.cost(settings.interventions)
    utility = summary.utility_before_cost - settings.utility.cost_weight * cost
    return {
        "mask": mask,
        "scenario": scenario.name,
        "utility": float(utility),
        "improvement": float(utility - baseline_utility),
        "relevance": float(summary.relevance),
        "relevance_delta": float(summary.relevance - baseline_relevance),
        "provider_disparity": float(summary.provider_disparity),
        "fatigue": float(summary.fatigue),
        "cost": float(cost),
        "intervention_stats": summary.intervention_stats,
    }


def run_integrated_game(settings, names: tuple[str, ...], seed: int) -> dict:
    started = time.time()
    masks = list(range(1 << len(names)))
    rows = []
    for scenario in settings.scenarios:
        base = rollout_value(settings, scenario, names, 0, 0.0, 0.0)
        base_relevance = base["relevance"]
        base_utility = base["utility"]
        base["improvement"] = 0.0
        base["relevance_delta"] = 0.0
        rows.append(base)
        for mask in masks[1:]:
            rows.append(rollout_value(settings, scenario, names, mask, base_utility, base_relevance))
    frame = pd.DataFrame(rows)
    robust = frame.groupby("mask")["improvement"].min().to_dict()
    duration = time.time() - started
    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {"frame": frame, "robust": robust, "duration_seconds": duration, "peak_memory_mb": peak_memory_mb, "seed": seed}


def select_maximin(game: dict, settings, names: tuple[str, ...]) -> dict:
    frame = game["frame"]
    margins = (
        frame.groupby("mask")
        .agg(
            cost=("cost", "first"),
            relevance_delta_lower=("relevance_delta", "min"),
            provider_disparity_upper=("provider_disparity", "max"),
            fatigue_upper=("fatigue", "max"),
        )
    )
    c = settings.constraints

    def feasible(mask: int) -> bool:
        m = margins.loc[mask]
        return bool(
            m["cost"] <= c.budget + 1e-12
            and m["relevance_delta_lower"] >= c.min_relevance_delta
            and m["provider_disparity_upper"] <= c.max_provider_disparity
            and m["fatigue_upper"] <= c.max_fatigue
        )

    robust = game["robust"]
    feasible_masks = [mask for mask in robust if feasible(mask)]
    base_feasible = feasible(0)
    if base_feasible:
        best = max(feasible_masks, key=lambda mask: (robust[mask], -mask))
        status = "improve_selected" if robust[best] > 0 else "abstain_keep_base"
        selected = best if robust[best] > 0 else 0
    else:
        repairs = [mask for mask in feasible_masks if mask != 0]
        if not repairs:
            status, selected = "no_feasible_portfolio", 0
        else:
            status = "repair_selected"
            selected = max(repairs, key=lambda mask: (robust[mask], -mask))
    best_overall = max(robust, key=lambda mask: robust[mask])
    best_feasible_value = max((robust[mask] for mask in feasible_masks), default=float("nan"))
    return {
        "mode": "improvement" if base_feasible else "repair",
        "status": status,
        "selected_mask": int(selected),
        "selected_portfolio": ";".join(n for i, n in enumerate(names) if selected & (1 << i)) or "abstain",
        "selected_robust_value": float(robust[selected]),
        "oracle_best_value": float(robust[best_overall]),
        "selection_regret_vs_oracle": float(robust[best_overall] - robust[selected]),
        "selection_regret_vs_best_feasible": float(best_feasible_value - robust[selected]),
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    names = player_library(n)
    out = ASSETS / f"players_{n}"
    out.mkdir(parents=True, exist_ok=True)

    settings = load_settings(CONFIG)
    settings.run.seed = seed
    settings.interventions.extended_players = n > 6
    for name, cost in EXTRA_COSTS.items():
        settings.interventions.costs[name] = cost

    print(f"INTEGRATED GAME n={n} seed={seed} masks={1 << n}", flush=True)
    game = run_integrated_game(settings, names, seed)
    game["frame"].to_csv(out / "coalition_values.csv", index=False)
    print(f"game done in {game['duration_seconds']:.0f}s, peak memory {game['peak_memory_mb']:.0f} MB", flush=True)

    shapley = exact_shapley_n(game["robust"], names)
    gap = sum(shapley.values()) - game["robust"][(1 << n) - 1]
    print(f"shapley efficiency gap: {gap:.2e}", flush=True)

    fidelity_rows = []
    for budget in SAMPLE_BUDGETS:
        estimate = sampled_shapley_n(game["robust"], names, budget, seed=seed)
        a = np.array([estimate[name] for name in names])
        b = np.array([shapley[name] for name in names])
        rank_a = pd.Series(a).rank().to_numpy()
        rank_b = pd.Series(b).rank().to_numpy()
        fidelity_rows.append({
            "players": n,
            "budget": budget,
            "mae": float(np.mean(np.abs(a - b))),
            "max_error": float(np.max(np.abs(a - b))),
            "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
            "rank_correlation": float(np.corrcoef(rank_a, rank_b)[0, 1]),
        })
    fidelity = pd.DataFrame(fidelity_rows)
    fidelity.to_csv(out / "sampled_fidelity.csv", index=False)

    selection = select_maximin(game, settings, names)
    summary = pd.DataFrame([{
        "players": n,
        "coalitions": 1 << n,
        "seed": seed,
        "wall_clock_seconds": game["duration_seconds"],
        "peak_memory_mb": game["peak_memory_mb"],
        "shapley_efficiency_gap": gap,
        **selection,
    }])
    summary.to_csv(out / "integrated_summary.csv", index=False)
    pd.DataFrame([{"intervention": name, "robust_phi": shapley[name]} for name in names]).to_csv(out / "robust_attribution.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "players": list(names),
        "seed": seed,
        "config_hash": settings.config_hash(),
        "extra_costs": EXTRA_COSTS,
        "sample_budgets": list(SAMPLE_BUDGETS),
        "claim_scope": "integrated CURE-Sim scalability; simulator-conditional, not external causal evidence",
    }
    (out / "revision_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("INTEGRATED SCALABILITY DONE", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(fidelity.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
