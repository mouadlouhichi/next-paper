#!/usr/bin/env python3
"""Run the recommendation-only ActionShap experiment.

This script intentionally does not download data or silently insert test items
into candidate sets. Prepare MovieLens-1M yourself, then pass the path to
ratings.dat. It writes JSON results and does not require notebook execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Allow execution as `python scripts/run_recommendation.py` from code/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actionshap.baselines import lime_attribution, monte_carlo_attribution, permutation_importance
from actionshap.candidates import fixed_evaluation_sets
from actionshap.evaluation import (
    aia,
    exhaustive_best_joint,
    joint_effect,
    single_player_effects,
    within_user_aia_null,
)
from actionshap.models.profile import fit_item_embeddings
from actionshap.recommendation import UserGame, select_joint_action, target_margin_utility
from actionshap.recommendation_data import load_movielens_1m, truncate_histories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ratings", required=True, help="path to MovieLens ratings.dat")
    p.add_argument("--output", default="results/raw/movielens_actionshap.json")
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--candidate-k", type=int, default=200, help="fixed sampled evaluation-set size; target is always included")
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--permutations", type=int, default=250)
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--max-users", type=int, default=0, help="0 means all eligible users")
    p.add_argument("--oracle-users", type=int, default=100, help="users receiving exhaustive B=2 oracle")
    p.add_argument("--full-catalog-users", type=int, default=0, help="replace sampled evaluation sets for the first N users")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--action-rho", type=float, default=0.5, help="primary feasible downweighting strength; rho=0 is deletion only")
    p.add_argument("--null-draws", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_movielens_1m(args.ratings)
    histories = truncate_histories(data, args.n_max)
    model = fit_item_embeddings(
        histories,
        n_items=data.n_items,
        dimension=args.embedding_dim,
        epochs=args.epochs,
        seed=args.seed,
    )
    candidates, candidate_recall = fixed_evaluation_sets(
        histories, data.test, data.n_items, size=args.candidate_k, seed=args.seed
    )
    users = sorted(data.test)
    if args.max_users:
        users = users[:args.max_users]
    if args.full_catalog_users:
        for u in users[:args.full_catalog_users]:
            candidates[u] = np.arange(data.n_items, dtype=int)
    if not users:
        raise RuntimeError("no eligible test users were found")

    method_aia: dict[str, list[float]] = {}
    method_faithfulness: dict[str, list[float]] = {}
    method_gap: dict[str, list[float]] = {}
    method_null: dict[str, list[float]] = {}
    method_regret: dict[str, list[float]] = {}
    method_joint_effect: dict[str, list[float]] = {}
    oracle_effects: list[float] = []
    user_records = []

    for user_position, u in enumerate(users):
        game = UserGame(
            players=histories[u],
            candidate_items=candidates[u],
            target_item=data.test[u],
            tie_break=np.arange(candidates[u].size, dtype=int),
        )
        utility = lambda coalition, g=game: target_margin_utility(model, g, coalition, args.k)
        deletion_effects = single_player_effects(model, game, k=args.k, rho=0.0, utility="target_margin")
        action_effects = single_player_effects(model, game, k=args.k, rho=args.action_rho, utility="target_margin")
        shap, efficiency_error = monte_carlo_attribution(
            utility, game.players.size, args.permutations, args.seed + u
        )
        attrs = {
            "shapley_mc": shap,
            "loo_oracle": permutation_importance(utility, game.players.size),
            "lime": lime_attribution(utility, game.players.size, seed=args.seed + u),
        }
        per_user = {"user": int(u), "n_players": int(game.players.size), "aia": {}, "faithfulness_alignment": {}, "joint": {}}
        oracle = None
        if user_position < args.oracle_users and game.players.size >= 2:
            oracle = exhaustive_best_joint(model, game, budget=2, rho_grid=(args.action_rho,), k=args.k, utility="target_margin")
            oracle_effects.append(float(oracle[2]))

        for name, attribution in attrs.items():
            alignment = aia(attribution, action_effects)
            faithfulness_alignment = aia(attribution, deletion_effects)
            null = within_user_aia_null(attribution, action_effects, args.null_draws, args.seed + 100000 + u)
            method_aia.setdefault(name, []).append(alignment)
            method_faithfulness.setdefault(name, []).append(faithfulness_alignment)
            method_gap.setdefault(name, []).append(alignment - faithfulness_alignment)
            method_null.setdefault(name, []).append(float(np.mean(null)) if null.size else float("nan"))
            action = select_joint_action(attribution, budget=min(2, game.players.size))
            achieved = joint_effect(model, game, action, rho=args.action_rho, k=args.k, utility="target_margin")
            method_joint_effect.setdefault(name, []).append(float(achieved))
            if oracle is not None:
                method_regret.setdefault(name, []).append(float(oracle[2] - achieved))
            per_user["aia"][name] = None if not np.isfinite(alignment) else float(alignment)
            per_user["faithfulness_alignment"][name] = None if not np.isfinite(faithfulness_alignment) else float(faithfulness_alignment)
            per_user["actionability_gap"] = per_user.get("actionability_gap", {})
            per_user["actionability_gap"][name] = None if not np.isfinite(alignment - faithfulness_alignment) else float(alignment - faithfulness_alignment)
            per_user["joint"][name] = {
                "players": list(action),
                "effect": float(achieved),
                "regret": None if oracle is None else float(oracle[2] - achieved),
                "normalized_regret": None if oracle is None else float((oracle[2] - achieved) / (abs(oracle[2]) + 1e-12)),
            }
            per_user["oracle_effect"] = None if oracle is None else float(oracle[2])
        per_user["efficiency_error"] = float(efficiency_error)
        user_records.append(per_user)

    def summary(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        out = {}
        for name, xs in values.items():
            arr = np.asarray(xs, dtype=float)
            arr = arr[np.isfinite(arr)]
            out[name] = {
                "mean": None if not arr.size else float(arr.mean()),
                "median": None if not arr.size else float(np.median(arr)),
                "n": int(arr.size),
            }
        return out

    result = {
        "config": vars(args),
        "dataset": {
            "users_after_filter": len(data.test),
            "items": data.n_items,
            "evaluated_users": len(users),
            "candidate_recall": None,
            "target_coverage": candidate_recall,
            "evaluation_size": args.candidate_k,
            "primary_action_rho": args.action_rho,
            "evaluation_mode": "full_catalogue" if args.full_catalog_users >= len(users) and args.full_catalog_users else "fixed_sampled_negatives_with_target_included",
            "full_catalog_users": min(args.full_catalog_users, len(users)),
        },
        "metrics": {
            "aia": summary(method_aia),
            "faithfulness_alignment": summary(method_faithfulness),
            "actionability_gap": summary(method_gap),
            "aia_null_mean": summary(method_null),
            "joint_effect_b2": summary(method_joint_effect),
            "joint_regret_b2_on_oracle_subset": summary(method_regret),
            "oracle_b2_effect": {
                "mean": None if not oracle_effects else float(np.mean(oracle_effects)),
                "n": len(oracle_effects),
            },
        },
        "users": user_records,
        "note": "Run convergence and statistical aggregation before treating these values as paper results.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
