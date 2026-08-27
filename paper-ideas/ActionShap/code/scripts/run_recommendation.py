#!/usr/bin/env python3
"""Run the corrected recommendation-only ActionShap experiment.

The runner enforces the protocol rather than silently repairing invalid inputs:
complete pre-test histories define unseen candidates, candidate and user seeds
are independent of the model seed, the real-data masking gate is blocking, and
benefit-seeking actions may abstain.  Raw outputs retain enough per-user detail
for hierarchical statistics and independent auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

# Allow execution as ``python scripts/run_recommendation.py`` from code/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actionshap.baselines import (
    greedy_counterfactual_attribution,
    lime_attribution,
    permutation_importance,
    random_attribution,
)
from actionshap.candidates import (
    fixed_evaluation_sets,
    full_unseen_evaluation_sets,
    global_item_priorities,
    tie_break_for_candidates,
)
from actionshap.evaluation import (
    aia,
    aia_null_summary,
    direction_accuracy,
    exhaustive_best_joint_multi,
    greedy_best_joint,
    joint_effect,
    masking_sensitivity_gate,
    model_mc_shapley,
    ranking_metrics_from_scores,
    recommendation_metrics,
    signed_alignment,
    single_player_effects,
    topk_intervention_precision,
)
from actionshap.models import fit_item_embeddings, fit_item_knn
from actionshap.recommendation import (
    UserGame,
    profile_utility,
    select_downweight_action,
    target_margin_utility,
)
from actionshap.recommendation_data import (
    load_interactions_csv,
    load_movielens_1m,
    sample_evaluation_users,
    truncate_histories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratings", required=True, help="MovieLens ratings.dat or timestamped CSV"
    )
    parser.add_argument("--dataset-format", choices=("ml1m", "csv"), default="ml1m")
    parser.add_argument("--dataset-name", default="MovieLens-1M")
    parser.add_argument("--user-column", default="user")
    parser.add_argument("--item-column", default="item")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--rating-column", default="rating")
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--minimum-interactions", type=int, default=4)
    parser.add_argument("--output", default="results/raw/movielens_actionshap.json")
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="run and save the real-data gate, then stop",
    )
    parser.add_argument(
        "--analysis-role",
        choices=("primary", "full_catalogue", "sensitivity"),
        default="primary",
    )
    parser.add_argument("--condition", default="primary")
    parser.add_argument("--model", choices=("profile", "itemknn"), default="itemknn")
    parser.add_argument(
        "--model-role", choices=("primary", "robustness"), default="primary"
    )
    parser.add_argument("--n-max", type=int, default=20)
    parser.add_argument(
        "--candidate-k",
        "--evaluation-size",
        dest="evaluation_size",
        type=int,
        default=200,
        help="target-plus-unseen-negatives evaluation-set size",
    )
    parser.add_argument(
        "--full-catalog", action="store_true", help="evaluate against every unseen item"
    )
    # Backward-compatible pilot option; non-zero means full catalogue for the whole selected run.
    parser.add_argument(
        "--full-catalog-users", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--profile-samples-per-user", type=int, default=1)
    parser.add_argument("--profile-learning-rate", type=float, default=0.03)
    parser.add_argument("--profile-regularization", type=float, default=1e-4)
    parser.add_argument("--itemknn-neighbours", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--lime-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="model/attribution seed")
    parser.add_argument("--candidate-seed", type=int, default=1729)
    parser.add_argument("--user-seed", type=int, default=2718)
    parser.add_argument("--tie-seed", type=int, default=31415)
    parser.add_argument(
        "--max-users", type=int, default=1000, help="0 means all eligible users"
    )
    parser.add_argument(
        "--user-pool-size",
        type=int,
        default=0,
        help="draw this larger seeded cohort before taking max-users",
    )
    parser.add_argument(
        "--oracle-users", type=int, default=0, help="0 means every evaluated user"
    )
    parser.add_argument("--gate-users", type=int, default=200)
    parser.add_argument("--gate-evaluation-size", type=int, default=200)
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="smoke tests only; output is marked non-paper",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--utility", choices=("ndcg", "target_margin"), default="target_margin"
    )
    parser.add_argument("--action-rho", type=float, default=0.5)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--null-draws", type=int, default=1000)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_builder_provenance(path: Path) -> dict[str, Any] | None:
    sidecar = path.with_suffix(".provenance.json")
    if not sidecar.exists():
        return None
    payload = json.loads(sidecar.read_text())
    if payload.get("output_sha256") != _file_sha256(path):
        raise ValueError(f"dataset provenance hash does not match {path}")
    return payload


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _hardware_metadata() -> dict[str, Any]:
    """Record lightweight host details without adding a dependency."""
    memory_kb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_kb = int(line.split()[1])
                break
    except (OSError, ValueError):
        pass
    return {
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or None,
        "machine": platform.machine(),
        "memory_total_kb": memory_kb,
    }


def _peak_rss_kb() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.  The experiment runner is
        # Linux in the release environment, but keep the conversion explicit.
        return value if sys.platform.startswith("linux") else int(value / 1024)
    except (OSError, ValueError):
        return None


def _git_commit(code_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=code_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _summary(
    values: dict[str, list[float]],
) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for method, method_values in values.items():
        array = np.asarray(method_values, dtype=float)
        valid = array[np.isfinite(array)]
        output[method] = {
            "mean": None if not valid.size else float(valid.mean()),
            "median": None if not valid.size else float(np.median(valid)),
            "n": int(valid.size),
            "missing": int(array.size - valid.size),
        }
    return output


def _load_data(args: argparse.Namespace):
    if args.dataset_format == "ml1m":
        return load_movielens_1m(
            args.ratings,
            rating_threshold=args.rating_threshold,
            minimum_interactions=args.minimum_interactions,
        )
    rating_column = args.rating_column or None
    return load_interactions_csv(
        args.ratings,
        user_column=args.user_column,
        item_column=args.item_column,
        timestamp_column=args.timestamp_column,
        rating_column=rating_column,
        rating_threshold=args.rating_threshold if rating_column else None,
        minimum_interactions=args.minimum_interactions,
    )


def _fit_model(args: argparse.Namespace, data):
    # Model fitting uses complete training histories.  Truncation is exclusively
    # an attribution-player decision and must not discard training evidence.
    if args.model == "profile":
        return fit_item_embeddings(
            data.train,
            n_items=data.n_items,
            dimension=args.embedding_dim,
            epochs=args.epochs,
            samples_per_user=args.profile_samples_per_user,
            learning_rate=args.profile_learning_rate,
            regularization=args.profile_regularization,
            seed=args.seed,
        )
    return fit_item_knn(
        data.train,
        n_items=data.n_items,
        neighbours=args.itemknn_neighbours,
    )


def _utility_function(name: str, model, game: UserGame, k: int):
    if name == "ndcg":
        return lambda coalition: profile_utility(model, game, coalition, k)
    return lambda coalition: target_margin_utility(model, game, coalition)


def main() -> None:
    args = parse_args()
    started = time.time()
    ratings_path = Path(args.ratings).resolve()
    if not ratings_path.exists():
        raise FileNotFoundError(ratings_path)
    if not 0.0 <= args.action_rho <= 1.0:
        raise ValueError("--action-rho must lie in [0, 1]")
    if args.budget < 1:
        raise ValueError("--budget must be positive")

    data = _load_data(args)
    player_histories = truncate_histories(data, args.n_max)
    complete_seen = {user: data.seen_before_test(user) for user in data.test}
    all_training_items = np.concatenate(
        [np.asarray(items, dtype=int) for items in data.train.values()]
    )
    item_popularity = np.bincount(all_training_items, minlength=data.n_items).astype(
        float
    )
    users = sample_evaluation_users(
        data,
        max_users=args.max_users,
        seed=args.user_seed,
        minimum_history=2,
        pool_size=args.user_pool_size,
    )
    if not users:
        raise RuntimeError("no eligible evaluation users were found")

    model = _fit_model(args, data)
    selected_seen = {user: complete_seen[user] for user in users}
    selected_targets = {user: data.test[user] for user in users}
    full_catalog = bool(args.full_catalog or args.full_catalog_users)
    if full_catalog:
        candidates, target_coverage = full_unseen_evaluation_sets(
            selected_seen, selected_targets, data.n_items
        )
        evaluation_mode = "full_unseen_catalogue"
    else:
        candidates, target_coverage = fixed_evaluation_sets(
            selected_seen,
            selected_targets,
            data.n_items,
            size=args.evaluation_size,
            seed=args.candidate_seed,
        )
        evaluation_mode = "sampled_unseen_negatives_with_target"

    priorities = global_item_priorities(data.n_items, args.tie_seed)
    games = {
        user: UserGame(
            players=player_histories[user],
            candidate_items=candidates[user],
            target_item=data.test[user],
            tie_break=tie_break_for_candidates(candidates[user], priorities),
        )
        for user in users
    }

    gate_rng = np.random.default_rng(args.user_seed + 99)
    gate_count = min(args.gate_users, len(users))
    gate_users = (
        np.sort(gate_rng.choice(users, size=gate_count, replace=False))
        .astype(int)
        .tolist()
    )
    # Gate sensitivity on one fixed sampled-ranking set independent of whether
    # the scientific condition uses 100, 500, or the full unseen catalogue.
    # Otherwise target sparsity in a full catalogue can make Delta-NDCG zero
    # even when history masking clearly changes the model output.
    gate_seen = {user: complete_seen[user] for user in gate_users}
    gate_targets = {user: data.test[user] for user in gate_users}
    gate_candidates, _ = fixed_evaluation_sets(
        gate_seen,
        gate_targets,
        data.n_items,
        size=args.gate_evaluation_size,
        seed=args.candidate_seed + 1,
    )
    gate_games = {
        user: UserGame(
            players=player_histories[user],
            candidate_items=gate_candidates[user],
            target_item=data.test[user],
            tie_break=tie_break_for_candidates(gate_candidates[user], priorities),
        )
        for user in gate_users
    }
    gate = masking_sensitivity_gate(
        model, gate_games, gate_users, k=args.k, seed=args.seed
    )
    gate["evaluation_mode"] = "fixed_sampled_gate"
    gate["evaluation_size"] = args.gate_evaluation_size
    gate["minimum_required_users"] = 200
    gate["sample_size_pass"] = bool(gate["users"] >= 200)
    gate["passed"] = bool(gate["passed"] and gate["sample_size_pass"])
    gate["blocking_required"] = (
        args.analysis_role != "sensitivity" and args.model_role == "primary"
    )
    if args.gate_only:
        gate_quality_rows: list[dict[str, float | int]] = []
        for user in users:
            game = games[user]
            model_quality = recommendation_metrics(model, game, args.k)
            popularity_quality = ranking_metrics_from_scores(
                item_popularity[game.candidate_items], game, args.k
            )
            gate_quality_rows.append(
                {
                    **{f"model_{key}": value for key, value in model_quality.items()},
                    **{
                        f"popularity_{key}": value
                        for key, value in popularity_quality.items()
                    },
                }
            )
        gate_quality_summary: dict[str, float | int] = {"n": len(gate_quality_rows)}
        for key in gate_quality_rows[0]:
            gate_quality_summary[key] = float(
                np.mean([float(row[key]) for row in gate_quality_rows])
            )
        safe_config = vars(args).copy()
        safe_config["ratings"] = ratings_path.name
        gate_payload = {
            "schema_version": 2,
            "status": "gate_only",
            "provenance": {
                "input_file": ratings_path.name,
                "input_sha256": _file_sha256(ratings_path),
                "dataset_builder": _dataset_builder_provenance(ratings_path),
                "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
                "python": sys.version,
                "platform": platform.platform(),
                "hardware": _hardware_metadata(),
                "dependencies": _dependency_versions(),
                "runtime_seconds": float(time.time() - started),
                "peak_rss_kb": _peak_rss_kb(),
            },
            "config": safe_config,
            "dataset": {
                "name": args.dataset_name,
                "users_after_filter": len(data.test),
                "items": data.n_items,
                "gate_users": gate_users,
            },
            "masking_gate": gate,
            "recommendation_quality": gate_quality_summary,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(gate_payload, indent=2, allow_nan=False))
        print(json.dumps(gate_payload, indent=2, allow_nan=False))
        if not gate["passed"] and gate["blocking_required"] and not args.skip_gate:
            raise RuntimeError("blocking real-data masking gate failed")
        return
    if not gate["passed"] and gate["blocking_required"] and not args.skip_gate:
        raise RuntimeError(
            "blocking real-data masking gate failed; rerun with --skip-gate only for a labelled smoke test: "
            + json.dumps(gate, sort_keys=True)
        )

    metric_store: dict[str, dict[str, list[float]]] = {
        name: {}
        for name in (
            "aia",
            "aia_ndcg",
            "faithfulness_alignment",
            "actionability_gap",
            "signed_alignment",
            "signed_alignment_ndcg",
            "direction_accuracy",
            "direction_accuracy_ndcg",
            "top1_precision",
            "top3_precision",
            "top5_precision",
            "aia_null_mean",
            "aia_null_p95",
            "aia_permutation_p",
            "joint_effect_primary",
            "joint_effect_ndcg",
            "joint_effect_target_margin",
            "joint_regret_primary",
            "normalized_regret_primary",
            "joint_regret_ndcg",
            "normalized_regret_ndcg",
            "intervention_success",
            "intervention_success_ndcg",
            "abstention",
        )
    }
    quality_rows: list[dict[str, float | int]] = []
    user_records: list[dict[str, Any]] = []
    oracle_limit = (
        len(users) if args.oracle_users <= 0 else min(args.oracle_users, len(users))
    )

    for user_position, user in enumerate(users):
        game = games[user]
        budget = min(args.budget, game.players.size)
        raw_utility = _utility_function(args.utility, model, game, args.k)
        utility_cache: dict[frozenset[int], float] = {}

        def utility(
            coalition: frozenset[int],
            _cache: dict[frozenset[int], float] = utility_cache,
            _raw=raw_utility,
        ) -> float:
            if coalition not in _cache:
                _cache[coalition] = float(_raw(coalition))
            return _cache[coalition]

        deletion_effects = single_player_effects(
            model, game, k=args.k, rho=0.0, utility=args.utility
        )
        action_effects = single_player_effects(
            model, game, k=args.k, rho=args.action_rho, utility=args.utility
        )
        if args.utility == "ndcg":
            deletion_effects_ndcg = deletion_effects
            action_effects_ndcg = action_effects
        else:
            deletion_effects_ndcg = single_player_effects(
                model, game, k=args.k, rho=0.0, utility="ndcg"
            )
            action_effects_ndcg = single_player_effects(
                model, game, k=args.k, rho=args.action_rho, utility="ndcg"
            )
        shapley, efficiency_error = model_mc_shapley(
            model,
            game,
            args.permutations,
            (int(args.seed), int(user), 2),
            k=args.k,
            utility=args.utility,
        )
        methods = {
            "shapley_mc": shapley,
            "loo": permutation_importance(utility, game.players.size),
            "lime": lime_attribution(
                utility,
                game.players.size,
                samples=max(args.lime_samples, game.players.size + 2),
                seed=(int(args.seed), int(user), 1),
            ),
            "greedy_cf": greedy_counterfactual_attribution(utility, game.players.size),
            "random": random_attribution(
                game.players.size, (int(args.seed), int(user), 1_000_000)
            ),
        }

        primary_oracle = None
        ndcg_oracle = None
        if user_position < oracle_limit:
            if budget <= 2:
                oracle_outputs = exhaustive_best_joint_multi(
                    model,
                    game,
                    budget,
                    rho_grid=(args.action_rho,),
                    k=args.k,
                    utilities=tuple(dict.fromkeys((args.utility, "ndcg"))),
                    allow_abstain=True,
                )
                primary_oracle = oracle_outputs[args.utility]
                ndcg_oracle = oracle_outputs["ndcg"]
            else:
                primary_oracle = greedy_best_joint(
                    model,
                    game,
                    budget,
                    rho_grid=(args.action_rho,),
                    k=args.k,
                    utility=args.utility,
                    allow_abstain=True,
                )
                ndcg_oracle = (
                    primary_oracle
                    if args.utility == "ndcg"
                    else greedy_best_joint(
                        model,
                        game,
                        budget,
                        rho_grid=(args.action_rho,),
                        k=args.k,
                        utility="ndcg",
                        allow_abstain=True,
                    )
                )

        model_quality = recommendation_metrics(model, game, args.k)
        popularity_quality = ranking_metrics_from_scores(
            item_popularity[game.candidate_items], game, args.k
        )
        quality = {
            **{f"model_{key}": value for key, value in model_quality.items()},
            **{f"popularity_{key}": value for key, value in popularity_quality.items()},
        }
        quality_rows.append(quality)
        per_user: dict[str, Any] = {
            "user": int(user),
            "source_user_id": int(data.user_ids[user])
            if np.issubdtype(data.user_ids.dtype, np.integer)
            else str(data.user_ids[user]),
            "n_players": int(game.players.size),
            "evaluation_size": int(game.candidate_items.size),
            "recommendation_quality": quality,
            "effects": {
                "deletion_primary": deletion_effects.tolist(),
                "feasible_primary": action_effects.tolist(),
                "deletion_ndcg": deletion_effects_ndcg.tolist(),
                "feasible_ndcg": action_effects_ndcg.tolist(),
            },
            "methods": {},
            "oracle": None,
            "efficiency_error": float(efficiency_error),
        }
        if primary_oracle is not None and ndcg_oracle is not None:
            oracle_type = "exact" if budget <= 2 else "greedy_lower_bound"
            per_user["oracle"] = {
                "type": oracle_type,
                "primary_utility": {
                    "name": args.utility,
                    "players": list(primary_oracle[0]),
                    "item_ids": (
                        game.players[list(primary_oracle[0])].astype(int).tolist()
                        if primary_oracle[0]
                        else []
                    ),
                    "rhos": list(primary_oracle[1]),
                    "effect": float(primary_oracle[2]),
                },
                "ndcg": {
                    "players": list(ndcg_oracle[0]),
                    "item_ids": (
                        game.players[list(ndcg_oracle[0])].astype(int).tolist()
                        if ndcg_oracle[0]
                        else []
                    ),
                    "rhos": list(ndcg_oracle[1]),
                    "effect": float(ndcg_oracle[2]),
                },
            }

        for method, attribution in methods.items():
            alignment = aia(attribution, action_effects)
            alignment_ndcg = aia(attribution, action_effects_ndcg)
            faithfulness = aia(attribution, deletion_effects)
            if method == "loo" and np.isfinite(faithfulness):
                # LOO is the exact deletion diagnostic, not an estimated
                # correlation.  Preserve the theorem in serialized results.
                faithfulness = 1.0
            signed = signed_alignment(attribution, action_effects)
            signed_ndcg = signed_alignment(attribution, action_effects_ndcg)
            directional = direction_accuracy(attribution, action_effects)
            directional_ndcg = direction_accuracy(attribution, action_effects_ndcg)
            null = aia_null_summary(
                attribution,
                action_effects,
                draws=args.null_draws,
                seed=(int(args.seed), int(user), 100_000),
            )
            action = select_downweight_action(
                attribution,
                budget,
                allow_abstain=True,
            )
            effect_primary = joint_effect(
                model, game, action, args.action_rho, args.k, args.utility
            )
            effect_ndcg = joint_effect(
                model, game, action, args.action_rho, args.k, "ndcg"
            )
            effect_margin = joint_effect(
                model, game, action, args.action_rho, args.k, "target_margin"
            )
            raw_regret = (
                None
                if primary_oracle is None
                else float(primary_oracle[2] - effect_primary)
            )
            raw_regret_ndcg = (
                None
                if ndcg_oracle is None
                else float(ndcg_oracle[2] - effect_ndcg)
            )
            # B<=2 uses an exhaustive oracle over exactly the same action space.
            # A negative regret beyond floating-point noise is therefore a
            # protocol bug, not a result to hide by clipping.
            if budget <= 2 and (
                (raw_regret is not None and raw_regret < -1e-12)
                or (raw_regret_ndcg is not None and raw_regret_ndcg < -1e-12)
            ):
                raise AssertionError(
                    f"exact oracle returned negative regret for user {user}, method {method}"
                )
            regret = None if raw_regret is None else max(0.0, raw_regret)
            normalized_regret = (
                None
                if regret is None or primary_oracle is None or primary_oracle[2] <= 0
                else float(regret / primary_oracle[2])
            )
            regret_ndcg = (
                None if raw_regret_ndcg is None else max(0.0, raw_regret_ndcg)
            )
            normalized_regret_ndcg = (
                None
                if regret_ndcg is None or ndcg_oracle is None or ndcg_oracle[2] <= 0
                else float(regret_ndcg / ndcg_oracle[2])
            )
            method_values = {
                "aia": alignment,
                "aia_ndcg": alignment_ndcg,
                "faithfulness_alignment": faithfulness,
                "actionability_gap": alignment - faithfulness,
                "signed_alignment": signed,
                "signed_alignment_ndcg": signed_ndcg,
                "direction_accuracy": directional,
                "direction_accuracy_ndcg": directional_ndcg,
                "top1_precision": topk_intervention_precision(
                    attribution, action_effects, 1
                ),
                "top3_precision": topk_intervention_precision(
                    attribution, action_effects, 3
                ),
                "top5_precision": topk_intervention_precision(
                    attribution, action_effects, 5
                ),
                "aia_null_mean": null["null_mean"],
                "aia_null_p95": null["null_p95"],
                "aia_permutation_p": null["p_value"],
                "joint_effect_primary": effect_primary,
                "joint_effect_ndcg": effect_ndcg,
                "joint_effect_target_margin": effect_margin,
                "joint_regret_primary": regret,
                "normalized_regret_primary": normalized_regret,
                "joint_regret_ndcg": regret_ndcg,
                "normalized_regret_ndcg": normalized_regret_ndcg,
                "intervention_success": float(effect_primary > 0),
                "intervention_success_ndcg": float(effect_ndcg > 0),
                "abstention": float(len(action) == 0),
            }
            for metric_name, value in method_values.items():
                if value is not None:
                    metric_store[metric_name].setdefault(method, []).append(
                        float(value)
                    )
            per_user["methods"][method] = {
                "attribution": attribution.tolist(),
                "aia": _finite(alignment),
                "aia_ndcg": _finite(alignment_ndcg),
                "faithfulness_alignment": _finite(faithfulness),
                "actionability_gap": _finite(alignment - faithfulness),
                "signed_alignment": _finite(signed),
                "signed_alignment_ndcg": _finite(signed_ndcg),
                "direction_accuracy": _finite(directional),
                "direction_accuracy_ndcg": _finite(directional_ndcg),
                "topk_precision": {
                    "1": method_values["top1_precision"],
                    "3": method_values["top3_precision"],
                    "5": method_values["top5_precision"],
                },
                "aia_null": null,
                "action": {
                    "players": list(action),
                    "item_ids": game.players[list(action)].astype(int).tolist()
                    if action
                    else [],
                    "rho": args.action_rho,
                    "abstained": not action,
                },
                "effect_primary": float(effect_primary),
                "effect_ndcg": float(effect_ndcg),
                "effect_target_margin": float(effect_margin),
                "success": bool(effect_primary > 0),
                "success_ndcg": bool(effect_ndcg > 0),
                "regret_primary": regret,
                "normalized_regret_primary": normalized_regret,
                "regret_ndcg": regret_ndcg,
                "normalized_regret_ndcg": normalized_regret_ndcg,
            }
        user_records.append(per_user)

    quality_summary: dict[str, float | int] = {"n": len(quality_rows)}
    for key in quality_rows[0]:
        quality_summary[key] = float(np.mean([float(row[key]) for row in quality_rows]))

    safe_config = vars(args).copy()
    safe_config["ratings"] = ratings_path.name
    result = {
        "schema_version": 2,
        "status": "smoke_only" if args.skip_gate else "paper_eligible",
        "provenance": {
            "input_file": ratings_path.name,
            "input_sha256": _file_sha256(ratings_path),
            "dataset_builder": _dataset_builder_provenance(ratings_path),
            "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
            "python": sys.version,
            "platform": platform.platform(),
            "hardware": _hardware_metadata(),
            "dependencies": _dependency_versions(),
            "runtime_seconds": float(time.time() - started),
            "peak_rss_kb": _peak_rss_kb(),
        },
        "config": safe_config,
        "attribution_sampling": {
            "base_permutations": int(args.permutations),
            "evaluated_orders": int(2 * args.permutations),
            "antithetic_reverse": True,
        },
        "dataset": {
            "name": args.dataset_name,
            "users_after_filter": len(data.test),
            "items": data.n_items,
            "evaluated_users": len(users),
            "selected_user_ids": users,
            "target_coverage": target_coverage,
            "candidate_recall": None,
            "evaluation_mode": evaluation_mode,
            "evaluation_size_min": min(
                game.candidate_items.size for game in games.values()
            ),
            "evaluation_size_max": max(
                game.candidate_items.size for game in games.values()
            ),
            "primary_action_rho": args.action_rho,
            "primary_utility": args.utility,
            "users_with_repeated_target": int(
                sum(
                    data.test[user] in set(complete_seen[user].tolist())
                    for user in users
                )
            ),
        },
        "masking_gate": gate,
        "recommendation_quality": quality_summary,
        "metrics": {name: _summary(values) for name, values in metric_store.items()},
        "users": user_records,
        "notes": [
            "The statistical unit is the distinct user; aggregate repeated seeds hierarchically.",
            "This is a retrospective target-conditioned audit, not a prospective deployment policy.",
            "No explainer inspects measured intervention effects or oracle actions during attribution.",
            "LOO is an oracle only for B=1 deletion, and is labelled LOO elsewhere.",
            "The feasible action space includes no action and every action size up to the budget.",
            "NDCG and target-margin effects are stored separately and must not be relabelled.",
            "Prefix-walk efficiency is a numerical identity, not a convergence certificate.",
            "permutations is the number of base orders; antithetic reversal evaluates twice as many orders.",
            "Exact B<=2 normalized regret is asserted non-negative before serialization."
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {"status": result["status"], "gate": gate, "metrics": result["metrics"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
