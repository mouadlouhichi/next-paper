#!/usr/bin/env python3
"""Run the predeclared Monte Carlo convergence study on fixed users."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actionshap.candidates import (
    fixed_evaluation_sets,
    global_item_priorities,
    tie_break_for_candidates,
)
from actionshap.convergence import (
    convergence_table,
    minimum_usable_permutations,
)
from actionshap.evaluation import model_mc_shapley
from actionshap.models import fit_item_embeddings, fit_item_knn
from actionshap.recommendation import (
    UserGame,
    profile_utility,
    target_margin_utility,
)
from actionshap.recommendation_data import (
    load_interactions_csv,
    load_movielens_1m,
    sample_evaluation_users,
    truncate_histories,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not budgets or budgets[0] < 1:
        raise argparse.ArgumentTypeError(
            "budgets must be comma-separated positive integers"
        )
    return budgets


def _runtime_metadata(started: float) -> dict[str, object]:
    memory_kb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_kb = int(line.split()[1])
                break
    except (OSError, ValueError):
        pass
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if not sys.platform.startswith("linux"):
        rss = int(rss / 1024)
    return {
        "platform": platform.platform(),
        "hardware": {
            "cpu_count": os.cpu_count(),
            "processor": platform.processor() or None,
            "machine": platform.machine(),
            "memory_total_kb": memory_kb,
        },
        "runtime_seconds": float(time.time() - started),
        "peak_rss_kb": rss,
    }


def main() -> None:
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--output", default="results/raw/convergence_seed42.json")
    parser.add_argument("--dataset-name", default="MovieLens-1M")
    parser.add_argument("--dataset-format", choices=("ml1m", "csv"), default="ml1m")
    parser.add_argument("--user-column", default="user")
    parser.add_argument("--item-column", default="item")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--rating-column", default="rating")
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--minimum-interactions", type=int, default=4)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--evaluation-size", type=int, default=200)
    parser.add_argument("--n-max", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--profile-samples-per-user", type=int, default=1)
    parser.add_argument("--profile-learning-rate", type=float, default=0.03)
    parser.add_argument("--profile-regularization", type=float, default=1e-4)
    parser.add_argument("--model", choices=("profile", "itemknn"), default="itemknn")
    parser.add_argument(
        "--utility", choices=("ndcg", "target_margin"), default="target_margin"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-seed", type=int, default=1729)
    parser.add_argument("--user-seed", type=int, default=2718)
    parser.add_argument("--tie-seed", type=int, default=31415)
    parser.add_argument(
        "--budgets", type=_parse_budgets, default=(25, 50, 100, 250, 500, 1000)
    )
    parser.add_argument("--reference", type=int, default=1000)
    args = parser.parse_args()

    data = (
        load_movielens_1m(
            args.ratings,
            rating_threshold=args.rating_threshold,
            minimum_interactions=args.minimum_interactions,
        )
        if args.dataset_format == "ml1m"
        else load_interactions_csv(
            args.ratings,
            user_column=args.user_column,
            item_column=args.item_column,
            timestamp_column=args.timestamp_column,
            rating_column=args.rating_column or None,
            rating_threshold=args.rating_threshold if args.rating_column else None,
            minimum_interactions=args.minimum_interactions,
        )
    )
    players = truncate_histories(data, args.n_max)
    model = (
        fit_item_embeddings(
            data.train,
            data.n_items,
            epochs=args.epochs,
            samples_per_user=args.profile_samples_per_user,
            learning_rate=args.profile_learning_rate,
            regularization=args.profile_regularization,
            seed=args.seed,
        )
        if args.model == "profile"
        else fit_item_knn(data.train, data.n_items)
    )
    users = sample_evaluation_users(data, args.users, args.user_seed, minimum_history=2)
    seen = {user: data.seen_before_test(user) for user in users}
    targets = {user: data.test[user] for user in users}
    evaluations, _ = fixed_evaluation_sets(
        seen,
        targets,
        data.n_items,
        args.evaluation_size,
        args.candidate_seed,
    )
    priorities = global_item_priorities(data.n_items, args.tie_seed)

    rows: list[dict[str, float | int | None]] = []
    for user in users:
        game = UserGame(
            players[user],
            evaluations[user],
            data.test[user],
            tie_break_for_candidates(evaluations[user], priorities),
        )
        utility = (
            (lambda coalition, g=game: profile_utility(model, g, coalition))
            if args.utility == "ndcg"
            else (lambda coalition, g=game: target_margin_utility(model, g, coalition))
        )
        user_rows = convergence_table(
            utility,
            game.players.size,
            budgets=args.budgets,
            seeds=(0, 1, 2, 3, 4),
            reference=args.reference,
            estimator=lambda permutations, seed, g=game: model_mc_shapley(
                model,
                g,
                permutations,
                seed,
                k=10,
                utility=args.utility,
            ),
        )
        for row in user_rows:
            row["user"] = int(user)
            rows.append(row)

    frame = pd.DataFrame(rows)
    metric_columns = [
        "mean_rank_correlation_to_reference",
        "std_rank_correlation_to_reference",
        "valid_rank_seeds",
        "mean_top1_agreement",
        "mean_top2_jaccard",
        "mean_top2_exact_agreement",
        "mean_efficiency_error",
    ]
    aggregate = frame.groupby(
        ["permutations", "reference_permutations"], as_index=False
    )[metric_columns].mean()
    aggregate["base_permutations"] = aggregate["permutations"]
    aggregate["evaluated_orders"] = 2 * aggregate["permutations"]
    aggregate["reference_evaluated_orders"] = 2 * aggregate["reference_permutations"]
    # The final budget must satisfy the thresholds at the aggregate level and
    # for at least 95% of users, rather than only on an easy average user.
    aggregate_rows = (
        aggregate.astype(object).where(pd.notna(aggregate), None).to_dict("records")
    )
    aggregate_minimum = minimum_usable_permutations(aggregate_rows)
    user_coverage: dict[str, float] = {}
    rank_valid_fraction: dict[str, float] = {}
    for budget in args.budgets:
        budget_frame = frame.loc[frame["permutations"] == budget]
        rank_valid = budget_frame.loc[
            budget_frame["mean_rank_correlation_to_reference"].notna()
        ]
        passing = rank_valid.loc[
            (rank_valid["mean_rank_correlation_to_reference"] >= 0.95)
            & (rank_valid["mean_top2_jaccard"] >= 0.80)
        ]
        user_coverage[str(budget)] = (
            float(len(passing) / len(rank_valid)) if len(rank_valid) else 0.0
        )
        rank_valid_fraction[str(budget)] = float(len(rank_valid) / len(users))
    selected = next(
        (
            budget
            for budget in args.budgets
            if aggregate_minimum is not None
            and budget >= aggregate_minimum
            and rank_valid_fraction[str(budget)] >= 0.95
        ),
        None,
    )
    ratings_path = Path(args.ratings).resolve()
    payload = {
        "schema_version": 2,
        "config": {**vars(args), "ratings": ratings_path.name},
        "attribution_sampling": {
            "budget_name": "base_permutations",
            "evaluated_orders": "2 * base_permutations (antithetic reverse walk)",
        },
        "provenance": {
            "input_file": ratings_path.name,
            "input_sha256": _file_sha256(ratings_path),
            **_runtime_metadata(started),
        },
        "rows": aggregate_rows,
        "per_user_rows": rows,
        "aggregate_minimum": aggregate_minimum,
        "user_threshold_coverage": user_coverage,
        "rank_valid_fraction": rank_valid_fraction,
        "selected_permutations": selected,
        "criterion": {
            "mean_rank_correlation": 0.95,
            "mean_top2_jaccard": 0.80,
            "minimum_rank_valid_fraction": 0.95,
            "per_user_threshold_coverage": "reported diagnostic, not a selection threshold",
        },
        "note": "Efficiency is numerical only and is not used to select M.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(aggregate.to_string(index=False))
    print("selected_permutations:", selected)


if __name__ == "__main__":
    main()
