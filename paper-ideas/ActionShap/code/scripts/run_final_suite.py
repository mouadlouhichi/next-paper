#!/usr/bin/env python3
"""Execute the complete predeclared ActionShap experiment matrix."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def dataset_arguments(dataset: dict, code_root: Path) -> list[str]:
    arguments = [
        "--ratings",
        str((code_root / dataset["path"]).resolve()),
        "--dataset-name",
        str(dataset["name"]),
        "--dataset-format",
        str(dataset["format"]),
        "--rating-threshold",
        str(dataset.get("rating_threshold", 4.0)),
    ]
    if dataset["format"] == "csv":
        for key, option in (
            ("user_column", "--user-column"),
            ("item_column", "--item-column"),
            ("timestamp_column", "--timestamp-column"),
            ("rating_column", "--rating-column"),
        ):
            arguments.extend(
                [option, str(dataset.get(key, key.removesuffix("_column")))]
            )
    return arguments


def recommendation_command_prefix(
    python: str,
    code_root: Path,
    dataset: dict,
    model: str,
    seed: int,
    config: dict,
    permutations: int,
    overrides: dict | None = None,
) -> list[str]:
    values = {**config, **(overrides or {})}
    return [
        python,
        str(code_root / "scripts" / "run_recommendation.py"),
        *dataset_arguments(dataset, code_root),
        "--model",
        model,
        "--model-role",
        "primary" if model == config["primary_model"] else "robustness",
        "--seed",
        str(seed),
        "--candidate-seed",
        str(config["candidate_seed"]),
        "--user-seed",
        str(config["user_seed"]),
        "--tie-seed",
        str(config["tie_seed"]),
        "--minimum-interactions",
        str(config["minimum_interactions"]),
        "--n-max",
        str(values["n_max"]),
        "--candidate-k",
        str(values["evaluation_size"]),
        "--k",
        str(config["k"]),
        "--utility",
        str(values.get("utility", config["utility"])),
        "--action-rho",
        str(values["action_rho"]),
        "--budget",
        str(values["budget"]),
        "--permutations",
        str(permutations),
        "--lime-samples",
        str(config["lime_samples"]),
        "--null-draws",
        str(values.get("null_draws", config["null_draws"])),
        "--gate-users",
        str(config["gate_users"]),
        "--gate-evaluation-size",
        str(config["evaluation_size"]),
        "--oracle-users",
        str(values.get("oracle_users", config["oracle_users"])),
        "--epochs",
        str(config["epochs"]),
        "--embedding-dim",
        str(config["embedding_dim"]),
        "--profile-samples-per-user",
        str(config["profile_samples_per_user"]),
        "--profile-learning-rate",
        str(config["profile_learning_rate"]),
        "--profile-regularization",
        str(config["profile_regularization"]),
        "--itemknn-neighbours",
        str(config["itemknn_neighbours"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    config_path = (code_root / args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    python = sys.executable
    raw_root = code_root / "results" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    missing = [
        str((code_root / dataset["path"]).resolve())
        for dataset in config["datasets"]
        if not (code_root / dataset["path"]).exists()
    ]
    if missing and not args.dry_run:
        raise FileNotFoundError(
            "Final suite requires every predeclared dataset; missing: "
            + ", ".join(missing)
        )

    selected_permutations: dict[tuple[str, str, str], int] = {}
    for dataset in config["datasets"]:
        dataset_slug = slug(dataset["name"])
        for model in config["models"]:
            convergence_path = raw_root / f"convergence_{dataset_slug}_{model}.json"
            command = [
                python,
                str(code_root / "scripts" / "run_convergence.py"),
                *dataset_arguments(dataset, code_root),
                "--output",
                str(convergence_path),
                "--model",
                model,
                "--utility",
                str(config["utility"]),
                "--users",
                str(config["convergence_users"]),
                "--evaluation-size",
                str(config["evaluation_size"]),
                "--n-max",
                str(config["n_max"]),
                "--minimum-interactions",
                str(config["minimum_interactions"]),
                "--epochs",
                str(config["epochs"]),
                "--profile-samples-per-user",
                str(config["profile_samples_per_user"]),
                "--profile-learning-rate",
                str(config["profile_learning_rate"]),
                "--profile-regularization",
                str(config["profile_regularization"]),
                "--candidate-seed",
                str(config["candidate_seed"]),
                "--user-seed",
                str(config["user_seed"]),
                "--tie-seed",
                str(config["tie_seed"]),
                "--budgets",
                ",".join(str(value) for value in config["convergence_budgets"]),
                "--reference",
                str(config["convergence_reference"]),
            ]
            run(command, code_root, args.dry_run)
            if args.dry_run:
                selected_permutations[(dataset["name"], model, config["utility"])] = (
                    int(config["permutations"])
                )
            else:
                study = json.loads(convergence_path.read_text())
                selected = study.get("selected_permutations")
                if selected is None:
                    raise RuntimeError(
                        f"Convergence thresholds not reached for {dataset['name']} / {model}"
                    )
                selected_permutations[(dataset["name"], model, config["utility"])] = (
                    max(int(config["permutations"]), int(selected))
                )

    # The alternate utility sensitivity has its own convergence study. Failure
    # is itself reportable, so the sensitivity falls back to M=reference rather
    # than silently borrowing the primary utility's selected budget.
    sensitivity_dataset = config["datasets"][0]
    sensitivity_model = str(config["primary_model"])
    alternate_utility = "ndcg" if config["utility"] != "ndcg" else "target_margin"
    sensitivity_slug = slug(sensitivity_dataset["name"])
    alternate_path = raw_root / (
        f"convergence_{sensitivity_slug}_{sensitivity_model}_{alternate_utility.replace('_', '-')}.json"
    )
    alternate_command = [
        python,
        str(code_root / "scripts" / "run_convergence.py"),
        *dataset_arguments(sensitivity_dataset, code_root),
        "--output",
        str(alternate_path),
        "--model",
        sensitivity_model,
        "--utility",
        alternate_utility,
        "--users",
        str(config["convergence_users"]),
        "--evaluation-size",
        str(config["evaluation_size"]),
        "--n-max",
        str(config["n_max"]),
        "--minimum-interactions",
        str(config["minimum_interactions"]),
        "--epochs",
        str(config["epochs"]),
        "--profile-samples-per-user",
        str(config["profile_samples_per_user"]),
        "--profile-learning-rate",
        str(config["profile_learning_rate"]),
        "--profile-regularization",
        str(config["profile_regularization"]),
        "--candidate-seed",
        str(config["candidate_seed"]),
        "--user-seed",
        str(config["user_seed"]),
        "--tie-seed",
        str(config["tie_seed"]),
        "--budgets",
        ",".join(str(value) for value in config["convergence_budgets"]),
        "--reference",
        str(config["convergence_reference"]),
    ]
    run(alternate_command, code_root, args.dry_run)
    if args.dry_run:
        selected_permutations[
            (sensitivity_dataset["name"], sensitivity_model, alternate_utility)
        ] = int(config["convergence_reference"])
    else:
        alternate_study = json.loads(alternate_path.read_text())
        alternate_selected = alternate_study.get("selected_permutations")
        selected_permutations[
            (sensitivity_dataset["name"], sensitivity_model, alternate_utility)
        ] = (
            int(config["convergence_reference"])
            if alternate_selected is None
            else max(int(config["permutations"]), int(alternate_selected))
        )

    for dataset in config["datasets"]:
        dataset_slug = slug(dataset["name"])
        for model in config["models"]:
            permutations = selected_permutations[
                (dataset["name"], model, config["utility"])
            ]
            for seed in config["seeds"]:
                common = recommendation_command_prefix(
                    python,
                    code_root,
                    dataset,
                    model,
                    seed,
                    config,
                    permutations,
                )
                primary_output = (
                    raw_root / f"{dataset_slug}_{model}_sampled_seed{seed}.json"
                )
                run(
                    [
                        *common,
                        "--output",
                        str(primary_output),
                        "--analysis-role",
                        "primary",
                        "--condition",
                        "primary",
                        "--max-users",
                        str(config["max_users"]),
                    ],
                    code_root,
                    args.dry_run,
                )
                robustness_common = recommendation_command_prefix(
                    python,
                    code_root,
                    dataset,
                    model,
                    seed,
                    config,
                    permutations,
                    {"null_draws": config["robustness_null_draws"]},
                )
                robustness_output = (
                    raw_root / f"{dataset_slug}_{model}_fullcatalog_seed{seed}.json"
                )
                run(
                    [
                        *robustness_common,
                        "--output",
                        str(robustness_output),
                        "--analysis-role",
                        "full_catalogue",
                        "--condition",
                        "full_catalogue",
                        "--max-users",
                        str(config["full_catalog_users"]),
                        "--user-pool-size",
                        str(config["max_users"]),
                        "--full-catalog",
                    ],
                    code_root,
                    args.dry_run,
                )

    # Predeclared one-factor-at-a-time robustness on the primary dataset/model.
    sensitivity_dataset = config["datasets"][0]
    sensitivity_model = str(config["primary_model"])
    sensitivity_slug = slug(sensitivity_dataset["name"])
    for condition in config.get("sensitivities", []):
        condition_name = str(condition["name"])
        overrides = {
            "null_draws": config["sensitivity_null_draws"],
            **{key: value for key, value in condition.items() if key != "name"},
        }
        condition_utility = str(overrides.get("utility", config["utility"]))
        sensitivity_permutations = max(
            selected_permutations[
                (sensitivity_dataset["name"], sensitivity_model, condition_utility)
            ],
            int(config["convergence_reference"]),
        )
        for seed in config["seeds"]:
            command = recommendation_command_prefix(
                python,
                code_root,
                sensitivity_dataset,
                sensitivity_model,
                seed,
                config,
                sensitivity_permutations,
                overrides,
            )
            output = (
                raw_root
                / f"{sensitivity_slug}_{sensitivity_model}_sensitivity-{condition_name}_seed{seed}.json"
            )
            run(
                [
                    *command,
                    "--output",
                    str(output),
                    "--analysis-role",
                    "sensitivity",
                    "--condition",
                    condition_name,
                    "--max-users",
                    str(config["sensitivity_users"]),
                    "--user-pool-size",
                    str(config["max_users"]),
                ],
                code_root,
                args.dry_run,
            )

    if not args.skip_assets:
        run(
            [python, str(code_root / "scripts" / "make_paper_assets.py")],
            code_root,
            args.dry_run,
        )
        run(
            [python, str(code_root / "scripts" / "validate_manuscript.py")],
            code_root,
            args.dry_run,
        )
        run(
            [python, str(code_root / "scripts" / "package_results.py")],
            code_root,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
