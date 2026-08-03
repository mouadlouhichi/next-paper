#!/usr/bin/env python3
"""Validate schema-v2 ActionShap runs and build Q1-grade paper assets.

Unlike the pilot notebook, this script never treats seed-user records as
independent users.  It first averages repeated seeds within each distinct user,
then bootstraps users and performs paired user-level sign permutation tests.
All source paths in the manifest are repository-relative and content-addressed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actionshap.stats import (
    holm_bonferroni,
    paired_user_seed_comparison,
)

METHOD_LABELS = {
    "shapley_mc": "Monte Carlo Shapley",
    "lime": "LIME",
    "loo": "Leave-one-out",
    "greedy_cf": "Greedy counterfactual",
    "random": "Random control",
}
METHOD_ORDER = ["shapley_mc", "lime", "loo", "greedy_cf", "random"]
PRIMARY_METRICS = [
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
    "joint_effect_primary",
    "joint_effect_ndcg",
    "joint_effect_target_margin",
    "intervention_success",
    "abstention",
    "joint_regret_primary",
    "normalized_regret_primary",
    "joint_regret_ndcg",
    "normalized_regret_ndcg",
    "intervention_success_ndcg",
]
SEED_RE = re.compile(r"seed(?P<seed>\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("could not locate repository root")


def load_results(raw_root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    files = sorted(raw_root.glob("*.json"))
    results: list[dict[str, Any]] = []
    selected_files: list[Path] = []
    for path in files:
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != 2 or "users" not in payload:
            continue
        match = SEED_RE.search(path.name)
        payload["_seed"] = (
            int(match.group("seed")) if match else int(payload["config"]["seed"])
        )
        payload["_path"] = path
        results.append(payload)
        selected_files.append(path)
    if not results:
        raise FileNotFoundError(
            f"No schema-v2 recommendation result JSON found in {raw_root}. "
            "Legacy pilot files are deliberately rejected."
        )
    return selected_files, results


def convergence_frame(
    studies: dict[tuple[str, str, str], dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model, utility), study in sorted(studies.items()):
        selected = study.get("selected_permutations")
        coverage = study.get("user_threshold_coverage", {})
        valid_fraction = study.get("rank_valid_fraction", {})
        for row in study.get("rows", []):
            permutations = int(row["permutations"])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "utility": utility,
                    **row,
                    "user_threshold_coverage": coverage.get(str(permutations)),
                    "rank_valid_fraction": valid_fraction.get(str(permutations)),
                    "selected": bool(
                        (selected is not None and permutations == int(selected))
                        or (
                            selected is None
                            and permutations == int(study["config"].get("reference", 0))
                        )
                    ),
                    "selection_status": (
                        "threshold_selected"
                        if selected is not None
                        else "max_budget_unconverged"
                    ),
                }
            )
    return pd.DataFrame(rows)


def experiment_key(result: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(result["dataset"]["name"]),
        str(result["config"]["model"]),
        str(result["dataset"]["evaluation_mode"]),
        str(result["dataset"]["primary_utility"]),
        str(result["config"].get("analysis_role", "primary")),
        str(result["config"].get("condition", "primary")),
    )


def load_convergence(raw_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    studies: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(raw_root.glob("convergence*.json")):
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != 2 or "selected_permutations" not in payload:
            continue
        config = payload["config"]
        key = (
            str(config.get("dataset_name", "MovieLens-1M")),
            str(config["model"]),
            str(config["utility"]),
        )
        payload["_path"] = path
        studies[key] = payload
    return studies


def validate(
    results: list[dict[str, Any]],
    convergence: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(experiment_key(result), []).append(result)
        if result.get("status") != "paper_eligible":
            errors.append(f"{result['_path'].name}: status is not paper_eligible")
        analysis_role = str(result["config"].get("analysis_role", "primary"))
        gate = result.get("masking_gate", {})
        if not gate.get("passed"):
            message = f"{result['_path'].name}: masking gate did not pass"
            if (
                analysis_role == "sensitivity"
                or result["config"].get("model_role") == "robustness"
            ):
                notes.append(message)
            else:
                errors.append(message)
        if (
            gate.get("evaluation_mode") != "fixed_sampled_gate"
            or int(gate.get("evaluation_size", 0)) != 200
        ):
            errors.append(
                f"{result['_path'].name}: masking gate did not use the fixed 200-item gate set"
            )
        if result["dataset"].get("target_coverage") != 1.0:
            errors.append(f"{result['_path'].name}: target coverage is not one")
        if result["config"].get("dataset_format") == "csv":
            builder = result.get("provenance", {}).get("dataset_builder")
            if not builder or not builder.get("source_sha256"):
                errors.append(
                    f"{result['_path'].name}: secondary CSV lacks raw-source provenance"
                )
        cutoff = int(result["config"].get("k", 10))
        quality = result.get("recommendation_quality", {})
        model_ndcg = quality.get(f"model_ndcg@{cutoff}")
        popularity_ndcg = quality.get(f"popularity_ndcg@{cutoff}")
        model_recall = quality.get(f"model_recall@{cutoff}")
        popularity_recall = quality.get(f"popularity_recall@{cutoff}")
        if None in {model_ndcg, popularity_ndcg, model_recall, popularity_recall}:
            errors.append(
                f"{result['_path'].name}: recommendation quality audit is incomplete"
            )
        elif model_ndcg < popularity_ndcg or model_recall < popularity_recall:
            message = (
                f"{result['_path'].name}: attributed model underperforms popularity "
                f"(NDCG {model_ndcg:.4g} vs {popularity_ndcg:.4g}; "
                f"Recall {model_recall:.4g} vs {popularity_recall:.4g})"
            )
            notes.append(message)
        repeated_targets = int(result["dataset"].get("users_with_repeated_target", 0))
        if repeated_targets:
            notes.append(
                f"{result['_path'].name}: {repeated_targets} users repeat the temporal target; disclose repeat-consumption semantics"
            )
        required_methods = {"shapley_mc", "lime", "loo", "greedy_cf", "random"}
        for user in result.get("users", []):
            observed_methods = set(user.get("methods", {}))
            if observed_methods != required_methods:
                errors.append(
                    f"{result['_path'].name}: user {user.get('user')} methods "
                    f"{sorted(observed_methods)} != {sorted(required_methods)}"
                )
                break
            n_players = int(user.get("n_players", 0))
            effects = user.get("effects", {})
            if any(
                len(effects.get(name, [])) != n_players
                for name in (
                    "deletion_primary",
                    "feasible_primary",
                    "deletion_ndcg",
                    "feasible_ndcg",
                )
            ):
                errors.append(
                    f"{result['_path'].name}: user {user.get('user')} effect vectors do not match players"
                )
                break
        if analysis_role in {"primary", "full_catalogue"}:
            missing_oracles = [
                user.get("user")
                for user in result.get("users", [])
                if user.get("oracle") is None
            ]
            if missing_oracles:
                errors.append(
                    f"{result['_path'].name}: {len(missing_oracles)} users lack the required oracle"
                )
            if int(result["config"].get("budget", 2)) <= 2 and any(
                (user.get("oracle") or {}).get("type") != "exact"
                for user in result.get("users", [])
            ):
                errors.append(
                    f"{result['_path'].name}: B<=2 oracle is not exact for every user"
                )
            if any(
                "primary_utility" not in (user.get("oracle") or {})
                or "ndcg" not in (user.get("oracle") or {})
                for user in result.get("users", [])
            ):
                errors.append(
                    f"{result['_path'].name}: utility-specific oracle outcomes are incomplete"
                )
    for key, group in groups.items():
        if key[4] in {"primary", "full_catalogue"} and key[3] != "target_margin":
            errors.append(f"{key}: final attribution utility must be target_margin")
        seeds = sorted({int(result["_seed"]) for result in group})
        if len(seeds) < 5:
            errors.append(f"{key}: fewer than five seeds ({seeds})")
        user_sets = [tuple(result["dataset"]["selected_user_ids"]) for result in group]
        if any(users != user_sets[0] for users in user_sets[1:]):
            errors.append(f"{key}: selected users differ across seeds")
        for field in (
            "candidate_seed",
            "user_seed",
            "tie_seed",
            "n_max",
            "action_rho",
            "budget",
            "epochs",
            "embedding_dim",
            "profile_samples_per_user",
            "profile_learning_rate",
            "profile_regularization",
            "itemknn_neighbours",
            "model_role",
        ):
            values = {result["config"].get(field) for result in group}
            if len(values) != 1:
                errors.append(
                    f"{key}: config {field} differs across seeds: {sorted(values)}"
                )
        input_hashes = {result["provenance"].get("input_sha256") for result in group}
        if len(input_hashes) != 1:
            errors.append(f"{key}: source dataset hashes differ across seeds")
        cutoff = int(group[0]["config"].get("k", 10))
        required_quality_keys = {
            f"model_ndcg@{cutoff}",
            f"popularity_ndcg@{cutoff}",
            f"model_recall@{cutoff}",
            f"popularity_recall@{cutoff}",
        }
        if (
            key[4] == "primary"
            and all(result["config"].get("model_role") == "primary" for result in group)
            and all(
                required_quality_keys <= set(result.get("recommendation_quality", {}))
                for result in group
            )
        ):
            model_ndcg = np.mean(
                [
                    result["recommendation_quality"][f"model_ndcg@{cutoff}"]
                    for result in group
                ]
            )
            popularity_ndcg = np.mean(
                [
                    result["recommendation_quality"][f"popularity_ndcg@{cutoff}"]
                    for result in group
                ]
            )
            model_recall = np.mean(
                [
                    result["recommendation_quality"][f"model_recall@{cutoff}"]
                    for result in group
                ]
            )
            popularity_recall = np.mean(
                [
                    result["recommendation_quality"][f"popularity_recall@{cutoff}"]
                    for result in group
                ]
            )
            if model_ndcg < popularity_ndcg or model_recall < popularity_recall:
                errors.append(
                    f"{key}: declared primary model underperforms popularity across seeds "
                    f"(NDCG {model_ndcg:.4g} vs {popularity_ndcg:.4g}; "
                    f"Recall {model_recall:.4g} vs {popularity_recall:.4g})"
                )
        n_users = len(user_sets[0])
        total_eligible = int(group[0]["dataset"]["users_after_filter"])
        required_users = (
            100
            if key[4] == "full_catalogue"
            else 250
            if key[4] == "sensitivity"
            else 1000
        )
        if n_users < required_users and n_users < total_eligible:
            errors.append(
                f"{key}: only {n_users} of {total_eligible} eligible users; this analysis role requires at least {required_users} or all eligible users"
            )
        convergence_key = (key[0], key[1], key[3])
        study = convergence.get(convergence_key)
        if study is None:
            errors.append(f"{key}: no independent schema-v2 convergence study")
            continue
        if study.get("provenance", {}).get("input_sha256") not in input_hashes:
            errors.append(f"{key}: convergence used a different dataset payload")
        reference = int(study["config"].get("reference", 0))
        if reference < 1000:
            errors.append(f"{key}: convergence reference is below 1000 permutations")
        expected_budgets = {25, 50, 100, 250, 500, 1000}
        observed_budgets = {
            int(row.get("permutations", -1)) for row in study.get("rows", [])
        }
        if observed_budgets != expected_budgets:
            errors.append(
                f"{key}: convergence budgets {sorted(observed_budgets)} "
                f"!= {sorted(expected_budgets)}"
            )
        criterion = study.get("criterion", {})
        if (
            criterion.get("mean_rank_correlation") != 0.95
            or criterion.get("mean_top2_jaccard") != 0.80
            or criterion.get("minimum_rank_valid_fraction") != 0.95
        ):
            errors.append(
                f"{key}: convergence criterion differs from the frozen contract"
            )
        required_convergence_fields = {
            "mean_rank_correlation_to_reference",
            "mean_top1_agreement",
            "mean_top2_jaccard",
            "mean_top2_exact_agreement",
            "mean_efficiency_error",
        }
        if any(
            not required_convergence_fields <= set(row) for row in study.get("rows", [])
        ):
            errors.append(f"{key}: convergence rows lack rank/action diagnostics")

        selected_raw = study.get("selected_permutations")
        utility_failure_is_expected = (
            key[4] == "sensitivity" and key[5] == "utility-ndcg"
        )
        if selected_raw is None:
            if utility_failure_is_expected:
                notes.append(
                    f"{key}: NDCG attribution did not satisfy convergence thresholds; "
                    f"the sensitivity uses the maximum M={reference} and reports failure"
                )
                selected = reference
            else:
                errors.append(f"{key}: convergence thresholds were not reached")
                continue
        else:
            selected = int(selected_raw)

        used = {int(result["config"]["permutations"]) for result in group}
        if any(value < selected for value in used):
            errors.append(f"{key}: M={sorted(used)} is below required M={selected}")
        selected_coverage = float(
            study.get("user_threshold_coverage", {}).get(str(selected), 0.0)
        )
        notes.append(
            f"{key}: M={selected} reaches both per-user thresholds for "
            f"{selected_coverage:.1%} of rank-valid users"
        )
        valid_fraction = float(
            study.get("rank_valid_fraction", {}).get(str(selected), 0.0)
        )
        if valid_fraction < 0.95 and not utility_failure_is_expected:
            errors.append(
                f"{key}: rank convergence is defined for only {valid_fraction:.1%} "
                f"of users at M={selected}"
            )
    claim_groups = {
        key: value
        for key, value in groups.items()
        if key[4] in {"primary", "full_catalogue"}
    }
    for key, full_group in groups.items():
        if key[4] != "full_catalogue":
            continue
        primary_groups = [
            group
            for primary_key, group in groups.items()
            if primary_key[0] == key[0]
            and primary_key[1] == key[1]
            and primary_key[3] == key[3]
            and primary_key[4] == "primary"
            and primary_key[5] == "primary"
        ]
        if len(primary_groups) != 1:
            errors.append(f"{key}: cannot identify one matched primary cohort")
            continue
        full_users = set(full_group[0]["dataset"]["selected_user_ids"])
        primary_users = set(primary_groups[0][0]["dataset"]["selected_user_ids"])
        if not full_users <= primary_users:
            errors.append(
                f"{key}: full-catalogue users are not a primary-cohort subset"
            )
    for key, sensitivity_group in groups.items():
        if key[4] != "sensitivity":
            continue
        primary_groups = [
            group
            for primary_key, group in groups.items()
            if primary_key[0] == key[0]
            and primary_key[1] == key[1]
            and primary_key[4] == "primary"
            and primary_key[5] == "primary"
        ]
        if len(primary_groups) != 1:
            errors.append(f"{key}: cannot identify one matched primary cohort")
            continue
        sensitivity_users = set(sensitivity_group[0]["dataset"]["selected_user_ids"])
        primary_users = set(primary_groups[0][0]["dataset"]["selected_user_ids"])
        if not sensitivity_users <= primary_users:
            errors.append(f"{key}: sensitivity users are not a primary-cohort subset")
    datasets = {key[0] for key in claim_groups}
    models = {key[1] for key in claim_groups}
    if len(datasets) < 2:
        errors.append(
            "Only one primary dataset is present; a sparse secondary dataset is required for final claims"
        )
    if len(models) < 2:
        errors.append(
            "Only one primary history-conditioned model is present; architecture robustness is required"
        )
    for dataset in datasets:
        primary_model_groups = [
            group
            for key, group in groups.items()
            if key[0] == dataset
            and key[4] == "primary"
            and all(result["config"].get("model_role") == "primary" for result in group)
        ]
        if len(primary_model_groups) != 1:
            errors.append(
                f"{dataset}: expected exactly one declared primary model, found {len(primary_model_groups)}"
            )
    required_sensitivities = {
        "nmax50",
        "nmax100",
        "rho025",
        "candidates100",
        "candidates500",
        "budget1",
        "budget3",
        "utility-ndcg",
    }
    observed_sensitivities = {key[5] for key in groups if key[4] == "sensitivity"}
    missing_sensitivities = sorted(required_sensitivities - observed_sensitivities)
    if missing_sensitivities:
        errors.append(
            f"Missing predeclared sensitivity conditions: {missing_sensitivities}"
        )
    expected_sensitivity_values = {
        "nmax50": ("n_max", 50),
        "nmax100": ("n_max", 100),
        "rho025": ("action_rho", 0.25),
        "candidates100": ("evaluation_size", 100),
        "candidates500": ("evaluation_size", 500),
        "budget1": ("budget", 1),
        "budget3": ("budget", 3),
        "utility-ndcg": ("utility", "ndcg"),
    }
    sensitivity_fields = {"n_max", "evaluation_size", "action_rho", "budget", "utility"}
    for key, group in groups.items():
        if key[4] != "sensitivity" or key[5] not in expected_sensitivity_values:
            continue
        field, expected = expected_sensitivity_values[key[5]]
        observed = {result["config"].get(field) for result in group}
        if observed != {expected}:
            errors.append(
                f"{key}: expected {field}={expected}, observed {sorted(observed)}"
            )
        baseline_groups = [
            baseline_group
            for baseline_key, baseline_group in groups.items()
            if baseline_key[0] == key[0]
            and baseline_key[1] == key[1]
            and baseline_key[4] == "primary"
            and baseline_key[5] == "primary"
        ]
        if len(baseline_groups) != 1:
            errors.append(f"{key}: cannot identify exactly one primary baseline group")
            continue
        baseline_config = baseline_groups[0][0]["config"]
        condition_config = group[0]["config"]
        for controlled_field in sensitivity_fields - {field}:
            if condition_config.get(controlled_field) != baseline_config.get(
                controlled_field
            ):
                errors.append(
                    f"{key}: sensitivity changed uncontrolled field {controlled_field}: "
                    f"{baseline_config.get(controlled_field)} -> {condition_config.get(controlled_field)}"
                )
    return {
        "status": "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "experiment_groups": [list(key) for key in sorted(groups)],
        "n_files": len(results),
    }


def flatten_users(results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    for result in results:
        dataset, model, mode, utility, analysis_role, condition = experiment_key(result)
        seed = int(result["_seed"])
        for user in result["users"]:
            base = {
                "dataset": dataset,
                "model": model,
                "evaluation_mode": mode,
                "utility": utility,
                "analysis_role": analysis_role,
                "condition": condition,
                "seed": seed,
                "user": int(user["user"]),
                "n_players": int(user["n_players"]),
                "evaluation_size": int(user["evaluation_size"]),
            }
            quality = user.get("recommendation_quality", {})
            for method, values in user["methods"].items():
                row = {
                    **base,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "aia": values.get("aia"),
                    "aia_ndcg": values.get("aia_ndcg"),
                    "faithfulness_alignment": values.get("faithfulness_alignment"),
                    "actionability_gap": values.get("actionability_gap"),
                    "signed_alignment": values.get("signed_alignment"),
                    "signed_alignment_ndcg": values.get("signed_alignment_ndcg"),
                    "direction_accuracy": values.get("direction_accuracy"),
                    "direction_accuracy_ndcg": values.get("direction_accuracy_ndcg"),
                    "top1_precision": values.get("topk_precision", {}).get("1"),
                    "top3_precision": values.get("topk_precision", {}).get("3"),
                    "top5_precision": values.get("topk_precision", {}).get("5"),
                    "aia_null_mean": values.get("aia_null", {}).get("null_mean"),
                    "aia_null_p95": values.get("aia_null", {}).get("null_p95"),
                    "aia_permutation_p": values.get("aia_null", {}).get("p_value"),
                    "joint_effect_primary": values.get("effect_primary"),
                    "joint_effect_ndcg": values.get("effect_ndcg"),
                    "joint_effect_target_margin": values.get("effect_target_margin"),
                    "intervention_success": float(bool(values.get("success"))),
                    "intervention_success_ndcg": float(
                        bool(values.get("success_ndcg"))
                    ),
                    "abstention": float(
                        bool(values.get("action", {}).get("abstained"))
                    ),
                    "joint_regret_primary": values.get("regret_primary"),
                    "normalized_regret_primary": values.get(
                        "normalized_regret_primary"
                    ),
                    "joint_regret_ndcg": values.get("regret_ndcg"),
                    "normalized_regret_ndcg": values.get("normalized_regret_ndcg"),
                    **{f"quality_{key}": value for key, value in quality.items()},
                }
                metric_rows.append(row)
                attribution_rows.append(
                    {
                        **base,
                        "method": method,
                        "attribution": values.get("attribution"),
                        "feasible_effects": user.get("effects", {}).get(
                            "feasible_primary"
                        ),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(attribution_rows)


def _bootstrap_mean(
    values: np.ndarray, draws: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 256):
        stop = min(start + 256, draws)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def method_summaries(frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "dataset",
        "model",
        "evaluation_mode",
        "utility",
        "analysis_role",
        "condition",
        "method",
        "method_label",
    ]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        for metric in PRIMARY_METRICS + [
            "aia_null_mean",
            "aia_null_p95",
            "aia_permutation_p",
        ]:
            pivot = group.pivot_table(
                index="user", columns="seed", values=metric, aggfunc="first"
            )
            user_values = pivot.mean(axis=1, skipna=True).to_numpy(float)
            mean, low, high = _bootstrap_mean(user_values, draws, seed=42 + len(rows))
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_users": int(np.isfinite(user_values).sum()),
                    "n_seeds": int(group["seed"].nunique()),
                    "missing_users": int((~np.isfinite(user_values)).sum()),
                }
            )
    return pd.DataFrame(rows)


def paired_tests(frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    experiment_columns = [
        "dataset",
        "model",
        "evaluation_mode",
        "utility",
        "analysis_role",
        "condition",
    ]
    for experiment, group in frame.groupby(experiment_columns, sort=True):
        methods = [method for method in METHOD_ORDER if method in set(group["method"])]
        users = sorted(group["user"].unique())
        seeds = sorted(group["seed"].unique())
        for metric in PRIMARY_METRICS:
            method_matrices: dict[str, np.ndarray] = {}
            for method in methods:
                subset = group.loc[group["method"] == method]
                pivot = subset.pivot_table(
                    index="user", columns="seed", values=metric, aggfunc="first"
                )
                method_matrices[method] = pivot.reindex(
                    index=users, columns=seeds
                ).to_numpy(float)
            metric_rows: list[dict[str, Any]] = []
            for left, right in combinations(methods, 2):
                try:
                    result = paired_user_seed_comparison(
                        method_matrices[left],
                        method_matrices[right],
                        bootstrap_draws=draws,
                        permutation_draws=draws,
                        seed=42 + len(rows) + len(metric_rows),
                    )
                except ValueError:
                    continue
                metric_rows.append(
                    {
                        **dict(zip(experiment_columns, experiment)),
                        "metric": metric,
                        "left": left,
                        "right": right,
                        **asdict(result),
                    }
                )
            if metric_rows:
                corrected = holm_bonferroni(
                    [row["permutation_p"] for row in metric_rows]
                )
                for row, adjusted in zip(metric_rows, corrected):
                    row["p_holm"] = float(adjusted)
                    rows.append(row)
    return pd.DataFrame(rows)


def attribution_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "dataset",
        "model",
        "evaluation_mode",
        "utility",
        "analysis_role",
        "condition",
        "method",
    ]
    for keys, group in frame.groupby(group_columns, sort=True):
        user_stability: list[float] = []
        for _, user_group in group.groupby("user"):
            vectors = [
                np.asarray(value, dtype=float)
                for value in user_group.sort_values("seed")["attribution"]
            ]
            correlations: list[float] = []
            for left, right in combinations(vectors, 2):
                if (
                    left.shape == right.shape
                    and left.size > 1
                    and left.std()
                    and right.std()
                ):
                    correlations.append(float(spearmanr(left, right).statistic))
            if correlations:
                user_stability.append(float(np.mean(correlations)))
        values = np.asarray(user_stability, dtype=float)
        mean, low, high = _bootstrap_mean(values, 10_000, seed=91 + len(rows))
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "method_label": METHOD_LABELS.get(keys[-1], keys[-1]),
                "mean_rank_stability": mean,
                "ci95_low": low,
                "ci95_high": high,
                "n_users": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def aggregate_aia_null(frame: pd.DataFrame, draws: int = 1000) -> pd.DataFrame:
    """Calibrate the headline mean AIA with shuffling inside each user and seed."""
    rows: list[dict[str, Any]] = []
    group_columns = [
        "dataset",
        "model",
        "evaluation_mode",
        "utility",
        "analysis_role",
        "condition",
        "method",
    ]
    primary = frame.loc[frame["analysis_role"] == "primary"]
    for keys, group in primary.groupby(group_columns, sort=True):
        null_by_user: dict[int, list[np.ndarray]] = {}
        observed_by_user: dict[int, list[float]] = {}
        for record in group.itertuples(index=False):
            attribution = np.abs(np.asarray(record.attribution, dtype=float))
            effects = np.abs(np.asarray(record.feasible_effects, dtype=float))
            if (
                attribution.shape != effects.shape
                or attribution.size < 2
                or attribution.std() == 0
                or effects.std() == 0
            ):
                continue
            attribution_rank = rankdata(attribution)
            effect_rank = rankdata(effects)
            left = attribution_rank - attribution_rank.mean()
            right = effect_rank - effect_rank.mean()
            denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
            if denominator == 0:
                continue
            observed = float(np.dot(left, right) / denominator)
            identity = "|".join(map(str, (*keys, record.user, record.seed)))
            local_seed = int.from_bytes(
                hashlib.sha256(identity.encode()).digest()[:8], "little"
            )
            rng = np.random.default_rng(local_seed)
            # Random-key sorting creates independent permutations in a compact,
            # vectorized operation; ties in effect ranks are preserved.
            orders = np.argsort(rng.random((draws, effects.size)), axis=1)
            null_values = (right[orders] @ left) / denominator
            null_by_user.setdefault(int(record.user), []).append(null_values)
            observed_by_user.setdefault(int(record.user), []).append(observed)
        common_users = sorted(set(null_by_user) & set(observed_by_user))
        if not common_users:
            continue
        user_null = np.vstack(
            [np.mean(null_by_user[user], axis=0) for user in common_users]
        )
        null_distribution = user_null.mean(axis=0)
        observed_mean = float(
            np.mean([np.mean(observed_by_user[user]) for user in common_users])
        )
        exceedances = int(np.count_nonzero(null_distribution >= observed_mean))
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "method_label": METHOD_LABELS.get(keys[-1], keys[-1]),
                "observed_mean_aia": observed_mean,
                "null_mean": float(null_distribution.mean()),
                "null_p95": float(np.quantile(null_distribution, 0.95)),
                "permutation_p": float((exceedances + 1) / (draws + 1)),
                "draws": int(draws),
                "n_users": len(common_users),
            }
        )
    return pd.DataFrame(rows)


def protocol_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(experiment_key(result), []).append(result)
    for key, group in sorted(groups.items()):
        first = group[0]
        quality_keys = sorted(first.get("recommendation_quality", {}))
        row: dict[str, Any] = {
            "dataset": key[0],
            "model": key[1],
            "model_role": first["config"].get("model_role"),
            "evaluation_mode": key[2],
            "utility": key[3],
            "analysis_role": key[4],
            "condition": key[5],
            "n_seeds": len({result["_seed"] for result in group}),
            "n_users": len(first["dataset"]["selected_user_ids"]),
            "n_items": int(first["dataset"]["items"]),
            "users_with_repeated_target": int(
                first["dataset"].get("users_with_repeated_target", 0)
            ),
            "n_max": int(first["config"]["n_max"]),
            "epochs": int(first["config"]["epochs"]),
            "profile_samples_per_user": int(
                first["config"].get("profile_samples_per_user", 1)
            ),
            "permutations": int(first["config"]["permutations"]),
            "action_rho": float(first["config"]["action_rho"]),
            "budget": int(first["config"]["budget"]),
            "gate_passed_seeds": int(
                sum(bool(result["masking_gate"]["passed"]) for result in group)
            ),
            "gate_changed_fraction": float(
                np.mean(
                    [result["masking_gate"]["changed_fraction"] for result in group]
                )
            ),
            "gate_mean_abs_ndcg": float(
                np.mean(
                    [result["masking_gate"]["mean_abs_ndcg_change"] for result in group]
                )
            ),
        }
        for quality_key in quality_keys:
            row[quality_key] = float(
                np.mean(
                    [result["recommendation_quality"][quality_key] for result in group]
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_tex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    latex = frame.to_latex(
        index=False, escape=True, float_format=lambda value: f"{value:.4g}"
    )
    path.write_text(
        "% Generated by scripts/make_paper_assets.py\n"
        "\\begin{table}[t]\\centering\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n" + latex + "}\n\\end{table}\n"
    )


def make_figures(summary: pd.DataFrame, figure_root: Path) -> None:
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300, "font.size": 10})
    for metric, title, ylabel, filename in (
        (
            "aia",
            "Target-margin Attribution–Intervention Alignment",
            "Spearman AIA",
            "aia_margin",
        ),
        (
            "aia_ndcg",
            "Cross-utility NDCG alignment",
            "Spearman AIA",
            "aia_ndcg",
        ),
        (
            "actionability_gap",
            "Faithfulness–actionability gap",
            "AIA − deletion alignment",
            "gap",
        ),
        (
            "joint_effect_ndcg",
            "Realized NDCG effect of selected action",
            "Change in NDCG",
            "joint_ndcg",
        ),
        (
            "normalized_regret_ndcg",
            "Normalized NDCG intervention regret",
            "NDCG regret",
            "normalized_regret_ndcg",
        ),
    ):
        data = summary.loc[summary["metric"] == metric].copy()
        if data.empty:
            continue
        for experiment, group in data.groupby(
            [
                "dataset",
                "model",
                "evaluation_mode",
                "utility",
                "analysis_role",
                "condition",
            ]
        ):
            order = {method: index for index, method in enumerate(METHOD_ORDER)}
            group = group.assign(_order=group["method"].map(order)).sort_values(
                "_order"
            )
            group = group.loc[
                np.isfinite(group["mean"])
                & np.isfinite(group["ci95_low"])
                & np.isfinite(group["ci95_high"])
            ]
            if group.empty:
                continue
            x = np.arange(len(group))
            figure, axis = plt.subplots(figsize=(6.4, 3.8))
            # Percentile bootstrap intervals need not contain the plug-in mean
            # in a very small diagnostic run. Matplotlib requires non-negative
            # bar lengths, so clip only the visual distance, not the CSV values.
            lower_error = np.maximum(0.0, group["mean"] - group["ci95_low"])
            upper_error = np.maximum(0.0, group["ci95_high"] - group["mean"])
            axis.errorbar(
                x,
                group["mean"],
                yerr=[lower_error, upper_error],
                fmt="o",
                capsize=4,
            )
            axis.set_xticks(x, group["method_label"], rotation=20, ha="right")
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            figure.tight_layout()
            suffix = "_".join(str(value).replace(" ", "-") for value in experiment)
            for extension in ("png", "pdf"):
                figure.savefig(
                    figure_root / f"{filename}_{suffix}.{extension}",
                    bbox_inches="tight",
                )
            plt.close(figure)


def make_convergence_figure(frame: pd.DataFrame, figure_root: Path) -> None:
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True, sharey=True)
    for keys, group in frame.groupby(["dataset", "model", "utility"], sort=True):
        group = group.sort_values("permutations")
        label = " / ".join(map(str, keys))
        axes[0].plot(
            group["permutations"],
            group["mean_rank_correlation_to_reference"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            group["permutations"],
            group["mean_top2_jaccard"],
            marker="o",
            label=label,
        )
    for axis, title, threshold in zip(
        axes,
        ("Rank convergence", "Signed B=2 action-set overlap"),
        (0.95, 0.80),
    ):
        axis.axhline(threshold, color="black", linestyle="--", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Permutation budget M")
        axis.set_title(title)
    axes[0].set_ylabel("Agreement with independent M=1000 reference")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(figure_root / f"convergence.{extension}", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="results/raw")
    parser.add_argument("--paper-root", default="../paper")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--null-draws", type=int, default=1000)
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    raw_root = (code_root / args.raw_root).resolve()
    paper_root = (code_root / args.paper_root).resolve()
    repo = repository_root(code_root)
    final_root = paper_root / "final"
    if final_root.exists():
        shutil.rmtree(final_root)
    data_root = final_root / "data"
    table_root = final_root / "tables"
    figure_root = final_root / "figures"
    manifest_root = final_root / "manifests"
    for directory in (data_root, table_root, figure_root, manifest_root):
        directory.mkdir(parents=True, exist_ok=True)

    files, results = load_results(raw_root)
    convergence = load_convergence(raw_root)
    preflight_files = sorted((paper_root / "preflight").glob("*.json"))
    validation = validate(results, convergence)
    if not preflight_files:
        validation["errors"].append("No content-addressed design preflight was found")
        validation["status"] = "FAIL"
    else:
        result_hashes = {
            result.get("provenance", {}).get("input_sha256") for result in results
        }
        for preflight_path in preflight_files:
            preflight = json.loads(preflight_path.read_text())
            if preflight.get("source_sha256") not in result_hashes:
                validation["errors"].append(
                    f"{preflight_path.name}: preflight dataset hash differs from final inputs"
                )
                validation["status"] = "FAIL"
    convergence_summary = convergence_frame(convergence)
    metrics, attributions = flatten_users(results)
    summaries = method_summaries(metrics, args.draws)
    tests = paired_tests(metrics, args.draws)
    stability = attribution_stability(attributions)
    aggregate_null = aggregate_aia_null(attributions, args.null_draws)
    protocol = protocol_table(results)

    metrics.to_csv(
        data_root / "user_seed_metrics.csv.gz", index=False, compression="gzip"
    )
    summaries.to_csv(table_root / "method_metrics.csv", index=False)
    tests.to_csv(table_root / "paired_tests.csv", index=False)
    stability.to_csv(table_root / "attribution_stability.csv", index=False)
    aggregate_null.to_csv(table_root / "aia_permutation_null.csv", index=False)
    protocol.to_csv(table_root / "protocol_audit.csv", index=False)
    convergence_summary.to_csv(table_root / "convergence.csv", index=False)
    selected_summary = summaries.loc[
        summaries["metric"].isin(
            [
                "aia",
                "aia_ndcg",
                "faithfulness_alignment",
                "actionability_gap",
                "joint_effect_ndcg",
                "intervention_success_ndcg",
                "joint_regret_ndcg",
                "normalized_regret_ndcg",
                "normalized_regret_primary",
            ]
        )
    ][
        [
            "dataset",
            "model",
            "evaluation_mode",
            "utility",
            "analysis_role",
            "condition",
            "metric",
            "method_label",
            "mean",
            "ci95_low",
            "ci95_high",
            "n_users",
            "missing_users",
        ]
    ]
    result_columns = [
        "dataset",
        "model",
        "utility",
        "metric",
        "method_label",
        "mean",
        "ci95_low",
        "ci95_high",
        "n_users",
        "missing_users",
    ]
    primary_summary = selected_summary.loc[
        selected_summary["analysis_role"] == "primary", result_columns
    ]
    robustness_summary = selected_summary.loc[
        selected_summary["analysis_role"] == "full_catalogue", result_columns
    ]
    sensitivity_summary = selected_summary.loc[
        selected_summary["analysis_role"] == "sensitivity",
        ["condition", *result_columns],
    ]
    write_tex(
        protocol,
        table_root / "protocol_audit.tex",
        "Dataset, model, gate, and recommendation-quality audit.",
        "tab:protocol-audit",
    )
    write_tex(
        convergence_summary,
        table_root / "convergence.tex",
        "Independent Monte Carlo rank and action convergence.",
        "tab:convergence",
    )
    aggregate_null_columns = [
        "dataset",
        "model",
        "utility",
        "method_label",
        "observed_mean_aia",
        "null_mean",
        "null_p95",
        "permutation_p",
        "n_users",
    ]
    aggregate_null_table = (
        aggregate_null[aggregate_null_columns]
        if not aggregate_null.empty
        else aggregate_null
    )
    write_tex(
        aggregate_null_table,
        table_root / "aia_permutation_null.tex",
        "Within-user, within-seed permutation calibration of target-margin AIA.",
        "tab:aia-null",
    )
    write_tex(
        primary_summary,
        table_root / "primary_results.tex",
        "Primary user-clustered ActionShap results.",
        "tab:primary-results",
    )
    write_tex(
        robustness_summary,
        table_root / "full_catalogue_results.tex",
        "Full-unseen-catalogue robustness.",
        "tab:full-catalogue",
    )
    write_tex(
        sensitivity_summary,
        table_root / "sensitivity_results.tex",
        "Predeclared one-factor-at-a-time sensitivity analysis.",
        "tab:sensitivity",
    )
    primary_tests = (
        tests.loc[tests["analysis_role"] == "primary"] if not tests.empty else tests
    )
    if not primary_tests.empty:
        primary_tests = primary_tests[
            [
                "dataset",
                "model",
                "metric",
                "left",
                "right",
                "mean_difference",
                "ci95_low",
                "ci95_high",
                "cohens_dz",
                "p_holm",
                "n_users",
            ]
        ]
    write_tex(
        primary_tests,
        table_root / "paired_tests.tex",
        "Paired user-level comparisons with Holm correction.",
        "tab:paired-tests",
    )
    stability_table = stability.loc[
        stability["analysis_role"] == "primary",
        [
            "dataset",
            "model",
            "utility",
            "method_label",
            "mean_rank_stability",
            "ci95_low",
            "ci95_high",
            "n_users",
        ],
    ]
    write_tex(
        stability_table,
        table_root / "attribution_stability.tex",
        "Attribution rank stability across model seeds.",
        "tab:stability",
    )
    make_figures(summaries, figure_root)
    make_convergence_figure(convergence_summary, figure_root)

    provenance_files = (
        files + [study["_path"] for study in convergence.values()] + preflight_files
    )

    def portable_source_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(repo))
        except ValueError:
            # Tests and external archives may be mounted outside the checkout;
            # never leak a machine-specific absolute path into the manifest.
            return f"external/{path.name}"

    source_files = [
        {
            "path": portable_source_path(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in provenance_files
    ]
    (manifest_root / "validation_report.json").write_text(
        json.dumps(validation, indent=2)
    )
    (final_root / "README.md").write_text(
        "# Final ActionShap assets\n\n"
        f"Validation status: **{validation['status']}**.\n\n"
        "These assets use schema-v2 runs and distinct-user hierarchical inference. "
        "Legacy pilot assets are not consumed.\n"
    )
    manifest_path = manifest_root / "asset_manifest.json"
    manifest = {
        "schema_version": 2,
        "validation": validation,
        "source_files": source_files,
        "analysis": {
            "unit": "distinct user after averaging repeated seeds",
            "bootstrap_draws": args.draws,
            "paired_permutation_draws": args.draws,
            "aia_null_draws": args.null_draws,
            "multiplicity": "Holm-Bonferroni within experiment and metric",
            "paired_p_value_floor": 1 / (args.draws + 1),
            "aia_null_p_value_floor": 1 / (args.null_draws + 1),
        },
        # Include the manifest itself by declared relative path. Its checksum is
        # intentionally omitted to avoid a self-referential hash.
        "generated_files": sorted(
            {
                str(path.relative_to(paper_root))
                for path in final_root.rglob("*")
                if path.is_file()
            }
            | {str(manifest_path.relative_to(paper_root))}
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(validation, indent=2))
    if validation["errors"]:
        raise SystemExit("Asset generation completed with blocking validation errors")


if __name__ == "__main__":
    main()
