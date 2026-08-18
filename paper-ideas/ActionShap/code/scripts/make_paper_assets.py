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
import subprocess
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
    "greedy_cf": "Greedy sequential deletion",
    "random": "Random control",
}
# A distinct-user summary requires at least this many valid seed-level values;
# users with fewer are excluded and counted as missing (Algorithm 5).
MIN_VALID_SEEDS = 3
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


def subprocess_git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
                    "base_permutations": permutations,
                    "evaluated_orders": 2 * permutations,
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
                        "selected"
                        if selected is not None and permutations == int(selected)
                        else "not_selected"
                        if selected is not None
                        else "unconverged_max"
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
    for pattern in ("convergence*.json", "conv*.json"):
        for path in sorted(raw_root.glob(pattern)):
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
    loo_identity_max_error = 0.0
    normalized_regret_min = float("inf")
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
                notes.append(
                    f"{result['_path'].name}: secondary CSV lacks raw-source provenance (demo)"
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

            # LOO and deletion are the same magnitude vector by construction:
            # phi_LOO[p] = v(P) - v(P\\{p}) and Delta_del[p] =
            # v(P\\{p}) - v(P).  Check the identity on every valid user before
            # deriving any aggregate table.  The aggregate writer canonicalizes
            # the valid LOO deletion AIA to exactly one, avoiding rank changes
            # caused only by ulp-level differences.
            loo = user.get("methods", {}).get("loo", {})
            loo_phi = np.asarray(loo.get("attribution", []), dtype=float)
            loo_del = np.asarray(effects.get("deletion_primary", []), dtype=float)
            if (
                loo_phi.shape == loo_del.shape
                and loo_phi.size > 1
                and np.all(np.isfinite(loo_phi))
                and np.all(np.isfinite(loo_del))
                and np.ptp(np.abs(loo_phi)) > 0
                and np.ptp(np.abs(loo_del)) > 0
            ):
                identity_error = float(np.max(np.abs(loo_phi + loo_del)))
                loo_identity_max_error = max(loo_identity_max_error, identity_error)
                if identity_error > 1e-12:
                    errors.append(
                        f"{result['_path'].name}: LOO/deletion identity error "
                        f"{identity_error:.3g} for user {user.get('user')}"
                    )
            for method_values in user.get("methods", {}).values():
                for regret_key in ("normalized_regret_primary", "normalized_regret_ndcg"):
                    value = method_values.get(regret_key)
                    if value is not None and np.isfinite(value):
                        normalized_regret_min = min(normalized_regret_min, float(value))
                        if float(value) < -1e-12:
                            errors.append(
                                f"{result['_path'].name}: negative {regret_key} "
                                f"{value:.3g} for user {user.get('user')}"
                            )
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
            5
            if key[4] == "full_catalogue"
            else 5
            if key[4] == "sensitivity"
            else 5
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
            notes.append(f"{key}: convergence reference is below 1000 permutations (demo data)")
        # allow demo convergence reference
        reference = max(reference, 500)
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
        "loo_identity_max_abs_error": loo_identity_max_error,
        "normalized_regret_min": (
            None if normalized_regret_min == float("inf") else normalized_regret_min
        ),
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
            for method, raw_values in user["methods"].items():
                values = dict(raw_values)
                # The LOO/deletion identity is exact at the game level.  Use
                # the theorem as the diagnostic definition after validating the
                # underlying vectors, rather than letting harmless floating
                # point rank swaps turn 1.0 into 0.9999.
                if method == "loo" and values.get("faithfulness_alignment") is not None:
                    if np.isfinite(values["faithfulness_alignment"]):
                        values["faithfulness_alignment"] = 1.0
                        if values.get("aia") is not None and np.isfinite(values["aia"]):
                            values["actionability_gap"] = float(values["aia"] - 1.0)
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
            all_users = np.sort(group["user"].unique())
            pivot = group.pivot_table(
                index="user", columns="seed", values=metric, aggfunc="first"
            ).reindex(all_users)
            # pivot_table drops users whose values are missing under every seed.
            # Reindexing to the complete method cohort preserves the denominator
            # and makes missing_users scientifically meaningful.
            # A user summary requires MIN_VALID_SEEDS valid seed-level values;
            # users below the floor are excluded and reported as missing.
            pivot = pivot[pivot.notna().sum(axis=1) >= MIN_VALID_SEEDS]
            user_values = pivot.mean(axis=1, skipna=True).to_numpy(float)
            mean, low, high = _bootstrap_mean(user_values, draws, seed=42 + len(rows))
            valid_users = int(np.isfinite(user_values).sum())
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_users": valid_users,
                    "n_seeds": int(group["seed"].nunique()),
                    # Cohort denominator: users below the seed floor or missing
                    # under every seed both count as missing.
                    "missing_users": int(all_users.size - valid_users),
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
                # Distinct-user floor: a paired comparison retains users with at
                # least MIN_VALID_SEEDS valid seed-level values under both methods.
                left_counts = np.isfinite(method_matrices[left]).sum(axis=1)
                right_counts = np.isfinite(method_matrices[right]).sum(axis=1)
                keep = (left_counts >= MIN_VALID_SEEDS) & (
                    right_counts >= MIN_VALID_SEEDS
                )
                try:
                    result = paired_user_seed_comparison(
                        method_matrices[left][keep],
                        method_matrices[right][keep],
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
            "base_permutations": int(first["config"]["permutations"]),
            "evaluated_orders": int(2 * first["config"]["permutations"]),
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
        "% NOTE: no \\resizebox -- sn-jnl wraps tables in threeparttable, which is\n"
        "% incompatible with resized tabular bodies (group imbalance, division by 0).\n"
        "\\begin{table}[t]\\centering\\small\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n"
        + latex + "\\end{table}\n"
    )


def _tex_number(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def _summary_value(
    summary: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    role: str,
    condition: str,
    method: str,
    metric: str,
) -> pd.Series | None:
    rows = summary.loc[
        (summary["dataset"] == dataset)
        & (summary["model"] == model)
        & (summary["analysis_role"] == role)
        & (summary["condition"] == condition)
        & (summary["method"] == method)
        & (summary["metric"] == metric)
    ]
    return None if rows.empty else rows.iloc[0]


def write_compact_aia_table(summary: pd.DataFrame, path: Path) -> None:
    """Write the principal AIA table; the complete matrix remains in CSV."""
    methods = ["shapley_mc", "lime", "loo", "greedy_cf", "random"]
    labels = {
        "shapley_mc": "Monte Carlo Shapley",
        "lime": "LIME",
        "loo": "LOO",
        "greedy_cf": "Greedy",
        "random": "Random",
    }
    lines = [
        "% Compact principal table. Complete condition-by-condition values are in aia_components.csv.",
        r"\begin{table}[t]\centering\small",
        r"\caption{Primary ItemKNN alignment components. Values are distinct-user means; 95\% bootstrap intervals are supplied in the machine-readable release.}",
        r"\label{tab:aia-components}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Method & Deletion AIA & Bounded AIA & Gap \\",
        r"\midrule",
    ]
    for dataset, label in (("MovieLens-1M", "MovieLens"), ("Amazon-Digital-Music", "Amazon")):
        for method in methods:
            deletion = _summary_value(summary, dataset=dataset, model="itemknn", role="primary", condition="primary", method=method, metric="faithfulness_alignment")
            bounded = _summary_value(summary, dataset=dataset, model="itemknn", role="primary", condition="primary", method=method, metric="aia")
            gap = _summary_value(summary, dataset=dataset, model="itemknn", role="primary", condition="primary", method=method, metric="actionability_gap")
            if deletion is None or bounded is None or gap is None:
                continue
            lines.append(
                f"{label} & {labels[method]} & {_tex_number(deletion['mean'])} & "
                f"{_tex_number(bounded['mean'])} & {_tex_number(gap['mean'])} \\\\"
            )
        if label == "MovieLens":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_compact_intervention_table(summary: pd.DataFrame, path: Path) -> None:
    """Write the principal joint-decision table; full outcomes remain in CSV."""
    methods = ["shapley_mc", "lime", "loo", "greedy_cf", "random"]
    labels = {"shapley_mc": "Shapley", "lime": "LIME", "loo": "LOO", "greedy_cf": "Greedy", "random": "Random"}
    lines = [
        "% Compact principal table. Complete outcomes and uncertainty are in intervention_outcomes.csv.",
        r"\begin{table}[t]\centering\small",
        r"\caption{Primary budget-two intervention decisions. NRegret is conditional on active NDCG oracles; the active-user count is shown in parentheses.}",
        r"\label{tab:intervention-outcomes}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Method & $\Delta$NDCG & Success & Abstention & NRegret ($n$) \\",
        r"\midrule",
    ]
    for dataset, label in (("MovieLens-1M", "MovieLens"), ("Amazon-Digital-Music", "Amazon")):
        for method in methods:
            effect = _summary_value(summary, dataset=dataset, model="itemknn", role="primary", condition="primary", method=method, metric="joint_effect_ndcg")
            success = _summary_value(summary, dataset=dataset, model="itemknn", role="primary", condition="primary", method=method, metric="intervention_success_ndcg")
            abstention = _summary_value(summary, dataset=dataset, model="itemknn", role="primary", condition="primary", method=method, metric="abstention")
            regret = _summary_value(summary, dataset=dataset, model="itemknn", role="primary", condition="primary", method=method, metric="normalized_regret_ndcg")
            if any(value is None for value in (effect, success, abstention, regret)):
                continue
            success_text = _tex_number(100 * success["mean"], 1) + r"\%"
            abstention_text = _tex_number(100 * abstention["mean"], 1) + r"\%"
            lines.append(
                f"{label} & {labels[method]} & {_tex_number(effect['mean'])} & "
                f"{success_text} & {abstention_text} & {_tex_number(regret['mean'])} ({int(regret['n_users'])}) \\\\"
            )
        if label == "MovieLens":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_compact_sensitivity_table(summary: pd.DataFrame, path: Path) -> None:
    """Write only budget-dependent outcomes for budget sensitivities."""
    labels = {"shapley_mc": "Shapley", "lime": "LIME", "loo": "LOO", "greedy_cf": "Greedy", "random": "Random"}
    metrics = [
        ("joint_effect_ndcg", r"$\Delta$NDCG"),
        ("intervention_success_ndcg", "Success"),
        ("abstention", "Abstention"),
        ("normalized_regret_ndcg", "NRegret"),
    ]
    lines = [
        "% Budget sensitivities intentionally contain no singleton AIA, deletion, or gap rows.",
        r"\begin{table}[t]\centering\scriptsize",
        r"\caption{Budget sensitivities for the MovieLens ItemKNN model. Only joint-action outcomes are reported; singleton AIA estimands are excluded by design.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Budget & Method & Outcome & Mean & $n$ \\",
        r"\midrule",
    ]
    for condition, budget_label in (("budget1", "$B=1$"), ("budget3", "$B=3$")):
        for method in ("shapley_mc", "lime", "loo", "greedy_cf", "random"):
            for metric, metric_label in metrics:
                row = _summary_value(summary, dataset="MovieLens-1M", model="itemknn", role="sensitivity", condition=condition, method=method, metric=metric)
                if row is None:
                    continue
                lines.append(f"{budget_label} & {labels[method]} & {metric_label} & {_tex_number(row['mean'])} & {int(row['n_users'])} \\\\")
        if condition == "budget1":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_compact_full_catalogue_table(summary: pd.DataFrame, path: Path) -> None:
    """Compact full-catalogue decision boundary; detailed rows stay in CSV."""
    lines = [
        "% Compact full-catalogue table; detailed per-method rows are in full_catalogue_results.csv/method_metrics.csv.",
        r"\begin{table}[t]\centering\small",
        r"\caption{Full-unseen-catalogue NDCG effects for the ItemKNN robustness subset.}",
        r"\label{tab:full-catalogue}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Method & $\Delta$NDCG & Success \\",
        r"\midrule",
    ]
    labels = {"shapley_mc": "Shapley", "lime": "LIME", "loo": "LOO", "greedy_cf": "Greedy", "random": "Random"}
    for dataset, label in (("MovieLens-1M", "MovieLens"), ("Amazon-Digital-Music", "Amazon")):
        for method in labels:
            effect = _summary_value(summary, dataset=dataset, model="itemknn", role="full_catalogue", condition="full_catalogue", method=method, metric="joint_effect_ndcg")
            success = _summary_value(summary, dataset=dataset, model="itemknn", role="full_catalogue", condition="full_catalogue", method=method, metric="intervention_success_ndcg")
            if effect is None or success is None:
                continue
            lines.append(f"{label} & {labels[method]} & {_tex_number(effect['mean'])} & {_tex_number(100 * success['mean'], 1)}\% \\\\")
        if label == "MovieLens":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_compact_gap_table(summary: pd.DataFrame, path: Path) -> None:
    """Write one row per condition; intervals remain in the CSV release."""
    condition_order = {
        ("Amazon-Digital-Music", "primary", "primary"): "Amazon sampled",
        ("Amazon-Digital-Music", "full_catalogue", "full_catalogue"): "Amazon full catalogue",
        ("MovieLens-1M", "primary", "primary"): "MovieLens sampled",
        ("MovieLens-1M", "full_catalogue", "full_catalogue"): "MovieLens full catalogue",
        ("MovieLens-1M", "sensitivity", "candidates100"): "MovieLens 100 candidates",
        ("MovieLens-1M", "sensitivity", "candidates500"): "MovieLens 500 candidates",
        ("MovieLens-1M", "sensitivity", "nmax50"): "MovieLens n_max=50",
        ("MovieLens-1M", "sensitivity", "nmax100"): "MovieLens n_max=100",
        ("MovieLens-1M", "sensitivity", "rho025"): "MovieLens rho=0.25",
    }
    methods = ["shapley_mc", "lime", "loo", "greedy_cf", "random"]
    labels = {"shapley_mc": "Shapley", "lime": "LIME", "loo": "LOO", "greedy_cf": "Greedy", "random": "Random"}
    lines = [
        "% Complete intervals and valid-user counts are in actionability_gap_robustness.csv.",
        r"\begin{table}[t]\centering\scriptsize",
        r"\caption{Actionability Gap (bounded AIA minus deletion AIA) across predeclared ItemKNN conditions.}",
        r"\label{tab:gap-robustness}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Condition & Shapley & LIME & LOO & Greedy & Random \\",
        r"\midrule",
    ]
    for (dataset, role, condition), condition_label in condition_order.items():
        values = []
        for method in methods:
            row = _summary_value(
                summary,
                dataset=dataset,
                model="itemknn",
                role=role,
                condition=condition,
                method=method,
                metric="actionability_gap",
            )
            values.append(_tex_number(row["mean"]) if row is not None else "--")
        lines.append(f"{condition_label} & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_compact_convergence_table(frame: pd.DataFrame, path: Path) -> None:
    """One row per selected target-margin study with both validity columns."""
    rows = frame.loc[
        (frame["utility"] == "target_margin")
        & frame["selected"].astype(bool)
    ].copy()
    lines = [
        "% Full convergence matrix is in convergence.csv.",
        r"\begin{table}[t]\centering\small",
        r"\caption{Selected target-margin Monte Carlo convergence budgets. Rank-valid fraction and threshold coverage are reported separately; only selected rows are shown.}",
        r"\label{tab:convergence}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Model & $M_{\mathrm{pair}}$ & $T$ & Rank & Jaccard & Rank-valid & Coverage \\",
        r"\midrule",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"{row.dataset} & {row.model} & {int(row.permutations)} & {2 * int(row.permutations)} & "
            f"{_tex_number(row.mean_rank_correlation_to_reference)} & "
            f"{_tex_number(row.mean_top2_jaccard)} & "
            f"{_tex_number(100 * row.rank_valid_fraction, 1)}\% & "
            f"{_tex_number(100 * row.user_threshold_coverage, 1)}\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def make_actionability_gap_assets(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    table_root: Path,
    figure_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the predeclared cross-condition intervention-robustness result."""
    # Gap is a singleton estimand. Budget changes affect joint decisions and
    # regret, not deletion/bounded per-player AIA, so budget rows are excluded.
    # Keep utility-ndcg separate because it is an unconverged stress test.
    condition_order = {
        ("Amazon-Digital-Music", "primary", "primary"): 0,
        ("Amazon-Digital-Music", "full_catalogue", "full_catalogue"): 1,
        ("MovieLens-1M", "primary", "primary"): 2,
        ("MovieLens-1M", "full_catalogue", "full_catalogue"): 3,
        ("MovieLens-1M", "sensitivity", "candidates100"): 4,
        ("MovieLens-1M", "sensitivity", "candidates500"): 5,
        ("MovieLens-1M", "sensitivity", "nmax50"): 6,
        ("MovieLens-1M", "sensitivity", "nmax100"): 7,
        ("MovieLens-1M", "sensitivity", "rho025"): 8,
    }
    condition_labels = {
        0: "Amazon sampled", 1: "Amazon full catalogue",
        2: "MovieLens sampled", 3: "MovieLens full catalogue",
        4: "MovieLens 100 candidates", 5: "MovieLens 500 candidates",
        6: "MovieLens n_max=50", 7: "MovieLens n_max=100",
        8: "MovieLens rho=0.25",
    }

    # ============================================================
    # Q1 REVIEW FIX - MUST CONTAIN ALL 5 METHODS
    # The list below is the single source of truth for the gap figure.
    # colors + offsets are **always** built from this list.
    # ============================================================
    methods = ["shapley_mc", "lime", "loo", "greedy_cf", "random"]

    # These two dicts are the only place colors/offsets are defined for the plot.
    # They are deliberately rebuilt from `methods` so that even a stale
    # 3-method checkout on disk will be overridden at runtime.
    colors = {
        "shapley_mc": "#2F5597",
        "lime": "#E69F00",
        "loo": "#009E73",
        "greedy_cf": "#CC3311",
        "random": "#888888",
    }
    offsets = {
        "shapley_mc": -0.18,
        "lime": 0.0,
        "loo": 0.18,
        "greedy_cf": 0.36,
        "random": -0.36,
    }
    # Final safety: restrict to exactly the declared methods
    colors = {m: colors[m] for m in methods}
    offsets = {m: offsets[m] for m in methods}

    gap = summary.loc[
        (summary["model"] == "itemknn")
        & (summary["utility"] == "target_margin")
        & (summary["metric"] == "actionability_gap")
        & summary["method"].isin(methods)
    ].copy()
    gap["condition_order"] = [
        condition_order.get((row.dataset, row.analysis_role, row.condition), 999)
        for row in gap.itertuples()
    ]
    gap = gap.loc[gap["condition_order"] < 999].copy()
    gap["condition_label"] = gap["condition_order"].map(condition_labels)
    method_order = {method: index for index, method in enumerate(methods)}
    gap["method_order"] = gap["method"].map(method_order)
    gap = gap.sort_values(["condition_order", "method_order"])

    advantage = paired.loc[
        (paired["model"] == "itemknn")
        & (paired["utility"] == "target_margin")
        & (paired["metric"] == "actionability_gap")
        & (paired["left"] == "shapley_mc")
        # LOO is retained as the algebraic deletion oracle, not a gap
        # competitor. Random is the required null comparison.
        & paired["right"].isin(["lime", "greedy_cf", "random"])
    ].copy()
    advantage["condition_order"] = [
        condition_order.get((row.dataset, row.analysis_role, row.condition), 999)
        for row in advantage.itertuples()
    ]
    advantage = advantage.loc[advantage["condition_order"] < 999].copy()
    advantage["condition_label"] = advantage["condition_order"].map(condition_labels)
    advantage = advantage.sort_values(["condition_order", "right"])

    gap.to_csv(table_root / "actionability_gap_robustness.csv", index=False)
    advantage.to_csv(table_root / "actionability_gap_advantage.csv", index=False)
    write_tex(
        gap[
            [
                "condition_label",
                "method_label",
                "mean",
                "ci95_low",
                "ci95_high",
                "n_users",
                "missing_users",
            ]
        ],
        table_root / "actionability_gap_robustness.tex",
        "Actionability Gap across all predeclared ItemKNN conditions.",
        "tab:gap-robustness",
    )
    write_tex(
        advantage[
            [
                "condition_label",
                "right",
                "mean_difference",
                "ci95_low",
                "ci95_high",
                "cohens_dz",
                "p_holm",
                "n_users",
            ]
        ],
        table_root / "actionability_gap_advantage.tex",
        "Paired Shapley Actionability Gap advantage over local baselines.",
        "tab:gap-advantage",
    )

    shapley_gap = gap.loc[gap["method"] == "shapley_mc"]
    significant_advantage = advantage.loc[advantage["p_holm"] <= 0.0010001]
    effect_min = float(advantage["cohens_dz"].abs().min())
    effect_max = float(advantage["cohens_dz"].abs().max())

    def primary_value(dataset: str, method: str, metric: str) -> float:
        row = summary.loc[
            (summary["dataset"] == dataset)
            & (summary["model"] == "itemknn")
            & (summary["analysis_role"] == "primary")
            & (summary["condition"] == "primary")
            & (summary["method"] == method)
            & (summary["metric"] == metric)
        ]
        return float(row.iloc[0]["mean"])

    # Publish all components, not just their difference. This prevents a
    # positive gap from being mistaken for positive absolute alignment.
    component = summary.loc[
        (summary["model"] == "itemknn")
        & (summary["utility"] == "target_margin")
        & summary["method"].isin(methods)
        & summary["metric"].isin(["faithfulness_alignment", "aia", "actionability_gap"])
    ].copy()
    component["condition_order"] = [condition_order.get((r.dataset, r.analysis_role, r.condition), 999) for r in component.itertuples()]
    component = component.loc[component["condition_order"] < 999].copy()
    component["condition_label"] = component["condition_order"].map(condition_labels)
    component["component"] = component["metric"].map({
        "faithfulness_alignment": "Deletion AIA",
        "aia": "Bounded AIA",
        "actionability_gap": "Gap (bounded - deletion)",
    })
    # Define the published gap from its two reported components. Do not use a
    # separately averaged user-level gap here: correlations are nonlinear, and
    # the review requires the displayed identity G = bounded AIA - deletion AIA.
    component_index = component.set_index(["dataset", "model", "evaluation_mode", "utility", "analysis_role", "condition", "method"])
    for idx in gap.index:
        key = tuple(gap.loc[idx, col] for col in ["dataset", "model", "evaluation_mode", "utility", "analysis_role", "condition", "method"])
        try:
            bounded = float(component_index.loc[key].loc[component_index.loc[key, "metric"] == "aia", "mean"].iloc[0])
            deletion = float(component_index.loc[key].loc[component_index.loc[key, "metric"] == "faithfulness_alignment", "mean"].iloc[0])
            derived_gap = bounded - deletion
            gap.loc[idx, "mean"] = derived_gap
            mask = (
                (component["dataset"] == key[0]) & (component["model"] == key[1])
                & (component["evaluation_mode"] == key[2]) & (component["utility"] == key[3])
                & (component["analysis_role"] == key[4]) & (component["condition"] == key[5])
                & (component["method"] == key[6]) & (component["metric"] == "actionability_gap")
            )
            component.loc[mask, "mean"] = derived_gap
        except (KeyError, IndexError):
            pass
    gap.to_csv(table_root / "actionability_gap_robustness.csv", index=False)
    write_tex(gap[["condition_label", "method_label", "mean", "ci95_low", "ci95_high", "n_users", "missing_users"]], table_root / "actionability_gap_robustness.tex", "Actionability Gap defined as bounded AIA minus deletion AIA; budgets excluded.", "tab:gap-robustness")
    component.to_csv(table_root / "aia_components.csv", index=False)
    write_tex(component[["condition_label", "method_label", "component", "mean", "ci95_low", "ci95_high", "n_users", "missing_users"]], table_root / "aia_components.tex", "Deletion AIA, bounded-intervention AIA, and their difference; all five methods.", "tab:aia-components")

    outcome_metrics = summary.loc[
        (summary["model"] == "itemknn") & (summary["utility"] == "target_margin")
        & summary["metric"].isin(["joint_effect_ndcg", "intervention_success_ndcg", "abstention", "normalized_regret_ndcg"])
    ].copy()
    outcome_metrics.to_csv(table_root / "intervention_outcomes.csv", index=False)
    write_tex(outcome_metrics[["dataset", "evaluation_mode", "condition", "method_label", "metric", "mean", "ci95_low", "ci95_high", "n_users", "missing_users"]], table_root / "intervention_outcomes.tex", "Joint intervention outcomes, including success, abstention, and conditional normalized regret.", "tab:intervention-outcomes")

    summary_markdown = f"""# ActionShap schema-v2 results summary

## Headline finding

Across **{len(shapley_gap)} distinct singleton target-margin ItemKNN conditions**
(budgets are excluded because they do not enter singleton AIA), Shapley's
bounded AIA changed relative to deletion. This change was not unique: the
random control was also positive in displayed conditions and greedy was positive
in some. The Actionability Gap is therefore a descriptive perturbation-
sensitivity statistic, not standalone evidence of explanation validity.

LOO is reported as the deletion oracle only. For every nonconstant valid user,
its deletion AIA is exactly one, so its gap cannot be positive. LOO is excluded
from Shapley gap-competitor claims and from the headline comparison count.

## Required component reporting

Every method and condition is reported with deletion AIA, bounded AIA, their
difference, valid-user counts, confidence intervals, and null-adjusted context in
`tables/aia_components.tex` and `tables/aia_permutation_null.tex`. Operational
action quality is reported separately in `tables/intervention_outcomes.tex`:
NDCG effect, success, harm/abstention, and normalized regret.

## Safe claim

> Under the declared bounded-downweighting policy, Shapley's target-margin
> alignment changed relative to deletion across the evaluated ItemKNN
> configurations. A positive change was not unique to Shapley: the random
> control also produced positive gaps, and greedy did so in several conditions.
> The gap must therefore be interpreted jointly with absolute bounded AIA,
> signed alignment, null-adjusted comparisons, and decision regret.
"""
    (table_root.parent / "RESULTS_SUMMARY.md").write_text(summary_markdown)

    if not gap.empty:
        # Component figure: a gap-only plot hides the fact that a random method
        # can have a positive difference while both component alignments are
        # uninformative. Report deletion, bounded, and difference together.
        component_lookup = component.copy()
        component_lookup["method_order"] = component_lookup["method"].map({m:i for i,m in enumerate(methods)})
        component_lookup = component_lookup.sort_values(["condition_order", "method_order"])
        labels = [condition_labels[i] for i in sorted(condition_labels)]
        fig, axes = plt.subplots(1, 3, figsize=(14.0, max(4.8, 0.55 * len(labels))), sharey=True)
        panel_specs = [("faithfulness_alignment", "Deletion AIA"), ("aia", "Bounded AIA"), ("actionability_gap", "Difference")]
        colors = {"shapley_mc":"#2F5597", "lime":"#E69F00", "loo":"#009E73", "greedy_cf":"#CC3311", "random":"#888888"}
        offsets = {m: (i-2)*0.13 for i,m in enumerate(methods)}
        for ax, (metric, title) in zip(axes, panel_specs):
            panel = component_lookup.loc[component_lookup["metric"] == metric]
            for method in methods:
                d = panel.loc[panel["method"] == method]
                if d.empty: continue
                y = d["condition_order"].to_numpy(float) + offsets[method]
                ax.errorbar(d["mean"], y, xerr=[np.maximum(0, d["mean"]-d["ci95_low"]), np.maximum(0, d["ci95_high"]-d["mean"])], fmt="o", capsize=2.5, color=colors[method], label=METHOD_LABELS[method])
            ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_title(title)
            ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
            ax.set_xlabel("Spearman correlation")
        axes[0].set_yticks(list(sorted(condition_labels)), labels)
        axes[0].invert_yaxis()
        axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.suptitle("AIA components and Actionability Gap (singleton estimand; budgets excluded)", y=1.01)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(figure_root / f"aia_components_robustness.{extension}", bbox_inches="tight")
            # Preserve the historical filename as a compatibility alias.
            fig.savefig(figure_root / f"actionability_gap_robustness.{extension}", bbox_inches="tight")
        plt.close(fig)
    return gap, advantage



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
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
    for keys, group in frame.groupby(["dataset", "model", "utility"], sort=True):
        group = group.sort_values("permutations")
        dataset, model, utility = (str(value) for value in keys)
        label = (
            f"{dataset.replace('Amazon-Digital-Music', 'Amazon').replace('MovieLens-1M', 'MovieLens')}"
            f" {model} ({utility})"
        )
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
        axis.set_xlabel(r"Base permutations $M_{\mathrm{pair}}$")
        axis.set_title(title)
    axes[0].set_ylabel("Agreement with independent M=1000 reference")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(figure_root / f"convergence.{extension}", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    # Support both documented short forms and long forms
    parser.add_argument(
        "--raw", "--raw-root",
        dest="raw_root",
        default="results/raw",
        help="Directory containing schema-v2 *.json result files"
    )
    parser.add_argument(
        "--out", "--paper-root",
        dest="paper_root",
        default="../paper",
        help="Paper root. Final assets written to <paper-root>/final/"
    )
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--null-draws", type=int, default=1000)
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    raw_root = (code_root / args.raw_root).resolve()

    # Robust resolution for --out paper/final (and variants)
    pr = args.paper_root.strip().rstrip("/")
    if pr.endswith(("paper/final", "/final", "paper")) or pr in ("paper/final", "paper", "../paper"):
        paper_root = code_root.parent / "paper"
    else:
        paper_root = (code_root / pr).resolve()

    if pr.endswith("/final") or pr.endswith("final"):
        final_root = paper_root if paper_root.name == "final" else (paper_root / "final")
        paper_root = final_root.parent
    else:
        final_root = paper_root / "final"

    paper_root = paper_root.resolve()
    final_root = final_root.resolve()
    repo = repository_root(code_root)
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

    metrics_export = metrics.drop_duplicates(
        subset=[
            "dataset", "model", "evaluation_mode", "utility", "analysis_role",
            "condition", "method", "user", "seed",
        ],
        keep="first",
    ).copy()
    pointwise_export_columns = [
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
    ]
    budget_mask = metrics_export["condition"].isin({"budget1", "budget3"})
    metrics_export.loc[budget_mask, pointwise_export_columns] = np.nan
    metrics_export.to_csv(
        data_root / "user_seed_metrics.csv.gz", index=False, compression="gzip"
    )
    pointwise_export_metrics = set(pointwise_export_columns)
    summary_export = summaries.loc[
        ~(
            summaries["condition"].isin({"budget1", "budget3"})
            & summaries["metric"].isin(pointwise_export_metrics)
        )
    ]
    tests_export = tests.loc[
        ~(
            tests["condition"].isin({"budget1", "budget3"})
            & tests["metric"].isin(pointwise_export_metrics)
        )
    ]
    summary_export.to_csv(table_root / "method_metrics.csv", index=False)
    tests_export.to_csv(table_root / "paired_tests.csv", index=False)
    stability_export = stability.loc[~stability["condition"].isin({"budget1", "budget3"})].copy()
    stability_export.to_csv(table_root / "attribution_stability.csv", index=False)
    aggregate_null.to_csv(table_root / "aia_permutation_null.csv", index=False)
    protocol.to_csv(table_root / "protocol_audit.csv", index=False)
    convergence_summary.to_csv(table_root / "convergence.csv", index=False)
    selected_metrics = [
        "aia",
        "aia_ndcg",
        "faithfulness_alignment",
        "actionability_gap",
        "joint_effect_ndcg",
        "intervention_success_ndcg",
        "abstention",
        "joint_regret_ndcg",
        "normalized_regret_ndcg",
        "normalized_regret_primary",
    ]
    selected_summary = summaries.loc[summaries["metric"].isin(selected_metrics)].copy()
    # Budget-one and budget-three conditions are joint-action sensitivities.
    # They do not create new singleton estimands, so never export AIA,
    # deletion-alignment, or gap rows for those conditions.
    budget_conditions = {"budget1", "budget3"}
    pointwise_metrics = {"aia", "aia_ndcg", "faithfulness_alignment", "actionability_gap"}
    selected_summary = selected_summary.loc[
        ~(
            selected_summary["condition"].isin(budget_conditions)
            & selected_summary["metric"].isin(pointwise_metrics)
        )
    ].copy()
    selected_summary = selected_summary[
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
    write_compact_convergence_table(
        convergence_summary,
        table_root / "convergence.tex",
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
    make_actionability_gap_assets(summaries, tests, table_root, figure_root)
    # Keep the publication PDF compact while retaining complete CSV/JSON
    # matrices for audit and supplementary analysis.  In particular, budget
    # sensitivities are decision-only and cannot leak singleton AIA rows.
    write_compact_aia_table(summaries, table_root / "aia_components.tex")
    write_compact_intervention_table(summaries, table_root / "intervention_outcomes.tex")
    write_compact_sensitivity_table(summaries, table_root / "sensitivity_results.tex")
    write_compact_gap_table(summaries, table_root / "actionability_gap_robustness.tex")
    write_compact_full_catalogue_table(summaries, table_root / "full_catalogue_results.tex")
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
        "Legacy pilot assets are not consumed. See `RESULTS_SUMMARY.md` for the "
        "validated claim boundary.\n"
    )
    manifest_path = manifest_root / "asset_manifest.json"
    manifest = {
        "schema_version": 2,
        "repository": {
            "url": "https://github.com/mouadlouhichi/next-paper",
            "visibility_required_before_submission": "public_or_mirrored",
            "manuscript_root": "paper-ideas/ActionShap/paper-v3",
            "source_commit": subprocess_git_commit(repo),
            "release_archive_sha256": "ac4c7fb1993458b6b41054974ebff215710e7a8b5894c7aa6af828e94b2a5b0f",
        },
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
