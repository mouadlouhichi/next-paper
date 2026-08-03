"""Protocol-level regression tests for issues found in the pilot experiment."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from actionshap.baselines import (
    greedy_counterfactual_attribution,
    leave_one_out,
    lime_attribution,
)
from actionshap.convergence import convergence_table, minimum_usable_permutations
from actionshap.evaluation import (
    aia,
    aia_null_summary,
    direction_accuracy,
    signed_alignment,
    topk_intervention_precision,
)
from actionshap.recommendation import mc_shapley, select_downweight_action
from actionshap.recommendation_data import (
    load_interactions_csv,
    sample_evaluation_users,
)


def test_temporal_csv_split_uses_stable_original_row_tie_break(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "user,item,timestamp,rating\n"
        "u1,i1,1,5\n"
        "u1,i2,2,5\n"
        "u1,i3,2,5\n"
        "u1,i4,2,5\n"
        "u2,i2,1,5\n"
        "u2,i3,2,5\n"
        "u2,i4,3,5\n"
        "u2,i5,4,5\n"
    )
    first = load_interactions_csv(path, minimum_interactions=4)
    second = load_interactions_csv(path, minimum_interactions=4)
    assert first.test == second.test
    assert first.validation == second.validation
    assert all(len(history) == 2 for history in first.train.values())
    # Complete candidate exclusion includes the validation event.
    assert all(len(first.seen_before_test(user)) == 3 for user in first.test)


def test_dataset_downloader_extracts_and_hashes_movielens_atomically(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "download_datasets.py"
    spec = importlib.util.spec_from_file_location("actionshap_download_datasets", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ratings = b"1::10::5::100\n2::11::4::101\n"
    archive = tmp_path / "ml-1m.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ml-1m/ratings.dat", ratings)
    expected = hashlib.sha256(ratings).hexdigest()
    code_root = tmp_path / "code"
    provenance = module.prepare_movielens(
        code_root, archive.as_uri(), expected, force=True
    )
    output = code_root / "data" / "ml-1m" / "ratings.dat"
    assert output.read_bytes() == ratings
    assert provenance["output_sha256"] == expected
    assert json.loads(output.with_suffix(".provenance.json").read_text())[
        "output_sha256"
    ] == expected
    assert not list(output.parent.glob("*.part"))

    fallback_source = tmp_path / "fallback.bin"
    fallback_source.write_bytes(b"fallback-payload")
    destination = tmp_path / "downloaded.bin"
    downloaded, successful_url = module.download_from_sources(
        [
            (tmp_path / "missing.bin").as_uri(),
            fallback_source.as_uri(),
        ],
        destination,
        attempts=1,
        timeout=1,
    )
    assert downloaded.read_bytes() == b"fallback-payload"
    assert successful_url == fallback_source.as_uri()

    if module.shutil.which("curl") is not None:
        curl_destination = tmp_path / "curl-downloaded.bin"
        module._download_with_curl(
            fallback_source.as_uri(), curl_destination, timeout=1
        )
        assert curl_destination.read_bytes() == b"fallback-payload"


def test_amazon_builder_writes_deterministic_provenance(tmp_path):
    source = tmp_path / "Digital_Music_5.json.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        for user in range(6):
            for item in range(6):
                stream.write(
                    json.dumps(
                        {
                            "reviewerID": f"u{user}",
                            "asin": f"i{item}",
                            "overall": 5,
                            "unixReviewTime": 100 + item,
                        }
                    )
                    + "\n"
                )
    output = tmp_path / "interactions.csv"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_amazon_digital_music.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--output",
            str(output),
            "--core",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())
    assert provenance["users"] == 6
    assert provenance["items"] == 6
    assert provenance["interactions"] == 36
    assert len(provenance["source_sha256"]) == 64
    assert len(provenance["output_sha256"]) == 64
    data = load_interactions_csv(output, minimum_interactions=4)
    assert len(data.test) == 6


def test_method_summary_counts_users_missing_under_every_seed():
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_paper_assets.py"
    spec = importlib.util.spec_from_file_location("actionshap_make_assets", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = []
    for user in (1, 2):
        for seed in (42, 43):
            row = {
                "dataset": "toy",
                "model": "itemknn",
                "evaluation_mode": "sampled",
                "utility": "target_margin",
                "analysis_role": "primary",
                "condition": "primary",
                "method": "shapley_mc",
                "method_label": "Monte Carlo Shapley",
                "user": user,
                "seed": seed,
            }
            for metric in module.PRIMARY_METRICS + [
                "aia_null_mean",
                "aia_null_p95",
                "aia_permutation_p",
            ]:
                row[metric] = np.nan
            if user == 1:
                row["aia"] = 0.5 + 0.01 * (seed - 42)
            rows.append(row)
    summary = module.method_summaries(pd.DataFrame(rows), draws=25)
    aia_row = summary.loc[summary["metric"] == "aia"].iloc[0]
    assert aia_row["n_users"] == 1
    assert aia_row["missing_users"] == 1


def test_canonical_manuscript_passes_static_integrity_checks():
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_manuscript.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["status"] in {"PASS", "PASS_WITH_WARNINGS"}
    assert report["errors"] == []


def test_user_sampling_is_random_seeded_not_first_ids(tmp_path):
    rows = ["user,item,timestamp"]
    for user in range(20):
        for event in range(4):
            rows.append(f"{user},{user * 10 + event},{event}")
    path = tmp_path / "events.csv"
    path.write_text("\n".join(rows) + "\n")
    data = load_interactions_csv(path, minimum_interactions=4)
    selected = sample_evaluation_users(data, max_users=5, seed=9)
    larger = sample_evaluation_users(data, max_users=10, seed=9)
    matched = sample_evaluation_users(data, max_users=5, seed=9, pool_size=10)
    assert selected == sample_evaluation_users(data, max_users=5, seed=9)
    assert selected != [0, 1, 2, 3, 4]
    assert set(matched) <= set(larger)


def test_loo_is_exact_only_for_single_player_deletion():
    weights = np.array([0.2, -0.5, 1.0])

    def additive(coalition):
        return float(weights[list(coalition)].sum()) if coalition else 0.0

    loo = leave_one_out(additive, 3)
    deletion_effect = -loo
    assert aia(loo, deletion_effect) == pytest.approx(1.0)


def test_signed_metrics_distinguish_magnitude_from_action_direction():
    attribution = np.array([3.0, -2.0, 1.0])
    effects = np.array([-1.0, 0.8, -0.2])
    assert aia(attribution, effects) == pytest.approx(1.0)
    assert signed_alignment(attribution, effects) == pytest.approx(1.0)
    assert direction_accuracy(attribution, effects) == pytest.approx(1.0)
    assert topk_intervention_precision(attribution, effects, 1) == pytest.approx(1.0)


def test_within_user_null_reports_finite_resolution_p_value():
    attribution = np.arange(12, dtype=float)
    effects = np.arange(12, dtype=float)
    summary = aia_null_summary(attribution, effects, draws=199, seed=3)
    assert summary["null_mean"] == pytest.approx(0.0, abs=0.08)
    assert summary["null_p95"] is not None
    assert 0.0 < summary["p_value"] <= 1.0
    assert summary["p_value"] >= 1 / 200


def test_greedy_counterfactual_recomputes_and_uses_signed_convention():
    # Removing player 0 produces the largest benefit. Returned phi is negative
    # so -phi predicts a beneficial downweight/removal.
    def utility(coalition):
        players = set(coalition)
        return float(-2 * (0 in players) + 1 * (1 in players) + 0.5 * (2 in players))

    attribution = greedy_counterfactual_attribution(utility, 3)
    assert attribution[0] < 0
    assert attribution[1] > 0


def test_locally_weighted_lime_recovers_additive_game():
    truth = np.array([0.2, -0.4, 0.8, 0.1])

    def utility(coalition):
        return float(truth[list(coalition)].sum()) if coalition else 0.0

    estimate = lime_attribution(
        utility,
        n_players=4,
        samples=256,
        seed=5,
        ridge_alpha=1e-8,
        kernel_width=0.5,
    )
    assert np.allclose(estimate, truth, atol=1e-4)


def test_synthetic_redundancy_separates_shapley_from_loo_at_budget_two():
    # Players 0 and 1 are redundant harmful signals: removing either alone has
    # no benefit, but removing both is decisive. LOO misses the pair; Shapley
    # allocates their coalition harm and identifies it.
    def utility(coalition):
        players = set(coalition)
        return float(
            (-10.0 if (0 in players or 1 in players) else 0.0)
            - (1.0 if 2 in players else 0.0)
            - (0.5 if 3 in players else 0.0)
        )

    shapley, _ = mc_shapley(utility, 4, permutations=200, seed=7)
    loo = leave_one_out(utility, 4)
    assert set(select_downweight_action(shapley, 2)) == {0, 1}
    assert set(select_downweight_action(loo, 2)) == {2, 3}
    full = utility(frozenset(range(4)))
    shapley_effect = utility(frozenset({2, 3})) - full
    loo_effect = utility(frozenset({0, 1})) - full
    assert shapley_effect == pytest.approx(10.0)
    assert loo_effect == pytest.approx(1.5)


def test_convergence_uses_action_agreement_not_efficiency_only():
    def utility(coalition):
        players = set(coalition)
        return float(sum((index + 1) for index in players) + 2 * ({0, 1} <= players))

    rows = convergence_table(
        utility,
        n_players=4,
        budgets=(5, 20),
        seeds=(0, 1, 2),
        reference=100,
    )
    assert all("mean_top1_agreement" in row for row in rows)
    assert all(row["mean_efficiency_error"] < 1e-12 for row in rows)
    selected = minimum_usable_permutations(
        rows, correlation_threshold=0.0, action_threshold=0.0
    )
    assert selected == 5


def test_mc_shapley_caches_repeated_coalitions():
    calls = 0

    def utility(coalition):
        nonlocal calls
        calls += 1
        return float(len(coalition))

    mc_shapley(utility, n_players=3, permutations=100, seed=2)
    assert calls <= 2**3
