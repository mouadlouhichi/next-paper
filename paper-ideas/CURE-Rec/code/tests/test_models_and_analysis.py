from __future__ import annotations

import pandas as pd

from cure_rec.analysis import analyze_dataset
from cure_rec.data import DatasetLoadResult
from cure_rec.models import BPRMFRecommender, PopularityRecommender, chronological_leave_one_out, evaluate_leave_one_out


def _temporal_fixture() -> pd.DataFrame:
    rows = []
    for user in range(4):
        for offset, item in enumerate([0, 1, 2, 3]):
            rows.append({
                "user_id": user,
                "item_id": (item + user) % 6,
                "rating": 5,
                "response": 1,
                "timestamp": 100 + offset,
                "split": "observed",
                "source_dataset": "fixture",
            })
    return pd.DataFrame(rows)


def test_cpu_baselines_fit_and_evaluate():
    split = chronological_leave_one_out(_temporal_fixture())
    popularity = PopularityRecommender().fit(split.train)
    popularity_metrics = evaluate_leave_one_out(popularity, split, k=3)
    assert popularity_metrics.evaluated_users == 3
    assert popularity_metrics.cold_test_items == 1
    assert popularity_metrics.candidate_coverage == 0.75

    bpr = BPRMFRecommender(factors=4, max_updates=100, seed=7).fit(split.train)
    bpr_metrics = evaluate_leave_one_out(bpr, split, k=3)
    assert bpr_metrics.evaluated_users == 3
    assert bpr_metrics.candidate_coverage == popularity_metrics.candidate_coverage


def test_external_data_analysis_generates_assets(tmp_path):
    result = DatasetLoadResult("fixture", _temporal_fixture(), {"has_exposure_log": False})
    analysis = analyze_dataset(result, output_root=tmp_path, run_bpr=True, bpr_updates=100, max_eval_users=4)
    assert (analysis.run_dir / "tables" / "data_table_summary.csv").exists()
    assert (analysis.run_dir / "tables" / "data_table_model_metrics.csv").exists()
    assert (analysis.run_dir / "figures" / "data_figure_activity_distributions.png").exists()
    models = set(analysis.model_metrics["model"])
    assert "popularity" in models
    assert "bpr_mf_numpy" in models
    assert any(name.startswith("bpr_popularity_hybrid_alpha_") for name in models)
