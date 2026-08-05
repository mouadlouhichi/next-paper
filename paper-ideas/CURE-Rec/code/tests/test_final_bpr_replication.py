from __future__ import annotations

import pandas as pd

from cure_rec.search import _hash, run_final_bpr_seed_replication


def test_final_bpr_replication_resumes_existing_seed_artifacts(tmp_path):
    search_root = tmp_path / "search"
    search_root.mkdir()
    (search_root / "bpr_search_manifest.json").write_text('{"best_config": {}, "search": {"final_epochs": 1}}')
    output = tmp_path / "replication"
    config_hash = _hash({})
    for seed, bpr_ndcg in [(42, 0.04), (43, 0.05)]:
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir(parents=True)
        pd.DataFrame([
            {"seed": seed, "model": "popularity", "selected_config_hash": config_hash, "recall_at_k": 0.05, "ndcg_at_k": 0.03},
            {"seed": seed, "model": "torch_bpr_mf_bias_final", "selected_config_hash": config_hash, "recall_at_k": 0.06, "ndcg_at_k": bpr_ndcg},
        ]).to_csv(seed_dir / "final_bpr_test_metrics.csv", index=False)

    paired = run_final_bpr_seed_replication(None, search_root, output, seeds=(42, 43))
    assert list(paired["seed"]) == [42, 43]
    assert paired["delta_recall_at_k"].round(8).tolist() == [0.01, 0.01]
    assert paired["delta_ndcg_at_k"].round(8).tolist() == [0.01, 0.02]
    assert (output / "final_bpr_seed_paired_metrics.csv").exists()
    assert (output / "final_bpr_seed_summary.csv").exists()


def test_final_bpr_audit_preserves_final_model_label(tmp_path, monkeypatch):
    """Metric dictionaries must not overwrite the explicit final-model label."""
    import cure_rec.search as search
    from cure_rec.models import RankingMetrics

    search_root = tmp_path / "search"
    search_root.mkdir()
    (search_root / "bpr_search_manifest.json").write_text(
        '{"best_config": {}, "search": {"final_epochs": 1}}'
    )

    class DummyBPR:
        name = "torch_bpr_mf_bias"
        best_validation_epoch = 1
        restored_checkpoint_epoch = 1
        loss_history = [{"epoch": 1, "bpr_loss": 0.1}]
        validation_history = [{"epoch": 1, "ndcg_at_k": 0.1}]

        def __init__(self, config):
            self.config = config

        def fit(self, train, **kwargs):
            return self

    class DummyPopularity:
        name = "popularity"

        def fit(self, train):
            return self

    def fake_evaluate(model, split, **kwargs):
        return RankingMetrics(
            model=model.name,
            evaluated_users=1,
            candidate_coverage=1.0,
            cold_test_items=0,
            recall_at_k=0.1,
            ndcg_at_k=0.1,
            hit_rate_at_k=0.1,
        )

    monkeypatch.setattr(search, "torch_available", lambda: True)
    monkeypatch.setattr(search, "TorchBPRMFWithBias", DummyBPR)
    monkeypatch.setattr(search, "PopularityRecommender", DummyPopularity)
    monkeypatch.setattr(search, "evaluate_leave_one_out", fake_evaluate)
    monkeypatch.setattr(search, "audit_evaluation", lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()))

    result = search.run_final_bpr_audit(None, search_root, tmp_path / "audit")
    assert set(result["model"]) == {"popularity", "torch_bpr_mf_bias_final"}
    assert (tmp_path / "audit" / "final_bpr_test_metrics.csv").exists()
