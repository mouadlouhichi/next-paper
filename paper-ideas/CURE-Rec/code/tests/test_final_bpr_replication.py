from __future__ import annotations

import pandas as pd

from cure_rec.search import run_final_bpr_seed_replication


def test_final_bpr_replication_resumes_existing_seed_artifacts(tmp_path):
    search_root = tmp_path / "search"
    search_root.mkdir()
    (search_root / "bpr_search_manifest.json").write_text('{"best_config": {}, "search": {"final_epochs": 1}}')
    output = tmp_path / "replication"
    for seed, bpr_ndcg in [(42, 0.04), (43, 0.05)]:
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir(parents=True)
        pd.DataFrame([
            {"seed": seed, "model": "popularity", "recall_at_k": 0.05, "ndcg_at_k": 0.03},
            {"seed": seed, "model": "torch_bpr_mf_bias_final", "recall_at_k": 0.06, "ndcg_at_k": bpr_ndcg},
        ]).to_csv(seed_dir / "final_bpr_test_metrics.csv", index=False)

    paired = run_final_bpr_seed_replication(None, search_root, output, seeds=(42, 43))
    assert list(paired["seed"]) == [42, 43]
    assert paired["delta_recall_at_k"].round(8).tolist() == [0.01, 0.01]
    assert paired["delta_ndcg_at_k"].round(8).tolist() == [0.01, 0.02]
    assert (output / "final_bpr_seed_paired_metrics.csv").exists()
    assert (output / "final_bpr_seed_summary.csv").exists()
