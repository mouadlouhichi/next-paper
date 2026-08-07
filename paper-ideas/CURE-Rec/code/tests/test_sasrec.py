from __future__ import annotations

import numpy as np
import pandas as pd

from cure_rec.sasrec import SASRecConfig, TorchSASRec
from cure_rec.sasrec_search import _hash, run_final_sasrec_seed_replication


def test_sasrec_sequence_builder_is_chronological_and_config_is_validated():
    interactions = pd.DataFrame([
        {"user_id": 1, "item_id": 8, "response": 1, "timestamp": 30},
        {"user_id": 1, "item_id": 5, "response": 1, "timestamp": 10},
        {"user_id": 1, "item_id": 3, "response": 1, "timestamp": 20},
        {"user_id": 2, "item_id": 2, "response": 0, "timestamp": 10},
    ])
    assert TorchSASRec._ordered_sequences(interactions) == {1: [5, 3, 8]}
    try:
        SASRecConfig(embedding_dim=63, num_heads=2)
    except ValueError as error:
        assert "divisible" in str(error)
    else:  # pragma: no cover - explicit defensive assertion
        raise AssertionError("Invalid attention dimensions should fail")


def test_final_sasrec_replication_resumes_complete_frozen_artifacts(tmp_path):
    search_root = tmp_path / "search"
    search_root.mkdir()
    (search_root / "sasrec_search_manifest.json").write_text('{"best_config": {}, "search": {"final_epochs": 1}}')
    output = tmp_path / "replication"
    config_hash = _hash({})
    for seed, ndcg in ((42, 0.04), (43, 0.05)):
        seed_root = output / f"seed-{seed}"
        seed_root.mkdir(parents=True)
        pd.DataFrame([
            {"seed": seed, "model": "popularity", "selected_config_hash": config_hash, "recall_at_k": 0.05, "ndcg_at_k": 0.03},
            {"seed": seed, "model": "torch_sasrec_final", "selected_config_hash": config_hash, "recall_at_k": 0.06, "ndcg_at_k": ndcg},
        ]).to_csv(seed_root / "final_sasrec_test_metrics.csv", index=False)
    paired = run_final_sasrec_seed_replication(None, search_root, output, seeds=(42, 43))
    assert paired["seed"].tolist() == [42, 43]
    assert paired["delta_recall_at_k"].round(8).tolist() == [0.01, 0.01]
    assert paired["delta_ndcg_at_k"].round(8).tolist() == [0.01, 0.02]
    assert (output / "final_sasrec_seed_summary.csv").exists()


def test_sasrec_prefixes_are_right_padded_for_causal_attention():
    model = TorchSASRec(SASRecConfig(max_sequence_length=5, embedding_dim=64, num_heads=2))
    model.examples = [(7, 2)]
    model.sequence_values = {7: np.asarray([11, 12, 13], dtype=np.int64)}
    sequences, positive, users = model._pad_prefixes(np.asarray([0]))
    assert sequences.tolist() == [[11, 12, 0, 0, 0]]
    assert positive.tolist() == [13]
    assert users.tolist() == [7]
    assert model._sequence_for_user(7).tolist() == [[11, 12, 13, 0, 0]]
