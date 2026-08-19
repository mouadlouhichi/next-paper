"""Phase D completion (reviewer revision): paired user-level inference for MovieLens-1M.

The archived MovieLens-1M evidence reports five-seed means and seed-level standard
deviations but no paired user-level confidence intervals or significance tests.
This script re-trains the frozen final BPR-MF and SASRec configurations for the
archived seeds 42-46, exports one row per evaluated warm user, and computes:

- paired bootstrap confidence intervals for mean Recall@10 and NDCG@10 differences;
- exact McNemar tests for hit outcomes (sign test on discordant user pairs);
- paired permutation tests for NDCG@10;
- Cohen d_z effect sizes;
- Holm correction across the model x metric family.

Claim scope: chronological warm-item ranking on MovieLens-1M. Nothing here is
causal policy evidence and nothing here validates CURE-Sim intervention selection.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cure_rec.data import load_dataset  # noqa: E402
from cure_rec.models import (  # noqa: E402
    PopularityRecommender,
    build_shared_candidates,
    chronological_leave_one_out,
)
from cure_rec.revision_suite import _holm, paired_user_statistics  # noqa: E402
from cure_rec.sasrec import SASRecConfig, TorchSASRec  # noqa: E402
from cure_rec.torch_models import TorchBPRConfig, TorchBPRMFWithBias, torch_available  # noqa: E402

ASSETS = ROOT / "results" / "reviewer_phase_assets" / "movielens_1m_paired"
DATA_ROOT = ROOT / "data" / "raw" / "ml-1m"
SEEDS = (42, 43, 44, 45, 46)
MAX_EVAL_USERS = 1_000

# Frozen final configurations (hashes 9875430c235b and 7aa511398a7f in the
# reproducibility snapshot). Training budgets match the archived final audits.
BPR_CONFIG = {"embedding_dim": 128, "batch_size": 8192, "learning_rate": 0.003, "weight_decay": 1e-05, "negative_strategy": "popularity_mixture"}
SASREC_CONFIG = {"embedding_dim": 128, "max_sequence_length": 25, "num_heads": 2, "num_layers": 2, "dropout": 0.1, "batch_size": 1024, "learning_rate": 0.0005, "weight_decay": 1e-05, "negative_strategy": "popularity_mixture"}


def export_user_metrics(model, split, k: int = 10) -> pd.DataFrame:
    cases, _cold = build_shared_candidates(split, use_validation=False, max_users=MAX_EVAL_USERS)
    prepare = getattr(model, "prepare_evaluation", None)
    finish = getattr(model, "finish_evaluation", None)
    if callable(prepare):
        prepare([case.user_id for case in cases])
    rows = []
    try:
        for case in cases:
            scores = model.score(case.user_id, case.candidate_items)
            order = np.lexsort((case.candidate_items, -scores))
            ranked = case.candidate_items[order[:k]]
            position = np.where(ranked == case.target_item)[0]
            hit = int(len(position) > 0)
            ndcg = float(1 / np.log2(int(position[0]) + 2)) if hit else 0.0
            rows.append({"user_id": case.user_id, "hit": hit, "ndcg": ndcg})
    finally:
        if callable(finish):
            finish()
    return pd.DataFrame(rows)


def permutation_p(differences: np.ndarray, permutations: int = 10_000, seed: int = 20260818) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(differences.mean())
    signs = rng.choice([-1.0, 1.0], size=(permutations, len(differences)))
    null_means = (signs * differences).mean(axis=1)
    return float((np.abs(null_means) >= observed).mean())


def main() -> None:
    if not torch_available():
        raise SystemExit("PyTorch is required for this revision script")
    ASSETS.mkdir(parents=True, exist_ok=True)
    started = time.time()

    dataset = load_dataset("movielens_1m", DATA_ROOT, download=True)
    split = chronological_leave_one_out(dataset.interactions)
    popularity = PopularityRecommender().fit(split.train)

    per_user_frames: list[pd.DataFrame] = []
    aggregate_rows: list[dict] = []

    pop_metrics = export_user_metrics(popularity, split)
    pop_metrics["model"] = "popularity"
    pop_metrics["seed"] = -1
    per_user_frames.append(pop_metrics)
    aggregate_rows.append({"model": "popularity", "seed": -1, "recall_at_10": float(pop_metrics["hit"].mean()), "ndcg_at_10": float(pop_metrics["ndcg"].mean()), "evaluated_users": len(pop_metrics)})
    print(f"popularity: recall={pop_metrics['hit'].mean():.5f} ndcg={pop_metrics['ndcg'].mean():.5f} ({time.time()-started:.0f}s)", flush=True)

    for seed in SEEDS:
        bpr = TorchBPRMFWithBias(TorchBPRConfig(max_epochs=200, seed=seed, **BPR_CONFIG))
        bpr.fit(split.train, validation_split=split, max_eval_users=MAX_EVAL_USERS)
        metrics = export_user_metrics(bpr, split)
        metrics["model"] = "torch_bpr_mf_bias"
        metrics["seed"] = seed
        per_user_frames.append(metrics)
        aggregate_rows.append({"model": "torch_bpr_mf_bias", "seed": seed, "recall_at_10": float(metrics["hit"].mean()), "ndcg_at_10": float(metrics["ndcg"].mean()), "evaluated_users": len(metrics), "restored_checkpoint_epoch": getattr(bpr, "restored_checkpoint_epoch", None)})
        print(f"BPR seed {seed}: recall={metrics['hit'].mean():.5f} ndcg={metrics['ndcg'].mean():.5f} epoch={getattr(bpr,'restored_checkpoint_epoch',None)} ({time.time()-started:.0f}s)", flush=True)
        del bpr

    for seed in SEEDS:
        sasrec = TorchSASRec(SASRecConfig(max_epochs=120, seed=seed, **SASREC_CONFIG))
        sasrec.fit(split.train, validation_split=split, max_eval_users=MAX_EVAL_USERS)
        metrics = export_user_metrics(sasrec, split)
        metrics["model"] = "torch_sasrec"
        metrics["seed"] = seed
        per_user_frames.append(metrics)
        aggregate_rows.append({"model": "torch_sasrec", "seed": seed, "recall_at_10": float(metrics["hit"].mean()), "ndcg_at_10": float(metrics["ndcg"].mean()), "evaluated_users": len(metrics), "restored_checkpoint_epoch": getattr(sasrec, "restored_checkpoint_epoch", None)})
        print(f"SASRec seed {seed}: recall={metrics['hit'].mean():.5f} ndcg={metrics['ndcg'].mean():.5f} epoch={getattr(sasrec,'restored_checkpoint_epoch',None)} ({time.time()-started:.0f}s)", flush=True)
        del sasrec

    per_user = pd.concat(per_user_frames, ignore_index=True)
    per_user.to_csv(ASSETS / "per_user_metrics_ml1m.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(ASSETS / "aggregate_seed_metrics_ml1m.csv", index=False)

    # ------------------------------------------------------------------
    # Paired inference. Seed 42 is the primary frozen audit seed; pooled
    # analysis averages each user's metric across the five seeds first, so the
    # user remains the independent unit and seeds are not pseudoreplicated.
    # ------------------------------------------------------------------
    rows_ci: list[dict] = []
    rows_tests: list[dict] = []

    def add_paired(tag: str, frame: pd.DataFrame) -> None:
        ci, tests = paired_user_statistics(frame)
        ci["analysis"] = tag
        tests["analysis"] = tag
        rows_ci.append(ci)
        rows_tests.append(tests)

    seed42 = per_user[(per_user["seed"].isin([-1, 42]))].drop(columns=["seed"])
    add_paired("seed_42", seed42)

    pooled = (
        per_user[per_user["seed"] >= 0]
        .groupby(["user_id", "model"], as_index=False)[["hit", "ndcg"]]
        .mean()
    )
    pooled = pd.concat([pop_metrics.assign(model="popularity"), pooled], ignore_index=True)
    add_paired("five_seed_user_mean", pooled)

    ci_table = pd.concat(rows_ci, ignore_index=True)
    tests_table = pd.concat(rows_tests, ignore_index=True)
    tests_table["holm_p"] = _holm(tests_table["raw_p"].to_numpy())

    # Permutation tests for NDCG differences (user-level paired differences).
    perm_rows = []
    for tag, frame in (("seed_42", seed42), ("five_seed_user_mean", pooled)):
        pivot = frame.pivot(index="user_id", columns="model", values="ndcg")
        for model in ("torch_bpr_mf_bias", "torch_sasrec"):
            diff = (pivot[model] - pivot["popularity"]).dropna().to_numpy(float)
            perm_rows.append({"analysis": tag, "model": model, "metric": "ndcg", "permutation_p": permutation_p(diff)})
    perm_table = pd.DataFrame(perm_rows)

    ci_table.to_csv(ASSETS / "paired_bootstrap_ci_ml1m.csv", index=False)
    tests_table.to_csv(ASSETS / "paired_tests_holm_ml1m.csv", index=False)
    perm_table.to_csv(ASSETS / "paired_permutation_tests_ml1m.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "movielens_1m",
        "positive_interaction_definition": "rating >= 4",
        "split": "chronological leave-one-out; users with >= 3 positives; last positive is test, previous is validation",
        "candidates": "warm training catalogue minus items seen in training; identical for all models",
        "evaluated_users": MAX_EVAL_USERS,
        "seeds": list(SEEDS),
        "frozen_configs": {"bpr": BPR_CONFIG, "sasrec": SASREC_CONFIG},
        "tests": {
            "hit": "exact McNemar test (sign test on discordant user pairs), Holm-adjusted",
            "ndcg": "paired bootstrap CI plus paired sign-flip permutation test",
        },
        "claim_scope": "chronological warm-item ranking robustness; not causal policy evidence",
        "runtime_seconds": time.time() - started,
    }
    (ASSETS / "revision_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("PHASE D ML-1M PAIRED STATS DONE", flush=True)
    print(ci_table.to_string(index=False), flush=True)
    print(tests_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
