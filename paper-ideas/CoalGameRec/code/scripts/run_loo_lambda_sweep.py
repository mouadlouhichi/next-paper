#!/usr/bin/env python3
"""LOO reranking-strength sweep + validation-tuned lambda comparison (round 6).

Closes the two lambda-related review gaps:

1. LOO lambda sweep. The released v3 lambda-sensitivity artifact contains
   uniform / additive-pref / shapley-mc but NOT loo-marginal. This script
   produces the missing LOO curve (and re-emits the other families so all
   curves come from one consistent re-execution), lambda in
   {0.00, 0.05, 0.10, 0.20, 0.40}, five seeds, both datasets, TEST metrics.

2. Validation-tuned lambda (fair per-method tuning). For each family the
   reranking strength is selected independently by VALIDATION NDCG@20
   (ranking the held-out validation item with training items masked, i.e. the
   same full-catalog protocol applied one step earlier in time), and TEST
   NDCG@20 at the selected lambda is reported. Nothing from the test split is
   used for selection, so this is the "independently tuned" comparison the
   reviewers requested, alongside the frozen shared-lambda=0.10 protocol.

Families: uniform, additive-pref, loo-marginal (+ shapley-mc if
C1_WITH_SHAPLEY=1). Frozen protocol otherwise (archived splits, seeds 42-46,
k=24, 100 validation negatives, native intervention).

Usage:
  COALGAME_DEVICE=mps python scripts/run_loo_lambda_sweep.py --dataset ml1m \
      --source-run results/journal_runs/ml1m_lightgcn_v3_prospective \
      --out results/journal_runs/ml1m_lightgcn_v6_lambda_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_matched_controls as rmc  # noqa: E402
import run_second_backbone as rsb   # noqa: E402  (shared-prop trainer)
from coalgamerec.attribution import compute_attribution_for_users, compute_shapley_for_users  # noqa: E402
from coalgamerec.data import item_user_vectors  # noqa: E402
from coalgamerec.metrics import evaluate  # noqa: E402
from coalgamerec.models import LightGCN, TrainConfig, build_lightgcn_graph, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all  # noqa: E402
from coalgamerec.utils import sparse_fingerprint, write_json  # noqa: E402

LAMBDAS = [0.00, 0.05, 0.10, 0.20, 0.40]
BACKBONE = rmc.BACKBONE
ATTR = rmc.ATTR
TAU_ATT = rmc.TAU_ATT
KS = rmc.KS


def run_seed(split, item_vectors, seed: int, out_dir: Path) -> None:
    seed_dir = out_dir / "raw" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if (seed_dir / "lambda_sensitivity.csv").exists():
        print(f"seed {seed} already complete, skipping", flush=True)
        return
    stages = {}

    t = time.time()
    train_cfg = TrainConfig(dim=BACKBONE["dim"], lr=BACKBONE["lr"], weight_decay=BACKBONE["weight_decay"],
                            epochs=BACKBONE["epochs"], batch_size=BACKBONE["batch_size"],
                            n_neg=BACKBONE["n_neg"], seed=seed, device=os.environ.get("COALGAME_DEVICE", "cpu"))
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    edge_index, edge_weight = build_lightgcn_graph(split.train, split.n_users, split.n_items, device)
    model = LightGCN(split.n_users, split.n_items, edge_index, edge_weight,
                     dim=BACKBONE["dim"], n_layers=BACKBONE["n_layers"]).to(device)
    model = rsb.train_shared_prop(model, split.train, split.n_users, split.n_items, train_cfg)
    base_scores = cache_full_scores(model, split.n_users, batch_size=BACKBONE["score_batch_size"],
                                    chunk_items=4096 if split.n_items > 8000 else None)
    item_embeddings = get_item_embeddings(model)
    stages["train_and_score_seconds"] = time.time() - t

    t = time.time()
    loo = compute_attribution_for_users(
        split, base_scores, item_vectors, method="loo-marginal",
        max_users=None, exact_threshold=8, seed=seed,
        max_players_per_user=ATTR["max_players_per_user"],
        player_selection=ATTR["player_selection"],
        checkpoint_path=None, save_every=25,
        alpha=1.0, beta=0.0, lambda_pref=0.0,
        lambda_attr_value=ATTR["lambda_attr_value"],
        value_mode=ATTR["value_mode"], n_val_negatives=ATTR["n_val_negatives"])
    stages["loo_seconds"] = time.time() - t

    shap = None
    if os.environ.get("C1_WITH_SHAPLEY") == "1":
        t = time.time()
        shap = compute_shapley_for_users(
            split, base_scores, item_vectors, max_users=None, m=64, exact_threshold=8, seed=seed,
            max_players_per_user=ATTR["max_players_per_user"], player_selection=ATTR["player_selection"],
            checkpoint_path=None, save_every=25, alpha=1.0, beta=0.0, lambda_pref=0.0,
            lambda_attr_value=ATTR["lambda_attr_value"], value_mode=ATTR["value_mode"],
            n_val_negatives=ATTR["n_val_negatives"], antithetic=True)
        stages["shapley_seconds"] = time.time() - t

    # validation-target split view: rank the held-out VALIDATION item with the
    # same full-catalog protocol (training items masked). Used only to SELECT
    # per-family lambda; test evaluation remains the primary reported metric.
    val_split = replace(split, test=split.val)

    fam_weights = {"uniform": None, "additive-pref": None, "loo-marginal": loo}
    if shap is not None:
        fam_weights["shapley-mc"] = shap

    t = time.time()
    rows = []
    for fam, attr in fam_weights.items():
        loo_arg = attr if fam == "loo-marginal" else None
        shap_arg = attr if fam == "shapley-mc" else None
        for lam in LAMBDAS:
            scores = rerank_all(base_scores, split, item_vectors, fam,
                                shapley_by_user=shap_arg, loo_by_user=loo_arg,
                                lambda_attr=lam, tau_att=TAU_ATT,
                                intervention="native", item_embeddings=item_embeddings)
            test_summary, _ = evaluate(scores, split, item_vectors, ks=KS)
            val_summary, _ = evaluate(scores, val_split, item_vectors, ks=KS)
            rows.append({
                "HitRate@5": test_summary["HitRate@5"], "NDCG@5": test_summary["NDCG@5"],
                "HitRate@10": test_summary["HitRate@10"], "NDCG@10": test_summary["NDCG@10"],
                "HitRate@20": test_summary["HitRate@20"], "NDCG@20": test_summary["NDCG@20"],
                "Coverage@20": test_summary["Coverage@20"], "ILD@20": test_summary["ILD@20"],
                "ValNDCG@20": val_summary["NDCG@20"], "ValHitRate@20": val_summary["HitRate@20"],
                "seed": seed, "family": fam, "lambda_attr": lam,
                "backbone": "lightgcn", "dataset": split.name})
    stages["rerank_eval_sensitivity_seconds"] = time.time() - t
    pd.DataFrame(rows).to_csv(seed_dir / "lambda_sensitivity.csv", index=False)
    write_json(seed_dir / "runtime.json", {"seed": seed, "stages": stages})
    print(f"seed {seed} done: " + ", ".join(f"{k}={v:.0f}s" for k, v in stages.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    source_run = Path(args.source_run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = rmc.load_split_from_run(source_run)
    item_vectors = item_user_vectors(split.train_csr)

    src_report = json.loads((rmc.resolve_source_dir(source_run) / "item_vectors_report.json").read_text())
    fp = sparse_fingerprint(item_vectors)
    shape_nnz_match = ("shape" in src_report and "nnz" in src_report
                       and list(item_vectors.shape) == src_report["shape"]
                       and int(item_vectors.nnz) == int(src_report["nnz"]))
    assert shape_nnz_match or fp == src_report.get("hash"), "item-vector integrity check failed"

    t0 = time.time()
    for seed in args.seeds:
        run_seed(split, item_vectors, seed, out_dir)

    frames = [pd.read_csv(out_dir / "raw" / f"seed_{s}" / "lambda_sensitivity.csv") for s in args.seeds]
    ldf = pd.concat(frames, ignore_index=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    ldf.to_csv(out_dir / "tables" / "lambda_sensitivity_all.csv", index=False)
    g = ldf.groupby(["family", "lambda_attr"])[["NDCG@20", "HitRate@20", "ValNDCG@20"]].agg(["mean", "std"])
    g.to_csv(out_dir / "tables" / "lambda_sensitivity_mean_std.csv")

    # validation-tuned lambda: select argmax of mean VALIDATION NDCG@20 per
    # family, then report TEST NDCG@20 at the selected lambda (per seed).
    tuned_rows = []
    for fam, gf in ldf.groupby("family"):
        val_means = gf.groupby("lambda_attr")["ValNDCG@20"].mean()
        lam_star = float(val_means.idxmax())
        sub = gf[gf.lambda_attr == lam_star]
        tuned_rows.append({
            "family": fam, "lambda_selected_on_validation": lam_star,
            "test_ndcg20_mean": float(sub["NDCG@20"].mean()),
            "test_ndcg20_std": float(sub["NDCG@20"].std(ddof=1)),
            "test_hr20_mean": float(sub["HitRate@20"].mean()),
            "protocol_ndcg20_mean": float(gf[gf.lambda_attr == 0.10]["NDCG@20"].mean()),
        })
    pd.DataFrame(tuned_rows).to_csv(out_dir / "tables" / "validation_tuned_lambda.csv", index=False)

    write_json(out_dir / "manifest.json", {
        "note": "Round-6 lambda sweep incl. LOO + validation-tuned lambda selection.",
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "device": os.environ.get("COALGAME_DEVICE", "cpu"),
        "lambdas": LAMBDAS, "seeds": args.seeds, "total_seconds": time.time() - t0,
    })
    print(f"ALL DONE {args.dataset} in {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
