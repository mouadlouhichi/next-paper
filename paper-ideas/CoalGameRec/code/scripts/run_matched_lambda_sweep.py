#!/usr/bin/env python3
"""Round-9 REQUIRED FIX #4: matched single-execution lambda sweep.

All compared families (uniform, additive-pref, attention, heuristic-pop,
valid-sim, valid-linear, loo-marginal, shapley-mc) are reranked on the SAME
fitted models, SAME cached base scores, SAME player sets and SAME lambda grid
in one execution per seed, so cross-family comparisons are within-model by
construction (unlike the mixed v3/v6 sweep in tab:ablation_lambda).

Attributions are computed once per seed (lambda only affects reranking), so
the sweep costs one Shapley + one LOO pass plus cheap rerank/eval passes.
Per-user metrics are emitted for every (family, lambda) so paired per-user
contrasts at each lambda can be computed downstream.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_matched_controls as rmc  # noqa: E402
from coalgamerec.attribution import compute_attribution_for_users  # noqa: E402
from coalgamerec.metrics import evaluate  # noqa: E402
from coalgamerec.models import TrainConfig, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all, valid_sim_scores_all, valid_linear_scores_all  # noqa: E402
from coalgamerec.utils import write_json  # noqa: E402

LAMBDAS = [0.00, 0.05, 0.10, 0.20, 0.40]
KS = (5, 10, 20)
FAMILIES = ["uniform", "additive-pref", "attention", "heuristic-pop",
            "valid-sim", "valid-linear", "loo-marginal", "shapley-mc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    split = rmc.load_split_from_run(Path(args.source_run))
    item_vectors = rmc.item_user_vectors(split.train_csr)
    val_by_user = dict(zip(split.val.user.astype(int), split.val.item.astype(int)))
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("COALGAME_THREADS", "2"))))

    for seed in args.seeds:
        seed_dir = out_dir / "raw" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        if (seed_dir / "lambda_sensitivity.csv").exists():
            print(f"seed {seed} already complete, skipping")
            continue
        torch.manual_seed(seed)
        cfg = TrainConfig(dim=64, lr=0.002, weight_decay=1e-5, epochs=15, batch_size=4096,
                          n_neg=2, seed=seed, device=str(device))
        t0 = time.time()
        model = rmc.train_lightgcn_shared_prop(split.train, split.n_users, split.n_items, cfg, n_layers=2)
        base_scores = cache_full_scores(model, split.n_users, batch_size=256,
                                        chunk_items=4096 if split.n_items > 8000 else None)
        item_embeddings = get_item_embeddings(model)
        print(f"seed {seed}: trained in {time.time()-t0:.0f}s")

        t0 = time.time()
        loo = compute_attribution_for_users(
            split, base_scores, item_vectors, method="loo-marginal", seed=seed,
            max_players_per_user=24, player_selection="stratified",
            lambda_attr_value=0.10, value_mode="pairwise_logsigmoid", n_val_negatives=100)
        shap = compute_attribution_for_users(
            split, base_scores, item_vectors, method="shapley-mc", m=64, seed=seed,
            max_players_per_user=24, player_selection="stratified", antithetic=True,
            lambda_attr_value=0.10, value_mode="pairwise_logsigmoid", n_val_negatives=100)
        print(f"seed {seed}: attributions in {time.time()-t0:.0f}s")

        rows, per_user_rows = [], []
        for fam in FAMILIES:
            for lam in LAMBDAS:
                if fam == "valid-sim":
                    s = valid_sim_scores_all(base_scores, split, val_by_user, item_embeddings, lam)
                elif fam == "valid-linear":
                    s = valid_linear_scores_all(base_scores, split, val_by_user, item_embeddings, lam)
                else:
                    s = rerank_all(base_scores, split, item_vectors, fam,
                                   shapley_by_user=(shap if fam == "shapley-mc" else None),
                                   loo_by_user=(loo if fam == "loo-marginal" else None),
                                   lambda_attr=lam, intervention="native",
                                   item_embeddings=item_embeddings)
                summary, per_user = evaluate(s, split, item_vectors, ks=KS)
                summary.update(seed=seed, family=fam, lambda_attr=lam,
                               backbone="lightgcn", dataset=split.name)
                rows.append(summary)
                for metric in ("NDCG@20", "HitRate@20"):
                    per_user_rows.append(pd.DataFrame({
                        "seed": seed, "family": fam, "lambda_attr": lam, "metric": metric,
                        "user": np.arange(len(per_user[metric])), "value": per_user[metric]}))
                print(f"  {fam} lam={lam}: NDCG@20={summary['NDCG@20']:.5f}")
        pd.DataFrame(rows).to_csv(seed_dir / "lambda_sensitivity.csv", index=False)
        pd.concat(per_user_rows, ignore_index=True).to_csv(
            seed_dir / "per_user_lambda.csv.gz", index=False, compression="gzip")

    # aggregate
    frames = [pd.read_csv(out_dir / "raw" / f"seed_{s}" / "lambda_sensitivity.csv") for s in args.seeds]
    ldf = pd.concat(frames, ignore_index=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    ldf.to_csv(out_dir / "tables" / "lambda_sensitivity_all.csv", index=False)
    ldf.groupby(["family", "lambda_attr"])[["NDCG@20", "HitRate@20"]].agg(["mean", "std"]) \
       .to_csv(out_dir / "tables" / "lambda_sensitivity_mean_std.csv")
    pu = pd.concat([pd.read_csv(out_dir / "raw" / f"seed_{s}" / "per_user_lambda.csv.gz")
                    for s in args.seeds], ignore_index=True)
    pu.to_csv(out_dir / "raw" / "per_user_lambda_all.csv.gz", index=False, compression="gzip")
    write_json(out_dir / "manifest.json", {
        "note": "Round-9 fix #4: matched single-execution lambda sweep — every family reranked "
                "on the same fitted models, cached scores, player sets, and lambda grid; "
                "per-user metrics included for paired per-lambda contrasts.",
        "seeds": args.seeds, "lambda_grid": LAMBDAS, "families": FAMILIES,
    })
    print("DONE ->", out_dir)


if __name__ == "__main__":
    main()
