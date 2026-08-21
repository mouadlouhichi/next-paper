#!/usr/bin/env python3
"""Round-9 REQUIRED FIX #3: leakage-free NESTED lambda tuning.

Replaces the circular validation-tuned experiment (Table tab:lambda_tuned,
exploratory). Nested timeline per user:
  1. tuning signal: attributions/weights constructed with calibration = the
     LAST TRAINING item (event t-2);
  2. tuning target: rank the validation item (event t-1) with training items
     masked; choose lambda per family by validation NDCG@20;
  3. freeze lambda; construct the FINAL signal with calibration = the
     validation item (event t-1), as in the protocol;
  4. evaluate the test target (event t), calibration item excluded from
     candidates (corrected v7 protocol).

Documented approximation: event t-2 remains in the frozen training graph
(per-user graph retraining is infeasible), so the tuning signal is not fully
independent of the graph; it IS independent of the tuning target (t-1) and of
the test target (t), which removes the circularity of the v6 experiment
(where the same item constructed the signal and was the tuning target).

Grid: lambda in {0.00, 0.05, 0.10, 0.20, 0.40} for tuning; the selected
lambda is applied once to the final signal.
"""
from __future__ import annotations

import argparse
import json
import os
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
from coalgamerec.attribution import compute_attribution_for_users  # noqa: E402
from coalgamerec.data import SplitData  # noqa: E402
from coalgamerec.metrics import evaluate, per_user_hit_ndcg, mask_seen  # noqa: E402
from coalgamerec.models import TrainConfig, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all, valid_sim_scores_all, valid_linear_scores_all  # noqa: E402
from coalgamerec.utils import write_json  # noqa: E402

LAMBDAS = [0.00, 0.05, 0.10, 0.20, 0.40]
KS = (5, 10, 20)


def validation_ndcg(scores, split, exclude_train_only=True):
    """NDCG@20 of ranking the VALIDATION item (training items masked)."""
    train_csr = split.train_csr
    targets = split.val.sort_values("user").item.values.astype(np.int64)
    pu = per_user_hit_ndcg(scores, targets, train_csr, ks=(20,))
    return float(np.mean(pu["NDCG@20"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--families", nargs="+",
                    default=["uniform", "additive-pref", "loo-marginal", "shapley-mc",
                             "valid-sim", "valid-linear"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = rmc.load_split_from_run(Path(args.source_run))
    item_vectors = rmc.item_user_vectors(split.train_csr)
    val_by_user = dict(zip(split.val.user.astype(int), split.val.item.astype(int)))
    excl_test = {u: val_by_user[u] for u in val_by_user}  # v7 candidate correction
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("COALGAME_THREADS", "2"))))

    # tuning split: calibration target = last training item per user
    train_csr = split.train_csr
    rows = []
    for seed in args.seeds:
        seed_cache = out_dir / f"raw_seed_{seed}.csv"
        if seed_cache.exists():
            print(f"seed {seed}: cached, loading {seed_cache.name}")
            rows.extend(pd.read_csv(seed_cache).to_dict("records"))
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

        # ---- step 1: tuning signal with calibration = last TRAINING item (t-2) ----
        # build a split whose val map is the last training item per user
        last_train = {u: int(train_csr[u].indices[-1]) for u in range(split.n_users)
                      if len(train_csr[u].indices) > 0}
        tuning_val_df = pd.DataFrame({"user": list(last_train.keys()),
                                      "item": list(last_train.values()),
                                      "timestamp": 0})
        split_tuning = replace(split, val=tuning_val_df)

        tuning_weights = {}
        if any(f in args.families for f in ["loo-marginal", "shapley-mc"]):
            tuning_weights["loo-marginal"] = compute_attribution_for_users(
                split_tuning, base_scores, item_vectors, method="loo-marginal", seed=seed,
                max_players_per_user=24, player_selection="stratified",
                lambda_attr_value=0.10, value_mode="pairwise_logsigmoid", n_val_negatives=100)
            if "shapley-mc" in args.families:
                tuning_weights["shapley-mc"] = compute_attribution_for_users(
                    split_tuning, base_scores, item_vectors, method="shapley-mc", m=64, seed=seed,
                    max_players_per_user=24, player_selection="stratified", antithetic=True,
                    lambda_attr_value=0.10, value_mode="pairwise_logsigmoid", n_val_negatives=100)

        # ---- step 2: tune lambda on validation target (t-1) ----
        selected = {}
        for fam in args.families:
            best_lam, best_val = 0.0, -np.inf
            for lam in LAMBDAS:
                if fam in ("valid-sim",):
                    s = valid_sim_scores_all(base_scores, split_tuning, last_train, item_embeddings, lam)
                elif fam == "valid-linear":
                    s = valid_linear_scores_all(base_scores, split_tuning, last_train, item_embeddings, lam)
                else:
                    w = tuning_weights.get(fam)
                    if w is None:
                        s = rerank_all(base_scores, split_tuning, item_vectors, fam,
                                       lambda_attr=lam, intervention="native",
                                       item_embeddings=item_embeddings)
                    else:
                        s = rerank_all(base_scores, split_tuning, item_vectors, fam,
                                       shapley_by_user=(w if fam == "shapley-mc" else None),
                                       loo_by_user=(w if fam == "loo-marginal" else None),
                                       lambda_attr=lam, intervention="native",
                                       item_embeddings=item_embeddings)
                v = validation_ndcg(s, split_tuning)
                if v > best_val:
                    best_val, best_lam = v, lam
            selected[fam] = (best_lam, best_val)
            print(f"  seed {seed} {fam}: selected lambda={best_lam} (val NDCG={best_val:.5f})")

        # ---- step 3+4: final signal with calibration = validation item; test eval ----
        final_weights = {}
        if any(f in args.families for f in ["loo-marginal", "shapley-mc"]):
            final_weights["loo-marginal"] = compute_attribution_for_users(
                split, base_scores, item_vectors, method="loo-marginal", seed=seed,
                max_players_per_user=24, player_selection="stratified",
                lambda_attr_value=0.10, value_mode="pairwise_logsigmoid", n_val_negatives=100)
            if "shapley-mc" in args.families:
                final_weights["shapley-mc"] = compute_attribution_for_users(
                    split, base_scores, item_vectors, method="shapley-mc", m=64, seed=seed,
                    max_players_per_user=24, player_selection="stratified", antithetic=True,
                    lambda_attr_value=0.10, value_mode="pairwise_logsigmoid", n_val_negatives=100)
        for fam in args.families:
            lam = selected[fam][0]
            if fam == "valid-sim":
                s = valid_sim_scores_all(base_scores, split, val_by_user, item_embeddings, lam,
                                         exclude_by_user=excl_test)
            elif fam == "valid-linear":
                s = valid_linear_scores_all(base_scores, split, val_by_user, item_embeddings, lam,
                                            exclude_by_user=excl_test)
            else:
                w = final_weights.get(fam)
                s = rerank_all(base_scores, split, item_vectors, fam,
                               shapley_by_user=(w if fam == "shapley-mc" else None),
                               loo_by_user=(w if fam == "loo-marginal" else None),
                               lambda_attr=lam, intervention="native",
                               item_embeddings=item_embeddings, exclude_by_user=excl_test)
            summary, _ = evaluate(s, split, item_vectors, ks=KS, exclude_by_user=excl_test)
            summary.update(seed=seed, family=fam, lambda_selected=lam)
            rows.append(summary)
            print(f"  seed {seed} {fam}: test NDCG@20={summary['NDCG@20']:.5f} at lambda={lam}")
        pd.DataFrame([r for r in rows if r.get("seed") == seed]).to_csv(seed_cache, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "nested_tuning.csv", index=False)
    g = df.groupby("family").agg(lambda_selected=("lambda_selected", lambda x: list(x)),
                                 test_ndcg20_mean=("NDCG@20", "mean"),
                                 test_ndcg20_std=("NDCG@20", lambda x: x.std(ddof=1)))
    g.to_csv(out_dir / "nested_tuning_summary.csv")
    write_json(out_dir / "manifest.json", {
        "note": "Round-9 fix #3: nested lambda tuning. Tuning signal built from event t-2 (last "
                "training item), lambda tuned to predict event t-1, final signal from event t-1, "
                "test evaluation under corrected candidate exclusions. Documented approximation: "
                "event t-2 remains in the frozen training graph.",
        "seeds": args.seeds, "lambda_grid": LAMBDAS,
    })
    print(g.to_string())
    print("DONE ->", out_dir)


if __name__ == "__main__":
    main()
