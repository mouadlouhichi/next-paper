#!/usr/bin/env python3
"""Round-9 REQUIRED FIX #11: CONTROLLED randomization sanity test.

The v22 randomization test changed the WHOLE untrained model; this one holds
the trained base scorer, candidate scores, item embeddings, player sets, and
reranking operation fixed, and swaps ONLY the attribution weights:

  trained        LOO weights from the trained model (reference)
  untrained-w    LOO weights computed from the untrained (initialized) model,
                 applied to the trained scorer/intervention
  shuffled       per-user random permutation of the trained LOO weights
  dist-matched   random weights matched to the trained weights' empirical
                 distribution (per-user resampling with replacement)
  selection-only weights = 1 on the 12 highest-trained-|w| players, 0 elsewhere
                 (isolates the selection information from the valuation)

Reports reranked NDCG@20/HitRate@20 under the corrected v7 candidate
exclusions, with user-level paired differences and 95% CIs (bootstrap) for
each swap vs trained.
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
from coalgamerec.metrics import per_user_hit_ndcg  # noqa: E402
from coalgamerec.models import LightGCN, TrainConfig, build_lightgcn_graph, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all  # noqa: E402
from coalgamerec.utils import write_json  # noqa: E402

KS = (5, 10, 20)
LAMBDA = 0.10
SEED = 42


def metrics_from_scores(scores, split, exclude_by_user):
    train_csr = split.train_csr
    targets = split.test.sort_values("user").item.values.astype(np.int64)
    pu = per_user_hit_ndcg(scores, targets, train_csr, ks=KS, exclude_by_user=exclude_by_user)
    return {k: float(np.mean(v)) for k, v in pu.items()}, pu


def bootstrap_ci(d: np.ndarray, rng, b=5000):
    n = len(d)
    boots = np.empty(b)
    for s in range(0, b, 500):
        m = min(500, b - s)
        idx = rng.integers(0, n, size=(m, n))
        boots[s:s + m] = d[idx].mean(axis=1)
    return np.quantile(boots, [0.025, 0.975]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "controlled_randomization.json").exists():
        print("controlled_randomization.json exists, skipping (delete it to re-run)")
        return
    split = rmc.load_split_from_run(Path(args.source_run))
    item_vectors = rmc.item_user_vectors(split.train_csr)
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("COALGAME_THREADS", "2"))))

    # --- trained model ---
    torch.manual_seed(args.seed)
    cfg = TrainConfig(dim=64, lr=0.002, weight_decay=1e-5, epochs=15, batch_size=4096,
                      n_neg=2, seed=args.seed, device=str(device))
    model = rmc.train_lightgcn_shared_prop(split.train, split.n_users, split.n_items, cfg, n_layers=2)
    base_scores = cache_full_scores(model, split.n_users, batch_size=256,
                                    chunk_items=4096 if split.n_items > 8000 else None)
    item_embeddings = get_item_embeddings(model)

    # --- untrained model (same initialization, zero gradient steps) ---
    torch.manual_seed(args.seed)
    edge_index, edge_weight = build_lightgcn_graph(split.train, split.n_users, split.n_items, device)
    model_untrained = LightGCN(split.n_users, split.n_items, edge_index, edge_weight, 64, n_layers=2).to(device)
    base_scores_untrained = cache_full_scores(model_untrained, split.n_users, batch_size=256,
                                              chunk_items=4096 if split.n_items > 8000 else None)
    item_embeddings_untrained = get_item_embeddings(model_untrained)

    # --- attributions ---
    t0 = time.time()
    loo_trained = compute_attribution_for_users(
        split, base_scores, item_vectors, method="loo-marginal", seed=args.seed,
        max_players_per_user=24, player_selection="stratified",
        lambda_attr_value=LAMBDA, value_mode="pairwise_logsigmoid", n_val_negatives=100)
    loo_untrained = compute_attribution_for_users(
        split, base_scores_untrained, item_vectors, method="loo-marginal", seed=args.seed,
        max_players_per_user=24, player_selection="stratified",
        lambda_attr_value=LAMBDA, value_mode="pairwise_logsigmoid", n_val_negatives=100)
    print(f"attributions: {time.time()-t0:.0f}s")

    train_csr = split.train_csr
    rng = np.random.default_rng(args.seed * 7919 + 3)
    shuffled, dist_matched, selection_only = {}, {}, {}
    for u, w in loo_trained.items():
        w = np.asarray(w, dtype=np.float32)
        perm = rng.permutation(len(w))
        shuffled[u] = w[perm]
        dist_matched[u] = rng.choice(w, size=len(w), replace=True).astype(np.float32)
        k = min(12, len(w))
        sel = np.zeros_like(w)
        sel[np.argsort(-np.abs(w))[:k]] = 1.0
        selection_only[u] = sel

    val_by_user = dict(zip(split.val.user.astype(int), split.val.item.astype(int)))
    excl = {u: val_by_user[u] for u in val_by_user}

    variants = {
        "trained": loo_trained,
        "untrained-w": loo_untrained,
        "shuffled": shuffled,
        "dist-matched": dist_matched,
        "selection-only": selection_only,
    }
    results, per_user_ndcg = {}, {}
    for name, weights in variants.items():
        scores = rerank_all(base_scores, split, item_vectors, "loo-marginal",
                            loo_by_user=weights, lambda_attr=LAMBDA, tau_att=0.10,
                            intervention="native", item_embeddings=item_embeddings,
                            exclude_by_user=excl)
        summary, pu = metrics_from_scores(scores, split, excl)
        results[name] = {"NDCG@20": summary["NDCG@20"], "HitRate@20": summary["HitRate@20"]}
        per_user_ndcg[name] = pu["NDCG@20"]
        print(f"  {name}: NDCG@20={summary['NDCG@20']:.5f}")

    # user-level paired contrasts vs trained
    rng2 = np.random.default_rng(args.seed + 101)
    contrasts = {}
    ref = per_user_ndcg["trained"]
    for name in variants:
        if name == "trained":
            continue
        d = per_user_ndcg[name] - ref
        contrasts[name] = {
            "mean_diff": float(d.mean()),
            "ci95": bootstrap_ci(d, rng2),
            "frac_users_worse": float((d < 0).mean()),
        }

    report = {"dataset": split.name, "seed": args.seed,
              "note": "Controlled randomization: trained base scorer, candidate scores, item "
                      "embeddings, player sets, and reranking operation held fixed; only the "
                      "attribution weights are swapped.",
              "metrics": results, "contrasts_vs_trained": contrasts}
    write_json(out_dir / "controlled_randomization.json", report)
    print(json.dumps(report, indent=1))
    print("DONE ->", out_dir / "controlled_randomization.json")


if __name__ == "__main__":
    main()
