#!/usr/bin/env python3
"""Validation-negative-set sensitivity (round-6 diagnostic).

The coalition value averages the pairwise log-sigmoid margin over a fixed set
of |N_u^-| = 100 validation negatives. Reviewers asked whether attributions
are stable when that set size changes. This script recomputes LOO and
Shapley-MC attributions with |N_u^-| in {50, 100, 500} on one training seed
(default 42) and reports:

  * per-user Spearman rank correlation of attributions against the protocol
    (100-negative) reference, averaged over users;
  * top-12 player-set overlap between settings;
  * reranked TEST NDCG@20 / HitRate@20 at lambda=0.10 for each setting.

Negatives for each size are drawn with the same deterministic per-user seeding
scheme as the protocol (seed + 100000 + u) via sample_validation_negatives
(uniform without replacement over the eligible pool; different sizes are
independent draws from the same scheme, which is exactly the robustness
question being tested).

Usage:
  COALGAME_DEVICE=mps python scripts/run_negset_sensitivity.py --dataset ml1m \
      --source-run results/journal_runs/ml1m_lightgcn_v3_prospective \
      --out results/journal_runs/ml1m_lightgcn_v6_negset_sensitivity \
      [--max-users N] [--sizes 50 100 500]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_matched_controls as rmc  # noqa: E402
import run_second_backbone as rsb   # noqa: E402
from coalgamerec.attribution import compute_attribution_for_users, compute_shapley_for_users  # noqa: E402
from coalgamerec.data import item_user_vectors  # noqa: E402
from coalgamerec.metrics import evaluate  # noqa: E402
from coalgamerec.models import LightGCN, TrainConfig, build_lightgcn_graph, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all  # noqa: E402
from coalgamerec.utils import write_json  # noqa: E402

BACKBONE = rmc.BACKBONE
ATTR = rmc.ATTR
TAU_ATT = rmc.TAU_ATT
LAMBDA_ATTR = rmc.LAMBDA_ATTR
KS = rmc.KS


def attribute(split, base_scores, item_vectors, seed, method, n_neg, max_users):
    fn = compute_shapley_for_users if method == "shapley-mc" else compute_attribution_for_users
    kwargs = dict(split=split, base_scores=base_scores, item_vectors=item_vectors,
                  max_users=max_users, exact_threshold=8, seed=seed,
                  max_players_per_user=ATTR["max_players_per_user"],
                  player_selection=ATTR["player_selection"],
                  checkpoint_path=None, save_every=25,
                  alpha=1.0, beta=0.0, lambda_pref=0.0,
                  lambda_attr_value=ATTR["lambda_attr_value"],
                  value_mode=ATTR["value_mode"], n_val_negatives=n_neg)
    if method == "shapley-mc":
        kwargs.update(m=64, antithetic=True)
    else:
        kwargs["method"] = "loo-marginal"
    return fn(**kwargs)


def stability_report(attr_ref: dict, attr_alt: dict, q: int = 12) -> dict:
    rhos, overlaps = [], []
    for u, a_ref in attr_ref.items():
        a_alt = attr_alt.get(u)
        if a_alt is None or len(a_ref) < 3:
            continue
        nz = (np.abs(a_ref) > 0) | (np.abs(a_alt) > 0)
        if nz.sum() < 3:
            continue
        rho = sps.spearmanr(a_ref[nz], a_alt[nz]).statistic
        if not np.isnan(rho):
            rhos.append(rho)
        k = min(q, len(a_ref))
        top_ref = set(np.argsort(-a_ref)[:k])
        top_alt = set(np.argsort(-a_alt)[:k])
        overlaps.append(len(top_ref & top_alt) / k)
    return {"mean_spearman": float(np.mean(rhos)) if rhos else float("nan"),
            "n_users": len(rhos),
            f"top{q}_overlap": float(np.mean(overlaps)) if overlaps else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 500])
    ap.add_argument("--max-users", type=int, default=None,
                    help="optional user subsample for feasibility (default: all users)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = rmc.load_split_from_run(Path(args.source_run))
    item_vectors = item_user_vectors(split.train_csr)

    train_cfg = TrainConfig(dim=BACKBONE["dim"], lr=BACKBONE["lr"], weight_decay=BACKBONE["weight_decay"],
                            epochs=BACKBONE["epochs"], batch_size=BACKBONE["batch_size"],
                            n_neg=BACKBONE["n_neg"], seed=args.seed, device=os.environ.get("COALGAME_DEVICE", "cpu"))
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    edge_index, edge_weight = build_lightgcn_graph(split.train, split.n_users, split.n_items, device)
    model = LightGCN(split.n_users, split.n_items, edge_index, edge_weight,
                     dim=BACKBONE["dim"], n_layers=BACKBONE["n_layers"]).to(device)
    model = rsb.train_shared_prop(model, split.train, split.n_users, split.n_items, train_cfg)
    base_scores = cache_full_scores(model, split.n_users, batch_size=BACKBONE["score_batch_size"],
                                    chunk_items=4096 if split.n_items > 8000 else None)
    item_embeddings = get_item_embeddings(model)

    report = {"dataset": split.name, "seed": args.seed, "sizes": args.sizes, "methods": {}}
    for method in ["loo-marginal", "shapley-mc"]:
        attrs, evals = {}, {}
        for n_neg in args.sizes:
            t = time.time()
            attr = attribute(split, base_scores, item_vectors, args.seed, method, n_neg, args.max_users)
            attrs[n_neg] = attr
            scores = rerank_all(base_scores, split, item_vectors,
                                "shapley-mc" if method == "shapley-mc" else "loo-marginal",
                                shapley_by_user=attr if method == "shapley-mc" else None,
                                loo_by_user=attr if method == "loo-marginal" else None,
                                lambda_attr=LAMBDA_ATTR, tau_att=TAU_ATT,
                                intervention="native", item_embeddings=item_embeddings)
            summary, _ = evaluate(scores, split, item_vectors, ks=KS)
            evals[n_neg] = {"NDCG@20": summary["NDCG@20"], "HitRate@20": summary["HitRate@20"],
                            "seconds": time.time() - t}
        ref = attrs[100]
        method_report = {"rerank_eval": evals, "stability_vs_100": {}}
        for n_neg in args.sizes:
            if n_neg == 100:
                continue
            method_report["stability_vs_100"][str(n_neg)] = stability_report(ref, attrs[n_neg])
        report["methods"][method] = method_report

    write_json(out_dir / "negset_sensitivity.json", report)
    write_json(out_dir / "manifest.json", {
        "note": "Validation-negative-set size sensitivity (single training seed, diagnostic).",
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "device": os.environ.get("COALGAME_DEVICE", "cpu"),
    })
    print(json.dumps(report, indent=1))
    print(f"DONE -> {out_dir / 'negset_sensitivity.json'}", flush=True)


if __name__ == "__main__":
    main()
