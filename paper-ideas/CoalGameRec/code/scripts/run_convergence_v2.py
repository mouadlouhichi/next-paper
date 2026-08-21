#!/usr/bin/env python3
"""Round-9 REQUIRED FIX #8: higher-budget Shapley convergence study (v2).

Improvements over run_estimator_convergence.py:
  * independent high-budget reference M=1024 (kept separate from the
    estimators under test; an estimator is never part of its own reference);
  * exact Shapley for users with |P_u| <= 8 as ground truth on small games;
  * downstream NDCG@20 / HitRate@20 of the M-budget reranking itself, on the
    same fitted models (does estimator budget change the intervention?);
  * attribution MAE / sign agreement / top-12 agreement vs the reference;
  * user-level uncertainty (mean +/- CI over users) and two estimator seeds.

Subsampled users for feasibility (--max-users, default 1000).
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
from scipy import stats as sps

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_matched_controls as rmc  # noqa: E402
from coalgamerec.attribution import (compute_attribution_for_users, exact_shapley,  # noqa: E402
                                     permutation_shapley, select_players,
                                     sample_validation_negatives)
from coalgamerec.metrics import evaluate  # noqa: E402
from coalgamerec.models import TrainConfig, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all  # noqa: E402
from coalgamerec.utils import write_json  # noqa: E402

LAMBDA = 0.10
KS = (5, 10, 20)
MS = [16, 32, 64, 128, 256]
M_REF = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--est-seeds", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--max-users", type=int, default=1000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = rmc.load_split_from_run(Path(args.source_run))
    item_vectors = rmc.item_user_vectors(split.train_csr)
    val_by_user = dict(zip(split.val.user.astype(int), split.val.item.astype(int)))
    excl = {u: val_by_user[u] for u in val_by_user}
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("COALGAME_THREADS", "2"))))

    torch.manual_seed(args.seed)
    cfg = TrainConfig(dim=64, lr=0.002, weight_decay=1e-5, epochs=15, batch_size=4096,
                      n_neg=2, seed=args.seed, device=str(device))
    model = rmc.train_lightgcn_shared_prop(split.train, split.n_users, split.n_items, cfg, n_layers=2)
    base_scores = cache_full_scores(model, split.n_users, batch_size=256,
                                    chunk_items=4096 if split.n_items > 8000 else None)
    item_embeddings = get_item_embeddings(model)

    train_csr = split.train_csr
    hlen = np.diff(train_csr.indptr)
    eligible = np.where(hlen >= 10)[0]
    step = max(1, len(eligible) // args.max_users)
    users = eligible[::step][: args.max_users]
    subset_split_users = set(int(u) for u in users)

    def compute_m(m, est_seed, method="perm"):
        if method == "ref":
            return compute_attribution_for_users(
                split, base_scores, item_vectors, method="shapley-mc", m=m, seed=est_seed,
                max_users=None, max_players_per_user=24, player_selection="stratified",
                lambda_attr_value=LAMBDA, value_mode="pairwise_logsigmoid",
                n_val_negatives=100, antithetic=True)
        return compute_attribution_for_users(
            split, base_scores, item_vectors, method="shapley-mc", m=m, seed=est_seed,
            max_players_per_user=24, player_selection="stratified",
            lambda_attr_value=LAMBDA, value_mode="pairwise_logsigmoid",
            n_val_negatives=100, antithetic=True)

    rows = []
    attrs = {}
    for est_seed in args.est_seeds:
        t0 = time.time()
        ref = compute_m(M_REF, est_seed, method="ref")
        t_ref = time.time() - t0
        print(f"est-seed {est_seed}: M={M_REF} reference in {t_ref:.0f}s")
        # exact ground truth for small games
        exact_u, exact_phi = [], []
        for u in users[:200]:
            u = int(u)
            items = train_csr[u].indices
            if len(items) == 0 or u not in val_by_user:
                continue
            sel = select_players(items, item_vectors, 24, strategy="stratified",
                                 val_target=val_by_user[u], seed=args.seed + u)
            if len(sel) > 8:
                continue
            ti = items[sel]
            vn = sample_validation_negatives(split.n_items, items, val_by_user[u], 100,
                                             args.seed + 100000 + u)
            phi = exact_shapley(base_scores[u], ti, val_by_user[u], item_vectors,
                                val_negatives=vn, lambda_attr_value=LAMBDA,
                                value_mode="pairwise_logsigmoid")
            exact_u.append(u)
            exact_phi.append(phi)
        for m in MS:
            t0 = time.time()
            a = compute_m(m, est_seed)
            t_m = time.time() - t0
            attrs[(est_seed, m)] = a
            # agreement vs reference over the subset
            maes, signs, tops, spear = [], [], [], []
            for u in users:
                u = int(u)
                w, wr = a.get(u), ref.get(u)
                if w is None or wr is None or len(w) < 3:
                    continue
                maes.append(float(np.mean(np.abs(w - wr))))
                signs.append(float(np.mean(np.sign(w) == np.sign(wr))))
                k = min(12, len(w))
                tops.append(len(set(np.argsort(-np.abs(w))[:k]) & set(np.argsort(-np.abs(wr))[:k])) / k)
                rho = sps.spearmanr(w, wr).statistic
                if not np.isnan(rho):
                    spear.append(rho)
            # exact-game MAE where available
            exact_mae = None
            if exact_u:
                em = []
                for u, phi_true in zip(exact_u, exact_phi):
                    items = train_csr[u].indices
                    sel = select_players(items, item_vectors, 24, strategy="stratified",
                                         val_target=val_by_user[u], seed=args.seed + u)
                    w_full = np.zeros(len(items), dtype=np.float32)
                    w_full[sel] = a.get(u, np.zeros(len(sel)))[sel] if u in a else 0.0
                    phi_full = np.zeros(len(items), dtype=np.float32)
                    phi_full[sel] = phi_true
                    em.append(float(np.mean(np.abs(w_full[sel] - phi_true))))
                exact_mae = float(np.mean(em))
            # downstream reranking metrics of this M-budget estimator
            scores = rerank_all(base_scores, split, item_vectors, "shapley-mc",
                                shapley_by_user=a, lambda_attr=LAMBDA,
                                intervention="native", item_embeddings=item_embeddings,
                                exclude_by_user=excl)
            summary, _ = evaluate(scores, split, item_vectors, ks=KS, exclude_by_user=excl)
            rows.append({
                "est_seed": est_seed, "M": m, "seconds": round(t_m, 1),
                "ref_seconds": round(t_ref, 1),
                "mae_vs_ref": float(np.mean(maes)),
                "ci95_mae": [float(np.quantile(maes, 0.025)), float(np.quantile(maes, 0.975))],
                "sign_agreement": float(np.mean(signs)),
                "top12_agreement": float(np.mean(tops)),
                "spearman_vs_ref": float(np.mean(spear)),
                "exact_game_mae": exact_mae,
                "NDCG@20": summary["NDCG@20"], "HitRate@20": summary["HitRate@20"],
            })
            print(f"  est-seed {est_seed} M={m}: mae={rows[-1]['mae_vs_ref']:.5f} "
                  f"top12={rows[-1]['top12_agreement']:.3f} NDCG={summary['NDCG@20']:.5f} ({t_m:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "convergence_v2.csv", index=False)
    write_json(out_dir / "manifest.json", {
        "note": "Round-9 fix #8: independent M=1024 reference (never includes the estimators "
                "under test), exact Shapley ground truth for |P_u|<=8 games, downstream "
                "NDCG/HitRate of each M-budget reranking, user-level uncertainty.",
        "training_seed": args.seed, "est_seeds": args.est_seeds,
        "M_grid": MS, "M_ref": M_REF, "max_users": args.max_users,
    })
    print(df.to_string(index=False))
    print("DONE ->", out_dir)


if __name__ == "__main__":
    main()
