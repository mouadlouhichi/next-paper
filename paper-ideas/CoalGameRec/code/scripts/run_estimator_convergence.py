#!/usr/bin/env python3
"""Round-4 required experiment: Shapley estimator convergence (M-budget).

For M in {16, 32, 64, 128, 256} (two estimator RNG seeds each):
  - runtime of the attribution stage
  - efficiency residual |sum(phi) - (v(P_u) - v(empty))|
  - per-user Spearman correlation of attributions vs the M=256 seed-mean reference
  - l1 / l2 attribution distances vs the reference
  - reranking NDCG@20 / HitRate@20 under each M

Usage (on the run machine, from code/):
  COALGAME_DEVICE=mps python scripts/run_estimator_convergence.py --dataset ml1m --seed 42
Outputs: results/journal_runs/<dataset>_lightgcn_v3_prospective/tables/estimator_convergence.csv (+ run log)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from coalgamerec.attribution import coalition_value, compute_shapley_for_users, select_players, sample_validation_negatives  # noqa: E402
from coalgamerec.data import item_user_vectors  # noqa: E402
from coalgamerec.metrics import evaluate  # noqa: E402
from coalgamerec.models import TrainConfig, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all  # noqa: E402
from scripts.run_matched_controls import load_split_from_run, train_lightgcn_shared_prop, RESULTS  # noqa: E402

SRC_NAME = {"ml1m": "ml1m_lightgcn_v3_prospective", "amazon": "amazon_books_lightgcn_v3_prospective"}
BACKBONE = dict(dim=64, n_layers=2, lr=0.002, weight_decay=1e-5, epochs=15, batch_size=4096, n_neg=2)
M_VALUES = [16, 32, 64, 128, 256]
EST_SEEDS_OFFSETS = [0, 1000]
K = 24
N_VAL_NEG = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m", choices=["ml1m", "amazon"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-users", type=int, default=1000,
                    help="subsample of users for attribution + diagnostics (runtime feasibility)")
    ap.add_argument("--source-run", default=None)
    args = ap.parse_args()

    source = Path(args.source_run) if args.source_run else RESULTS / SRC_NAME[args.dataset]
    out_tables = source / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    log = source / "estimator_convergence.log"

    def say(msg):
        print(msg, flush=True)
        with open(log, "a") as f:
            f.write(msg + "\n")

    split = load_split_from_run(source)
    item_vectors = item_user_vectors(split.train_csr)
    train_cfg = TrainConfig(**BACKBONE, seed=args.seed, device=os.environ.get("COALGAME_DEVICE", "cpu"))
    model = train_lightgcn_shared_prop(split.train, split.n_users, split.n_items, train_cfg, n_layers=BACKBONE["n_layers"])
    base_scores = cache_full_scores(model, split.n_users, batch_size=256, chunk_items=4096 if split.n_items > 8000 else None)
    item_embeddings = get_item_embeddings(model)
    say(f"trained {args.dataset} seed {args.seed}; computing Shapley for M={M_VALUES} on first {args.max_users} users")

    train_csr = split.train_csr
    val_targets = split.val.sort_values("user").set_index("user").item.to_dict()
    users = np.arange(split.n_users)

    attributions = {}
    rows = []
    for M in M_VALUES:
        for off in EST_SEEDS_OFFSETS:
            rseed = args.seed + off
            t0 = time.time()
            shap = compute_shapley_for_users(
                split, base_scores, item_vectors, max_users=args.max_users, m=M,
                exact_threshold=8, seed=rseed, max_players_per_user=K,
                player_selection="stratified", checkpoint_path=None, save_every=25,
                alpha=1.0, beta=0.0, lambda_pref=0.0, lambda_attr_value=0.10,
                value_mode="pairwise_logsigmoid", n_val_negatives=N_VAL_NEG, antithetic=True)
            dt = time.time() - t0
            attributions[(M, rseed)] = shap
            # efficiency residual on a user subsample (every 6th user, capped 1000)
            resid = []
            for u in users[::6][:1000]:
                items = train_csr[int(u)].indices
                if len(items) == 0 or int(u) not in val_targets:
                    continue
                P = select_players(items, item_vectors, int(val_targets[int(u)]), K, "stratified")
                idx_all = np.where(np.isin(items, P))[0]
                vt = int(val_targets[int(u)])
                negs = sample_validation_negatives(split.n_items, items, vt, N_VAL_NEG, seed=rseed)
                kw = dict(alpha=1.0, beta=0.0, lambda_pref=0.0, lambda_attr_value=0.10,
                          value_mode="pairwise_logsigmoid", val_negatives=negs)
                v_full = coalition_value(base_scores[int(u)], items, idx_all, vt, item_vectors, **kw)
                v_empty = coalition_value(base_scores[int(u)], items, np.array([], dtype=np.int64), vt, item_vectors, **kw)
                phi = np.asarray([shap[int(u)][i] for i in idx_all], dtype=np.float64)
                resid.append(abs(float(phi.sum()) - (v_full - v_empty)))
            rows.append(dict(M=M, est_seed=rseed, attribution_seconds=round(dt, 1),
                             efficiency_residual_mean=float(np.mean(resid)),
                             efficiency_residual_p95=float(np.quantile(resid, 0.95)),
                             n_residual_users=len(resid)))
            say(f"  M={M} seed={rseed}: {dt:.0f}s, eff-residual mean={np.mean(resid):.2e}")

    # reference: mean of the two M=256 runs
    ref = {u: (attributions[(256, args.seed)][u].astype(np.float64) + attributions[(256, args.seed + 1000)][u].astype(np.float64)) / 2
           for u in attributions[(256, args.seed)]}
    for r in rows:
        M, rseed = r["M"], r["est_seed"]
        rhos, l1s, l2s = [], [], []
        for u in users[::6][:1000]:
            a, b = attributions[(M, rseed)].get(int(u)), ref.get(int(u))
            if a is None or b is None or len(a) < 3:
                continue
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            rhos.append(stats.spearmanr(a, b).correlation)
            l1s.append(np.abs(a - b).sum())
            l2s.append(np.sqrt(((a - b) ** 2).sum()))
        r.update(spearman_vs_M256_mean=float(np.nanmean(rhos)), l1_vs_M256_mean=float(np.mean(l1s)),
                 l2_vs_M256_mean=float(np.mean(l2s)))

    # reranking metrics per M (first estimator seed) — only meaningful when all users computed
    if args.max_users is None or args.max_users >= split.n_users:
        for M in M_VALUES:
            scores = rerank_all(base_scores, split, item_vectors, "shapley-mc",
                                shapley_by_user=attributions[(M, args.seed)], loo_by_user=None,
                                lambda_attr=0.10, tau_att=0.10, intervention="native", item_embeddings=item_embeddings)
            summary, _ = evaluate(scores, split, item_vectors, ks=(5, 10, 20))
            for r in rows:
                if r["M"] == M and r["est_seed"] == args.seed:
                    r["NDCG@20"] = summary["NDCG@20"]
                    r["HitRate@20"] = summary["HitRate@20"]

    df = pd.DataFrame(rows)
    df.to_csv(out_tables / "estimator_convergence.csv", index=False)
    say("SAVED estimator_convergence.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
