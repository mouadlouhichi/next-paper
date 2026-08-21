#!/usr/bin/env python3
"""Round-4 required experiment: the five planned game-design ablations.

Executes (single training seed, declared exploratory):
  1. k-sweep:          k in {8, 16, 24, 32} for Shapley and LOO
  2. player selection: stratified vs similarity vs random (Shapley, k=24)
  3. utility:          smooth pairwise log-sigmoid vs hard top-K NDCG value
  4. intervention:     native embeddings vs external train-only kernel
  5. M reference:      (covered separately by run_estimator_convergence.py)

Usage (from code/):
  COALGAME_DEVICE=mps python scripts/run_design_ablations.py --dataset ml1m --seed 42
  COALGAME_DEVICE=mps python scripts/run_design_ablations.py --dataset amazon --seed 42
Outputs: results/journal_runs/<dataset>_lightgcn_v3_prospective/tables/design_ablations.csv (+ log)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from coalgamerec.attribution import compute_attribution_for_users, compute_shapley_for_users  # noqa: E402
from coalgamerec.data import item_user_vectors  # noqa: E402
from coalgamerec.metrics import evaluate  # noqa: E402
from coalgamerec.models import TrainConfig, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all  # noqa: E402
from scripts.run_matched_controls import load_split_from_run, train_lightgcn_shared_prop, RESULTS  # noqa: E402

SRC_NAME = {"ml1m": "ml1m_lightgcn_v3_prospective", "amazon": "amazon_books_lightgcn_v3_prospective"}
BACKBONE = dict(dim=64, n_layers=2, lr=0.002, weight_decay=1e-5, epochs=15, batch_size=4096, n_neg=2)
LAM, TAU, M, NNEG = 0.10, 0.10, 64, 100


MAX_USERS = None  # optional subsample (documented in output caption when used)


def attrib_args(**over):
    base = dict(max_users=MAX_USERS, exact_threshold=8, checkpoint_path=None, save_every=25,
                alpha=1.0, beta=0.0, lambda_pref=0.0, lambda_attr_value=LAM,
                value_mode="pairwise_logsigmoid", n_val_negatives=NNEG)
    base.update(over)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m", choices=["ml1m", "amazon"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source-run", default=None)
    ap.add_argument("--max-users", type=int, default=None,
                    help="optional user subsample for feasibility (documented in output)")
    args = ap.parse_args()
    global MAX_USERS
    MAX_USERS = args.max_users

    source = Path(args.source_run) if args.source_run else RESULTS / SRC_NAME[args.dataset]
    out_tables = source / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    per_seed_csv = out_tables / f"design_ablations_seed_{args.seed}.csv"
    if per_seed_csv.exists():
        print(f"seed {args.seed} already complete ({per_seed_csv.name}), skipping")
        return
    log = source / "design_ablations.log"

    def say(msg):
        print(msg, flush=True)
        with open(log, "a") as f:
            f.write(msg + "\n")

    split = load_split_from_run(source)
    item_vectors = item_user_vectors(split.train_csr)
    cfg = TrainConfig(dim=BACKBONE["dim"], lr=BACKBONE["lr"], weight_decay=BACKBONE["weight_decay"],
                      epochs=BACKBONE["epochs"], batch_size=BACKBONE["batch_size"], n_neg=BACKBONE["n_neg"],
                      seed=args.seed, device=os.environ.get("COALGAME_DEVICE", "cpu"))
    model = train_lightgcn_shared_prop(split.train, split.n_users, split.n_items, cfg, n_layers=BACKBONE["n_layers"])
    base_scores = cache_full_scores(model, split.n_users, batch_size=256, chunk_items=4096 if split.n_items > 8000 else None)
    item_embeddings = get_item_embeddings(model)
    say(f"trained {args.dataset} seed {args.seed}")

    rows = []

    def record(ablation, variant, family, shap, loo, intervention="native"):
        t0 = time.time()
        scores = rerank_all(base_scores, split, item_vectors, family,
                            shapley_by_user=shap, loo_by_user=loo,
                            lambda_attr=LAM, tau_att=TAU, intervention=intervention,
                            item_embeddings=item_embeddings)
        s, _ = evaluate(scores, split, item_vectors, ks=(5, 10, 20))
        rows.append(dict(ablation=ablation, variant=variant, family=family, intervention=intervention,
                         **{m: s[m] for m in ["NDCG@20", "HitRate@20", "Coverage@20"]},
                         eval_seconds=round(time.time() - t0, 1)))
        say(f"  {ablation} | {variant} | {family} | {intervention}: NDCG@20={s['NDCG@20']:.5f}")

    # 1. k-sweep
    for k in [8, 16, 24, 32]:
        say(f"k={k}: LOO")
        loo = compute_attribution_for_users(split, base_scores, item_vectors, method="loo-marginal",
                                            seed=args.seed, max_players_per_user=k,
                                            player_selection="stratified", **attrib_args())
        record("k_sweep", f"k={k}", "loo-marginal", None, loo)
        say(f"k={k}: Shapley")
        shap = compute_shapley_for_users(split, base_scores, item_vectors, m=M, seed=args.seed,
                                         max_players_per_user=k, player_selection="stratified",
                                         antithetic=True, **attrib_args())
        record("k_sweep", f"k={k}", "shapley-mc", shap, None)

    # 2. player selection (Shapley k=24)
    for sel in ["stratified", "similarity", "random"]:
        say(f"selection={sel}: Shapley")
        shap = compute_shapley_for_users(split, base_scores, item_vectors, m=M, seed=args.seed,
                                         max_players_per_user=24, player_selection=sel,
                                         antithetic=True, **attrib_args())
        record("player_selection", sel, "shapley-mc", shap, None)

    # 3. hard vs smooth utility (Shapley + LOO, k=24, stratified)
    for vm in ["pairwise_logsigmoid", "ndcg_ild"]:
        say(f"utility={vm}: Shapley")
        shap = compute_shapley_for_users(split, base_scores, item_vectors, m=M, seed=args.seed,
                                         max_players_per_user=24, player_selection="stratified",
                                         antithetic=True, **attrib_args(value_mode=vm))
        record("utility", vm, "shapley-mc", shap, None)
        say(f"utility={vm}: LOO")
        loo = compute_attribution_for_users(split, base_scores, item_vectors, method="loo-marginal",
                                            seed=args.seed, max_players_per_user=24,
                                            player_selection="stratified", **attrib_args(value_mode=vm))
        record("utility", vm, "loo-marginal", None, loo)

    # 4. native vs external-kernel intervention (stratified k=24)
    say("intervention ablation")
    loo = compute_attribution_for_users(split, base_scores, item_vectors, method="loo-marginal",
                                        seed=args.seed, max_players_per_user=24,
                                        player_selection="stratified", **attrib_args())
    shap = compute_shapley_for_users(split, base_scores, item_vectors, m=M, seed=args.seed,
                                     max_players_per_user=24, player_selection="stratified",
                                     antithetic=True, **attrib_args())
    for fam, sh, lo in [("loo-marginal", None, loo), ("shapley-mc", shap, None)]:
        record("intervention", "native", fam, sh, lo, intervention="native")
        record("intervention", "kernel", fam, sh, lo, intervention="kernel")

    df = pd.DataFrame(rows)
    df.insert(0, "seed", args.seed)
    df.insert(1, "max_users", args.max_users if args.max_users is not None else -1)
    df.to_csv(out_tables / f"design_ablations_seed_{args.seed}.csv", index=False)
    # merge into the canonical multi-seed table (replace any rows from this seed)
    canon = out_tables / "design_ablations.csv"
    if canon.exists():
        prev = pd.read_csv(canon)
        if "seed" not in prev.columns:
            prev["seed"] = 42  # canonical table predates multi-seed support (single seed 42)
        if "max_users" not in prev.columns:
            prev["max_users"] = np.nan  # released full-user rows
        mu = args.max_users if args.max_users is not None else -1
        prev = prev[~((prev["seed"] == args.seed) & (prev["max_users"].fillna(-1) == mu))]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(canon, index=False)
    say(f"SAVED design_ablations_seed_{args.seed}.csv + merged design_ablations.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
