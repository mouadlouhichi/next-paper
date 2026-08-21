#!/usr/bin/env python3
"""Round-9 REQUIRED FIX #2: 2x2 factorial separating player-SELECTION effects
from VALUATION effects.

Factors:
  selection: calibration-guided stratified (protocol) vs train-only profile
             similarity selection (no calibration item anywhere in selection)
  valuation: non-game (uniform weights on the selected set) vs LOO-marginal

Cells (per dataset, seed 42; more seeds append):
  calib  x uniform   = primary 'uniform' family (indirect calibration access)
  calib  x loo       = protocol LOO
  train  x uniform   = clean no-calibration control
  train  x loo       = LOO valuation over train-only-selected players

All evaluated under the corrected v7 candidate exclusions.
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
from coalgamerec.rerank import rerank_all  # noqa: E402
from coalgamerec.utils import write_json  # noqa: E402

LAMBDA = 0.10
KS = (5, 10, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
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

    rows = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        cfg = TrainConfig(dim=64, lr=0.002, weight_decay=1e-5, epochs=15, batch_size=4096,
                          n_neg=2, seed=seed, device=str(device))
        t0 = time.time()
        model = rmc.train_lightgcn_shared_prop(split.train, split.n_users, split.n_items, cfg, n_layers=2)
        base_scores = cache_full_scores(model, split.n_users, batch_size=256,
                                        chunk_items=4096 if split.n_items > 8000 else None)
        item_embeddings = get_item_embeddings(model)
        print(f"seed {seed}: trained in {time.time()-t0:.0f}s")

        cells = {}
        for selection in ["calibration-guided", "train-only"]:
            strategy = "stratified" if selection == "calibration-guided" else "similarity"
            # train-only selection: hide the calibration target from player selection only
            split_sel = split
            use_val = selection == "calibration-guided"
            for valuation in ["uniform", "loo-marginal"]:
                name = f"{selection} x {valuation}"
                t1 = time.time()
                if valuation == "uniform":
                    # uniform weights on the selected player set: emulate by LOO API with
                    # selection strategy applied via compute_attribution_for_users? The
                    # uniform family weights all of P_u equally; we build P_u explicitly.
                    from coalgamerec.attribution import select_players
                    train_csr = split.train_csr
                    unif = {}
                    for u in range(split.n_users):
                        items = train_csr[u].indices
                        if len(items) == 0:
                            unif[u] = np.zeros(0, dtype=np.float32)
                            continue
                        vt = val_by_user.get(u) if use_val else None
                        sel = select_players(items, item_vectors, 24, strategy=strategy,
                                             val_target=vt, seed=seed + u)
                        w = np.zeros(len(items), dtype=np.float32)
                        w[sel] = 1.0 / max(1, len(sel))
                        unif[u] = w
                    scores = rerank_all(base_scores, split, item_vectors, "loo-marginal",
                                        loo_by_user=unif, lambda_attr=LAMBDA,
                                        intervention="native", item_embeddings=item_embeddings,
                                        exclude_by_user=excl)
                else:
                    # strategy 'similarity' ignores the calibration target in selection,
                    # while coalition values still use the calibration positive (valuation
                    # cannot be blind to it by construction)
                    loo = compute_attribution_for_users(
                        split_sel, base_scores, item_vectors, method="loo-marginal", seed=seed,
                        max_players_per_user=24, player_selection=strategy,
                        lambda_attr_value=LAMBDA, value_mode="pairwise_logsigmoid",
                        n_val_negatives=100)
                    scores = rerank_all(base_scores, split, item_vectors, "loo-marginal",
                                        loo_by_user=loo, lambda_attr=LAMBDA,
                                        intervention="native", item_embeddings=item_embeddings,
                                        exclude_by_user=excl)
                summary, _ = evaluate(scores, split, item_vectors, ks=KS, exclude_by_user=excl)
                summary.update(seed=seed, cell=name)
                rows.append(summary)
                cells[name] = summary["NDCG@20"]
                print(f"  {name}: NDCG@20={summary['NDCG@20']:.5f} ({time.time()-t1:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "selection_factorial.csv", index=False)
    write_json(out_dir / "manifest.json", {
        "note": "Round-9 fix #2: 2x2 factorial (player selection x valuation) under corrected "
                "v7 candidate exclusions. train-only selection uses profile-similarity strategy "
                "with no calibration item; LOO coalitions still use the calibration target as "
                "the value-function target (valuation cannot be blind to it by construction).",
        "seeds": args.seeds,
    })
    print("DONE ->", out_dir)


if __name__ == "__main__":
    main()
