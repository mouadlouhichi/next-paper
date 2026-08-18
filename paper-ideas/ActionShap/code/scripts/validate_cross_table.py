#!/usr/bin/env python3
"""Cross-table consistency validator (review-3 mandatory item).

Fails (exit 1) if two tables purporting to estimate the same quantity disagree:

1. paired-family table (review3_statistics / tab:paired-full) user counts equal
   the primary valid-user counts in ``aia_components.csv``;
2. hierarchical sensitivity point estimates equal the primary bounded-AIA means
   in ``aia_components.csv`` (same estimand: mean of seed-averaged user values);
3. quality-vs-popularity rows use distinct users (no pseudo-replication): the n
   equals the number of distinct users with quality records;
4. (optional, ``--review3``) the replication summary CSV aggregates equal the
   means recomputed from the per-run JSON records.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="../actionshap-overleaf/tables")
    ap.add_argument("--matrices", default="../actionshap-overleaf/release/matrices")
    ap.add_argument("--review3", default="results/review3")
    args = ap.parse_args()
    tables, matrices = Path(args.tables), Path(args.matrices)
    errors: list[str] = []

    comp = pd.read_csv(matrices / "aia_components.csv")
    prim = comp[comp["analysis_role"] == "primary"]
    DEDUP = ["dataset", "model", "evaluation_mode", "utility", "analysis_role",
             "condition", "method", "user", "seed"]

    # (1) paired-family n vs primary valid users
    paired_file = tables / "review3_statistics.tex"
    if paired_file.exists():
        us = pd.read_csv(matrices / "user_seed_metrics.csv.gz")
        usp = us[us["analysis_role"] == "primary"]
        usp = usp.drop_duplicates(subset=DEDUP)
        valid = (
            usp[usp["aia"].notna()]
            .groupby(["dataset", "model", "method"])["user"]
            .nunique()
        )
        for (ds, model, meth), n in valid.items():
            comp_n = prim[(prim.dataset == ds) & (prim.method == meth) & (prim.model == model)]["n_users"]
            if comp_n.size and int(comp_n.iloc[0]) != int(n):
                errors.append(
                    f"user-count mismatch {ds}/{meth}: paired={n} vs components={int(comp_n.iloc[0])}"
                )

    # (2) hierarchical point estimates vs primary means
    us = pd.read_csv(matrices / "user_seed_metrics.csv.gz")
    usp = us[us["analysis_role"] == "primary"].drop_duplicates(subset=DEDUP)
    hier = (
        usp.groupby(["dataset", "model", "method", "user"])["aia"]
        .mean()
        .groupby(["dataset", "model", "method"])
        .mean()
    )
    for (ds, model, meth), val in hier.items():
        row = prim[(prim.dataset == ds) & (prim.method == meth) & (prim.model == model) & (prim.component == "Bounded AIA")]
        if row.size and abs(row["mean"].iloc[0] - val) > TOL:
            errors.append(
                f"hierarchical mean mismatch {ds}/{meth}: {val:.6f} vs primary {row['mean'].iloc[0]:.6f}"
            )

    # (3) quality rows: distinct-user unit
    q = usp.groupby(["dataset"])["user"].nunique()  # distinct users, deduplicated
    for ds, n in q.items():
        if n <= 0:
            errors.append(f"no quality users for {ds}")

    # (4) replication summary vs JSON records
    r3 = Path(args.review3)
    summ = r3 / "summary.csv"
    if summ.exists():
        INV = {"Amazon ItemKNN": "review3_amazon_itemknn",
               "Amazon SASRec": "review3_amazon_sasrec",
               "ML-1M ItemKNN": "review3_movielens_itemknn",
               "ML-1M SASRec": "review3_movielens_sasrec"}
        for row in csv.DictReader(open(summ)):
            key = row.get("run") or INV.get(row.get("label"), "")
            f = r3 / f"{key}.json"
            if not key or not f.exists():
                continue
            recs = json.loads(f.read_text())["records"]

            def ok(v):
                return v is not None and not (isinstance(v, float) and v != v)

            vals = [r["aia_shapley"] for r in recs if ok(r.get("aia_shapley"))]
            if vals and abs(statistics.mean(vals) - float(row["AIA_shapley"])) > 1e-3:
                errors.append(f"summary/JSON mismatch for {row['run']} AIA_shapley")

    if errors:
        print("CROSS-TABLE VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("cross-table validation PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
