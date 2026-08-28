#!/usr/bin/env python3
"""Verify the supplement's decision-quality block against the frozen release matrices.

``tables/review3_statistics.tex`` carries a ``tab:prop-quality`` block that its own generator only
partially owns, so ``make_review3_stats.py --check`` never looks at it. This script recomputes every
row of that block from the released matrices and fails when a printed value is unsupported, so the
block stops being the one place a number can live without a file behind it.

The convention is fixed by the release, not by preference: ``intervention_success_ndcg`` is the
binary per-user-and-seed indicator (``float(effect > 0)`` in ``run_recommendation.py``), and the
published cohort statistic averages it over the seed runs and then over users. Because a user's value
is itself a mean over ``R_seed`` seeds, a cohort rate over ``n`` users moves on a ``1/(n R_seed)``
grid - multiples of 0.0002 for the 1,000-user primary ItemKNN cohort - which is why four-decimal
values such as 0.2742 are correct rather than suspicious. The alternative pooling (the indicator of
the seed-averaged effect) is a different quantity, and the two are reported nowhere as substitutes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
PAPER = CODE.parent
MATRICES = PAPER / "actionshap-ipm" / "release" / "matrices"
TABLE = PAPER / "acmart-primary" / "tables" / "review3_statistics.tex"
MAIN = PAPER / "acmart-primary" / "acmmanuscript.tex"
OUT = CODE / "results" / "review9" / "success_estimand_audit.json"

METHOD = {"Shapley": "shapley_mc", "LIME": "lime", "LOO": "loo",
          "Greedy": "greedy_cf", "Random": "random"}
QUANTITY = {"Success": "intervention_success_ndcg", "Abstention": "abstention",
            "NDCG$-$Pop.": "__model_minus_popularity__"}
DERIVED = {"__model_minus_popularity__": ("quality_model_ndcg@10", "quality_popularity_ndcg@10")}
ROW = re.compile(r"^([A-Za-z0-9-]+) & (\w+) & (Success|Abstention|NDCG\$-\$Pop\.) & "
                 r"([0-9.]+) & \[([0-9.]+),([0-9.]+)\] & (\d+)", re.M)


def slice_of(frame: pd.DataFrame, dataset: str, method_id: str) -> pd.DataFrame:
    """The published slice: primary cohort, the primary ItemKNN model, one method."""
    return frame[(frame.dataset == dataset) & (frame.condition == "primary")
                 & (frame.model == "itemknn") & (frame.method == method_id)]


def values_of(sub: pd.DataFrame, column: str) -> pd.Series:
    """Per-user seed-mean of a stored column, or of the released model-minus-reference difference."""
    if column in DERIVED:
        high, low = DERIVED[column]
        return (sub[high] - sub[low]).groupby(sub["user"]).mean()
    return sub.groupby("user")[column].mean()


def cohort(frame: pd.DataFrame, dataset: str, method_id: str, column: str) -> dict:
    sub = slice_of(frame, dataset, method_id)
    per_user = values_of(sub, column)
    indicator_of_seed_mean = (sub.groupby("user")["joint_effect_ndcg"].mean() > 0).astype(float)
    return {
        "n_users": int(sub["user"].nunique()),
        "n_seeds": int(sub.groupby("user").size().max()),
        "published_convention": round(float(per_user.mean()), 6),
        "other_pooling_indicator_of_seed_mean": round(float(indicator_of_seed_mean.mean()), 6),
    }


def bootstrap_ci(frame: pd.DataFrame, dataset: str, method_id: str, column: str,
                 draws: int = 10_000, seed: int = 13) -> list[float]:
    values = values_of(slice_of(frame, dataset, method_id), column).to_numpy()
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    means = values[idx].mean(axis=1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit non-zero if a printed row is unsupported")
    args = parser.parse_args()

    frame = pd.read_csv(MATRICES / "user_seed_metrics.csv.gz")
    text = TABLE.read_text(encoding="utf-8")
    rows = [dict(zip(("dataset", "method", "quantity", "printed", "ci_low", "ci_high", "n"),
                     (d, m, q, float(mean), float(lo), float(hi), int(n))))
            for d, m, q, mean, lo, hi, n in ROW.findall(text)]
    if not rows:
        print(f"PROBLEM: no tab:prop-quality rows parsed from {TABLE.name}", file=sys.stderr)
        return 1

    report, problems = [], []
    for row in rows:
        column = QUANTITY[row["quantity"]]
        method_id = METHOD.get(row["method"], row["method"])
        stats = cohort(frame, row["dataset"], method_id, column)
        entry = {**row, **stats,
                 "recomputed_ci": bootstrap_ci(frame, row["dataset"], method_id, column)}
        delta = abs(stats["published_convention"] - row["printed"])
        entry["abs_delta"] = round(delta, 6)
        ok = delta <= 0.0006 and stats["n_users"] == row["n"]
        ci_ok = (entry["recomputed_ci"][0] - 0.0015 <= row["ci_low"]
                 and row["ci_high"] <= entry["recomputed_ci"][1] + 0.0015)
        entry["supported"] = bool(ok)
        entry["ci_within_recomputed"] = bool(ci_ok)
        report.append(entry)
        flag = "OK" if ok else "UNSUPPORTED"
        print(f"  {row['dataset']:<21} {row['method']:<8} {row['quantity']:<18} "
              f"printed {row['printed']:.4f}  recomputed {stats['published_convention']:.4f}  "
              f"other pooling {stats['other_pooling_indicator_of_seed_mean']:.4f}  "
              f"n={stats['n_users']}  CI {'inside' if ci_ok else 'OUTSIDE'}  {flag}")
        if not ok:
            problems.append(f"{row['dataset']}/{row['method']}/{row['quantity']}: printed "
                            f"{row['printed']} but the frozen matrix gives "
                            f"{stats['published_convention']} (n={stats['n_users']})")

    # the grid argument the review asked about: values live on a 1/(n * R_seed) lattice
    n_users, r_seed = 1000, int(report[0]["n_seeds"] or 5)
    grid = 1.0 / (n_users * r_seed)
    on_grid = [abs(round(r["printed"] / grid) * grid - r["printed"]) <= 1e-9 for r in report]
    print(f"grid: 1/(n*R_seed) = {grid:.6f}; printed rows on the lattice: {sum(on_grid)}/{len(on_grid)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"matrix": "user_seed_metrics.csv.gz", "slice": {
        "condition": "primary", "model": "itemknn", "grid": grid, "n_users": n_users,
        "r_seed": r_seed}, "rows": report, "unsupported": len(problems)}, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(PAPER)} ({len(report)} rows, {len(problems)} unsupported)")

    if problems and args.check:
        print("\n".join(f"PROBLEM: {p}" for p in problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
