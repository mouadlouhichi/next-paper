#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coalgamerec.stats import bootstrap_ci, cohen_dz, holm_bonferroni
from coalgamerec.utils import write_json


def paired_seed_diffs(per_user: pd.DataFrame, metric: str, treatment: str, control: str) -> list[np.ndarray]:
    df = per_user[per_user.metric == metric]
    diffs = []
    for seed, g in df.groupby("seed"):
        piv = g.pivot_table(index="user", columns="family", values="value", aggfunc="first")
        if treatment not in piv.columns or control not in piv.columns:
            continue
        d = (piv[treatment] - piv[control]).dropna().to_numpy(dtype=float)
        diffs.append(d)
    return diffs


def bootstrap_p_value(diff_by_seed: list[np.ndarray], n_boot: int = 2000, seed: int = 42) -> float:
    # Percentile bootstrap around observed mean; descriptive two-sided sign probability.
    rng = np.random.default_rng(seed)
    obs = np.mean([d.mean() for d in diff_by_seed if len(d)])
    vals = []
    for _ in range(n_boot):
        vals.append(np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diff_by_seed if len(d)]))
    vals = np.asarray(vals)
    if obs >= 0:
        return float(2 * min(np.mean(vals <= 0), np.mean(vals >= 0)))
    return float(2 * min(np.mean(vals >= 0), np.mean(vals <= 0)))


def main() -> None:
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description="Analyze CoalGameRec per-user result artifacts.")
    ap.add_argument("--run-dir", required=True, help="Run output directory from run_q1_pipeline.py")
    ap.add_argument("--treatment", default="shapley-mc")
    ap.add_argument("--control", default="uniform")
    ap.add_argument("--metrics", nargs="+", default=["NDCG@20", "HitRate@20"])
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    per_user_path = run_dir / "raw" / "per_user_metrics_all.csv"
    if not per_user_path.exists():
        raise FileNotFoundError(per_user_path)
    per_user = pd.read_csv(per_user_path)
    rows = []
    pvals = {}
    for metric in args.metrics:
        diffs = paired_seed_diffs(per_user, metric, args.treatment, args.control)
        if not diffs:
            continue
        obs, lo, hi = bootstrap_ci(diffs, n_boot=args.bootstrap_samples, seed=args.seed)
        pooled = np.concatenate(diffs)
        p = bootstrap_p_value(diffs, n_boot=args.bootstrap_samples, seed=args.seed)
        key = f"{args.treatment}_vs_{args.control}_{metric}"
        pvals[key] = p
        rows.append(
            {
                "contrast": key,
                "metric": metric,
                "treatment": args.treatment,
                "control": args.control,
                "mean_diff_conditional_user": obs,
                "ci95_low": lo,
                "ci95_high": hi,
                "median_user_diff": float(np.median(pooled)),
                "cohen_dz_user_conditional_descriptive": cohen_dz(pooled),
                "bootstrap_p_descriptive": p,
                "n_seeds": len(diffs),
                "n_user_diffs": int(sum(len(d) for d in diffs)),
            }
        )
    out_dir = run_dir / "tables"
    out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    if rows:
        holm = holm_bonferroni(pvals)
        df["holm_reject_0.05"] = df["contrast"].map(lambda k: holm[k]["reject_0.05"])
        df["holm_threshold"] = df["contrast"].map(lambda k: holm[k]["holm_threshold"])
        write_json(out_dir / "holm_primary.json", holm)
    df.to_csv(out_dir / "paired_bootstrap_contrasts.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
