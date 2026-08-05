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


def paired_seed_diffs(per_user: pd.DataFrame, metric: str, treatment: str, control: str) -> dict[int, np.ndarray]:
    """Return paired user-level differences for each seed."""
    df = per_user[per_user.metric == metric]
    diffs: dict[int, np.ndarray] = {}
    for seed, g in df.groupby("seed"):
        piv = g.pivot_table(index="user", columns="family", values="value", aggfunc="first")
        if treatment not in piv.columns or control not in piv.columns:
            continue
        d = (piv[treatment] - piv[control]).dropna().to_numpy(dtype=float)
        diffs[int(seed)] = d
    return diffs


def bootstrap_p_value(diff_by_seed: list[np.ndarray], n_boot: int = 2000, seed: int = 42) -> float:
    """Descriptive two-sided bootstrap sign probability.

    This is not a literal exact p-value; with finite bootstrap samples, report a
    zero value as p < 1 / n_boot in manuscripts.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        vals.append(np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diff_by_seed if len(d)]))
    vals = np.asarray(vals)
    obs = np.mean([d.mean() for d in diff_by_seed if len(d)])
    if obs >= 0:
        return float(2 * min(np.mean(vals <= 0), np.mean(vals >= 0)))
    return float(2 * min(np.mean(vals >= 0), np.mean(vals <= 0)))


def contrast_summary(
    per_user: pd.DataFrame,
    metric: str,
    treatment: str,
    control: str,
    n_boot: int,
    seed: int,
) -> tuple[dict, list[dict]] | tuple[None, list[dict]]:
    by_seed = paired_seed_diffs(per_user, metric, treatment, control)
    diffs = [d for d in by_seed.values() if len(d)]
    if not diffs:
        return None, []
    obs, lo, hi = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    pooled = np.concatenate(diffs)
    p = bootstrap_p_value(diffs, n_boot=n_boot, seed=seed)
    eps = 1e-12
    row = {
        "contrast": f"{treatment}_vs_{control}_{metric}",
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "mean_diff_conditional_user": obs,
        "ci95_low": lo,
        "ci95_high": hi,
        "median_user_diff": float(np.median(pooled)),
        "prop_users_improved": float(np.mean(pooled > eps)),
        "prop_users_harmed": float(np.mean(pooled < -eps)),
        "prop_users_unchanged": float(np.mean(np.abs(pooled) <= eps)),
        "cohen_dz_user_conditional_descriptive": cohen_dz(pooled),
        "bootstrap_p_descriptive": p,
        "bootstrap_p_report": f"< {1 / n_boot:.4g}" if p == 0 else f"{p:.6g}",
        "n_seeds": len(diffs),
        "n_user_diffs": int(sum(len(d) for d in diffs)),
    }
    seed_rows = []
    for s, d in by_seed.items():
        seed_rows.append(
            {
                "contrast": row["contrast"],
                "metric": metric,
                "treatment": treatment,
                "control": control,
                "seed": s,
                "mean_diff": float(np.mean(d)),
                "median_diff": float(np.median(d)),
                "prop_users_improved": float(np.mean(d > eps)),
                "prop_users_harmed": float(np.mean(d < -eps)),
                "prop_users_unchanged": float(np.mean(np.abs(d) <= eps)),
                "n_users": int(len(d)),
            }
        )
    return row, seed_rows


def main() -> None:
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description="Analyze CoalGameRec per-user result artifacts.")
    ap.add_argument("--run-dir", required=True, help="Run output directory from run_q1_pipeline.py")
    ap.add_argument("--treatment", default="shapley-mc")
    ap.add_argument("--control", default="uniform", help="Single control retained for backward compatibility")
    ap.add_argument("--controls", nargs="+", default=None, help="Controls to compare against treatment")
    ap.add_argument("--metrics", nargs="+", default=["NDCG@20", "HitRate@20"])
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--output-prefix", default="paired_bootstrap_contrasts")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    per_user_path = run_dir / "raw" / "per_user_metrics_all.csv"
    if not per_user_path.exists():
        raise FileNotFoundError(per_user_path)
    per_user = pd.read_csv(per_user_path)
    controls = args.controls if args.controls is not None else [args.control]

    rows = []
    seed_rows = []
    pvals = {}
    for control in controls:
        for metric in args.metrics:
            row, sr = contrast_summary(
                per_user,
                metric=metric,
                treatment=args.treatment,
                control=control,
                n_boot=args.bootstrap_samples,
                seed=args.seed,
            )
            if row is None:
                continue
            rows.append(row)
            seed_rows.extend(sr)
            pvals[row["contrast"]] = row["bootstrap_p_descriptive"]

    out_dir = run_dir / "tables"
    out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    if rows:
        holm = holm_bonferroni(pvals)
        df["holm_reject_0.05"] = df["contrast"].map(lambda k: holm[k]["reject_0.05"])
        df["holm_threshold"] = df["contrast"].map(lambda k: holm[k]["holm_threshold"])
        write_json(out_dir / f"{args.output_prefix}_holm.json", holm)
    df.to_csv(out_dir / f"{args.output_prefix}.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out_dir / f"{args.output_prefix}_by_seed.csv", index=False)
    # Backward-compatible names for the default primary Shapley-vs-uniform call.
    if controls == ["uniform"] and args.output_prefix == "paired_bootstrap_contrasts":
        if rows:
            write_json(out_dir / "holm_primary.json", holm)
        df.to_csv(out_dir / "paired_bootstrap_contrasts.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
