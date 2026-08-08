#!/usr/bin/env python3
"""Paired-contrast analysis for a matched-controls (v4) confirmatory run.

Primary family (Holm per dataset): LOO-marginal vs every matched control
(uniform, additive-pref, attention, heuristic-pop, valid-sim, valid-linear)
x {NDCG@20, HitRate@20}  ->  F = 12.

Secondary family (Holm per dataset): the validation-access effect itself,
{valid-sim, valid-linear} vs uniform x 2 metrics  ->  F = 4.

Methodology is identical to the v3 analyses (coalgamerec.stats:
within-seed user bootstrap B=2000, seed-mean estimand, percentile CIs,
descriptive d_z, Holm-Bonferroni).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coalgamerec.stats import bootstrap_ci, cohen_dz, holm_bonferroni  # noqa: E402
from coalgamerec.utils import write_json  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 20260804  # identical to the frozen v3 analysis config
METRICS = ["NDCG@20", "HitRate@20"]
EPS = 1e-12


def paired_seed_diffs(per_user: pd.DataFrame, metric: str, treatment: str, control: str) -> dict[int, np.ndarray]:
    df = per_user[per_user.metric == metric]
    diffs: dict[int, np.ndarray] = {}
    for seed, g in df.groupby("seed"):
        piv = g.pivot_table(index="user", columns="family", values="value", aggfunc="first")
        if treatment not in piv.columns or control not in piv.columns:
            continue
        d = (piv[treatment] - piv[control]).dropna().to_numpy(dtype=float)
        diffs[int(seed)] = d
    return diffs


def contrast_row(per_user: pd.DataFrame, metric: str, treatment: str, control: str) -> dict | None:
    by_seed = paired_seed_diffs(per_user, metric, treatment, control)
    diffs = [d for d in by_seed.values() if len(d)]
    if not diffs:
        return None
    obs, lo, hi = bootstrap_ci(diffs, n_boot=N_BOOT, seed=BOOT_SEED)
    pooled = np.concatenate(diffs)
    rng = np.random.default_rng(BOOT_SEED)
    vals = []
    for _ in range(N_BOOT):
        vals.append(np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs]))
    vals = np.asarray(vals)
    if obs >= 0:
        p = float(2 * min(np.mean(vals <= 0), np.mean(vals >= 0)))
    else:
        p = float(2 * min(np.mean(vals >= 0), np.mean(vals <= 0)))
    return {
        "contrast": f"{treatment}_vs_{control}_{metric}",
        "metric": metric, "treatment": treatment, "control": control,
        "mean_diff_conditional_user": obs, "ci95_low": lo, "ci95_high": hi,
        "median_user_diff": float(np.median(pooled)),
        "prop_users_improved": float(np.mean(pooled > EPS)),
        "prop_users_harmed": float(np.mean(pooled < -EPS)),
        "prop_users_unchanged": float(np.mean(np.abs(pooled) <= EPS)),
        "cohen_dz_user_conditional_descriptive": cohen_dz(pooled),
        "bootstrap_p_descriptive": p,
        "bootstrap_p_report": f"< {1 / N_BOOT:.4g}" if p == 0 else f"{p:.6g}",
        "n_seeds": len(diffs), "n_user_diffs": int(sum(len(d) for d in diffs)),
    }


def run_family(per_user: pd.DataFrame, treatment: str, controls: list[str], prefix: str, run_dir: Path) -> None:
    rows = []
    for control in controls:
        for metric in METRICS:
            row = contrast_row(per_user, metric, treatment, control)
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows)
    holm = holm_bonferroni({r["contrast"]: r["bootstrap_p_descriptive"] for r in rows})
    df["holm_threshold"] = df.contrast.map(lambda c: holm[c]["holm_threshold"])
    df["holm_reject_0.05"] = df.contrast.map(lambda c: holm[c]["reject_0.05"])
    (run_dir / "tables").mkdir(exist_ok=True)
    df.to_csv(run_dir / "tables" / f"{prefix}.csv", index=False)
    write_json(run_dir / "tables" / f"{prefix}_holm.json", holm)
    print(df[["contrast", "mean_diff_conditional_user", "ci95_low", "ci95_high",
              "bootstrap_p_report", "holm_reject_0.05"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    per_user = pd.read_csv(run_dir / "raw" / "per_user_metrics_all.csv.gz")
    controls = ["uniform", "additive-pref", "attention", "heuristic-pop", "valid-sim", "valid-linear"]
    run_family(per_user, "loo-marginal", controls, "paired_bootstrap_loo_vs_matched_controls", run_dir)
    run_family(per_user, "valid-sim", ["uniform"], "paired_bootstrap_valid_access_sim", run_dir)
    run_family(per_user, "valid-linear", ["uniform"], "paired_bootstrap_valid_access_linear", run_dir)


if __name__ == "__main__":
    main()
