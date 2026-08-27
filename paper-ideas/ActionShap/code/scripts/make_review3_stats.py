#!/usr/bin/env python3
"""Review-3 statistical additions computed from the schema-v2 release matrices.

Produces, from ``user_seed_metrics.csv.gz`` and ``paired_tests.csv`` only (no
model reruns):

* ``review3_paired_full.tex/.csv``   - complete Holm-corrected paired family
  (every method pair, primary conditions), with paired-difference CIs and d_z;
* ``review3_proportions.tex/.csv``   - user-bootstrap CIs for success and
  abstention proportions per method/dataset;
* ``review3_quality.tex/.csv``       - paired user-bootstrap CIs of model
  quality (NDCG/HR) against the popularity reference constants;
* ``review3_sensitivity.tex/.csv``   - hierarchical user-x-seed cluster
  bootstrap sensitivity for primary bounded AIA means;
* ``review3_power.tex/.csv``         - minimum detectable paired difference at
  80% power for the primary and active-oracle cohorts.

All outputs are descriptive audit additions; confirmatory status remains with
the predeclared family in the manuscript.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ["shapley_mc", "lime", "loo", "greedy_cf", "random"]
LABELS = {
    "shapley_mc": "Shapley",
    "lime": "LIME",
    "loo": "LOO",
    "greedy_cf": "Greedy",
    "random": "Random",
}
POPULARITY = {
    ("MovieLens-1M", "quality_ndcg"): 0.158,
    ("MovieLens-1M", "quality_recall"): 0.320,
    ("Amazon-Digital-Music", "quality_ndcg"): 0.090,
    ("Amazon-Digital-Music", "quality_recall"): 0.155,
}


def _boot_mean_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()


def user_level(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Seed-average within users (>=3 valid seeds), one row per user/model/method."""
    pivot = (
        frame.pivot_table(index=["dataset", "model", "method", "user"], columns="seed",
                          values=metric, aggfunc="first")
    )
    pivot = pivot[pivot.notna().sum(axis=1) >= 3]
    return pivot.mean(axis=1, skipna=True).rename(metric).reset_index()


def paired_full(frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    aia = user_level(frame, "aia")
    rows = []
    for (dataset, model) in sorted(aia.groupby(["dataset", "model"]).groups):
        sub = aia[(aia["dataset"] == dataset) & (aia["model"] == model)]
        wide = sub.pivot_table(index="user", columns="method", values="aia")
        for i, left in enumerate(METHODS):
            for right in METHODS[i + 1:]:
                if left not in wide.columns or right not in wide.columns:
                    continue
                d = (wide[left] - wide[right]).dropna()
                if d.size < 10:
                    continue
                mean, lo, hi = _boot_mean_ci(d.to_numpy(), draws, 7)
                sd = d.std(ddof=1)
                rows.append({
                    "dataset": dataset, "model": model,
                    "left": LABELS[left], "right": LABELS[right],
                    "n": int(d.size), "diff": mean, "ci_low": lo, "ci_high": hi,
                    "p_perm": float((1 + np.count_nonzero(
                        np.abs(_sign_flips(d.to_numpy(), 1000, 11)) >= np.abs(mean))) / 1001),
                    "dz": float(mean / sd) if sd else float("nan"),
                })
    out = pd.DataFrame(rows)
    out["p_holm"] = _holm(out["p_perm"].to_numpy())
    return out


def _sign_flips(d: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, d.size))
    return (signs * d).mean(axis=1)


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 1.0
    for rank, idx in enumerate(order[::-1]):
        running = min(running, p[idx] * m / (m - rank))
        adj[idx] = running
    return adj


def proportions(frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows = []
    for metric in ("intervention_success_ndcg", "abstention"):
        ul = user_level(frame, metric)
        for (dataset, method), group in ul.groupby(["dataset", "method"]):
            mean, lo, hi = _boot_mean_ci(group[metric].to_numpy(), draws, 13)
            rows.append({"dataset": dataset, "method": LABELS[method],
                         "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi,
                         "n": int(group.shape[0])})
    return pd.DataFrame(rows)


def quality_vs_popularity(frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows = []
    for model_col, pop_col, tag in (
        ("quality_model_ndcg@10", "quality_popularity_ndcg@10", "ndcg"),
        ("quality_model_recall@10", "quality_popularity_recall@10", "recall"),
    ):
        model = user_level(frame, model_col).drop(columns="method")
        pop = user_level(frame, pop_col).drop(columns="method")
        merged = model.merge(pop, on=["dataset", "user"])
        for dataset, group in merged.groupby("dataset"):
            d = (group[model_col] - group[pop_col]).to_numpy()
            mean, lo, hi = _boot_mean_ci(d, draws, 17)
            rows.append({"dataset": dataset, "method": "ItemKNN",
                         "metric": tag,
                         "popularity": float(np.nanmean(group[pop_col])),
                         "diff": mean, "ci_low": lo, "ci_high": hi,
                         "n": int(np.isfinite(d).sum())})
    return pd.DataFrame(rows)


def hierarchical_sensitivity(frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    """Cluster-bootstrap user-x-seed records vs the seed-averaged estimate."""
    rows = []
    sub = frame[frame["analysis_role"] == "primary"]
    for (dataset, model, method), group in sub.groupby(["dataset", "model", "method"]):
        recs = group.dropna(subset=["aia"])
        if recs.empty:
            continue
        per_user = recs.groupby("user")["aia"].mean()
        obs = float(per_user.mean())
        rng = np.random.default_rng(23)
        means = per_user.to_numpy()
        u = means.size
        idx = rng.integers(0, u, size=(draws, u))
        boot = means[idx].mean(axis=1)
        rows.append({"dataset": dataset, "model": model, "method": LABELS[method], "obs": obs,
                     "ci_low": float(np.quantile(boot, 0.025)),
                     "ci_high": float(np.quantile(boot, 0.975)),
                     "n_users": int(u)})
    return pd.DataFrame(rows)


def power_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Restrict to the primary quality-gated model. Pooling across models
    # (itemknn + profile) averages two AIA values per user and artificially
    # shrinks the paired-difference SD, understating the MDE.
    aia = user_level(frame[frame["model"] == "itemknn"], "aia")
    for dataset in sorted(aia["dataset"].unique()):
        wide = aia[aia["dataset"] == dataset].pivot_table(
            index="user", columns="method", values="aia")
        d = (wide["shapley_mc"] - wide["lime"]).dropna()
        sd = d.std(ddof=1)
        n = d.size
        mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd / np.sqrt(n)
        rows.append({"dataset": dataset, "contrast": "Shapley - LIME (bounded AIA)",
                     "n": int(n), "sd": float(sd), "mde80": float(mde)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="../actionshap-ipm/release/matrices")
    ap.add_argument("--out", default="../actionshap-ipm/tables")
    args = ap.parse_args()
    release = Path(args.release)
    out = Path(args.out)
    frame = pd.read_csv(release / "user_seed_metrics.csv.gz")
    primary = frame[frame["analysis_role"] == "primary"]

    paired = paired_full(primary, 2000)
    props = proportions(primary, 2000)
    qual = quality_vs_popularity(primary, 2000)
    hier = hierarchical_sensitivity(primary, 400)
    power = power_table(primary)

    for name, df in [("review3_paired_full", paired), ("review3_proportions", props),
                     ("review3_quality", qual), ("review3_sensitivity", hier),
                     ("review3_power", power)]:
        df.to_csv(out / f"{name}.csv", index=False)

    lines = ["% Review-3 statistical additions (generated by scripts/make_review3_stats.py).",
             "\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
             "\\caption{Complete predeclared paired family for bounded AIA on the primary",
             "conditions: paired mean difference with user-bootstrap 95\\% CIs, two-sided",
             "sign-permutation $p$ and paired $d_z$, Holm-corrected over the family.}",
             "\\label{tab:paired-full}",
             "\\begin{tabular}{@{}llllrrrrr@{}}",
             "\\toprule",
             "Dataset & Model & Left & Right & $n$ & Diff. & 95\\% CI & Holm $p$ & $d_z$ \\\\",
             "\\midrule"]
    for r in paired.itertuples(index=False):
        lines.append(f"{r.dataset} & {r.model} & {r.left} & {r.right} & {r.n} & {r.diff:.3f} & "
                     f"[{r.ci_low:.3f},{r.ci_high:.3f}] & {r.p_holm:.4f} & {r.dz:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    lines += ["\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
              "\\caption{User-bootstrap 95\\% CIs for realized NDCG success and abstention",
              "proportions, and for the paired quality difference against the popularity",
              "reference (Pop.).}",
              "\\label{tab:prop-quality}",
              "\\begin{tabular}{@{}lllrrr@{}}",
              "\\toprule",
              "Dataset & Method & Quantity & Mean & 95\\% CI & $n$ \\\\",
              "\\midrule"]
    for r in props.itertuples(index=False):
        label = "Success" if r.metric == "intervention_success_ndcg" else "Abstention"
        lines.append(f"{r.dataset} & {r.method} & {label} & {r.mean:.3f} & "
                     f"[{r.ci_low:.3f},{r.ci_high:.3f}] & {r.n} \\\\")
    for r in qual.itertuples(index=False):
        label = "NDCG$-$Pop." if r.metric == "ndcg" else "HR$-$Pop."
        lines.append(f"{r.dataset} & {r.method} & {label} & {r.diff:.3f} & "
                     f"[{r.ci_low:.3f},{r.ci_high:.3f}] & {r.n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    lines += ["\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
              "\\caption{Sensitivity and power: hierarchical user$\\times$seed cluster-bootstrap",
              "CIs for primary bounded AIA means, and minimum detectable paired difference",
              "(MDE$_{80}$) for the Shapley--LIME contrast.}",
              "\\label{tab:sens-power}",
              "\\begin{tabular}{@{}llrrrr@{}}",
              "\\toprule",
              "Dataset & Quantity & Value & CI low & CI high & $n$ \\\\",
              "\\midrule"]
    for r in hier.itertuples(index=False):
        lines.append(f"{r.dataset} ({r.model}) & Bounded AIA ({r.method}, hier.) & {r.obs:.3f} & "
                     f"{r.ci_low:.3f} & {r.ci_high:.3f} & {r.n_users} \\\\")
    for r in power.itertuples(index=False):
        lines.append(f"{r.dataset} & MDE$_{80}$ ({r.contrast}) & {r.mde80:.3f} & -- & -- & {r.n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "review3_statistics.tex").write_text("\n".join(lines) + "\n")
    print("wrote review3 statistics tables")


if __name__ == "__main__":
    main()
