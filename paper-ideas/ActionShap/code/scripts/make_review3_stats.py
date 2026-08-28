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
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ["shapley_mc", "lime", "loo", "greedy_cf", "random"]
# The declared protocol in the manuscript: 10,000 plus-one two-sided sign
# permutations and 10,000 user-bootstrap resamples.
PERMUTATION_DRAWS = 10_000
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
                    # Review-9 Issue 18: the audit tables must use the SAME
                    # inference protocol as the release, otherwise a contrast
                    # printed here disagrees with the same contrast printed in
                    # the predeclared family for a purely procedural reason
                    # (1,000 versus 10,000 sign flips changes both the p-value
                    # floor and the Holm multiplier).
                    "p_perm": float((1 + np.count_nonzero(
                        np.abs(_sign_flips(d.to_numpy(), PERMUTATION_DRAWS, 11)) >= np.abs(mean)))
                        / (PERMUTATION_DRAWS + 1)),
                    "permutation_draws": PERMUTATION_DRAWS,
                    "exceedances": int(np.count_nonzero(
                        np.abs(_sign_flips(d.to_numpy(), PERMUTATION_DRAWS, 11)) >= np.abs(mean))),
                    "dz": float(mean / sd) if sd else float("nan"),
                })
    out = pd.DataFrame(rows)
    # Holm within each (dataset, model) block, i.e. within the declared
    # dataset-model-condition-metric family of ten pairs -- previously the
    # correction ran over every block at once, which inflated adjusted p-values
    # in a way no document described.
    if len(out):
        adjusted = pd.Series(index=out.index, dtype=float)
        for _, group in out.groupby(["dataset", "model"], sort=False):
            adjusted.loc[group.index] = _holm(group["p_perm"].to_numpy())
        out["p_holm"] = adjusted.to_numpy()
    return out


def _sign_flips(d: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, d.size))
    return (signs * d).mean(axis=1)


def _holm(p: np.ndarray) -> np.ndarray:
    """Standard step-down Holm: cumulative maximum of (m-rank) p, capped at 1.

    The previous formulation took a running *minimum* from the largest p-value
    downwards, which silently returned the uncorrected smallest p whenever the
    whole family sat at the permutation floor -- i.e. it printed "Holm" $p$
    values that were not Holm-adjusted.
    """
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
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


def parse_paired_full(path: Path) -> dict[tuple[str, str, str, str], tuple[float, float]]:
    """Read the committed ``tab:paired-full`` block: (dataset, model, left, right)."""
    labels = {v: k for k, v in LABELS.items()}
    rows: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    if not path.exists():
        return rows
    inside = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("Dataset & Model & Left"):
            inside = True
            continue
        if inside and line.startswith("\\bottomrule"):
            inside = False
            continue
        if not inside or "&" not in line or not line.endswith("\\\\"):
            continue
        cells = [c.strip() for c in line.rstrip("\\").split("&")]
        if len(cells) < 9:
            continue
        try:
            rows[(cells[0], cells[1], labels.get(cells[2], cells[2]), labels.get(cells[3], cells[3]))] = (
                float(cells[5]),
                float(cells[7]),
            )
        except ValueError:
            continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="../actionshap-ipm/release/matrices")
    ap.add_argument("--out", default="../actionshap-ipm/tables")
    ap.add_argument("--mirror", default="../acmart-primary/tables",
                    help="second table directory kept byte-identical")
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare against the committed tables; write nothing")
    args = ap.parse_args()
    release = Path(args.release)
    out = Path(args.out)
    frame = pd.read_csv(release / "user_seed_metrics.csv.gz")
    primary = frame[frame["analysis_role"] == "primary"]

    paired = paired_full(primary, 2000)
    if args.check:
        published = parse_paired_full(Path(args.mirror) / "review3_statistics.tex")
        problems: list[str] = []
        for (dataset, model, left, right), (diff, holm) in published.items():
            # the recomputation stores display labels, not method keys
            row = paired[
                (paired["dataset"] == dataset)
                & (paired["model"] == model)
                & (paired["left"] == LABELS.get(left, left))
                & (paired["right"] == LABELS.get(right, right))
            ]
            if row.empty:
                problems.append(f"committed row absent from recomputation: {dataset}/{model}/{left}-{right}")
                continue
            rec = row.iloc[0]
            if abs(float(rec["diff"]) - diff) > 6e-4:
                problems.append(
                    f"mean difference {dataset}/{model}/{left}-{right}: table {diff} vs recomputed {rec['diff']:.4f}"
                )
            if abs(float(rec["p_holm"]) - holm) > 6e-5:
                problems.append(
                    f"Holm p {dataset}/{model}/{left}-{right}: table {holm} vs recomputed {float(rec['p_holm']):.4f}"
                )
        print(json.dumps(
            {"status": "PASS" if not problems else "FAIL",
             "compared_rows": len(published), "problems": problems[:20]}, indent=1))
        if problems:
            raise SystemExit(1)
        return
    props = proportions(primary, 2000)
    qual = quality_vs_popularity(primary, 2000)
    hier = hierarchical_sensitivity(primary, 400)
    power = power_table(primary)

    for name, df in [("review3_paired_full", paired), ("review3_proportions", props),
                     ("review3_quality", qual), ("review3_sensitivity", hier),
                     ("review3_power", power)]:
        df.to_csv(out / f"{name}.csv", index=False)

    lines = ["% Statistical additions generated by scripts/make_review3_stats.py.",
             "\\begin{table}[!htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
             "\\caption{Complete predeclared paired family for bounded AIA on the primary",
             "conditions: paired mean difference with user-bootstrap 95\\% CIs, two-sided",
             "sign-permutation $p$ and paired $d_z$, Holm-corrected within the family.",
             r"($10{,}000$ plus-one sign permutations, i.e.\ the same",
             "declared budget as the release; the minimum attainable adjusted $p$ in a",
             "ten-pair family is therefore $10/10{,}001=0.0010$.)}",
             "\\label{tab:paired-full}",
             "\\begin{tabular}{@{}llllrrrrr@{}}",
             "\\toprule",
             "Dataset & Model & Left & Right & $n$ & Diff. & 95\\% CI & Holm $p$ & $d_z$ \\\\",
             "\\midrule"]
    for r in paired.itertuples(index=False):
        lines.append(f"{r.dataset} & {r.model} & {r.left} & {r.right} & {r.n} & {r.diff:.3f} & "
                     f"[{r.ci_low:.3f},{r.ci_high:.3f}] & {r.p_holm:.4f} & {r.dz:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    lines += ["\\begin{table}[!htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
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
    lines += ["\\begin{table}[!htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
              "\\caption{Sensitivity and power: hierarchical user$\\times$seed cluster-bootstrap",
              "CIs for primary bounded AIA means, and minimum detectable paired difference",
              r"(MDE$_{80}$) for the Shapley--LIME contrast, $\mathrm{MDE}_{80}=(z_{0.975}+z_{0.8})\,s_d/\sqrt{n}$",
             "with $s_d$ the SD of the seed-averaged paired user differences over distinct",
             "users (primary ItemKNN only; $n=993$ on Amazon after the constant-vector",
             "exclusion). These values match the main-text MDE statement.}",
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
    mirror = Path(args.mirror)
    if mirror.is_dir():
        (mirror / "review3_statistics.tex").write_text("\n".join(lines) + "\n")
    print("wrote review3 statistics tables (both mirrors)")


if __name__ == "__main__":
    main()
