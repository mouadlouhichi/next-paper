#!/usr/bin/env python3
"""Review-9 inference audits derived from the frozen release matrices.

Every table here answers a reviewer request that can be settled with data that
is already in the reproducibility release, i.e. without new recommender runs:

* S31 inclusion-flow table + all-user sensitivity                  [#10]
* S32 confirmatory multiplicity map with raw exceedance counts     [#18]
* S33 attribution-utility x outcome-utility factorial              [#4]
* S34 studentized (bootstrap-t) robustness of the paired sign-flip
  tests, with the skewness that motivates it                        [#15, #18]
* S35 per-user Monte Carlo error propagated into the reported
  alignment statistic                                              [#9]

Writes:

* code/results/review9/review9_statistics.json
* tables/review9_statistics.tex in BOTH the acmart-primary and the
  actionshap-ipm mirrors, written from one string so the two documents
  cannot drift (Issue #17's cross-document consistency requirement).

Usage:  python3 scripts/make_review9_stats.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT.parent / "actionshap-ipm" / "release" / "matrices"
# AES_REVIEW9_RESULTS lets a pilot's scratch payloads be pushed through the real
# generators before a 10-hour cohort is spent: a run type whose table builder
# crashes on the payload shape is a defect in this file, and the only cheap place
# to find that out is while the cohort is still 12 users.
OUT = Path(os.environ.get("AES_REVIEW9_RESULTS") or (ROOT / "results" / "review9"))
MIRRORS = [
    ROOT.parent / "acmart-primary" / "tables",
    ROOT.parent / "actionshap-ipm" / "tables",
]

DATASETS = ("MovieLens-1M", "Amazon-Digital-Music")
SHORT = {"MovieLens-1M": "MovieLens", "Amazon-Digital-Music": "Amazon"}
METHODS = {
    "shapley_mc": "MC Shapley",
    "lime": "LIME",
    "loo": "LOO",
    "greedy_cf": "Greedy seq. del.",
    "random": "Random control",
}
_ROW_END = chr(92) * 2  # the LaTeX row terminator, built without source-level escapes
BOOT = 10_000
FLIP = 50_000  # sign-flip draws for the paired ablation contrasts (see S36)
FAMILY_KEYS = [
    "dataset",
    "model",
    "evaluation_mode",
    "utility",
    "analysis_role",
    "condition",
    "metric",
]


def key_of(*parts: str) -> int:
    """Deterministic integer for entropy (never ``hash``, which is salted)."""
    text = "|".join(str(p) for p in parts)
    value = 0
    for character in text:
        value = (value * 1_315_423_911 + ord(character)) % (2**61 - 1)
    return value


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([key_of(*[str(p) for p in parts])]))


def bootstrap_ci(values: np.ndarray, *seed_parts) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    gen = rng_for(*seed_parts, v.size)
    idx = gen.integers(0, v.size, size=(BOOT, v.size))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def studentized_p(values: np.ndarray, *seed_parts, draws: int = 4000) -> float:
    """Two-sided studentized bootstrap-t p-value for a mean.

    Needs no symmetry assumption on the per-user paired differences and is
    second-order accurate under skew, which is why Issue #15 asks for it as the
    robust alternative to the sign-flip test.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n < 3:
        return float("nan")
    gen = rng_for(*seed_parts, n)
    idx = gen.integers(0, n, size=(draws, n))
    samples = v[idx]
    means = samples.mean(axis=1)
    ses = samples.std(axis=1, ddof=1) / np.sqrt(n)
    sd = v.std(ddof=1)
    if sd == 0:
        return 1.0
    t_obs = v.mean() / (sd / np.sqrt(n))
    keep = ses > 0
    if not keep.any():
        return 1.0
    t_star = (means[keep] - v.mean()) / ses[keep]
    lower = float((t_star <= -abs(t_obs)).mean())
    upper = float((t_star >= abs(t_obs)).mean())
    return float(min(1.0, 2 * min(lower, upper) + 1 / (t_star.size + 1)))


def sign_flip_p(values: np.ndarray, *seed_parts, draws: int = 10_000) -> float:
    """Plus-one paired sign-flip p-value (the published test), recomputed."""
    v = np.asarray(values, dtype=float)
    gen = rng_for(*seed_parts, v.size)
    signs = gen.choice(np.array([-1.0, 1.0]), size=(draws, v.size))
    null = (signs * v).mean(axis=1)
    return float((1 + int((np.abs(null) >= abs(v.mean())).sum())) / (draws + 1))


def user_means(frame: pd.DataFrame, column: str) -> pd.Series:
    """Seed-mean within distinct user, the release's aggregation order."""
    if column not in frame:
        return pd.Series(dtype=float)
    return frame.groupby("user")[column].mean()


def primary(d: pd.DataFrame, dataset: str, method: str, condition: str = "primary") -> pd.DataFrame:
    return d[
        (d.dataset == dataset)
        & (d.model == "itemknn")
        & (d.condition == condition)
        & (d.method == method)
    ]


# --------------------------------------------------------------------------
# S31  inclusion flow + all-user sensitivity (Issue #10)
# --------------------------------------------------------------------------

def inclusion_flow(d: pd.DataFrame) -> list[dict]:
    rows = []
    for ds in DATASETS:
        base = d[(d.dataset == ds) & (d.model == "itemknn") & (d.condition == "primary")]
        users = pd.Index(sorted(base.user.unique()))
        per_user_seeds = base.groupby(["user", "seed"]).aia.apply(lambda s: s.notna().any()).unstack()
        n_seed_valid = int((per_user_seeds.sum(axis=1) >= 3).reindex(users).fillna(False).sum())
        shap = primary(d, ds, "shapley_mc")
        rows.append(
            {
                "dataset": SHORT[ds],
                "cohort": int(len(users)),
                "seed_valid": n_seed_valid,
                "aia_defined": int(shap[shap.aia.notna()].user.nunique()),
                "positive_oracle_tm": int(shap[shap.normalized_regret_primary.notna()].user.nunique()),
                "finite_ndcg_effect": int(shap[shap.joint_effect_ndcg.notna()].user.nunique()),
                "active_oracle_ndcg": int(shap[shap.normalized_regret_ndcg.notna()].user.nunique()),
            }
        )
    return rows


def all_user_sensitivity(d: pd.DataFrame) -> list[dict]:
    rows = []
    quantities = [
        ("bounded AIA", "aia", False),
        ("realized target-margin effect", "joint_effect_target_margin", False),
        ("realized NDCG effect", "joint_effect_ndcg", False),
        ("normalized regret", "normalized_regret_primary", True),
    ]
    for ds in DATASETS:
        for left, right in (("shapley_mc", "lime"), ("shapley_mc", "loo")):
            a, b = primary(d, ds, left), primary(d, ds, right)
            for label, column, oracle_conditional in quantities:
                la, lb = user_means(a, column), user_means(b, column)
                if la.empty or lb.empty:
                    continue
                diff = (la - lb)[la.notna() & lb.notna()]
                if diff.empty:
                    continue
                if oracle_conditional:
                    ok = user_means(a, "joint_regret_primary").notna()
                    pub = diff[ok.reindex(diff.index).fillna(False)]
                else:
                    pub = diff
                v_all = diff.to_numpy(dtype=float)
                if not np.isfinite(v_all).all():
                    keep = np.isfinite(v_all)
                    v_all, pub = v_all[keep], diff[keep]
                if v_all.size < 3:
                    continue
                lo, hi = bootstrap_ci(v_all, "sens", ds, label)
                rows.append(
                    {
                        "dataset": SHORT[ds],
                        "contrast": f"{METHODS[left]} -- {METHODS[right]}",
                        "quantity": label,
                        "n_published": int(pub.size),
                        "mean_published": float(pub.mean()),
                        "n_all_user": int(v_all.size),
                        "mean_all_user": float(v_all.mean()),
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )
        # Gap-vs-regret falsification statistic, with and without the
        # positive-oracle conditioning (zero-oracle users enter with regret 0).
        shap = primary(d, ds, "shapley_mc")
        gap, regret = user_means(shap, "actionability_gap"), user_means(shap, "normalized_regret_primary")
        pub = pd.concat([gap, regret], axis=1).dropna()
        filled = regret.fillna(0.0)
        allp = pd.concat([gap, filled], axis=1)[gap.notna()].dropna()
        r_pub = spearmanr(pub.iloc[:, 0], pub.iloc[:, 1])
        r_all = spearmanr(allp.iloc[:, 0], allp.iloc[:, 1])
        rows.append(
            {
                "dataset": SHORT[ds],
                "contrast": "Shapley gap vs. target-margin regret",
                "quantity": "Spearman $\\rho$ (Spearman, not a mean difference)",
                "n_published": int(len(pub)),
                "mean_published": float(r_pub.statistic),
                "n_all_user": int(len(allp)),
                "mean_all_user": float(r_all.statistic),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
            }
        )
    return rows


# --------------------------------------------------------------------------
# S32  multiplicity map with raw exceedance counts (Issue #18)
# --------------------------------------------------------------------------

def holm(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    adjusted = np.empty_like(raw)
    running = 0.0
    m = raw.size
    for rank, index in enumerate(np.argsort(raw, kind="stable")):
        running = max(running, (m - rank) * raw[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def family_label(rec) -> str:
    dataset = SHORT.get(rec.dataset, rec.dataset)
    return f"{dataset} / {rec.condition} / {rec.utility} / {rec.metric}"


def multiplicity_map(paired: pd.DataFrame, review7: pd.DataFrame) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    sub = paired[paired.model == "itemknn"].copy()
    mismatches = 0
    for _, group in sub.groupby(FAMILY_KEYS, dropna=False):
        raw = group.permutation_p.to_numpy(dtype=float)
        recomputed = holm(raw)
        m = len(group)
        for (_, rec), adj in zip(group.iterrows(), recomputed):
            draws = int(rec.permutation_draws)
            exceed = int(round(float(rec.permutation_p) * (draws + 1))) - 1
            if abs(adj - float(rec.p_holm)) > 5e-4:
                mismatches += 1
            rows.append(
                {
                    "source": "per-metric family",
                    "family": family_label(rec),
                    "family_size": m,
                    "left": METHODS.get(rec.left, rec.left),
                    "right": METHODS.get(rec.right, rec.right),
                    "n_users": int(rec.n_users),
                    "mean_difference": float(rec.mean_difference),
                    "raw_p": float(rec.permutation_p),
                    "exceedances": exceed,
                    "draws": draws,
                    "published_holm_p": float(rec.p_holm),
                    "recomputed_holm_p": float(adj),
                }
            )
    # The predeclared success/abstention family is a single 12-contrast family
    # spanning both datasets (that is what makes its Holm multiplier differ
    # from the per-metric families'), so it is Holm-corrected as one group and
    # listed here to show the same contrast under both corrections.
    for _, group in review7.groupby(lambda _: 0):
        raw = group.p_plusone.to_numpy(dtype=float)
        recomputed = holm(raw)
        m = len(group)
        for (_, rec), adj in zip(group.iterrows(), recomputed):
            rows.append(
                {
                    "source": "predeclared 12-contrast family",
                    "family": f"{SHORT.get(rec.dataset, rec.dataset)} / success+abstention / {rec.quantity}",
                    "family_size": m,
                    "left": METHODS.get(rec.left, rec.left),
                    "right": METHODS.get(rec.right, rec.right),
                    "n_users": int(rec.n),
                    "mean_difference": float(rec.mean_diff),
                    "raw_p": float(rec.p_plusone),
                    "exceedances": int(round(float(rec.p_plusone) * (10_000 + 1))) - 1,
                    "draws": 10_000,
                    "published_holm_p": float(rec.holm_p),
                    "recomputed_holm_p": float(adj),
                }
            )
    stats = {
        "tests": len(rows),
        "families": len({r["family"] for r in rows}),
        "holm_recomputation_mismatches": mismatches,
        "floor_raw_p_min": float(min(r["raw_p"] for r in rows)),
        "family_sizes": sorted({r["family_size"] for r in rows}),
        "shapley_lime_success_rows": [
            r
            for r in rows
            if {r["left"], r["right"]} == {"MC Shapley", "LIME"}
            and r["family"].endswith(("intervention_success", "Success"))
            and SHORT["MovieLens-1M"] in r["family"]
        ],
    }
    return rows, stats


# --------------------------------------------------------------------------
# S4   regenerate the disputed effects table from the release (Issue #18)
# --------------------------------------------------------------------------

S4_METHOD = {
    "MC Shapley": "shapley_mc",
    "LIME": "lime",
    "LOO": "loo",
    "Greedy seq.\\ del.": "greedy_cf",
    "Random": "random",
}
S4_METRIC = {"Success": "intervention_success_ndcg", "$\\Delta$NDCG": "joint_effect_ndcg"}
S4_DATASET = {"MovieLens": "MovieLens-1M", "Amazon": "Amazon-Digital-Music"}


def parse_s4(path: Path) -> dict[tuple[str, str, str, str], float]:
    """Read the Holm column of an already committed S4 file, for the audit only.

    Rows are never *generated* from this file -- the release is the source -- but
    comparing against it tells us exactly which printed values the review-9
    regeneration changes.
    """
    published: dict[tuple[str, str, str, str], float] = {}
    if not path or not path.exists():
        return published
    for line in path.read_text().splitlines():
        line = line.strip()
        if "&" not in line or not line.endswith("\\\\"):
            continue
        cells = [c.strip() for c in line.rstrip("\\").split("&")]
        if len(cells) < 10 or cells[0] not in S4_DATASET:
            continue
        try:
            published[(cells[0], cells[1], cells[2], cells[3])] = float(cells[7])
        except ValueError:
            continue
    return published


def s4_rows(paired: pd.DataFrame) -> pd.DataFrame:
    """The S4 grid: both datasets, both NDCG-utility quantities, ten pairs."""
    labels = {v: k for k, v in S4_METHOD.items()}
    wanted = list(S4_METRIC.values())
    sub = paired[
        paired.model.eq("itemknn")
        & paired.condition.eq("primary")
        & paired.metric.isin(wanted)
        & paired.dataset.isin(S4_DATASET.values())
    ].copy()
    sub["left_label"] = sub.left.map(labels)
    sub["right_label"] = sub.right.map(labels)
    sub["metric_label"] = sub.metric.map({k: v for v, k in S4_METRIC.items()})
    sub["dataset_label"] = sub.dataset.map({v: k for k, v in S4_DATASET.items()})
    return sub.sort_values(["dataset_label", "metric_label", "left_label", "right_label"], kind="stable")


def regenerate_s4(paired: pd.DataFrame, review7: pd.DataFrame, legacy: Path) -> tuple[str, dict]:
    """Rebuild S4 from ``paired_tests.csv``, publishing exceedance counts.

    Issue #18 noted that the same success contrast carries different
    Holm-adjusted $p$-values in the replication tables (.0066) and in the
    predeclared family (.0216). The cause is provenance, not arithmetic: that
    table came from an audit generator using 1,000 sign flips, Holm-corrected
    over every dataset/model block at once, whereas the release uses the
    declared 10,000 draws and the dataset--model--condition--metric families.
    Rebuilding the table from the released per-test file removes the
    disagreement and prints the raw exceedance count and the authoritative
    adjusted $p$ side by side. Rows are generated from the release, so the
    function is idempotent; the previously committed file is read only to
    report which printed values the regeneration changes.
    """
    twelve = {
        (SHORT.get(rec.dataset, rec.dataset), rec.quantity, rec.left, rec.right): float(rec.holm_p)
        for _, rec in review7.iterrows()
    }
    quantity_of = {"Success": "Success", "$\\Delta$NDCG": "Delta"}
    previously = parse_s4(legacy)
    report = {"rows": 0, "matched_previous_rows": 0, "holm_changes": [], "unmatched": []}
    row_end = "\\" + "\\"  # the LaTeX row terminator
    body: list[str] = []
    for rec in s4_rows(paired).itertuples(index=False):
        report["rows"] += 1
        draws = int(rec.permutation_draws)
        exceed = int(round(float(rec.permutation_p) * (draws + 1))) - 1
        key = (rec.dataset_label, rec.metric_label, rec.left_label, rec.right_label)
        old = previously.get(key)
        if old is None:
            report["unmatched"].append("|".join(key))
        else:
            report["matched_previous_rows"] += 1
            if abs(old - float(rec.p_holm)) > 6e-5:
                report["holm_changes"].append(
                    {
                        "row": "|".join(key),
                        "published_previously": old,
                        "released_now": float(rec.p_holm),
                        "raw_p": float(rec.permutation_p),
                        "exceedances": exceed,
                    }
                )
        holm12 = twelve.get((rec.dataset_label, quantity_of[rec.metric_label], rec.left, rec.right))
        holm12_str = "--" if holm12 is None else f"{holm12:.4f}"
        body.append(
            f"{rec.dataset_label} & {rec.metric_label} & {rec.left_label} & {rec.right_label} & "
            f"{rec.mean_difference:.4f} & {rec.ci95_low:.4f} & {rec.ci95_high:.4f} & "
            f"{rec.p_holm:.4f} & {exceed} & {holm12_str} & "
            f"{rec.cohens_dz:.3f} & {int(rec.n_users)} \\\\\\n"[:-2]
        )
    report["holm_changes_count"] = len(report["holm_changes"])

    L = [
        "% Generated by scripts/make_review9_stats.py from release/paired_tests.csv.",
        "% Rebuilt for the review-9 response: identical estimands, declared",
        "% inference protocol, published exceedance counts and the authoritative",
        "% 12-contrast Holm value (Issue #18).",
        "\\begin{longtable}{@{}p{.085\\textwidth}p{.105\\textwidth}p{.095\\textwidth}p{.095\\textwidth}"
        "rrrrcrrr@{}}",
        "\\caption{Joint NDCG effect and success comparisons for the primary ItemKNN",
        "cohort, rebuilt from the frozen release matrices so that every printed value is",
        "reproducible. Inference is the declared protocol: $10{,}000$ plus-one two-sided",
        "sign permutations on seed-averaged paired user differences, Holm-corrected",
        "within the dataset--model--condition--metric family of ten declared pairs.",
        "\\emph{Exceed.} is the raw count of permuted statistics at least as extreme as",
        "the observed one, so the printed $p$ equals $(1+\\#)/(R+1)$ exactly; $\\#=0$",
        "with $R=10{,}000$ in a ten-test family is the floor $10/10{,}001=0.0010$ and",
        "not censoring. \\emph{Holm $p$ (12)} is the adjusted $p$ inside the single",
        "predeclared 12-contrast success/abstention family, which is authoritative for",
        "\\emph{Success} rows; a dash marks contrasts outside that family.}",
        "\\label{tab:S4}\\\\",
        "\\toprule",
        "Dataset & Metric & Left & Right & $\\Delta$ & CI low & CI high & Holm $p$ & "
        "Exceed. & Holm $p$ (12) & $d_z$ & $n$ \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{12}{c}{Table S4 -- continued from previous page}\\\\",
        "\\toprule",
        "Dataset & Metric & Left & Right & $\\Delta$ & CI low & CI high & Holm $p$ & "
        "Exceed. & Holm $p$ (12) & $d_z$ & $n$ \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{12}{r}{Continued on next page}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    L.extend(body)
    L.append("\\end{longtable}")
    return "\n".join(L) + "\n", report


# --------------------------------------------------------------------------
# S36  Monte Carlo precision of the plus-one permutation p-values (Issue #18)
# --------------------------------------------------------------------------

def permutation_precision(paired: pd.DataFrame, d: pd.DataFrame, draws: int = 50_000) -> list[dict]:
    """Re-estimate the disputed raw permutation p-values with many more draws.

    The reviewer asks whether a printed $p$ reflects censoring or the actual
    number of permutation draws. For the two pipelines that report the same
    contrast, the raw $p$-values differ slightly (15 versus 23 exceedances of
    $10{,}000$): both are Monte Carlo estimates of the same sign-flip $p$, whose
    sampling error at that magnitude is $\\approx\\sqrt{p(1-p)/R}=0.0004$. This
    recomputes the statistic from the released per-user differences with
    $50{,}000$ draws and reports the Monte Carlo standard error, so a reader can
    check that each published value is within Monte Carlo error of the
    converged one.
    """
    disputed = [
        ("MovieLens-1M", "shapley_mc", "lime", "intervention_success_ndcg", 0.0016, 0.0024),
        ("MovieLens-1M", "shapley_mc", "loo", "intervention_success_ndcg", 0.0036, None),
        ("MovieLens-1M", "shapley_mc", "lime", "joint_effect_ndcg", 0.0053, None),
        ("MovieLens-1M", "shapley_mc", "lime", "aia", 0.0001, None),
        ("Amazon-Digital-Music", "shapley_mc", "lime", "intervention_success_ndcg", None, 0.725127),
    ]
    rows = []
    for ds, left, right, metric, p_metric, p12 in disputed:
        column = {
            "intervention_success_ndcg": "intervention_success_ndcg",
            "joint_effect_ndcg": "joint_effect_ndcg",
            "aia": "aia",
        }[metric]
        a, b = primary(d, ds, left), primary(d, ds, right)
        la, lb = user_means(a, column), user_means(b, column)
        v = (la - lb)[la.notna() & lb.notna()].to_numpy(dtype=float)
        if v.size < 3:
            continue
        gen = rng_for("precision", ds, metric)
        signs = gen.choice(np.array([-1.0, 1.0]), size=(draws, v.size))
        null = (signs * v).mean(axis=1)
        exceed = int((np.abs(null) >= abs(v.mean())).sum())
        p_hat = (1 + exceed) / (draws + 1)
        mc_se = float(np.sqrt(max(p_hat, 1 / (draws + 1)) * (1 - p_hat) / draws))
        # Tolerance is the Monte Carlo error of the PUBLISHED statistic, i.e. of
        # a 10,000-draw estimate, since that is the experiment that produced it;
        # judging agreement at 50,000 draws would demand precision the published
        # values never claimed.
        se_published = float(np.sqrt(max(p_hat, 1 / (draws + 1)) * (1 - p_hat) / 10_000))
        rows.append(
            {
                "dataset": SHORT[ds],
                "contrast": f"{METHODS[left]} -- {METHODS[right]}",
                "metric": metric,
                "n_users": int(v.size),
                "draws": int(draws),
                "exceedances": exceed,
                "p_reestimated": float(p_hat),
                "mc_se": mc_se,
                "published_per_metric_p": p_metric,
                "published_12_contrast_p": p12,
                "se_at_published_draws": se_published,
                "per_metric_within_2se": (
                    None if p_metric is None else bool(abs(p_metric - p_hat) <= 2 * se_published + 1e-12)
                ),
                "family_12_within_2se": (
                    None if p12 is None else bool(abs(p12 - p_hat) <= 2 * se_published + 1e-12)
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------
# S33  attribution-utility x outcome-utility factorial (Issue #4)
# --------------------------------------------------------------------------

def utility_factorial(d: pd.DataFrame) -> list[dict]:
    rows = []
    cells = [
        ("target margin", "aia", "joint_effect_target_margin"),
        ("target margin", "aia", "joint_effect_ndcg"),
        ("NDCG", "aia_ndcg", "joint_effect_target_margin"),
        ("NDCG", "aia_ndcg", "joint_effect_ndcg"),
    ]
    for ds in DATASETS:
        for method in ("shapley_mc", "lime", "loo"):
            base = primary(d, ds, method)
            columns = {c for _, a, c in cells} | {a for _, a, _ in cells}
            per_user = {c: user_means(base, c) for c in columns}
            for attr_label, attr_col, out_col in cells:
                a, o = per_user.get(attr_col), per_user.get(out_col)
                if a is None or o is None or a.empty or o.empty:
                    continue
                frame = pd.concat([a, o], axis=1).dropna()
                if len(frame) < 10:
                    continue
                r = spearmanr(frame.iloc[:, 0], frame.iloc[:, 1])
                rows.append(
                    {
                        "dataset": SHORT[ds],
                        "method": METHODS[method],
                        "attribution_utility": attr_label,
                        "outcome_utility": "target margin" if "target" in out_col else "NDCG",
                        "n_users": int(len(frame)),
                        "spearman": float(r.statistic),
                        "p_value": float(r.pvalue),
                    }
                )
    return rows


# --------------------------------------------------------------------------
# S34  studentized robustness of the paired sign-flip tests (Issue #15)
# --------------------------------------------------------------------------

def studentized_table(d: pd.DataFrame) -> list[dict]:
    rows = []
    targets = [
        ("bounded AIA (Shapley -- LIME)", "aia", "shapley_mc", "lime", False),
        ("bounded AIA (Shapley -- LOO)", "aia", "shapley_mc", "loo", False),
        ("realized NDCG effect (Shapley -- LIME)", "joint_effect_ndcg", "shapley_mc", "lime", False),
        ("target-margin effect (Shapley -- LIME)", "joint_effect_target_margin", "shapley_mc", "lime", False),
        ("normalized regret (Shapley -- LIME)", "normalized_regret_primary", "shapley_mc", "lime", True),
        ("bounded AIA (LIME -- LOO)", "aia", "lime", "loo", False),
    ]
    for ds in DATASETS:
        for label, column, left, right, oracle in targets:
            a, b = primary(d, ds, left), primary(d, ds, right)
            la, lb = user_means(a, column), user_means(b, column)
            if la.empty or lb.empty:
                continue
            diff = (la - lb)[la.notna() & lb.notna()]
            if oracle:
                ok = user_means(a, "joint_regret_primary").notna()
                diff = diff[ok.reindex(diff.index).fillna(False)]
            v = diff.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size < 3:
                continue
            rows.append(
                {
                    "dataset": SHORT[ds],
                    "quantity": label,
                    "n_users": int(v.size),
                    "mean_difference": float(v.mean()),
                    "skew": float(pd.Series(v).skew()),
                    "sign_flip_p": sign_flip_p(v, "sf", ds, label),
                    "studentized_p": studentized_p(v, "st", ds, label),
                }
            )
    return rows


# --------------------------------------------------------------------------
# S35  Monte Carlo error propagated into alignment (Issue #9)
# --------------------------------------------------------------------------

def mc_propagation() -> list[dict]:
    rows = []
    for ds, slug in (("MovieLens", "movielens"), ("Amazon", "amazon")):
        path = ROOT / "results" / "review8" / f"mcse_n20_{slug}.json"
        if not path.exists():
            continue
        records = json.loads(path.read_text())["records"]
        delta = np.array(
            [
                abs(r["bounded_aia_250"] - r["bounded_aia_1000"])
                for r in records
                if r.get("bounded_aia_250") is not None and r.get("bounded_aia_1000") is not None
            ],
            dtype=float,
        )
        rel = np.array([r["rel_se"] for r in records if r.get("rel_se") is not None], dtype=float)
        ranks = np.array(
            [r["cross_budget_rank_corr"] for r in records if r.get("cross_budget_rank_corr") is not None],
            dtype=float,
        )
        if delta.size == 0:
            continue
        lo, hi = bootstrap_ci(delta, "mc", ds)
        rows.append(
            {
                "dataset": ds,
                "n_users": int(delta.size),
                "mean_abs_shift": float(delta.mean()),
                "ci95_low": lo,
                "ci95_high": hi,
                "max_abs_shift": float(delta.max()),
                "mean_relative_se": float(rel.mean()),
                "min_rank_corr": float(ranks.min()),
            }
        )
    return rows


# --------------------------------------------------------------------------
# S37  fixed-denominator (pure suppression) ablation  (Critical Issue #1)
# --------------------------------------------------------------------------

def _bootstrapped_means(d: np.ndarray, rng, draws: int) -> np.ndarray:
    """Resampled-mean distribution, blocked so large cohorts stay in memory bounds."""
    out = np.empty(draws, dtype=float)
    block = max(1, min(draws, 4_000_000 // max(d.size, 1)))
    for start in range(0, draws, block):
        stop = min(start + block, draws)
        idx = rng.integers(0, d.size, size=(stop - start, d.size))
        out[start:stop] = d[idx].mean(axis=1)
    return out


def _sign_flipped_means(d: np.ndarray, rng, draws: int) -> np.ndarray:
    """Plus-one sign-flip null: independent Rademacher signs per user, exact under
    symmetry of the paired difference distribution about its null value."""
    out = np.empty(draws, dtype=float)
    block = max(1, min(draws, 4_000_000 // max(d.size, 1)))
    signs_np = np.array([-1.0, 1.0])
    for start in range(0, draws, block):
        stop = min(start + block, draws)
        signs = rng.choice(signs_np, size=(stop - start, d.size))
        out[start:stop] = (signs * d).mean(axis=1)
    return out


def _paired_contrasts(payload: dict) -> dict:
    """User-level paired normalized-minus-fixed contrasts, recomputed here.

    Reading the per-user records rather than the run's summary keeps the
    published table reproducible from the released data alone: bootstrap
    percentile intervals plus a plus-one sign-flip test with independent
    Rademacher signs on the per-user differences, with the permutation seed
    derived from the contrast name so re-runs are byte-identical.
    """
    records = payload.get("records") or []
    by_user: dict[str, dict[int, dict]] = {}
    for rec in records:
        by_user.setdefault(rec.get("scorer"), {})[int(rec["user"])] = rec
    shared = sorted(set(by_user.get("normalized", {})) & set(by_user.get("fixed_denominator", {})))
    if len(shared) < 2:
        return {}
    keys = tuple(k for k in (
        "aia_shapley_bounded", "aia_lime_bounded", "aia_loo_bounded",
        "aia_shapley_deletion", "gap_shapley", "signed_shapley_bounded", "mean_abs_effect",
    ) if any(k in r for r in records))
    out: dict[str, dict] = {}
    for key in keys:
        diff = [
            float(by_user["normalized"][u][key]) - float(by_user["fixed_denominator"][u][key])
            for u in shared
            if by_user["normalized"][u].get(key) is not None
            and by_user["fixed_denominator"][u].get(key) is not None
        ]
        d = np.asarray(diff, dtype=float)
        if d.size < 2:
            continue
        rng = rng_for("fixed-denominator-paired", key, int(d.size))
        mean = float(d.mean())
        sd = float(d.std(ddof=1))
        boots = _bootstrapped_means(d, rng, BOOT)
        flipped = _sign_flipped_means(d, rng, FLIP)
        out[key] = {
            "n": int(d.size),
            "share_nonzero": float(np.mean(np.abs(d) > 1e-12)),
            "mean_difference": mean,
            "ci95_low": float(np.percentile(boots, 2.5)),
            "ci95_high": float(np.percentile(boots, 97.5)),
            "sign_flip_p": float(
                (1 + int((np.abs(flipped) >= abs(mean) - 1e-15).sum())) / (FLIP + 1)),
            "cohens_dz": float(mean / sd) if sd > 1e-12 else 0.0,  # degenerate => undefined, report 0
            "sign_flip_draws": int(FLIP),
        }
    return out


def utility_factorial_replication() -> tuple[list[str], dict]:
    """Utility-matched versus utility-crossed AIA on the replication benchmark.

    Issue 4 asks whether the H2 adjudication is confounded by scoring an
    attribution under one utility while evaluating the selected action under
    another. The primary-cohort answer is the rank factorial in
    Table~\\ref{tab:r9-utility-factorial}; this float adds the same design on the
    benchmark that ships inside the artifact, as a mean bounded AIA per cell plus
    the paired matched-minus-crossed difference. The cell counts are the finding:
    an NDCG-conditioned arm is nearly empty on this cohort, so the means are
    descriptive and carry no inferential weight, which is itself the reason the
    factorial is adjudicated on the primary cohort.
    """
    rows: list[str] = []
    per_dataset: dict[str, dict] = {}
    for path in sorted(OUT.glob("utility_factorial_*.json")):
        payload = json.loads(path.read_text())
        dataset_raw = payload.get("dataset", path.stem.split("_")[-1])
        dataset = {"gowalla": "Gowalla", "movielens": "MovieLens-1M",
                   "amazon": "Amazon"}.get(dataset_raw, dataset_raw)
        records = payload.get("records") or []
        users_sampled = len({int(r["user"]) for r in records}) or 0
        by_cell: dict[tuple[str, str], dict[int, dict]] = {}
        for rec in records:
            by_cell.setdefault((rec["attr_utility"], rec["outcome_utility"]), {})[
                int(rec["user"])] = rec
        cells = payload.get("summary") or {}
        for cell, block in sorted(cells.items()):
            attr, _, outcome = cell.partition("__x__")
            matched = block.get("aia_matched") or {}
            crossed = block.get("aia_cross") or {}
            regret = block.get("regret") or {}
            diff = np.asarray([
                rec["aia_matched"] - rec["aia_cross"]
                for rec in by_cell.get((attr, outcome), {}).values()
                if rec.get("aia_matched") is not None and rec.get("aia_cross") is not None
            ], dtype=float)
            lo = hi = None
            if diff.size >= 2:
                rng = rng_for("utility-factorial-replication", dataset, attr, outcome, diff.size)
                boots = _bootstrapped_means(diff, rng, BOOT)
                lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            rows.append(
                f"{dataset} & {attr} & {outcome} & {int(matched.get('n') or crossed.get('n') or 0)}"
                f" & {users_sampled} & {num(matched.get('mean'), 4)} & {num(crossed.get('mean'), 4)}"
                f" & {num(float(diff.mean()) if diff.size else None, 4)} & {num(lo, 4)}"
                f" & {num(hi, 4)} & {num(regret.get('mean'), 6)}"
                f" & {int(block.get('active_oracles') or 0)}"
            )
            per_dataset.setdefault(dataset, {})[f"{attr} x {outcome}"] = {
                "n_users": int(matched.get("n") or crossed.get("n") or 0),
                "users_sampled": users_sampled,
                "aia_matched": matched.get("mean"),
                "aia_crossed": crossed.get("mean"),
                "paired_difference": float(diff.mean()) if diff.size else None,
                "paired_n": int(diff.size),
                "paired_ci95": [lo, hi] if diff.size >= 2 else None,
                "mean_regret": regret.get("mean"),
                "active_oracles": block.get("active_oracles"),
            }
    if not rows:
        return [], {}
    L = table(
        r"Dataset & Attribution & Outcome & $n$ & sampled & AIA matched & AIA crossed"
        r" & Matched $-$ crossed & CI low & CI high & Mean regret & Oracles",
        r"Attribution-utility $\times$ outcome-utility factorial on the replication "
        r"benchmark (Issue~4), run on the same fitted ItemKNN models and the same "
        r"$M_{\mathrm{pair}}=250$ budget and intervention strength as the "
        r"fixed-denominator ablation in Table~\ref{tab:r9-fixed-denominator}, for the "
        r"Monte Carlo Shapley attribution. Each cell scores the attribution game under "
        r"one utility and evaluates the selected action under one utility; a matched "
        r"cell uses the same utility on both sides. $n$ is the number of users with a "
        r"defined bounded AIA in that cell, out of \emph{sampled}, and \emph{Oracles} "
        r"is the number whose optimum under the outcome utility is nonzero. The "
        r"counts are the substantive point: the NDCG-conditioned arm is almost empty "
        r"on this cohort, so these means are descriptive and the factorial is "
        r"adjudicated on the primary cohort (Table~\ref{tab:r9-utility-factorial}). "
        r"Matched $-$ crossed is a user-level paired mean with a $10{,}000$-draw "
        r"bootstrap interval, computed on the users where both arms are defined.",
        "tab:r9-utility-factorial-replication",
        "lllrrrrrrrrr",
        rows,
    )
    return L, per_dataset


def stratified_null_tables() -> tuple[list[str], dict]:
    """Within-user nulls stratified by the structure that could fake alignment.

    Issue 12 objects that a plain within-user permutation null treats every
    reordering of a profile as equally implausible, so recency or popularity
    structure in the history can inflate the observed alignment. Each released
    run supplies the observed mean bounded AIA and a null distribution generated
    under one stratification; the reported $z$ is the observed mean in units of
    that null's standard deviation, which is the quantity that changes when the
    null is made stricter. The plus-one $p$-value uses the run's own draw count,
    so on a $1000$-draw null it saturates at $1/1001$ and $z$ is the informative
    statistic.
    """
    rows: list[str] = []
    stats: dict[str, dict] = {}
    for path in sorted(OUT.glob("stratified_null_*.json")):
        payload = json.loads(path.read_text())
        dataset_raw = payload.get("dataset", path.stem.split("_")[-1])
        dataset = {"gowalla": "Gowalla", "movielens": "MovieLens-1M",
                   "amazon": "Amazon"}.get(dataset_raw, dataset_raw)
        for key, block in sorted((payload.get("summary") or {}).items()):
            null = block.get("null_mean") or {}
            obs = block.get("observed_mean")
            sd = null.get("sd")
            draws = null.get("n")
            z = None
            if obs is not None and sd:
                z = (float(obs) - float(null.get("mean", 0.0))) / float(sd)
            label = {
                "free": "unstratified (any reordering)",
                "recency_blocks": "recency blocks",
                "popularity_blocks": "popularity blocks",
            }.get(key, key.replace("_", " "))
            draw_txt = f"{draws}" if draws else "--"
            rows.append(
                f"{dataset} & {label} & {int(block.get('n_users') or 0)} & "
                f"{num(obs, 4)} & {num(null.get('mean'), 4)} & {num(sd, 4)} & "
                f"{num(null.get('min'), 4)} & {num(null.get('max'), 4)} & "
                f"{num(z, 1)} & {pstr(block.get('plus_one_p'))} & {draw_txt}"
            )
            stats.setdefault(dataset, {})[label] = {
                "n_users": block.get("n_users"),
                "observed_mean": obs,
                "null_mean": null.get("mean"),
                "null_sd": sd,
                "null_range": [null.get("min"), null.get("max")],
                "z": z,
                "plus_one_p": block.get("plus_one_p"),
                "null_draws": draws,
            }
    if not rows:
        return [], {}
    L = table(
        "Dataset & Null stratification & Users & Observed mean & Null mean & Null SD "
        "& Null min & Null max & $z$ & Plus-one $p$ & Draws",
        "Within-user nulls stratified by the profile structure that could produce "
        "alignment on its own (Issue 12), on the replication benchmark at the same "
        "$M_{\\mathrm{pair}}=250$ budget as Tables~\ref{tab:r9-fixed-denominator} "
        "and~\ref{tab:r9-utility-factorial-replication}. The observed mean is the "
        "bounded AIA of the Monte Carlo Shapley attribution over the users with a "
        "defined game; each null is one thousand redraws under the named "
        "stratification. Tightening the null raises its mean from essentially zero "
        "to $0.19$, so the unstratified test does understate the baseline; the "
        "observed value still sits about nineteen null standard deviations above "
        "even the strictest stratification, which is the $z$ column. The plus-one "
        "$p$-value at one thousand draws cannot go below $1/1001$, so $z$ is the "
        "comparable quantity across rows.",
        "tab:r9-stratified-null",
        "llrrrrrrrcr",
        rows,
    )
    return L, stats


def _dataset_label(raw: str) -> str:
    return {"gowalla": "Gowalla", "movielens": "MovieLens-1M",
            "amazon": "Amazon"}.get(raw, raw)


def _runs(pattern: str):
    """Each released run of one experiment, as (label, payload) pairs."""
    for path in sorted(OUT.glob(pattern)):
        payload = json.loads(path.read_text())
        raw = payload.get("dataset", path.stem.split("_")[-1])
        yield _dataset_label(raw), payload


def prospective_tables() -> tuple[list[str], dict]:
    """Non-target-conditioned (prospective) cohort audit, if it has run.

    Issue 5 objects that conditioning the attribution game on the held-out target
    makes "alignment" partly definitional. The released audit rebuilds each game
    from the model's own top-1 prospective recommendation and re-runs the same
    attribution, so alignment is measured against a target the scorer chose. One
    row per dataset, with the share of users whose prospective target actually
    covers the held-out item, because that share bounds how much the two
    conditioning schemes can differ.
    """
    rows: list[str] = []
    stats: dict[str, dict] = {}
    for dataset, payload in _runs("prospective_*.json"):
        summary = payload.get("summary") or {}
        block = summary.get("aia_shapley") or {}
        rows.append(
            f"{dataset} & {int(payload.get('users_total') or 0)} & "
            f"{int(payload.get('users_audited') or 0)} & "
            f"{num((payload.get('covers_heldout_target_fraction') or 0) * 100, 1)}\\% & "
            f"{num(block.get('mean'), 4)} & "
            f"{num(summary.get('aia_lime', {}).get('mean'), 4)} & "
            f"{num(summary.get('aia_loo', {}).get('mean'), 4)} & "
            f"{num(summary.get('signed_shapley', {}).get('mean'), 4)}"
        )
        stats[dataset] = {
            "users_total": payload.get("users_total"),
            "users_audited": payload.get("users_audited"),
            "covers_heldout_target_fraction": payload.get("covers_heldout_target_fraction"),
            "summary": summary,
        }
    if not rows:
        return [], {}
    L = table(
        "Dataset & Sampled & Audited & Prospective target covers held-out item & "
        "AIA Shapley & AIA LIME & AIA LOO & Signed Shapley",
        "Prospective (non-target-conditioned) replication of the audit (Issue 5). "
        "Each user's game is rebuilt from the model's own top-1 prospective "
        "recommendation rather than from the held-out target, and the same Monte "
        "Carlo Shapley, LIME and leave-one-out attributions are scored against the "
        "same realized effects; $n_{\\max}$, permutation budget and intervention "
        "strength are unchanged. The third column counts users for whom the "
        "prospective candidate set contains the held-out item at all, which is the "
        "fraction on which the two conditioning schemes are even comparable. If the "
        "ordering of the three attributions is preserved here, alignment is not an "
        "artefact of constructing the game around the answer.",
        "tab:r9-prospective-replication",
        "lrrrrrrr",
        rows,
    )
    return L, stats


def candidate_redraw_tables() -> tuple[list[str], dict]:
    """Independent candidate-set resamples, if they have run.

    Issue 8 asks whether the reported contrasts survive a different draw of the
    Bernoulli candidate set rather than a different number of permutations. Each
    redraw re-draws the candidate sets from the same fitted model with a fresh
    candidate seed and re-runs the whole attribution, so spread across redraws is
    sampling variability of the design itself, not of the estimator.
    """
    rows: list[str] = []
    stats: dict[str, dict] = {}
    for dataset, payload in _runs("candidate_redraw_*.json"):
        between = payload.get("between_redraw") or {}
        for method in ("shapley", "lime", "loo"):
            block = between.get(method) or {}
            if not block:
                continue
            spread = ""
            per_redraw = [
                (r.get("summary") or {}).get(method, {}).get("mean")
                for r in payload.get("redraws") or []
            ]
            values = [v for v in per_redraw if v is not None]
            if len(values) >= 2:
                spread = f"{min(values):.4f}--{max(values):.4f}"
            rows.append(
                f"{dataset} & {method.upper() if method != 'loo' else 'LOO'} & "
                f"{int(block.get('n') or 0)} & {int(payload.get('redraws') and len(payload['redraws']) or 0)} & "
                f"{num(block.get('mean'), 4)} & {num(block.get('sd'), 4)} & "
                f"{spread or '--'}"
            )
        stats[dataset] = {
            "between_redraw": between,
            "redraws": len(payload.get("redraws") or []),
            "per_redraw_means": {
                m: [(r.get("summary") or {}).get(m, {}).get("mean")
                    for r in payload.get("redraws") or []]
                for m in ("shapley", "lime", "loo")
            },
        }
    if not rows:
        return [], {}
    L = table(
        "Dataset & Method & Users & Redraws & Mean AIA & SD across redraws & "
        "Min--max redraw mean",
        "Independent candidate-set resamples (Issue 8). Every redraw draws the "
        "Bernoulli candidate sets afresh from the same fitted model with a new "
        "candidate seed and re-runs the attribution and the realized-effect "
        "evaluation, so the spread reported here is design variability rather than "
        "Monte Carlo variability of a fixed design. Compare the last column against "
        "the Monte Carlo propagation in Table~\\ref{tab:r9-mc-propagation}: "
        "candidate-set variability is the larger of the two if the redraw means "
        "move more than the per-user estimator noise, and that is the quantity any "
        "cross-method claim has to clear.",
        "tab:r9-candidate-redraw",
        "llrrrcr",
        rows,
    )
    return L, stats


def compute_matched_tables() -> tuple[list[str], dict]:
    """Budget-response curves at matched scorer-call counts, if they have run.

    Issue 13 objects that comparing Monte Carlo Shapley at $M_{\\mathrm{pair}}$
    permutations against LIME at its declared mask budget compares two different
    amounts of work. Each row evaluates both at the same number of scorer calls:
    the reverse-paired prefix walk costs about $2M_{\\mathrm{pair}}(n_u+1)$
    coalition requests, so LIME is given that many masks at the same row.
    """
    rows: list[str] = []
    stats: dict[str, list[dict]] = {}
    for dataset, payload in _runs("compute_matched_*.json"):
        for curve in payload.get("curves") or []:
            shapley = curve.get("shapley") or {}
            lime = curve.get("lime_matched") or {}
            rows.append(
                f"{dataset} & {int(curve.get('m_pair') or 0)} & "
                f"{int(curve.get('matched_scorer_calls_per_user') or 0):,} & "
                f"{num(shapley.get('mean'), 4)} & {num(shapley.get('sd'), 4)} & "
                f"{num(lime.get('mean'), 4)} & {num(lime.get('sd'), 4)} & "
                f"{num((lime.get('mean') or 0) - (shapley.get('mean') or 0), 4)}"
            )
        stats.setdefault(dataset, []).append(payload.get("curves") or [])
    if not rows:
        return [], {}
    L = table(
        "Dataset & $M_{\\mathrm{pair}}$ & Scorer calls per user & Shapley AIA & "
        "Shapley SD & LIME AIA & LIME SD & LIME $-$ Shapley",
        "Compute-matched budget-response curves (Issue 13). Each row is one budget "
        "at which Monte Carlo Shapley and bounded LIME are given the \emph{same} "
        "number of recommender evaluations: the reverse-paired prefix walk at "
        "$M_{\\mathrm{pair}}$ costs about $2M_{\\mathrm{pair}}(n_u+1)$ coalition "
        "requests, and LIME receives that many masks. Rows are therefore "
        "equal-computation comparisons, while the primary configuration in the main "
        "paper is not; the question this answers is whether the Shapley--LIME gap "
        "narrows, holds or reverses as both methods are given more work. A "
        "difference that shrinks monotonically toward zero is a budget artefact.",
        "tab:r9-compute-matched",
        "llcrrrrr",
        rows,
    )
    return L, stats


def hardware_tables() -> tuple[list[str], dict]:
    """Machine, library versions, peak memory and repeated per-method timings.

    Issue 16 asks for a reproducible environment statement and Issue 17 for the
    artifact to describe the run it came from. These are recorded from the same
    process that produced the review-9 numbers, at the same candidate and
    permutation seeds, so the timings are attributable rather than decorative:
    each method is timed over repeated single-user attribution calls and the median
    is reported, which is what makes the budget choices in the paper legible.
    """
    rows: list[str] = []
    stats: dict[str, dict] = {}
    for dataset, payload in _runs("hardware_*.json"):
        hardware = payload.get("hardware") or {}
        timings = payload.get("timings_seconds") or {}
        def _median(values):
            # `run_review9_experiments.py` serialises a summary dict
            # ({"n", "mean", "median", ...}) per method, not the raw sample list;
            # accept either shape so a real run cannot crash the table.
            if isinstance(values, dict):
                for key in ("median", "mean"):
                    if values.get(key) is not None:
                        return float(values[key])
                return None
            if values is None or len(values) == 0:
                return None
            return float(np.median(np.asarray(values, dtype=float)))

        medians = {method: _median(values) for method, values in timings.items()}
        rows.append(
            f"{dataset} & {hardware.get('machine', '?')} & "
            f"{hardware.get('python', '?')} & {hardware.get('numpy', '?')} & "
            f"{int(hardware.get('cpu_count') or 0)} & "
            f"{num(payload.get('peak_rss_mb'), 0)} & "
            f"{int(payload.get('timing_repeats') or 0)} & "
            f"{num(medians.get('shapley'), 3)} & {num(medians.get('lime'), 3)} & "
            f"{num(medians.get('loo'), 3)}"
        )
        stats[dataset] = {"hardware": hardware, "median_seconds": medians,
                          "peak_rss_mb": payload.get("peak_rss_mb"),
                          "platform": hardware.get("platform")}
    if not rows:
        return [], {}
    L = table(
        "Dataset & Machine & Python & NumPy & Cores & Peak RSS (MB) & Repeats & "
        "Shapley s/user & LIME s/user & LOO s/user",
        "Execution environment and repeated per-user attribution timings recorded "
        "with the review-9 runs (Issues 16/17). Times are medians over the declared "
        "number of repeats of one user's attribution at the primary budget, measured "
        "in the same process as the released numbers; peak RSS is the whole run. "
        "Together with the result-manifest hash quoted in both documents and the "
        "seeds in each run's \\texttt{config}, this is what a re-runner needs to "
        "reproduce the tables without the original machine.",
        "tab:r9-hardware",
        "lllccrcrrr",
        rows,
    )
    return L, stats


def _benchmark_extras(L: list[str], stats: dict) -> None:
    """Append every benchmark float that is independent of the ablation, in place.

    These used to be reachable only after the fixed-denominator run had produced
    rows, so a collection with, say, only a compute-matched run rendered nothing
    at all. Each builder still returns nothing when its own run is absent, so no
    placeholder float is ever emitted for work that has not been done.
    """
    for (extra, extra_stats), key in (
        (utility_factorial_replication(), "utility_factorial_replication"),
        (stratified_null_tables(), "stratified_null_replication"),
        (prospective_tables(), "prospective_replication"),
        (candidate_redraw_tables(), "candidate_redraw"),
        (compute_matched_tables(), "compute_matched"),
        (hardware_tables(), "hardware"),
    ):
        if extra:
            L += [""] + extra
            stats[key] = extra_stats


def benchmark_replication_tables() -> tuple[str, dict]:
    """Every float for the third (Gowalla-style) benchmark runs.

    Currently the fixed-denominator construct-validity ablation and, when the
    utility-factorial run exists, its replication on the same cohort.
    """
    """Render the normalized-versus-fixed-denominator comparison, if it has run.

    The reviewer's first critical point is that dividing by the retained weight
    sum makes the intervention a relative reallocation of profile mass rather
    than isolated suppression, so the bounded-minus-deletion contrast may be a
    normalization artifact. The ablation re-fits nothing and changes only the
    denominator, which isolates exactly that mechanism. Returns an empty table
    when no run exists yet, so the generator stays usable on a fresh checkout.
    """
    runs = sorted((OUT).glob("fixed_denominator_*.json"))
    pretty = {"gowalla": "Gowalla", "movielens": "MovieLens-1M", "amazon": "Amazon"}
    rows: list[dict] = []
    paired_rows: list[dict] = []
    mismatches: list[dict] = []
    for path in runs:
        payload = json.loads(path.read_text())
        dataset_raw = payload.get("dataset", path.stem.split("_")[-1])
        dataset = pretty.get(dataset_raw, dataset_raw)
        summary = payload.get("summary", {})
        # Prefer the recomputation; fall back to the block the run wrote only if
        # the per-user records are unavailable (e.g. a trimmed release file).
        paired = _paired_contrasts(payload) or payload.get("paired", {})
        for key, entry in (payload.get("paired") or {}).items():
            got = paired.get(key)
            if not got or not entry:
                continue
            # The point estimate is deterministic and must agree exactly; the p
            # value is Monte Carlo and the run uses fewer sign-flip draws than the
            # table does, so allow a difference the smaller draw count can explain.
            if abs(float(entry["mean_difference"]) - got["mean_difference"]) > 1e-9:
                mismatches.append({"dataset": dataset, "key": key,
                                   "run": entry, "recomputed": got})
            if entry.get("sign_flip_p") is not None:
                se = float(np.sqrt(max(got["sign_flip_p"], 1e-12) / got["n"]) or 0.0)
                if abs(float(entry["sign_flip_p"]) - got["sign_flip_p"]) > max(2e-3, 10 * se):
                    mismatches.append({"dataset": dataset, "key": key, "field": "sign_flip_p",
                                       "run": entry, "recomputed": got})
        for scorer, title in (("normalized", "normalized (relative reweighting)"),
                             ("fixed_denominator", "fixed denominator (pure suppression)")):
            block = summary.get(scorer)
            if not block:
                continue
            rows.append({
                "dataset": dataset,
                "scorer": title,
                "n": block["aia_shapley_bounded"].get("n", 0),
                "n_sampled": payload.get("n_users_sampled", block["mean_abs_effect"].get("n", 0)),
                "bounded": block["aia_shapley_bounded"].get("mean"),
                "deletion": block["aia_shapley_deletion"].get("mean"),
                "gap": block.get("gap_shapley", {}).get("mean"),
                "lime": block["aia_lime_bounded"].get("mean"),
                "loo": block["aia_loo_bounded"].get("mean"),
                "signed": block["signed_shapley_bounded"].get("mean"),
                "scale": block["mean_abs_effect"].get("mean"),
            })
        for key, label in (
            ("aia_shapley_bounded", "Shapley bounded AIA"),
            ("aia_shapley_deletion", "Shapley deletion AIA"),
            ("gap_shapley", "bounded-minus-deletion gap (Shapley)"),
            ("aia_lime_bounded", "LIME bounded AIA"),
            ("aia_loo_bounded", "LOO bounded AIA"),
            ("signed_shapley_bounded", "signed alignment (Shapley bnd.)"),
            ("mean_abs_effect", "mean $|\\Delta|$ (effect scale)"),
        ):
            entry = paired.get(key)
            if not entry:
                continue
            share_nz = entry.get("share_nonzero")
            nz = 0 if share_nz is None else int(round(share_nz * entry["n"]))
            paired_rows.append({
                "dataset": dataset,
                "quantity": label,
                "n": entry["n"],
                "nonzero": f"{nz}/{entry['n']}",
                "dag": r"\,$^{\dagger}$" if share_nz is not None and share_nz < 0.9 else "",
                "share_nonzero": share_nz,
                "diff": entry["mean_difference"],
                "lo": entry["ci95_low"],
                "hi": entry["ci95_high"],
                "p": entry.get("sign_flip_p"),
                "dz": entry.get("cohens_dz"),
            })
    if not rows:
        L = ["% Generated by scripts/make_review9_stats.py: benchmark floats only, "
             "no results/review9/fixed_denominator_*.json yet."]
        stats: dict = {"runs": 0}
        _benchmark_extras(L, stats)
        return "\n".join(L) + "\n", stats

    level_rows = [
        f"{r['dataset']} & {r['scorer']} & {r['n']} & {num(r['bounded'], 3)} & "
        f"{num(r['deletion'], 3)} & {num(r['gap'], 4)} & {num(r['lime'], 3)} & "
        f"{num(r['loo'], 3)} & {num(r['signed'], 3)} & {num(r['scale'], 5)} \\\\"
        for r in rows
    ]
    paired_text_rows = [
        f"{r['dataset']} & {r['quantity']} & {r['n']} & {r['nonzero']} & "
        f"{num(r['diff'], 4)} & {num(r['lo'], 4)} & {num(r['hi'], 4)} & "
        f"{pstr(r['p'])}{r['dag']} & {num(r['dz'], 3)} \\\\"
        for r in paired_rows
    ]
    head = ("% Generated by scripts/make_review9_stats.py from "
            "results/review9/fixed_denominator_*.json.")
    L = [head, ""]
    L += table(
        r"Dataset & Scorer & $n$ & Shapley bnd. & Shapley del. & Gap & LIME bnd. "
        r"& LOO bnd. & Signed & Effect scale \\",
        r"Construct-validity ablation for the intervention semantics (Issue~1). The "
        r"same fitted ItemKNN neighbourhoods are scored twice: with the normalized "
        r"interface used everywhere else in this paper, in which downweighting one "
        r"interaction raises every other interaction's share (relative profile "
        r"reweighting), and with a fixed denominator equal to the full history length, "
        r"in which downweighting suppresses exactly the targeted contribution and "
        r"reallocates nothing (pure suppression). Nothing is retrained, and the "
        r"candidates, permutation seed, intervention strength and evaluated users are "
        r"identical, so any difference is the normalization itself. $n$ counts users "
        r"whose attribution game is defined under the $n_{\max}=20$ cap. Effect scale "
        r"is the mean absolute singleton effect, which the normalized interface shrinks "
        r"by construction, so a bounded AIA near $1$ at a small scale is saturation "
        r"rather than strength.",
        "tab:r9-fixed-denominator",
        "llrrrrrrrr",
        level_rows,
    )
    L += table(
        r"Dataset & Quantity & $n$ & Nonzero & Normalized $-$ fixed & CI low & CI high "
        r"& Sign-flip $p$ & $d_z$ \\",
        r"Paired user-level contrasts for the same ablation: normalized relative "
        r"reweighting minus fixed-denominator pure suppression, so a positive value "
        r"means the normalized interface reports the larger attribution. Intervals are "
        r"$10{,}000$-draw user-bootstrap percentile intervals and $p$ is the plus-one "
        r"sign-flip test with $50{,}000$ draws, whose smallest attainable value is "
        r"$2{\times}10^{-5}$. Nonzero is the number of users with a nonzero paired "
        r"difference; a dagger marks a zero-inflated contrast, where a floor "
        r"$p$-value shows that the few informative users agree in sign and says nothing "
        r"about the size of the effect in the population.",
        "tab:r9-fixed-denominator-paired",
        "lllcrrrrr",
        paired_text_rows,
    )
    stats = {
        "runs": len(runs),
        "datasets": sorted({r["dataset"] for r in rows}),
        "users_with_defined_aia": max((r["n"] for r in rows), default=0),
        "users_sampled": max((r.get("n_sampled", 0) for r in rows), default=0),
        "paired": {
            f"{r['dataset']}|{r['quantity']}": {
                k: r[k] for k in ("n", "share_nonzero", "diff", "lo", "hi", "p", "dz")
            }
            for r in paired_rows
        },
        "levels": {
            f"{r['dataset']}|{r['scorer']}": {
                k: r[k] for k in ("bounded", "deletion", "gap", "lime", "loo",
                                  "signed", "scale", "n", "n_sampled")
            }
            for r in rows
        },
        "cross_check_mismatches": mismatches,
    }
    _benchmark_extras(L, stats)
    return "\n".join(L) + "\n", stats


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def num(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def pstr(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if value < 1e-4:
        return "$<10^{-4}$"
    return f"{value:.4f}"


def tick(ok) -> str:
    r"""Tick when a published p is within Monte Carlo error.

    Uses \ensuremath{\surd} rather than \checkmark so the generated tables do
    not require amssymb in either of the two document preambles that input them.
    """
    return "\\ensuremath{\\surd}" if ok else ""  # no amssymb dependency


def _ends_row(line: str) -> bool:
    return line.rstrip().endswith(_ROW_END)


def table(header: str, caption: str, label: str, colspec: str, body: list[str]) -> list[str]:
    """One float with a single tabular, in the same house style as the hand-written
    tables (``@{}<spec>@{}`` preamble so the first and last columns have no padding).

    The LaTeX row terminator is appended here rather than at each call site: writing
    it in a Python literal is exactly the escaping trap that produced a malformed
    third-benchmark asset once already, so callers pass bare rows and
    any row that already ends in ``\\\\`` is left alone.
    """
    rows = [header if _ends_row(header) else header + " " + _ROW_END]
    rows += [line if _ends_row(line) else line + " " + _ROW_END for line in body]
    return [
        "\\begin{table}[!htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
        "\\toprule",
        *rows[:1],
        "\\midrule",
        *rows[1:],
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]


def render(flow, sens, mult, mult_stats, factorial, stud, mcp, prec) -> str:
    L: list[str] = [
        "% Generated by scripts/make_review9_stats.py -- do not edit by hand.",
        "% Review-9 inference audits computed from the frozen release matrices.",
        "",
    ]

    L += table(
        "Dataset & Cohort & $\\ge3/5$ valid seeds & Non-constant AIA & "
        "Positive exact oracle & Finite NDCG effect & Active NDCG oracle \\\\",
        "Inclusion flow for the primary ItemKNN cohort (target-margin attribution "
        "game, five seeds, seed-mean within distinct user). Each inferential "
        "statement in the manuscript is labelled by the stage it uses; the counts "
        "here are recomputed from the released per-user matrices, so the "
        "denominators $1000$, $993$, $987$, and $339$/$196$ quoted in the main "
        "paper are reproducible rather than asserted.",
        "tab:r9-inclusion-flow",
        "lrrrrrr",
        [
            f"{r['dataset']} & {r['cohort']} & {r['seed_valid']} & {r['aia_defined']} & "
            f"{r['positive_oracle_tm']} & {r['finite_ndcg_effect']} & {r['active_oracle_ndcg']} \\\\"
            for r in flow
        ],
    )

    L += table(
        "Dataset & Contrast & Quantity & $n$ pub. & Mean pub. & $n$ all & Mean all & "
        "CI low & CI high \\\\",
        "All-user sensitivity of every population-conditional claim. "
        "\\emph{Published} is the statistic on the population used in the main "
        "paper; \\emph{All-user} recomputes the same estimand over all users with "
        "finite values under both methods (normalized regret without the "
        "positive-oracle conditioning; for the gap--regret correlation, "
        "zero-oracle users enter with regret $0$). CIs are $10{,}000$-draw "
        "user-bootstrap intervals on the all-user population; the two columns "
        "overlapping is the requested evidence that no headline conclusion "
        "depends on the exclusion rule.",
        "tab:r9-all-user-sensitivity",
        "lllrrrrrr",
        [
            f"{r['dataset']} & {r['contrast']} & {r['quantity']} & {r['n_published']} & "
            f"{num(r['mean_published'])} & {r['n_all_user']} & {num(r['mean_all_user'])} & "
            f"{num(r['ci95_low'])} & {num(r['ci95_high'])} \\\\"
            for r in sens
        ],
    )

    headline = [
        r
        for r in mult
        if r["source"] == "per-metric family"
        and r["family"].endswith(("aia", "intervention_success"))
        and ("/ primary /" in r["family"] or "/ full_catalogue /" in r["family"])
        and "MC Shapley" in (r["left"], r["right"])
    ] + [r for r in mult if r["source"] == "predeclared 12-contrast family"]
    L += table(
        "Family & $m$ & Contrast & $n$ & Mean diff. & Raw $p$ & Exceed. & "
        "Holm pub. & Holm recomp. \\\\",
        "Confirmatory multiplicity map with raw exceedance counts for the "
        "headline contrasts. A family is one "
        "dataset--model--condition--metric group of declared pairs of size $m$; "
        "the exceedance count is recovered exactly from the plus-one raw "
        "$p$-value as $\\#=p(R+1)-1$ with $R=10{,}000$ declared draws, so "
        "$\\#=0$ and a displayed adjusted $p$ of $0.0010$ in a ten-test family is "
        "the resolution floor $10/10{,}001$ rather than censoring. The two "
        "$p_{\\mathrm{holm}}$ columns are the published value and the value "
        f"recomputed here from the raw $p$-values; across all "
        f"{mult_stats['families']} families and {mult_stats['tests']} tests the "
        f"two disagree in {mult_stats['holm_recomputation_mismatches']} cases. "
        "The last block lists the complete predeclared 12-contrast "
        "success/abstention family, which spans both datasets: the MovieLens "
        "Shapley--LIME success contrast therefore carries the same raw evidence "
        "(23 exceedances of $10{,}000$) as the $0.0066$ in the per-metric "
        "replication table but a $\\times9$ rather than $\\times5$ multiplier, "
        "giving $0.0216$. The 12-contrast family is authoritative for success "
        "and abstention.",
        "tab:r9-multiplicity-map",
        "llcrrrrrr",
        [
            f"{r['family']} & {r['family_size']} & {r['left']}--{r['right']} & {r['n_users']} & "
            f"{num(r['mean_difference'])} & {num(r['raw_p'], 6)} & {r['exceedances']} & "
            f"{num(r['published_holm_p'])} & {num(r['recomputed_holm_p'])} \\\\"
            for r in headline
        ]
        + [
            f"{r['family']} & {r['family_size']} & {r['left']}--{r['right']} & {r['n_users']} & "
            f"{num(r['mean_difference'])} & {num(r['raw_p'], 6)} & {r['exceedances']} & "
            f"{num(r['published_holm_p'])} & {num(r['recomputed_holm_p'])} \\\\"
            for r in mult_stats["shapley_lime_success_rows"][:6]
        ],
    )

    L += table(
        "Dataset & Method & Attribution utility & Outcome utility & $n$ & Spearman & $p$ \\\\",
        "Attribution-utility $\\times$ outcome-utility factorial on the primary "
        "ItemKNN cohort, computed from the released per-user matrices rather "
        "than from a new run. Each cell is the user-level Spearman association "
        "between an alignment score measured under one utility and the realized "
        "effect of the selected action under one utility; comparing across rows "
        "isolates utility mismatch from additivity/interaction effects, which is "
        "the confound raised against H2. The interaction-aware arm of the "
        "reviewer's $2\\times2\\times2$ design is reported for the nonlinear "
        "scorers in the replication section (pair-interaction ratios "
        "$0.21$--$0.78$ under SASRec), so the full triple interaction remains a "
        "run on the datasets machine.",
        "tab:r9-utility-factorial",
        "lllcrrr",
        [
            f"{r['dataset']} & {r['method']} & {r['attribution_utility']} & {r['outcome_utility']} & "
            f"{r['n_users']} & {num(r['spearman'], 3)} & {pstr(r['p_value'])} \\\\"
            for r in factorial
        ],
    )

    L += table(
        "Dataset & Quantity & $n$ & Mean diff. & Skew & Sign-flip $p$ & Studentized $p$ \\\\",
        "Robustness of the paired randomization inference. \\emph{Sign-flip} "
        "reproduces the published plus-one test, which assumes the per-user "
        "paired differences are exchangeable under sign flip (equivalently, "
        "symmetric about zero under the null). \\emph{Studentized} is a two-sided "
        "bootstrap-$t$ test requiring no symmetry assumption and second-order "
        "accurate under skew, which the \\emph{Skew} column shows is not "
        "innocuous for normalized regret. Agreement of the two columns is the "
        "validation the reviewer asked for; where they differ, the studentized "
        "value is the one to read.",
        "tab:r9-studentized",
        "llcrrrr",
        [
            f"{r['dataset']} & {r['quantity']} & {r['n_users']} & {num(r['mean_difference'])} & "
            f"{num(r['skew'], 2)} & {pstr(r['sign_flip_p'])} & {pstr(r['studentized_p'])} \\\\"
            for r in stud
        ],
    )

    L += table(
        "Dataset & Users & Mean $|\\Delta$AIA| & CI low & CI high & Max & "
        "Mean rel.\\ SE & Min rank corr. \\\\",
        "Per-user Monte Carlo error propagated into the reported alignment "
        "statistic for the $n_{\\max}=20$ cap users: absolute change in bounded "
        "AIA between the diagnostic budget $M_{\\mathrm{pair}}=250$ and the "
        "independent $M_{\\mathrm{pair}}=1000$ reference. This bounds how far "
        "estimator noise alone can move the headline alignment numbers, which is "
        "the propagation the reviewer asked for beyond the rank-agreement "
        "diagnostics; adaptive stopping for unstable users remains an extension.",
        "tab:r9-mc-propagation",
        "lrrrrrrr",
        [
            f"{r['dataset']} & {r['n_users']} & {num(r['mean_abs_shift'], 5)} & "
            f"{num(r['ci95_low'], 5)} & {num(r['ci95_high'], 5)} & "
            f"{num(r['max_abs_shift'], 5)} & {num(r['mean_relative_se'], 3)} & "
            f"{num(r['min_rank_corr'], 3)} \\\\"
            for r in mcp
        ],
    )

    L += table(
        "Dataset & Contrast & Metric & $n$ & Draws & Exceed. & Re-estimated $p$ & "
        "MC SE & $2$-SE band at $R=10^4$ & Per-metric $p$ pub. & 12-contrast $p$ pub. \\\\",
        "Is a printed $p$-value censored, or does it reflect the actual number of "
        "permutation draws? Each row recomputes the declared plus-one sign-flip "
        "test for a disputed contrast from the released per-user differences with "
        "$50{,}000$ draws. A raw $p$ of $10^{-4}$ with zero exceedances is the "
        "resolution floor, not censoring; and the two published values for the "
        "same contrast, which differ in their third decimal, both lie within two "
        "Monte Carlo standard errors of the re-estimated one, so their "
        "disagreement is permutation sampling error of the same order as the "
        "floor rather than a different test. The acceptance band is twice the "
        "Monte Carlo error of a $10{,}000$-draw estimate, because that is the "
        "experiment that produced the published values; ticks mark the published "
        "values falling inside it.",
        "tab:r9-permutation-precision",
        "lllccrrrrrr",
        [
            (
                f"{r['dataset']} & {r['contrast']} & {r['metric']} & {r['n_users']} & "
                f"{r['draws']} & {r['exceedances']} & {num(r['p_reestimated'], 6)} & "
                f"{num(r['mc_se'], 6)} & {num(r['se_at_published_draws'], 6)} & "
                f"{num(r['published_per_metric_p'], 6)}{tick(r['per_metric_within_2se'])} & "
                f"{num(r['published_12_contrast_p'], 6)}{tick(r['family_12_within_2se'])} \\\\"
            )
            for r in prec
        ],
    )
    return "\n".join(L) + "\n"


def main() -> None:
    dry_run = "--dry-run" in __import__("sys").argv
    d = pd.read_csv(MAT / "user_seed_metrics.csv.gz")
    paired = pd.read_csv(MAT / "paired_tests.csv")
    review7 = pd.read_csv(MAT / "review7_success_abstention_tests.csv")

    fd_tex, fd_stats = benchmark_replication_tables()
    s4_tex, s4_report = regenerate_s4(
        paired, review7, MIRRORS[0] / "appendix_s3b_effects.tex"
    )

    flow = inclusion_flow(d)
    sens = all_user_sensitivity(d)
    mult, mult_stats = multiplicity_map(paired, review7)
    factorial = utility_factorial(d)
    stud = studentized_table(d)
    mcp = mc_propagation()
    prec = permutation_precision(paired, d)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/make_review9_stats.py",
        "inclusion_flow": flow,
        "all_user_sensitivity": sens,
        "multiplicity_map": mult,
        "multiplicity_summary": {k: v for k, v in mult_stats.items() if k != "shapley_lime_success_rows"},
        "utility_factorial": factorial,
        "studentized": stud,
        "mc_propagation": mcp,
        "permutation_precision": prec,
        "fixed_denominator": fd_stats,
        "s4_regeneration": {k: v for k, v in s4_report.items() if k != "holm_changes"},
        "s4_holm_changes": s4_report["holm_changes"],
    }
    if dry_run:
        # A check must not write anything: --dry-run with AES_REVIEW9_RESULTS pointed at
        # the pilot directory used to drop a review9_statistics.json there, which a
        # later reader could easily mistake for a completed run.
        import re
        # fd_tex is one rendered string of floats, not a list of lines.
        labels = re.findall(r"\\label\{([^}]*)\}", fd_tex)
        print("DRY_RUN_FLOATS: " + json.dumps(sorted(labels)))
        print("DRY_RUN_RESULTS_DIR: " + str(OUT))
        print("DRY_RUN_RESULTS_JSON_COUNT: " + str(len(list(OUT.glob("*.json")))))
        return
    pd.DataFrame(mult).to_csv(OUT / "review9_multiplicity_map.csv", index=False)
    payload["multiplicity_map"] = [
        r for r in mult if r["family_size"] in (10, 12) and "MC Shapley" in (r["left"], r["right"])
    ]
    (OUT / "review9_statistics.json").write_text(json.dumps(payload, indent=1))
    print("wrote", OUT / "review9_statistics.json")
    print(
        json.dumps(
            {
                "s4": {
                    k: v
                    for k, v in s4_report.items()
                    if k in {"rows", "matched_release_rows", "holm_changes_count", "unmatched", "point_estimate_disagreements"}
                },
                "multiplicity_summary": payload["multiplicity_summary"],
            },
            indent=1,
        )
    )

    if fd_tex:
        print("fixed-denominator ablation:", fd_stats)
    tex = render(flow, sens, mult, mult_stats, factorial, stud, mcp, prec)
    for mirror in MIRRORS:
        if mirror.is_dir():
            (mirror / "review9_statistics.tex").write_text(tex)
            (mirror / "appendix_s3b_effects.tex").write_text(s4_tex)
            if fd_tex:
                (mirror / "review9_benchmark_replications.tex").write_text(fd_tex)
            print("wrote", mirror / "review9_statistics.tex")
            print("wrote", mirror / "appendix_s3b_effects.tex")


if __name__ == "__main__":
    main()
