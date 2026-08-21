#!/usr/bin/env python3
"""Paired inference for the v7 corrected-protocol runs (round-9 fix #1).

Same corrected statistics as analyze_round6_stats.py (joint seed-mean user
differences, sign-flip tests with +1 correction B=10,000, Wilcoxon
sensitivity, dz bootstrap CIs, Holm families, equivalence, Friedman-Nemenyi),
applied to *_v7_corrected_protocol per-user artifacts. Runs for whichever
datasets have completed v7 runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "results" / "journal_runs"
OUT = RUNS / "round9_v7_analysis"
TEX = HERE.parent.parent / "manuscript_assets" / "round9"
OUT.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)

B_PERM = 10_000
B_BOOT = 10_000
SEED = 20260821
SEEDS = [42, 43, 44, 45, 46]
CHUNK = 1000
METRICS = ["NDCG@20", "HitRate@20"]
V7 = {"ml1m": RUNS / "ml1m_lightgcn_v7_corrected_protocol",
      "amazon_books": RUNS / "amazon_books_lightgcn_v7_corrected_protocol"}

FAMILIES = {
    "v7_loo": ("loo-marginal", ["uniform", "additive-pref", "attention",
                                "heuristic-pop", "valid-sim", "valid-linear"]),
    "v7_shapley": ("shapley-mc", ["uniform", "valid-sim", "valid-linear", "loo-marginal"]),
}


def available(ds):
    return (V7[ds] / "raw" / "per_user_metrics_all.csv.gz").exists()


def load_per_user(run_dir: Path):
    df = pd.read_csv(run_dir / "raw" / "per_user_metrics_all.csv.gz")
    df = df[df["metric"].isin(METRICS)]
    users = np.sort(df["user"].unique())
    n = len(users)
    uidx = {int(u): i for i, u in enumerate(users)}
    sidx = {s: i for i, s in enumerate(SEEDS)}
    out = {}
    for (fam, met), g in df.groupby(["family", "metric"]):
        mat = np.full((n, len(SEEDS)), np.nan)
        for s, gs in g.groupby("seed"):
            mat[[uidx[int(u)] for u in gs["user"]], sidx[int(s)]] = gs["value"].to_numpy()
        out[(fam, met)] = mat
    return out


def permutation_p(d, rng, b=B_PERM):
    n = len(d)
    obs = abs(d.mean())
    ge = 0
    for start in range(0, b, CHUNK):
        m = min(CHUNK, b - start)
        signs = rng.choice([-1.0, 1.0], size=(m, n))
        t = np.abs(signs @ d) / n
        ge += int((t >= obs - 1e-15).sum())
    return (ge + 1) / (b + 1)


def bootstrap_ci_dz(d, rng, b=B_BOOT):
    n = len(d)
    means = np.empty(b)
    dzs = np.empty(b)
    for start in range(0, b, CHUNK):
        m = min(CHUNK, b - start)
        idx = rng.integers(0, n, size=(m, n))
        samp = d[idx]
        mu = samp.mean(axis=1)
        sd = samp.std(axis=1, ddof=1)
        means[start:start + m] = mu
        dzs[start:start + m] = np.where(sd > 0, mu / sd, 0.0)
    return (np.quantile(means, [0.025, 0.975]).tolist(),
            np.quantile(means, [0.05, 0.95]).tolist(),
            np.quantile(dzs, [0.025, 0.975]).tolist())


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvals[idx] * (m - rank)))
        adj[idx] = running
    return adj.tolist()


def contrast_stats(matA, matB, rng):
    d = np.nanmean(matA - matB, axis=1)
    d = d[~np.isnan(d)]
    delta = float(d.mean())
    p_perm = permutation_p(d, rng)
    ci95, ci90, dz_ci = bootstrap_ci_dz(d, rng)
    sd = float(d.std(ddof=1))
    try:
        p_w = float(sps.wilcoxon(d, zero_method="wilcox", correction=True, method="auto").pvalue)
    except ValueError:
        p_w = float("nan")
    return {"n_users": int(len(d)), "mean_diff": delta, "ci95": ci95, "ci90": ci90,
            "p_perm": p_perm, "p_wilcoxon": p_w,
            "dz": delta / sd if sd > 0 else 0.0, "dz_ci95": dz_ci}


def nemenyi(mean_ranks, k, n_blocks):
    pmat = np.ones((k, k))
    denom = np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(mean_ranks[i] - mean_ranks[j]) / denom
            pmat[i, j] = pmat[j, i] = min(1.0, float(sps.studentized_range.sf(q * np.sqrt(2), k, np.inf)))
    return pmat


def main():
    rng_master = np.random.default_rng(SEED)
    rows, friedman, eq = [], {}, {}
    for ds in V7:
        if not available(ds):
            print(f"skip {ds}: per-user artifact missing")
            continue
        mats = load_per_user(V7[ds])
        print(f"loaded {ds}: {len(mats)} (family, metric) matrices")
        for fam_name, (treat, comps) in FAMILIES.items():
            for met in METRICS:
                for comp in comps:
                    r = contrast_stats(mats[(treat, met)], mats[(comp, met)],
                                       np.random.default_rng(rng_master.integers(2**32)))
                    r.update(family=fam_name, dataset=ds, treatment=treat, comparator=comp, metric=met)
                    rows.append(r)
        # equivalence Shapley vs LOO
        for met in METRICS:
            d = np.nanmean(mats[("shapley-mc", met)] - mats[("loo-marginal", met)], axis=1)
            d = d[~np.isnan(d)]
            rng = np.random.default_rng(rng_master.integers(2**32))
            _, ci90, _ = bootstrap_ci_dz(d, rng)
            margin = 0.001 if met == "NDCG@20" else 0.002
            eq[f"{ds}:{met}"] = {"mean": float(d.mean()), "ci90": ci90,
                                 "inside": bool(ci90[0] > -margin and ci90[1] < margin),
                                 "all_negative": bool(ci90[1] < 0)}
        # friedman
        for met in METRICS:
            fams = sorted({f for (f, m) in mats if m == met})
            seedmeans = np.stack([np.nanmean(mats[(f, met)], axis=1) for f in fams], axis=1)
            N, k = seedmeans.shape
            chi2, p = sps.friedmanchisquare(*[seedmeans[:, i] for i in range(k)])
            mean_ranks = np.apply_along_axis(sps.rankdata, 1, seedmeans).mean(axis=0)
            friedman[f"{ds}:{met}"] = {"chi2": float(chi2), "p": float(p),
                                       "top": [fams[i] for i in np.argsort(mean_ranks)[::-1]][:3]}
    dfc = pd.DataFrame(rows)
    dfc["p_perm_holm"] = dfc.groupby(["family", "dataset"])["p_perm"].transform(lambda s: holm(s.tolist()))
    dfc["p_wilcoxon_holm"] = dfc.groupby(["family", "dataset"])["p_wilcoxon"].transform(lambda s: holm(s.tolist()))
    dfc.to_csv(OUT / "v7_paired_contrasts.csv", index=False)
    (OUT / "v7_equivalence.json").write_text(json.dumps(eq, indent=1))
    (OUT / "v7_friedman.json").write_text(json.dumps(friedman, indent=1))
    print(dfc.round(5).to_string(index=False))
    print(json.dumps(eq, indent=1))
    print(json.dumps(friedman, indent=1))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
