#!/usr/bin/env python3
"""Round-6 inferential upgrade (responds to the three Discovery-AI reviews).

Recomputes ALL paired inferential tables from the released per-user artifacts
(v3 primary runs + v4b matched C1b runs) with the corrected statistics the
reviewers require:

  1. JOINT user resampling across seeds: per-user seed-mean paired difference
     d_u = (1/S) sum_s (m^A_{u,s} - m^B_{u,s}); the same users are resampled,
     preserving cross-seed dependence (Reviewer-1 stat item 2).
  2. Paired sign-flip PERMUTATION p-values with the +1 Monte-Carlo correction
     p = (#{|T_b| >= |T_obs|} + 1) / (B + 1), B = 10,000 -- replaces the
     zero-count "<1/B" bootstrap sign-count p-values (Reviewer-1 stat item 3).
  3. Percentile bootstrap CIs (95% and 90%) and bootstrap CIs for d_z.
  4. Wilcoxon signed-rank sensitivity p-values.
  5. Friedman + Nemenyi omnibus ranking test per dataset/metric
     (users as blocks; requested by Reviewers 2 and 3).
  6. Minimum detectable effect (paired z approximation) and TOST power for the
     declared SESOI margins (Reviewer-1 stat items 6/9).
  7. Proportion of users whose top-20 hit status changes (practical impact).
  8. Runtime medians/IQRs from the per-seed runtime.json artifacts.
  9. Oracle best-lambda sensitivity table from the released lambda-sweep
     artifacts (explicitly labelled test-oracle; validation-tuned lambda is a
     separate user-side run).
 10. Sensitivity: the OLD within-seed bootstrap re-run at B=10,000 for the
     primary family, to show conclusions are procedure-robust.

Everything is deterministic (seed 20260818). No new model outputs are used.

Usage:
  python scripts/analyze_round6_stats.py
Outputs:
  results/journal_runs/round6_analysis/*.csv|json
  ../../manuscript_assets/round6/*.tex (table fragments) + numbers_round6.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "results" / "journal_runs"
OUT = RUNS / "round6_analysis"
TEX = HERE.parent.parent / "manuscript_assets" / "round6"
OUT.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)

B_PERM = 10_000
B_BOOT = 10_000
SEED = 20260818
SEEDS = [42, 43, 44, 45, 46]
CHUNK = 1000

V3 = {"ml1m": RUNS / "ml1m_lightgcn_v3_prospective",
      "amazon_books": RUNS / "amazon_books_lightgcn_v3_prospective"}
V4B = {"ml1m": RUNS / "ml1m_lightgcn_v4b_matched_controls",
       "amazon_books": RUNS / "amazon_books_lightgcn_v4b_matched_controls"}

METRICS = ["NDCG@20", "HitRate@20"]


def load_per_user(run_dir: Path, gz: bool) -> dict[tuple[str, str], np.ndarray]:
    """Return {(family, metric): array (n_users, 5) aligned to SEEDS}."""
    path = run_dir / "raw" / ("per_user_metrics_all.csv.gz" if gz else "per_user_metrics_all.csv")
    df = pd.read_csv(path)
    df = df[df["metric"].isin(METRICS)]
    out: dict[tuple[str, str], np.ndarray] = {}
    users = np.sort(df["user"].unique())
    n = len(users)
    uidx = {int(u): i for i, u in enumerate(users)}
    sidx = {s: i for i, s in enumerate(SEEDS)}
    for (fam, met), g in df.groupby(["family", "metric"]):
        mat = np.full((n, len(SEEDS)), np.nan)
        for s, gs in g.groupby("seed"):
            mat[[uidx[int(u)] for u in gs["user"]], sidx[int(s)]] = gs["value"].to_numpy()
        out[(fam, met)] = mat
    return out


def permutation_p(d: np.ndarray, rng: np.random.Generator, b: int = B_PERM) -> float:
    n = len(d)
    obs = abs(d.mean())
    ge = 0
    for start in range(0, b, CHUNK):
        m = min(CHUNK, b - start)
        signs = rng.choice([-1.0, 1.0], size=(m, n))
        t = np.abs(signs @ d) / n
        ge += int((t >= obs - 1e-15).sum())
    return (ge + 1) / (b + 1)


def bootstrap_ci_dz(d: np.ndarray, rng: np.random.Generator, b: int = B_BOOT):
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
    ci95 = np.quantile(means, [0.025, 0.975])
    ci90 = np.quantile(means, [0.05, 0.95])
    dz_ci = np.quantile(dzs, [0.025, 0.975])
    return ci95, ci90, dz_ci


def within_seed_bootstrap(diff_by_seed: list[np.ndarray], rng: np.random.Generator, b: int = B_BOOT) -> float:
    """Old procedure (users resampled independently within each seed): two-sided sign-count p."""
    le = ge = 0
    k = len(diff_by_seed)
    for start in range(0, b, CHUNK):
        m = min(CHUNK, b - start)
        vals = np.zeros(m)
        for d in diff_by_seed:
            idx = rng.integers(0, len(d), size=(m, len(d)))
            vals += d[idx].mean(axis=1)
        vals /= k
        le += int((vals <= 0).sum())
        ge += int((vals >= 0).sum())
    return 2.0 * min(le, ge) / b


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - rank))
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def contrast_stats(matA: np.ndarray, matB: np.ndarray, rng: np.random.Generator) -> dict:
    d = np.nanmean(matA - matB, axis=1)  # joint seed-mean difference
    d = d[~np.isnan(d)]
    delta = float(d.mean())
    p_perm = permutation_p(d, rng)
    ci95, ci90, dz_ci = bootstrap_ci_dz(d, rng)
    sd = float(d.std(ddof=1))
    dz = delta / sd if sd > 0 else 0.0
    try:
        w = sps.wilcoxon(d, zero_method="wilcox", correction=True, method="auto")
        p_wilcox = float(w.pvalue)
    except ValueError:
        p_wilcox = float("nan")
    return {"n_users": int(len(d)), "mean_diff": delta, "ci95": ci95.tolist(),
            "ci90": ci90.tolist(), "p_perm": p_perm, "p_wilcoxon": p_wilcox,
            "dz": float(dz), "dz_ci95": dz_ci.tolist()}


def nemenyi(ranks_mean: np.ndarray, k: int, n_blocks: int) -> np.ndarray:
    """Pairwise Nemenyi p-values from mean ranks (studentized range)."""
    pmat = np.ones((k, k))
    denom = np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(ranks_mean[i] - ranks_mean[j]) / denom
            p = sps.studentized_range.sf(q * np.sqrt(2), k, np.inf)
            pmat[i, j] = pmat[j, i] = min(1.0, float(p))
    return pmat


def friedman_nemenyi(mats: dict[str, np.ndarray], metric: str) -> dict:
    fams = list(mats.keys())
    seedmeans = np.stack([np.nanmean(mats[f], axis=1) for f in fams], axis=1)
    N, k = seedmeans.shape
    chi2, p = sps.friedmanchisquare(*[seedmeans[:, i] for i in range(k)])
    row_ranks = np.apply_along_axis(sps.rankdata, 1, seedmeans)
    mean_ranks = row_ranks.mean(axis=0)
    pmat = nemenyi(mean_ranks, k, N)
    flat = [pmat[i, j] for i in range(k) for j in range(i + 1, k)]
    adj = holm(flat)
    pairs = {}
    t = 0
    for i in range(k):
        for j in range(i + 1, k):
            pairs[f"{fams[i]} vs {fams[j]}"] = {"p": pmat[i, j], "p_holm": adj[t]}
            t += 1
    order = np.argsort(mean_ranks)[::-1]
    return {"friedman_chi2": float(chi2), "p": float(p), "k": k, "n_users": N,
            "mean_ranks": {fams[i]: float(mean_ranks[i]) for i in range(k)},
            "rank_order": [fams[i] for i in order], "nemenyi_pairs": pairs}


def main():
    rng_master = np.random.default_rng(SEED)
    report: dict = {"seed": SEED, "B_perm": B_PERM, "B_boot": B_BOOT}

    data = {}
    for ds in V3:
        data[("v3", ds)] = load_per_user(V3[ds], gz=False)
    for ds in V4B:
        data[("v4b", ds)] = load_per_user(V4B[ds], gz=True)

    # ---------------- paired contrast families ----------------
    families = {
        "primary": ("v3", "shapley-mc", ["uniform", "additive-pref", "attention", "loo-marginal"]),
        "loo_treatment": ("v3", "loo-marginal", ["uniform", "additive-pref", "attention", "heuristic-pop", "shapley-mc"]),
        "c1_loo": ("v4b", "loo-marginal", ["uniform", "additive-pref", "attention", "heuristic-pop", "valid-sim", "valid-linear"]),
        "c1_shapley": ("v4b", "shapley-mc", ["uniform", "valid-sim", "valid-linear", "loo-marginal"]),
    }
    all_contrasts = []
    for fam_name, (ver, treat, comps) in families.items():
        for ds in ["ml1m", "amazon_books"]:
            mats = data[(ver, ds)]
            for met in METRICS:
                for comp in comps:
                    if (treat, met) not in mats or (comp, met) not in mats:
                        continue
                    r = contrast_stats(mats[(treat, met)], mats[(comp, met)],
                                       np.random.default_rng(rng_master.integers(2**32)))
                    r.update(family=fam_name, dataset=ds, treatment=treat, comparator=comp, metric=met)
                    all_contrasts.append(r)

    dfc = pd.DataFrame(all_contrasts)
    # Holm within each pre-specified family x dataset (metrics pooled as declared)
    dfc["p_perm_holm"] = dfc.groupby(["family", "dataset"])["p_perm"].transform(lambda s: holm(s.tolist()))
    dfc["p_wilcoxon_holm"] = dfc.groupby(["family", "dataset"])["p_wilcoxon"].transform(lambda s: holm(s.tolist()))
    dfc.to_csv(OUT / "paired_contrasts_round6.csv", index=False)

    # old within-seed bootstrap sensitivity for the primary family
    old_rows = []
    for ds in ["ml1m", "amazon_books"]:
        mats = data[("v3", ds)]
        for met in METRICS:
            for comp in families["primary"][2]:
                diffs = [mats[("shapley-mc", met)][:, i] - mats[(comp, met)][:, i] for i in range(len(SEEDS))]
                p_old = within_seed_bootstrap(diffs, np.random.default_rng(rng_master.integers(2**32)))
                old_rows.append({"dataset": ds, "comparator": comp, "metric": met, "p_old_within_seed_boot": p_old})
    pd.DataFrame(old_rows).to_csv(OUT / "sensitivity_old_bootstrap_primary.csv", index=False)

    # ---------------- Friedman / Nemenyi ----------------
    friedman = {}
    for ver, dss in [("v3", V3), ("v4b", V4B)]:
        for ds in dss:
            mats = data[(ver, ds)]
            fams = sorted({f for (f, m) in mats if m == "NDCG@20"})
            for met in METRICS:
                key = f"{ver}:{ds}:{met}"
                friedman[key] = friedman_nemenyi({f: mats[(f, met)] for f in fams}, met)
    with open(OUT / "friedman_nemenyi.json", "w") as f:
        json.dump(friedman, f, indent=1)

    # ---------------- MDE + TOST power ----------------
    mde = {}
    for ds in ["ml1m", "amazon_books"]:
        mats = data[("v4b", ds)]
        n = mats[("loo-marginal", "NDCG@20")].shape[0]
        mde_dz = (sps.norm.ppf(0.975) + sps.norm.ppf(0.80)) / np.sqrt(n)
        d = np.nanmean(mats[("shapley-mc", "NDCG@20")] - mats[("loo-marginal", "NDCG@20")], axis=1)
        sd, delta, n_u = d.std(ddof=1), d.mean(), len(d)
        z05 = sps.norm.ppf(0.95)
        power_tost = (sps.norm.cdf((0.001 - delta) * np.sqrt(n_u) / sd - z05)
                      - sps.norm.cdf((-0.001 - delta) * np.sqrt(n_u) / sd + z05))
        mde[ds] = {"n_users": int(n), "mde_dz_user_level": float(mde_dz),
                   "mde_ndcg_user_level": float(mde_dz * sd),
                   "tost_power_delta001": float(power_tost)}
    mde["seed_level_5_seeds"] = {"mde_dz": float((sps.norm.ppf(0.975) + sps.norm.ppf(0.80)) / np.sqrt(5))}
    with open(OUT / "power_mde.json", "w") as f:
        json.dump(mde, f, indent=1)

    # ---------------- top-20 crossing rates ----------------
    crossing = {}
    for ds in ["ml1m", "amazon_books"]:
        for ver, pairs in [("v3", [("loo-marginal", "uniform"), ("loo-marginal", "shapley-mc")]),
                           ("v4b", [("loo-marginal", "valid-linear"), ("loo-marginal", "valid-sim")])]:
            mats = data[(ver, ds)]
            for a, b in pairs:
                ma = np.nanmean(mats[(a, "HitRate@20")], axis=1)
                mb = np.nanmean(mats[(b, "HitRate@20")], axis=1)
                gained = ((ma > 0.5) & (mb <= 0.5)).mean()
                lost = ((ma <= 0.5) & (mb > 0.5)).mean()
                crossing[f"{ver}:{ds}:{a}-vs-{b}"] = {"gained": float(gained), "lost": float(lost),
                                                      "changed": float(gained + lost)}
    with open(OUT / "top20_crossing.json", "w") as f:
        json.dump(crossing, f, indent=1)

    # ---------------- runtime medians/IQRs ----------------
    runtimes = {}
    for ds, run in V3.items():
        rows = []
        for s in SEEDS:
            rj = json.load(open(run / "raw" / f"seed_{s}" / "runtime.json"))
            st = rj.get("stages", {})
            rows.append({"seed": s, "loo": st.get("loo_seconds"), "shapley": st.get("shapley_seconds")})
        rdf = pd.DataFrame(rows)
        runtimes[ds] = {}
        for col in ["loo", "shapley"]:
            v = rdf[col].dropna()
            runtimes[ds][col] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                 "median": float(v.median()),
                                 "iqr": [float(v.quantile(0.25)), float(v.quantile(0.75))],
                                 "total": float(v.sum())}
    with open(OUT / "runtime_stats.json", "w") as f:
        json.dump(runtimes, f, indent=1)

    # ---------------- oracle best-lambda (test-oracle sensitivity) ----------------
    oracle = []
    for ds, run in V3.items():
        ldfs = [pd.read_csv(run / "raw" / f"seed_{s}" / "lambda_sensitivity.csv") for s in SEEDS]
        ldf = pd.concat(ldfs)
        g = ldf.groupby(["family", "lambda_attr"])["NDCG@20"].agg(["mean", "std"])
        for fam, gf in g.groupby(level=0):
            best = gf["mean"].idxmax()
            oracle.append({"dataset": ds, "family": fam, "oracle_lambda": float(best[1]),
                           "ndcg20_at_oracle": float(gf.loc[best, "mean"]),
                           "sd": float(gf.loc[best, "std"]),
                           "ndcg20_at_protocol": float(gf.loc[(fam, 0.1), "mean"])})
    pd.DataFrame(oracle).to_csv(OUT / "oracle_lambda.csv", index=False)

    print(json.dumps({"contrasts": len(dfc), "friedman_keys": list(friedman.keys())}, indent=1))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
