#!/usr/bin/env python3
"""Round-7: paired inference for the NGCF second-backbone study (R7-1).

Same corrected methodology as analyze_round6_stats.py: joint seed-mean user
differences, sign-flip permutation p-values with +1 correction (B=10,000),
Wilcoxon sensitivity, d_z with bootstrap CIs, Holm within pre-specified
families, Friedman-Nemenyi omnibus.

Families (mirroring the C1b declarations):
  ngcf_loo      F=12: loo-marginal vs {uniform, additive-pref, attention,
                heuristic-pop, valid-sim, valid-linear} x {NDCG@20, HR@20}
  ngcf_shapley  F=8 : shapley-mc vs {uniform, valid-sim, valid-linear,
                loo-marginal} x {NDCG@20, HR@20}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "results" / "journal_runs"
OUT = RUNS / "round7_ngcf_analysis"
TEX = HERE.parent.parent / "manuscript_assets" / "round7"
OUT.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)

B_PERM = 10_000
B_BOOT = 10_000
SEED = 20260819
SEEDS = [42, 43, 44, 45, 46]
CHUNK = 1000
METRICS = ["NDCG@20", "HitRate@20"]
NGCF = {"ml1m": RUNS / "ml1m_ngcf_v6_second_backbone",
        "amazon_books": RUNS / "amazon_books_ngcf_v6_second_backbone"}


def load_per_user(run_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    df = pd.read_csv(run_dir / "raw" / "per_user_metrics_all.csv.gz")
    df = df[df["metric"].isin(METRICS)]
    users = np.sort(df["user"].unique())
    n = len(users)
    uidx = {int(u): i for i, u in enumerate(users)}
    sidx = {s: i for i, s in enumerate(SEEDS)}
    out: dict[tuple[str, str], np.ndarray] = {}
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
    return (np.quantile(means, [0.025, 0.975]).tolist(),
            np.quantile(means, [0.05, 0.95]).tolist(),
            np.quantile(dzs, [0.025, 0.975]).tolist())


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvals[idx] * (m - rank)))
        adj[idx] = running
    return adj.tolist()


def contrast_stats(matA: np.ndarray, matB: np.ndarray, rng) -> dict:
    d = np.nanmean(matA - matB, axis=1)
    d = d[~np.isnan(d)]
    delta = float(d.mean())
    p_perm = permutation_p(d, rng)
    ci95, ci90, dz_ci = bootstrap_ci_dz(d, rng)
    sd = float(d.std(ddof=1))
    dz = delta / sd if sd > 0 else 0.0
    try:
        p_w = float(sps.wilcoxon(d, zero_method="wilcox", correction=True, method="auto").pvalue)
    except ValueError:
        p_w = float("nan")
    return {"n_users": int(len(d)), "mean_diff": delta, "ci95": ci95, "ci90": ci90,
            "p_perm": p_perm, "p_wilcoxon": p_w, "dz": float(dz), "dz_ci95": dz_ci}


def nemenyi(mean_ranks: np.ndarray, k: int, n_blocks: int) -> np.ndarray:
    pmat = np.ones((k, k))
    denom = np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(mean_ranks[i] - mean_ranks[j]) / denom
            pmat[i, j] = pmat[j, i] = min(1.0, float(sps.studentized_range.sf(q * np.sqrt(2), k, np.inf)))
    return pmat


def main():
    rng_master = np.random.default_rng(SEED)
    data = {ds: load_per_user(run) for ds, run in NGCF.items()}

    families = {
        "ngcf_loo": ("loo-marginal", ["uniform", "additive-pref", "attention",
                                      "heuristic-pop", "valid-sim", "valid-linear"]),
        "ngcf_shapley": ("shapley-mc", ["uniform", "valid-sim", "valid-linear", "loo-marginal"]),
    }
    rows = []
    for fam_name, (treat, comps) in families.items():
        for ds in NGCF:
            mats = data[ds]
            for met in METRICS:
                for comp in comps:
                    r = contrast_stats(mats[(treat, met)], mats[(comp, met)],
                                       np.random.default_rng(rng_master.integers(2**32)))
                    r.update(family=fam_name, dataset=ds, treatment=treat, comparator=comp, metric=met)
                    rows.append(r)
    dfc = pd.DataFrame(rows)
    dfc["p_perm_holm"] = dfc.groupby(["family", "dataset"])["p_perm"].transform(lambda s: holm(s.tolist()))
    dfc["p_wilcoxon_holm"] = dfc.groupby(["family", "dataset"])["p_wilcoxon"].transform(lambda s: holm(s.tolist()))
    dfc.to_csv(OUT / "ngcf_paired_contrasts.csv", index=False)

    # Friedman-Nemenyi over all nine NGCF families
    friedman = {}
    for ds in NGCF:
        mats = data[ds]
        fams = sorted({f for (f, m) in mats if m == "NDCG@20"})
        for met in METRICS:
            seedmeans = np.stack([np.nanmean(mats[(f, met)], axis=1) for f in fams], axis=1)
            N, k = seedmeans.shape
            chi2, p = sps.friedmanchisquare(*[seedmeans[:, i] for i in range(k)])
            row_ranks = np.apply_along_axis(sps.rankdata, 1, seedmeans)
            mean_ranks = row_ranks.mean(axis=0)
            pmat = nemenyi(mean_ranks, k, N)
            order = [fams[i] for i in np.argsort(mean_ranks)[::-1]]
            friedman[f"{ds}:{met}"] = {
                "friedman_chi2": float(chi2), "p": float(p),
                "rank_order": order[:4],
                "loo_vs_shapley_nemenyi_p": float(pmat[fams.index("loo-marginal"), fams.index("shapley-mc")])}
    with open(OUT / "ngcf_friedman.json", "w") as f:
        json.dump(friedman, f, indent=1)

    # equivalence check for Shapley vs LOO on NGCF (descriptive, same margins)
    eq = {}
    for ds in NGCF:
        mats = data[ds]
        d = np.nanmean(mats[("shapley-mc", "NDCG@20")] - mats[("loo-marginal", "NDCG@20")], axis=1)
        rng = np.random.default_rng(rng_master.integers(2**32))
        _, ci90, _ = bootstrap_ci_dz(d, rng)
        eq[ds] = {"mean_diff": float(d.mean()), "ci90": ci90,
                  "inside_margin": bool(ci90[0] > -0.001 and ci90[1] < 0.001),
                  "all_negative": bool(ci90[1] < 0)}
    with open(OUT / "ngcf_equivalence.json", "w") as f:
        json.dump(eq, f, indent=1)
    print(json.dumps({"contrasts": len(dfc), "equivalence": eq}, indent=1, default=str))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
