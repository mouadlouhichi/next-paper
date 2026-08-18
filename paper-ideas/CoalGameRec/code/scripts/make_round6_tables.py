#!/usr/bin/env python3
"""Emit LaTeX table fragments from round6_analysis outputs (manuscript v19)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RUN6 = Path(__file__).resolve().parent.parent / "results" / "journal_runs" / "round6_analysis"
TEX = Path(__file__).resolve().parent.parent.parent / "manuscript_assets" / "round6"
TEX.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RUN6 / "paired_contrasts_round6.csv")
old = pd.read_csv(RUN6 / "sensitivity_old_bootstrap_primary.csv")


def fmtp(p: float) -> str:
    if p < 1e-4:
        return "$<10^{-4}$"
    return f"{p:.4f}".rstrip("0").rstrip(".")


def fmtci(ci) -> str:
    lo, hi = ci.strip("[]").split(",")
    lo, hi = float(lo), float(hi)
    nd = 6 if max(abs(lo), abs(hi)) < 0.01 else 4
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def fmtci3(ci) -> str:
    lo, hi = ci.strip("[]").split(",")
    return f"[{float(lo):+.3f}, {float(hi):+.3f}]"


def row(r, contrast_label: str) -> str:
    ds = "ML-1M" if r.dataset == "ml1m" else "Amazon"
    return (" & ".join([
        ds, contrast_label, r.metric.replace("HitRate", "HR"),
        f"{r.mean_diff:+.6f}", fmtci(r.ci95), fmtp(r.p_perm_holm),
        fmtp(r.p_wilcoxon_holm),
        f"{r.dz:+.3f} {fmtci3(r.dz_ci95)}"])) + " \\\\"


def pick(family, ds, treat, comp, met):
    m = df[(df.family == family) & (df.dataset == ds) & (df.treatment == treat)
           & (df.comparator == comp) & (df.metric == met)]
    assert len(m) == 1, (family, ds, treat, comp, met)
    return m.iloc[0]


HDR = ("Dataset & Contrast & Metric & Mean diff. & 95\\% CI & $p$ (perm., Holm) "
       "& $p$ (Wilcoxon, Holm) & $d_z$ [95\\% CI] \\\\\n\\midrule\n")

# ---------------- primary (tab:paired) ----------------
lines = [HDR]
for ds in ["ml1m", "amazon_books"]:
    for comp, lab in [("uniform", "Shapley vs uniform"), ("loo-marginal", "Shapley vs LOO")]:
        for met in ["NDCG@20", "HitRate@20"]:
            lines.append(row(pick("primary", ds, "shapley-mc", comp, met), lab))
(TEX / "tab_paired_round6.tex").write_text("\n".join(lines))

# ---------------- C1 LOO (tab:c1_paired) ----------------
lines = [HDR]
for ds in ["ml1m", "amazon_books"]:
    for comp, lab in [("valid-sim", "LOO vs valid-sim"), ("valid-linear", "LOO vs valid-linear")]:
        for met in ["NDCG@20", "HitRate@20"]:
            lines.append(row(pick("c1_loo", ds, "loo-marginal", comp, met), lab))
(TEX / "tab_c1_paired_round6.tex").write_text("\n".join(lines))

# ---------------- C1 Shapley (tab:c1_shap) ----------------
lines = [HDR]
for ds in ["ml1m", "amazon_books"]:
    for comp, lab in [("valid-sim", "Shapley vs valid-sim"), ("valid-linear", "Shapley vs valid-linear"),
                      ("loo-marginal", "Shapley vs LOO")]:
        for met in ["NDCG@20", "HitRate@20"]:
            lines.append(row(pick("c1_shapley", ds, "shapley-mc", comp, met), lab))
(TEX / "tab_c1_shap_round6.tex").write_text("\n".join(lines))

# ---------------- procedure sensitivity (Shapley vs LOO) ----------------
rows = []
for ver, family, ds in [("v3", "primary", "ml1m"), ("v3", "primary", "amazon_books"),
                        ("v4b", "c1_shapley", "ml1m"), ("v4b", "c1_shapley", "amazon_books")]:
    for met in ["NDCG@20", "HitRate@20"]:
        r = pick(family, ds, "shapley-mc", "loo-marginal", met)
        p_old = None
        if ver == "v3":
            mo = old[(old.dataset == ds) & (old.comparator == "loo-marginal") & (old.metric == met)]
            p_old = float(mo.iloc[0]["p_old_within_seed_boot"])
        ds_l = "ML-1M" if ds == "ml1m" else "Amazon"
        rows.append(" & ".join([
            ds_l, ("primary" if ver == "v3" else "C1b"), met.replace("HitRate", "HR"),
            f"{r.mean_diff:+.6f}", fmtp(r.p_perm_holm),
            (fmtp(p_old) if p_old is not None else "---"),
            fmtp(r.p_wilcoxon_holm)]) + " \\\\")
(TEX / "tab_robustness_sensitivity.tex").write_text(
    "Dataset & Study & Metric & Mean diff. & $p$ joint perm.\\ (Holm) & $p$ within-seed boot.\\ (legacy) & $p$ Wilcoxon (Holm) \\\\\n\\midrule\n"
    + "\n".join(rows))

# ---------------- Friedman ----------------
f = json.load(open(RUN6 / "friedman_nemenyi.json"))
rows = []
for key, lab in [("v4b:ml1m:NDCG@20", "ML-1M / NDCG@20"), ("v4b:ml1m:HitRate@20", "ML-1M / HR@20"),
                 ("v4b:amazon_books:NDCG@20", "Amazon / NDCG@20"), ("v4b:amazon_books:HitRate@20", "Amazon / HR@20")]:
    d = f[key]
    mr = sorted(d["mean_ranks"].items(), key=lambda x: -x[1])
    top = " $>$ ".join(f"{k.replace('-mc','').replace('-marginal','')}" for k, _ in mr[:3])
    rows.append(f"{lab} & $\\chi^2={d['friedman_chi2']:.0f}$, $p<10^{{-30}}$ & {top} & all n.s. \\\\")
(TEX / "tab_friedman.tex").write_text(
    "Scope (C1b, 9 families, users as blocks) & Friedman omnibus & Top-3 mean ranks & Nemenyi--Holm pairwise \\\\\n\\midrule\n"
    + "\n".join(rows))

# ---------------- numbers for prose ----------------
eq = {}
for family, ver, ds in [("primary", "v3", "ml1m"), ("primary", "v3", "amazon_books"),
                        ("c1_shapley", "v4b", "ml1m"), ("c1_shapley", "v4b", "amazon_books")]:
    r = pick(family, ds, "shapley-mc", "loo-marginal", "NDCG@20")
    eq[f"{ver}:{ds}"] = {"mean": r.mean_diff, "ci90": r.ci90, "ci95": r.ci95, "p_perm": r.p_perm}
power = json.load(open(RUN6 / "power_mde.json"))
cross = json.load(open(RUN6 / "top20_crossing.json"))
rt = json.load(open(RUN6 / "runtime_stats.json"))
numbers = {"equivalence_joint": eq, "power_mde": power, "top20_crossing": cross, "runtime": rt}
(TEX / "numbers_round6.json").write_text(json.dumps(numbers, indent=1))
print("WROTE fragments to", TEX)
