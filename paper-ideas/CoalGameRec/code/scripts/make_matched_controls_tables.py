#!/usr/bin/env python3
"""Generate manuscript-ready LaTeX snippets from the matched-controls (v4) runs.

Outputs (printed to stdout, also saved to paper-ideas/CoalGameRec/manuscript_assets/):
  - C1 main-results table rows (5-seed mean±SD, grouped)
  - C1 paired LOO-vs-matched-controls contrast rows (after analyze_matched_controls.py)
  - Faithfulness-curve table (family x fraction, both datasets, 5-seed mean±SD)
  - Percentage-gain arithmetic checks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "MovieLens-1M": ROOT / "results/journal_runs/ml1m_lightgcn_v4_matched_controls",
    "Amazon-Book": ROOT / "results/journal_runs/amazon_books_lightgcn_v4_matched_controls",
}
METRICS = ["HitRate@20", "NDCG@20", "Coverage@20", "ILD@20"]
FAM_ORDER = ["unreranked", "uniform", "additive-pref", "attention", "heuristic-pop",
             "valid-sim", "valid-linear", "loo-marginal"]


def load_summary(run: Path) -> pd.DataFrame | None:
    p = run / "tables" / "summary_by_seed_family.csv"
    return pd.read_csv(p) if p.exists() else None


def fmt(mean: float, sd: float) -> str:
    return f"{mean:.5f} $\\pm$ {sd:.5f}"


def main_results_rows() -> str:
    lines = []
    for ds_name, run in RUNS.items():
        df = load_summary(run)
        if df is None:
            lines.append(f"% {ds_name}: summary not ready yet")
            continue
        n_seeds = int(df.seed.nunique())
        g = df.groupby("family")[METRICS].agg(["mean", "std"])
        lines.append(r"\multicolumn{7}{l}{\textit{%s --- unreranked reference ($\lambda_{\text{attr}}=0$)}}\\" % ds_name)
        for fam in ["unreranked"]:
            r = g.loc[fam]
            lines.append(" & ".join([ds_name, "LightGCN", fam] + [fmt(r[m]["mean"], r[m]["std"]) for m in METRICS]) + r" \\")
        lines.append(r"\multicolumn{7}{l}{\textit{%s --- non-game reweighting (no validation access)}}\\" % ds_name)
        for fam in ["uniform", "additive-pref", "attention", "heuristic-pop"]:
            r = g.loc[fam]
            lines.append(" & ".join([ds_name, "LightGCN", fam] + [fmt(r[m]["mean"], r[m]["std"]) for m in METRICS]) + r" \\")
        lines.append(r"\multicolumn{7}{l}{\textit{%s --- validation-informed non-game controls (matched validation access)}}\\" % ds_name)
        for fam in ["valid-sim", "valid-linear"]:
            r = g.loc[fam]
            lines.append(" & ".join([ds_name, "LightGCN", fam] + [fmt(r[m]["mean"], r[m]["std"]) for m in METRICS]) + r" \\")
        lines.append(r"\multicolumn{7}{l}{\textit{%s --- cooperative-game attribution (validation-guided)}}\\" % ds_name)
        r = g.loc["loo-marginal"]
        bold = [f"\\textbf{{{fmt(r[m]['mean'], r[m]['std'])}}}" for m in METRICS]
        lines.append(" & ".join([ds_name, "LightGCN", r"\textbf{CoalGameRec (LOO)}"] + bold) + r" \\")
        lines.append(f"% {ds_name}: {n_seeds} seeds")
    return "\n".join(lines)


def paired_rows() -> str:
    lines = []
    for ds_name, run in RUNS.items():
        p = run / "tables" / "paired_bootstrap_loo_vs_matched_controls.csv"
        if not p.exists():
            lines.append(f"% {ds_name}: paired contrasts not ready yet")
            continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            md = r["mean_diff_conditional_user"]
            md_s = f"{md:.6f}" if md >= 0 else f"$-${abs(md):.6f}"
            lo, hi = r["ci95_low"], r["ci95_high"]
            ci = f"[{lo:.6f}, {hi:.6f}]" if lo >= 0 else (
                f"[$-${abs(lo):.6f}, {hi:.6f}]" if hi >= 0 else f"[$-${abs(lo):.6f}, $-${abs(hi):.6f}]")
            p_rep = r["bootstrap_p_report"]
            p_cell = f"$<{p_rep.split('< ')[1]}$" if str(p_rep).startswith("<") else f"{p_rep}"
            holm = "rej" if r["holm_reject_0.05"] else "n.s."
            control = r["control"].replace("loo-marginal", "Shapley-MC")
            lines.append(" & ".join([
                ds_name, f"LOO vs {r['control']}", r["metric"], md_s, ci,
                f"{p_cell} ({holm})", f"{r['cohen_dz_user_conditional_descriptive']:.4f}"]) + r" \\")
    return "\n".join(lines)


def faithfulness_table() -> str:
    lines = []
    for ds_name, run in RUNS.items():
        p = run / "tables" / "faithfulness_curves_all.csv"
        if not p.exists():
            lines.append(f"% {ds_name}: faithfulness curves not ready yet")
            continue
        f = pd.read_csv(p)
        cols = ["DeletionDelta_NDCG@20", "Insertion_NDCG@20"]
        g = f.groupby(["family", "fraction"])[cols].agg(["mean", "std"])
        for fam in ["loo-marginal", "uniform", "random"]:
            for frac in [0.05, 0.10, 0.20, 0.30]:
                r = g.loc[(fam, frac)]
                lines.append(" & ".join([
                    ds_name, fam, f"{frac:.2f}",
                    fmt(r["DeletionDelta_NDCG@20"]["mean"], r["DeletionDelta_NDCG@20"]["std"]),
                    fmt(r["Insertion_NDCG@20"]["mean"], r["Insertion_NDCG@20"]["std"])]) + r" \\")
    return "\n".join(lines)


def gain_checks() -> str:
    lines = []
    for ds_name, run in RUNS.items():
        df = load_summary(run)
        if df is None:
            continue
        g = df.groupby("family")["NDCG@20"].mean()
        loo = g["loo-marginal"]
        for ctrl in ["valid-sim", "valid-linear", "uniform"]:
            pct = (loo - g[ctrl]) / g[ctrl] * 100
            lines.append(f"% {ds_name}: LOO NDCG@20 {loo:.5f} vs {ctrl} {g[ctrl]:.5f} -> +{pct:.2f}%")
        p = run / "tables" / "paired_bootstrap_loo_vs_matched_controls_holm.json"
        if p.exists():
            holm = json.loads(p.read_text())
            n_rej = sum(v["reject_0.05"] for v in holm.values())
            lines.append(f"% {ds_name}: Holm rejected {n_rej}/{len(holm)} LOO-vs-control contrasts")
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = ROOT.parent / "manuscript_assets"
    out_dir.mkdir(exist_ok=True)
    sections = {
        "c1_main_results_rows.tex": main_results_rows(),
        "c1_paired_rows.tex": paired_rows(),
        "c1_faithfulness_rows.tex": faithfulness_table(),
        "c1_gain_checks.txt": gain_checks(),
    }
    for name, content in sections.items():
        (out_dir / name).write_text(content + "\n")
        print(f"===== {name} =====")
        print(content)
        print()
