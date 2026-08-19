#!/usr/bin/env python3
"""Aggregate review-5 full-catalogue raw runs into user-level statistics.

Reads results/raw/amazon_full_catalogue_1000_seed{42..46}.json, averages each
metric within user across seeds, then aggregates across distinct users with
user-bootstrap 95% CIs and paired differences (Holm-uncorrected here; the
confirmatory family is declared in the manuscript). Writes:

* results/review5/full_catalogue_1000_summary.json
* paper-ideas/ActionShap/actionshap-overleaf/tables/review5_full_catalogue.tex
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "review5"
TABLES = ROOT.parent / "actionshap-overleaf" / "tables"

METHODS = [
    ("shapley_mc", "Monte Carlo Shapley"),
    ("lime", "LIME"),
    ("loo", "Leave-one-out"),
    ("greedy_cf", "Greedy sequential deletion"),
    ("random", "Random control"),
]
SEEDS = [42, 43, 44, 45, 46]


def load_all():
    runs = {}
    for s in SEEDS:
        p = RAW / f"amazon_full_catalogue_1000_seed{s}.json"
        runs[s] = json.load(open(p))
    return runs


def user_metric(runs, method, field):
    """seed-mean within user, then dict user -> value (None if all missing)."""
    per_user = {}
    users = runs[SEEDS[0]]["users"]
    for u in users:
        uid = u["user"]
        vals = []
        for s in SEEDS:
            rec = next(x for x in runs[s]["users"] if x["user"] == uid)
            v = rec["methods"][method].get(field)
            if v is not None and v == v:
                vals.append(v)
        per_user[uid] = float(np.mean(vals)) if vals else None
    return per_user


def boot_ci(values, n_boot=10000, seed=7):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    runs = load_all()
    users0 = runs[SEEDS[0]]["users"]
    n_users = len(users0)

    # active-oracle denominators (oracle positive on target margin / NDCG)
    active_tm, active_ndcg = [], []
    for u in users0:
        o = u["oracle"]
        eff_tm = o.get("primary_utility", {}).get("effect") if isinstance(o.get("primary_utility"), dict) else None
        if eff_tm is None and isinstance(o.get("primary_utility"), dict):
            eff_tm = o["primary_utility"].get("oracle_effect")
        eff_nd = o.get("ndcg", {}).get("effect") if isinstance(o.get("ndcg"), dict) else None
        if eff_nd is None and isinstance(o.get("ndcg"), dict):
            eff_nd = o["ndcg"].get("oracle_effect")
        active_tm.append(eff_tm is not None and eff_tm > 0)
        active_ndcg.append(eff_nd is not None and eff_nd > 0)

    summary = {
        "dataset": "Amazon-Digital-Music",
        "evaluation_mode": "full_unseen_catalogue",
        "n_users": n_users,
        "n_seeds": len(SEEDS),
        "active_oracle_target_margin": int(sum(active_tm)),
        "active_oracle_ndcg": int(sum(active_ndcg)),
        "methods": {},
    }

    fields = {
        "bounded_aia": "aia",
        "deletion_aia": "faithfulness_alignment",
        "gap": "actionability_gap",
        "signed_alignment": "signed_alignment",
        "effect_target_margin": "effect_target_margin",
        "normalized_regret": "normalized_regret_primary",
    }
    per_user_cache = {}
    for mk, label in METHODS:
        entry = {"label": label, "metrics": {}}
        for name, field in fields.items():
            pu = user_metric(runs, mk, field)
            per_user_cache[(mk, name)] = pu
            vals = np.array([v for v in pu.values() if v is not None], float)
            if field == "normalized_regret_primary":
                vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                entry["metrics"][name] = None
                continue
            lo, hi = boot_ci(vals)
            entry["metrics"][name] = {
                "mean": float(vals.mean()),
                "ci95_low": lo,
                "ci95_high": hi,
                "n": int(len(vals)),
            }
        summary["methods"][mk] = entry

    # paired differences on bounded AIA and effect (seed-mean within user)
    pairs = [("shapley_mc", "lime"), ("shapley_mc", "loo"), ("lime", "loo")]
    summary["paired"] = []
    for a, b in pairs:
        for name in ["bounded_aia", "effect_target_margin"]:
            pa, pb = per_user_cache[(a, name)], per_user_cache[(b, name)]
            common = [u for u in pa if pa[u] is not None and pb[u] is not None]
            diff = np.array([pa[u] - pb[u] for u in common], float)
            lo, hi = boot_ci(diff)
            summary["paired"].append({
                "left": a, "right": b, "metric": name, "n": len(common),
                "mean": float(diff.mean()), "ci95_low": lo, "ci95_high": hi,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "full_catalogue_1000_summary.json").write_text(
        json.dumps(summary, indent=1, allow_nan=False))
    print("wrote", OUT / "full_catalogue_1000_summary.json")

    # ---------------- LaTeX table ----------------
    def f3(x):
        return "--" if x is None else f"{x:.3f}"

    def cell(e, w=2):
        if e is None:
            return "--"
        return f"{e['mean']:.{w}f} [{e['ci95_low']:.{w}f},{e['ci95_high']:.{w}f}]"

    BS = "\\\\"
    L = []
    L.append("% Review-5 full-catalogue aggregate (generated by scripts/make_review5_stats.py).")
    L.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{2.5pt}")
    L.append(r"""\caption{Amazon Digital Music full-unseen-catalogue audit at 1,000 users
(five seeds, seed-mean within user; median catalogue size 8,572 unseen items).
Bounded and deletion AIA, their difference, signed alignment, realized
target-margin effect of the selected action, and conditional normalized
regret; intervals are user-bootstrap 95\% CIs. Active-oracle denominators:
$""" + str(summary["active_oracle_target_margin"]) + r"""$ (target margin) and $""" +
             str(summary["active_oracle_ndcg"]) + r"""$ (NDCG).}""")
    L.append(r"\label{tab:review5-fullcat}")
    L.append(r"\begin{tabular}{@{}lrrrrrr@{}}")
    L.append(r"\toprule")
    L.append("Method & Bounded AIA & Deletion AIA & Gap & Signed & $\\Delta$ effect & Norm.\\ regret " + BS)
    L.append(r"\midrule")
    for mk, label in METHODS:
        m = summary["methods"][mk]["metrics"]
        cells = []
        for name in ["bounded_aia", "deletion_aia", "gap", "signed_alignment",
                     "effect_target_margin", "normalized_regret"]:
            cells.append(cell(m[name], w=4 if name == "effect_target_margin" else 2))
        L.append(label + " & " + " & ".join(cells) + " " + BS)
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append(r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{2.5pt}")
    L.append(r"\caption{Paired seed-mean differences on the same 1,000 full-catalogue users (user-bootstrap 95\% CIs).}")
    L.append(r"\label{tab:review5-fullcat-paired}")
    L.append(r"\begin{tabular}{@{}lllrrr@{}}")
    L.append(r"\toprule")
    L.append("Left & Right & Metric & $n$ & Diff. & 95\\% CI " + BS)
    L.append(r"\midrule")
    for p in summary["paired"]:
        metric = "bounded AIA" if p["metric"] == "bounded_aia" else "$\\Delta$ effect"
        L.append(f"{summary['methods'][p['left']]['label']} & {summary['methods'][p['right']]['label']} & "
                 f"{metric} & {p['n']} & {p['mean']:+.3f} & [{p['ci95_low']:.3f},{p['ci95_high']:.3f}] {BS}")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "review5_full_catalogue.tex").write_text("\n".join(L) + "\n")
    print("wrote", TABLES / "review5_full_catalogue.tex")


if __name__ == "__main__":
    main()
