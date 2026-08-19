#!/usr/bin/env python3
"""Regenerate the lambda-sensitivity figure including the LOO curve (round 6, v20).

Sources:
  - v6 dedicated LOO sweep (uniform, additive-pref, loo-marginal; 5-seed re-execution)
  - v3 released sweep (shapley-mc; 5 seeds)
Plots five-seed mean NDCG@20 vs lambda_attr, one panel per dataset.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path(__file__).resolve().parent.parent / "results" / "journal_runs"
OUTS = [Path(__file__).resolve().parent.parent.parent / "paper_package" / "assets" / "figures",
        Path(__file__).resolve().parent.parent.parent / "springer_latex" / "assets" / "figures"]

STYLE = {"uniform": ("#95a5a6", "o"), "additive-pref": ("#9b59b6", "s"),
         "loo-marginal": ("#e74c3c", "D"), "shapley-mc": ("#e67e22", "^")}
LABEL = {"uniform": "uniform", "additive-pref": "additive-pref",
         "loo-marginal": "LOO (CoalGameRec)", "shapley-mc": "Shapley-MC"}


def load(ds: str) -> pd.DataFrame:
    v6 = pd.read_csv(RUNS / f"{ds}_lightgcn_v6_lambda_sweep" / "tables" / "lambda_sensitivity_all.csv")
    rows = []
    for s in [42, 43, 44, 45, 46]:
        f = RUNS / f"{ds}_lightgcn_v3_prospective" / "raw" / f"seed_{s}" / "lambda_sensitivity.csv"
        d3 = pd.read_csv(f)
        rows.append(d3[d3.family == "shapley-mc"][["family", "lambda_attr", "NDCG@20"]])
    shap = pd.concat(rows)
    v6 = v6[v6.family.isin(["uniform", "additive-pref", "loo-marginal"])][["family", "lambda_attr", "NDCG@20"]]
    return pd.concat([v6, shap])


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, (ds, title) in zip(axes, [("ml1m", "MovieLens-1M"), ("amazon_books", "Amazon-Book")]):
        df = load(ds)
        g = df.groupby(["family", "lambda_attr"])["NDCG@20"].mean()
        for fam in ["uniform", "additive-pref", "shapley-mc", "loo-marginal"]:
            ys = [g.loc[(fam, l)] for l in [0.0, 0.05, 0.1, 0.2, 0.4]]
            c, m = STYLE[fam]
            ax.plot([0.0, 0.05, 0.1, 0.2, 0.4], ys, marker=m, color=c, label=LABEL[fam],
                    linewidth=1.6, markersize=5)
        ax.set_xlabel(r"$\lambda_{\mathrm{attr}}$")
        ax.set_ylabel("NDCG@20")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, linewidth=0.5)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "lambda_sensitivity.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / "lambda_sensitivity.svg", bbox_inches="tight")
    print("WROTE lambda_sensitivity.{png,svg}")


if __name__ == "__main__":
    main()
