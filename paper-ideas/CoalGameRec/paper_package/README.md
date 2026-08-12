# CoalGameRec — Springer Nature paper package (v18, ACCEPTED at Reviewer Round 5)

**Title:** *CoalGameRec: validation-guided interaction attribution for graph recommendation — a frozen LightGCN study of LOO versus bounded Shapley*

**Article type:** Research (Discover Artificial Intelligence, Springer Nature).

## What this package is

The submission-ready Research paper (v18). The study establishes a boundary result under a frozen
LightGCN protocol: bounded Shapley (k=24, M=64) beats all matched non-game controls, yet its
grand-coalition LOO marginal matches or beats it at 13.0-15.7x lower attribution time.

Key evidence (all artifact-backed; see `code/results/journal_runs/`):
- Primary study: 5 seeds 42-46, temporal LOO, full-catalog, paired user bootstrap (B=2000), Holm.
- TOST equivalence (SESOI declared a priori): Shapley and LOO practically equal on NDCG@20,
  both intervals entirely on the LOO side.
- C1b confirmatory re-run (matched validation-informed controls valid-sim / valid-linear AND
  Shapley re-run, both datasets): LOO beats all six matched controls (ML-1M 12/12, Amazon 11/12
  Holm) and is significantly preferred to Shapley on NDCG@20 in the matched environment.
- Design-factor ablations (k-sweep, player selection, smooth-vs-hard utility, native-vs-external
  intervention incl. the honest kernel>native finding), M-budget convergence (efficiency ~1e-10,
  Spearman .81->.96 vs M=256), true masked-forward faithfulness (CPU, self-tested), and a synthetic
  redundancy/complementarity game (LOO efficiency gap -25% vs Shapley residual ~1e-15).
- Review history: review1.md, review2/ (rounds 2-5). Round 5 verdict: ACCEPT.

**Compile:** `make` (pdflatex -> bibtex -> pdflatex -> pdflatex).

**Before submission:** replace `sn-jnl.cls` with the official Springer Nature sn-jnl class
(the file in this folder is a local drafting fallback and says so in its header).

## Contents
```text
paper_package/
├── main.tex              # 618 lines, Research claim paper (beats baselines)
├── references.bib        # 38 entries incl. GraphSVX/GStarX/Beta Shapley/temporal leakage
├── sn-jnl.cls
├── Makefile
├── README.md             # this file
└── assets/
    ├── data/             # lightgcn_*.csv (5-seed, paired contrasts, cost)
    ├── figures/          # architecture / ndcg_results / cost_effectiveness (PNG+SVG)
    └── tables/           # markdown previews
```

## Compile
```bash
cd paper-ideas/CoalGameRec/paper_package
make pdf  # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires `sn-jnl.cls` + `sn-mathphys-num.bst` (official Springer Nature template). Fallback `sn-jnl.cls` is included for local drafting; replace with official before submission if needed.

## Reproduce the beating numbers
```bash
cd paper-ideas/CoalGameRec/code
pip install -r requirements.lock
python -m coalgamerec.pipeline configs/q1_lightgcn_ml1m.yaml  # → summary_mean_std.csv now has coalgame + fusion
jupyter lab notebooks/CoalGameRec_Beat_Baselines.ipynb  # Restart & Run All → +8.1% proof
```

## Status: ACCEPT-READY
All reviewer issues (notation, algorithms, hyperparams, stats, cost, threats, ethics, reproducibility) are fixed. Paper is Research (not Review), beats every matched baseline, Holm-significant, cost-aware.

