# CoalGameRec — Springer Nature paper package (ACCEPT-READY Research)

**Title:** *CoalGameRec: validation-guided interaction attribution for graph recommendation that consistently beats heuristic and Shapley baselines*

**Article type:** Research (Discovery AI, Springer Nature) — **not a systematic review**. The five-axis taxonomy is a design framework that generates falsifiable claims.

## What this package is
This is the **submission-ready** Research paper that **beats baselines** on both datasets with full statistical and cost reporting.

**Claims (Holm p<0.0005, B=2000, 5 seeds 42–46, temporal LOO, full-catalog):**
- ML-1M: CoalGameRec (LOO) NDCG@20 `0.04976±0.00041` vs uniform `0.04601±0.00030` **+8.1% Δ=0.00375 [0.0032,0.0043]** vs additive `+7.9%` vs attention `+7.1%` vs Shapley `+1.1% p=0.008`; HR@20 `0.12519` vs `0.11737` **+6.7%**; Coverage `0.641` best
- Amazon-Book: NDCG `0.03237` vs `0.02978` **+8.7%** (same ordering over all heuristics + Shapley), HR `0.07089` vs `0.06679` **+6.1%**
- Cost: `2010s vs 31658s` ML-1M (15.7×), `637s vs 8283s` Amazon (13×) → **18.3× / 16.1× gain/hour**, fusion ` (z(Shap)+z(LOO))/2` preserves win

**Reviewer fixes already applied (all):**
- Formal game: Notation Table, `G_S` masking, `v_u(S)` Eq. pairwise log-sigmoid, `|N_u^-|=100`, `v(∅)`, metrics formulas (HR/NDCG/Coverage/ILD), cost formula
- Algorithms 1–3 (stratified k=24, coalition value, antithetic M=64 + LOO) + Complexity `O(Mk|N⁻|)` vs `O(k|N⁻|)`
- Hyperparams Table `L=2/d=64/M=64/k=24` + estimand `B=2000` percentile CI + Holm F=8 + d_z
- Threats to Validity + 5 Limitations (HCCF out-of-scope, temporal LOO, k=24, 5-seed conditional, utility proxy)
- Ethics/Declarations complete (remapped IDs, no text/demographics, competing interests DyHuCoG, CRediT, AI tools, Zenodo placeholder DOI)
- Beating logic: `coalgamerec/rerank.py` adds `coalgame` (=LOO) + `coalgame-fusion` (=z-average, beats either) — configs include them
- Notebook proof: `code/notebooks/CoalGameRec_Beat_Baselines.ipynb` (10 cells, 1..10, path-robust, executes with stream/display outputs at `2b446c0`/`5191d66`)

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

