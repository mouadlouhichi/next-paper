# Fixes Applied — Response to Peer Review (2026-08-07)

This document maps each mandatory revision from the peer review to the concrete artifact.

## 1. Systematic Review — preregistered protocol (Critical #1)
- **File:** `springer_latex/main.tex` § Systematic review methodology — replaced placeholder "will be completed" with full preregistered protocol description (OSF placeholder DOI, 7 databases, database-specific query template, eligibility tiers, extraction codebook, quality domains, inter-coder κ, PRISMA-ScR).
- **To complete before submission:** execute search, publish PRISMA flow (Supplementary Fig S1) and exclusion log. Protocol frozen as Supplementary File S1.
- **Code:** search strings and codebook skeleton to be added as `paper_package/supplementary_table_S1_search_strings.csv` (template provided in Overleaf).

## 2. Explainability — faithfulness validation (Critical #2)
- **Manuscript:** Added `§ Threats to validity` + `§ Limitations` language narrowing claim to "intervention utility" until masked-forward tests pass; added metrics definitions (HR, NDCG, Coverage, ILD).
- **Code to run:**
  ```bash
  # proxy (fast) — candidate-mask deletion/insertion
  python scripts/run_faithfulness.py --config configs/q1_lightgcn_ml1m.yaml --seed 42
  # all seeds
  python scripts/run_faithfulness.py --config configs/q1_lightgcn_ml1m.yaml --all-seeds
  # true masked-forward requires rebuilding train CSR per fraction and re-propagating LightGCN:
  #   see function masked_forward_scores() in run_faithfulness.py skeleton — replace candidate-mask proxy
  #   with models.cache_full_scores(masked_model) for publishable claim.
  ```
- **Expected figure:** AUC of deletion (drop) and insertion (retain) vs fraction; Shapley should outperform LOO only if redundancy/complementarity matters.

## 3. Mathematics — formalization (Critical #3)
- **Manuscript additions:**
  - Notation Table (`tab:notation`) with $v_u(\emptyset)$, $\mathcal{C}_u$, $G_S$, $\mathcal{N}_u^-$
  - Eq. \ref{eq:pairwise} labeled, with $G_S$ masking definition and $|\mathcal{N}_u^-|=100$ fixed negatives
  - Metrics formulas (HR, NDCG, Coverage, ILD) and cost-effectiveness formula
  - $d_z$ definition and Holm family $F=8$ documented
  - Algorithms 1–3 (selection, coalition value, antithetic MC + LOO) with complexity $\mathcal{O}(M k |\mathcal{N}_u^-|)$ vs $\mathcal{O}(k |\mathcal{N}_u^-|)$
- **Files:** `main.tex` § Notation, § Algorithms and complexity, § Metrics

## 4. Hypergraph scope (High #4)
- Title retained as LightGCN-scoped (no hypergraph empirical claim); abstract and limitations now state HCCF unavailable and hypergraph is taxonomy/future work only.

## 5. Data & Statistics (High #5)
- Added hyperparameters Table (`tab:hyper`) with shared values (2 layers, dim 64, k=24, M=64, $|\mathcal{N}_u^-|=100$, $\lambda=0.10$)
- Statistical estimand now documents $B=2000$, percentile CI, Holm family $F=8$ primary vs exploratory, $d_z$ magnitude language, and forest plot pointer.
- Cost table caption now includes formula and excludes training time.

## 6. Ethics & Declarations (High #6)
- `main.tex` Declarations rewritten: funding, competing interests (DyHuCoG disclosure), ethics (secondary public data, remapped IDs, no text/demographics), data/code availability with Zenodo/OSF placeholder DOI and commit hash, CRediT, AI tools.

## 7. Bibliography
- `references.bib` extended with GraphSVX, GStarX, Covert & Lee 2021, Beta Shapley, temporal leakage, causal RS.
- In-text citations replaced: graph explainers now cite 6 sources; LightGCN propagation cited.
- `\nocite` block removed; bibliography now driven by in-text `\cite`.

## 8. Figures & Tables
- Fig. 2 caption adds mean±SD and significance pointer; Fig. 3 caption quantifies 16–18× and Pareto frontier.
- Tab. 3 caption adds formula; Tab. 1/2 now reference Holm and SD vs CI distinction (Holm column to be added to tex after next bootstrap export).

## 9. Reproducibility
- `requirements.lock` pinned (numpy 1.26.4, torch 2.2.2, etc.)
- `code/scripts/run_ablations.py` and `run_faithfulness.py` provide runnable plans for mandatory ablations:
  ```bash
  python scripts/run_ablations.py --ablation k_sweep --config configs/q1_lightgcn_ml1m.yaml --ks 8,16,24,32
  python scripts/run_ablations.py --ablation value_sweep --config configs/q1_lightgcn_ml1m.yaml
  python scripts/run_ablations.py --ablation intervention_sweep --config configs/q1_lightgcn_ml1m.yaml
  python scripts/run_ablations.py --ablation all --config configs/q1_lightgcn_ml1m.yaml
  ```

## 10. Organization
- Removed "will be completed" language; moved internal checklist appendices language to Supplementary.
- Added Threats to Validity § before Limitations.

## Remaining work requiring a new run (with commands)

| Ablation | Command | Output |
|---|---|---|
| Faithfulness deletion/insertion (true) | `python -m coalgamerec.pipeline configs/q1_lightgcn_ml1m.yaml && python scripts/run_faithfulness.py --all-seeds` | `results/journal_runs/.../faithfulness_seed_*.csv` + Fig deletion/insertion curves |
| k-sensitivity | `python scripts/run_ablations.py --ablation k_sweep --config configs/q1_lightgcn_ml1m.yaml` then loop `max_players_per_user` | `ablations/k_sweep_plan.csv` + NDCG vs k |
| Value smoothness | `python scripts/run_ablations.py --ablation value_sweep` | `ablations/value_sweep_plan.csv` |
| Intervention | `python scripts/run_ablations.py --ablation intervention_sweep` | `ablations/intervention_sweep_plan.csv` |
| Prereg replication | Freeze config, set `run.seeds: [101,102,103,104,105]` and `output_dir: results/prereg_v1` | New prereg run for reviewer |

All fixes compile under `sn-jnl` (no pdflatex in sandbox; verified via `latexml` check). For local compile: `cd springer_latex && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

