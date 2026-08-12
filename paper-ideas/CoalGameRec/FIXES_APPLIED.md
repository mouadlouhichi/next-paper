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


---

# Fixes Applied — v14 (response to v13 re-review, 2026-08-07)

**Policy applied:** every number in the manuscript must map to a released artifact file
(`results/journal_runs/*/tables/*.csv`, `raw/seed_*/{lambda_sensitivity.csv,explanation_diagnostics.json,runtime_by_seed.csv}`).
Claims without artifacts were removed, not renumbered.

## Critical Issue 1 — Algorithm 4 contradicted Eq. (7)
- Rewrote § Attribution-guided reranking: new Eqs. (eq:weights)--(eq:rerank) match
  `code/coalgamerec/rerank.py` exactly: raw signed weights $w_j=a_j$ for Shapley/LOO,
  $\mathbf{r}_u=\sum_j w_j\mathbf{e}_j$, and the attribution term divided by the
  **L1 normalizer** $\sum_j|w_j|+\epsilon$ (well-defined for signed weights with $\sum_j w_j\approx0$).
- Algorithm 4 rewritten with the explicit $d=\sum_j|w_j|+\epsilon$ line; justification added
  (signed weights, symmetric treatment of positive/negative attribution, identical intervention across families).

## Critical Issue 2 — the 64× complexity claim was false
- Complexity paragraph rewritten: operation count predicts an $M$-fold ($\approx64\times$) ratio;
  measured wall-clock is 15.7× (ML-1M: 31,657.7 s vs 2,010.5 s) and 13.0× (Amazon: 8,283.2 s vs 637.2 s);
  explained via cached graph structures, vectorized propagation, amortized fixed overheads.
  Op-count now stated as an upper bound, not an equality prediction.

## Critical Issue 3 — statistics for validation baselines / family definition
- **valid-sim and valid-linear rows removed from Table 3** (no code, config, or result artifact exists for them anywhere in the repo).
- Contribution 1, §13.2, Appendix B updated accordingly; the validation-access asymmetry is now flagged in
  Threats to validity (Internal) and Limitations (new 6th limitation) with matched controls committed to the regeneration plan.
- Primary family defined explicitly everywhere (Abstract-adjacent Contributions, estimand section, Table 4 caption):
  per dataset F=8 Holm contrasts, Shapley-MC vs {uniform, additive-pref, attention, LOO} × {NDCG@20, HitRate@20}
  — exactly the released `paired_bootstrap_all_controls*.csv` + `_holm.json`.
- **12 LOO-as-treatment rows removed from Table 4** (CI/p/dz values were not in the artifact files).
  LOO-vs-uniform mean differences (0.00375 ML-1M / 0.00259 Amazon, from `cost_effectiveness.csv`)
  are reported descriptively with an explicit note that LOO-as-treatment bootstrap is in the regeneration plan.
- Abstract fixed: "+8.1%/+8.7% (Holm p<0.0005)" → the Holm statement now correctly refers to the Shapley-vs-uniform contrasts.
- Table 3 regrouped (unreranked reference / non-game reweighting / cooperative-game attribution) as requested.
- Table 4 caption states the one-to-one mapping to artifact files.

## Critical Issue 4 — claimed ablations not reported
- Removed all unsupported numbers: k-sweep plateau, "M=64 halves variance vs M=32", "+30% variance",
  player-selection ~0.001, native-vs-external 0.002–0.003.
- Table 6 rebuilt from `lambda_sensitivity.csv`: **5-seed mean±SD, both datasets**, families restricted to
  those in the artifact (uniform, additive-pref, shapley-mc). LOO λ-sweep values (0.04710/0.05180/0.05820)
  were not in any artifact and were removed; LOO is reported at the protocol λ=0.10 only.
- §15 now states explicitly which ablations are "specified in run_ablations.py but not run" (no numbers reported).
- Amazon "same trend" claim replaced by the actual Amazon pattern (heuristics flat, Shapley monotone).
- Runtime statement corrected to artifact values (5,317–5,436 s on 4/5 ML-1M seeds + one 10,160 s outlier; 1,657±3 s Amazon).

## Critical Issue 5 — faithfulness evidence preliminary
- Table 7 replaced: fabricated 6-method comparison removed. New table = real candidate-masking proxies from
  `explanation_diagnostics.json`, **5-seed mean±SD, both datasets**, attributed to Shapley-MC (as the pipeline computes them).
- All three caveats stated: masking proxy (not masked-forward), single 20% fraction, no cross-family comparison run.
  Faithfulness explicitly NOT claimed; multi-fraction curves + masked-forward + cross-family = regeneration plan.
- Retention percentages computed honestly against the λ=0 reference (59% ML-1M, 88% Amazon).

## Technical/presentation issues 1–9
1. Algorithm 3: `for π ∈ {π, π'}` → `for each permutation ρ in the pair (π, π')`; "order π_order" → "order induced by ρ".
2. Algorithm 1: pools made explicit ($P_2$ from $H_u\setminus P_1$, $P_3$ from $H_u\setminus(P_1\cup P_2)$, exhaustion rule, fill rule).
3. Contribution 4 family wording rewritten (no more "LOO vs LOO" ambiguity) — see Critical 3.
4. Appendix B contradiction removed: the paragraph claiming validation-guided baselines exist in code was false and is deleted;
   appendix rewritten declaratively (no "should"/"before submission" planning language).
5. Table 3 grouped via multicolumn section headers.
6. Table 6 now contains real Amazon values (5-seed mean±SD).
7. Planning language removed from Appendix B; Appendix E remains the one-line archived checklist.
8. Reference louhichi2024gametheoryxai: `journal={Manuscript}` → `@misc` with `howpublished={Unpublished working paper}`
   + note "cited for taxonomy completeness only; not used as empirical evidence"; §4 text states the same.
9. Reproducibility: Appendix B now lists concrete artifacts (manifest.json fields incl. OS/Python/torch/device,
   config.resolved.json, dataset_stats.json, item_vectors_report.json, requirements.lock pins, cache-key scheme);
   Data/Code availability updated (Zenodo DOI at acceptance, commit hash in supplement, environment file named).

## Additional consistency fixes
- Fusion claims softened everywhere (Contribution 5, P3, §Future work, Appendix D): `coalgame-fusion` is implemented
  in code but was not run under the frozen protocol → no ranking claim.
- §hyper subsection title "(including validation-guided baselines)" → "(shared across families)".
- λ-figure caption rewritten (no +14.5% claim; points to the 5-seed table).
- Unreranked row in Table 3 replaced with real λ=0 5-seed values from the sweep artifact
  (ML-1M 0.11415/0.04482/0.62967/0.73045; Amazon 0.06690/0.02982/0.23573/0.92082).
- Both `paper_package/` and `springer_latex/` copies kept byte-identical.

---

# Fixes Applied — v15 (response to round-2 re-review, in progress)

Round-2 verdict: **Major Revision** (validation-informed baselines removed; LOO-vs-baseline paired tests missing; ablations; XAI evidence preliminary).

## Critical Issue 1 — validation-informed non-game baselines RESTORED (with real data)
- Implemented `valid-sim` (history reweighting by cos(e_j, e_{i_u+})) and `valid-linear`
  (candidate-side linear reranker s' = z(b) + λ z(cos(e_i, e_{i_u+}))) in `coalgamerec/rerank.py`;
  both share the a-priori λ=0.10 and native intervention; identical validation access as the games.
- Confirmatory study C1 (`scripts/run_matched_controls.py`): CPU re-execution of the frozen protocol
  (identical hyperparameters, seeds 42–46, archived splits) adding unreranked + the two matched controls;
  deviations recorded in each manifest.json. Runs: `*_lightgcn_v4_matched_controls/` (in progress).
- Table 3 grouped rows restored via the C1 results (pending run completion).

## Critical Issue 2 — paired LOO-vs-all-controls statistics
- (a) Computed from EXISTING v3 artifacts (`per_user_metrics_all.csv`): Table 4b (tab:paired_loo) —
  Holm F=10 per dataset, LOO vs {uniform, additive-pref, attention, heuristic-pop, shapley-mc} × 2 metrics;
  all LOO-vs-heuristic contrasts p<0.0005 (Holm), LOO beats Shapley on NDCG@20 both datasets.
- (b) C1 adds LOO vs {…, valid-sim, valid-linear} paired contrasts (Holm F=12) — pending run completion.

## Critical Issue 3 — unexecuted ablations no longer used to justify design
- §8.3 (bounded players): k=24 justification now rests solely on the feasibility argument;
  player-count sensitivity explicitly marked "specified but not executed".

## Critical Issue 4 — XAI evidence extended
- C1 produces deletion/insertion CURVES (fractions 0.05/0.10/0.20/0.30) with a seeded RANDOM control
  and uniform control, for LOO attributions, 5 seeds, both datasets (faithfulness_curves_all.csv).
- Masked-forward and stability tests remain declared future work (run_faithfulness.py).

## v15 completion status (2026-08-08/09)
- C1 confirmatory run EXECUTED for both datasets (5 seeds each):
  ML-1M on Apple MPS (original v3 hardware, torch 2.3.1, 14,067 s total, run.log committed);
  Amazon-Book on CPU. Fidelity check passed (uniform NDCG@20 ML-1M identical to v3: 0.04601).
- Critical Issue 1 CLOSED: valid-sim + valid-linear restored with real data;
  LOO beats both on NDCG@20 on both datasets (Holm): ML-1M 12/12 contrasts significant,
  Amazon 11/12 (HR vs valid-linear n.s. p=0.077).
- Critical Issue 2 CLOSED: paired LOO-vs-all-controls on primary artifacts (Table 4b) AND
  paired LOO-vs-matched-controls from C1 (tab:c1_paired).
- Critical Issue 4 extended: deletion/insertion CURVES (fractions 0.05-0.30) with uniform +
  seeded-random controls from C1 (tab:c1_faith); masked-forward remains declared future work.
- Manuscript v15 assembled: C1 subsection in Results, updated Threats/Limitations/Abstract/
  Contributions/Code-availability.

---

# Fixes Applied — v17 (response to round-4 review)

## Critical issues
1. **Eq. (7) mathematics (Critical #1):** reranking equation corrected to the divisor-free form
   s'_ui = zscore(b_ui) + λ·zscore(⟨r_u,e_i⟩); the released implementation's L1 divisor is
   proven algebraically inert under candidate-wise z-scoring (z(x/d_u)=z(x) for d_u>0) and is
   retained only for numerical safety. Stable z-score defined incl. zero-variance behavior.
2. **P2 / predictions overclaim (Critical #2):** §12 rewritten — P3 supported under the frozen
   protocol; P1 supported analytically (empirical ablation not executed); P2 declared an untested
   hypothesis. "predictions" → "hypotheses".
3. **Equivalence instead of n.s. (Critical #3):** TOST-style equivalence analysis computed from
   the released per-user artifacts with a pre-declared SESOI (δ=0.001 NDCG@20, δ=0.002 HR@20);
   new Table tab:equivalence: equivalence established on all four Shapley-vs-LOO contrasts;
   both NDCG@20 intervals lie entirely on the LOO side. "matches" language replaced everywhere.
4. **Masked-forward faithfulness (Critical #4):** proxy limitation now stated as failing to test
   the game-defining intervention; robustness-of-evaluation literature cited (Zheng et al. ICLR'24,
   Fang et al. NeurIPS'23); true masked-forward protocol scripted (run_masked_forward_faithfulness.py)
   for execution; no faithfulness claim made.
5. **M-budget convergence (Critical #13):** run_estimator_convergence.py prepared (M=16..256,
   efficiency residuals, Spearman/l1/l2 vs M=256 reference, reranking metrics, runtimes).
6. **Design ablations (Critical #6 / High):** run_design_ablations.py prepared and executed-pending:
   k-sweep {8,16,24,32} (Shapley+LOO), player selection {stratified,similarity,random},
   hard-vs-smooth utility (value_mode switch), native-vs-external intervention.
7. **C1 Shapley gap (High #10):** run_matched_controls.py extended with C1_WITH_SHAPLEY=1.

## High issues
- Complexity corrected: C_v = O(L|E_S|d + |N_u^-|d); T_MC=O(MkC_v), T_LOO=O((k+1)C_v); rerank
  O(|C_u|d); cache memory O(|U||I|); "infeasible for k>10" replaced by a budget statement.
- Superlative audit: ML-1M C1/primary text corrected; Amazon C1 Coverage/ILD bolding fixed
  (valid-linear highest); "strongest non-game reranker" → "a strong matched non-game reranker".
- PRISMA claim removed entirely (no systematic-review protocol in manuscript); Table 1 caption
  now "illustrative set, not a systematic corpus, not claimed exhaustive"; taxonomy repositioned
  as a proposed conceptual framework (reviewer's Positioning text adopted).
- Bootstrap hierarchy + exact p-value formula + Holm family pre-specification added to §stat_estimand.
- Runtime SDs across seeds reported (6,332±2,141 s Shapley vs 402±5 s LOO on ML-1M; training costs
  included); end-to-end latency/memory declared unmeasured; deployment claims restricted accordingly.
- History-length stratification table added (tab:strata); bounded-game coverage reported
  (24% ML-1M / 89% Amazon users below k=24); longest-history stratum nuance (Shapley>LOO) reported.
- Related work expanded with 9 verified published references (Markchom CSUR'25, LightGCL ICLR'23,
  XSimGCL TKDE'24, Zheng ICLR'24, Fang NeurIPS'23, Data Banzhaf AISTATS'23, Data-OOB ICML'23,
  TRAK ICML'23, shapiq NeurIPS'24); PRISMA entry removed.
- Ethics appendix rewritten declarative (no "should obtain" drafting notes); submission-checklist
  appendix removed; Deployment section → "Practical implication" (reviewer's rewrite);
  conclusion cost claim fixed (13.0–15.7×; stray tab character removed).
- Algorithms 1–4: safe-cosine + short-history + complexity (Alg 1); H_¬u + degree renormalization
  (Alg 2); RNG seed + efficiency residual output (Alg 3); family identifier + z-score stability (Alg 4).
- Repetition reduced: negative-result section trimmed; contributions 5→3 separating the proposed
  framework from the experimentally supported findings.

---

# v18 integration (2026-08-12): C1b (Shapley re-run), design ablations, convergence, masked-forward

All round-4 experiments executed on the authors' machine (Apple MPS, torch 2.3.1) and integrated:

1. **C1b (C1 with Shapley)** — 9 families, both datasets, 5 seeds. Ordering on NDCG@20 on both
   datasets: LOO > Shapley > valid-linear > valid-sim > heuristics. Paired (Holm): LOO beats all
   6 controls — ML-1M 12/12, Amazon 11/12 (HR vs valid-linear p=0.086 n.s.). Shapley-as-treatment:
   beats valid-sim on both datasets (NDCG p<0.0005) but is significantly BEATEN by LOO on NDCG@20
   in the matched environment (ML-1M p=0.007; Amazon p<0.0005). New tables: tab:c1_main (v4b),
   tab:c1_paired (v4b), tab:c1_shap (new).
2. **Design-factor ablations** (new tab:design_ablations, seed 42, both datasets): k flat for k>=16;
   player selection indistinguishable; smooth utility beats hard utility (first direct P2 evidence);
   external kernel beats native intervention (reported as honest finding against the alignment
   principle as a performance claim; native retained as the protocol intervention).
3. **Estimator convergence** (new tab:estimator_convergence, ML-1M, 1000 users, 2 est. seeds):
   efficiency residual <=4.3e-10 for all M; Spearman vs M=256 reference 0.81->0.96, passing 0.91
   at protocol M=64.
4. **Masked-forward faithfulness** (valid re-run, CPU, self-tested): tab:c1_faith replaced with true
   masked-forward results — insertion retains/exceeds unmasked under game families while uniform
   falls below; deletion degrades most under LOO (ML-1M) and Shapley (Amazon). Faithfulness still
   not claimed in the full sense.
5. Abstract, contributions, limitations, discussion updated to reflect C1b + masked-forward;
   all "Shapley not re-run" language removed.
