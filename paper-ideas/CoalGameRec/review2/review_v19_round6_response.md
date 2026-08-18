# Round 6 — Implementer response to the three Discovery-AI reviews (v19)

Date: 2026-08-18. Manuscript: `paper_package/main.tex` (mirrored in `springer_latex/`).
Three reviewer reports received; all three recommend **Major Revision**. This document maps every
critical/high/medium item to a concrete action. Items marked **DONE** are already in the
manuscript + released artifacts; items marked **RUN PENDING** have released scripts and await
execution on the authors' Mac (MPS) — results will be appended on return.

## A. Items executed directly in this revision (manuscript v19)

### A1. Critical — Table/cross-reference integrity
- Table 11 mismatch: the condensed v18/v19 layout already separates the tables; this revision
  re-verifies every label: `tab:c1_paired` (LOO contrasts), `tab:c1_shap` (Shapley contrasts),
  `tab:c1_faith` (masked-forward deletion/insertion) — captions and bodies now all match, and
  both metric families (NDCG@20 and HitRate@20) are included in the Shapley table.
- No `??` references remain (verified by grep); Figures 2 (family bars) and the lambda-sensitivity
  figure were re-inserted after the condensation had dropped them.
- All cited keys resolve: 6 previously-undefined citations added to `references.bib`
  (cohen1988power, ghazimatin2020prince, jethani2022fastshap, koh2017influence,
  tran2021counterfactual, xian2019reinforcement).

### A2. Critical — Statistical procedure corrected (all tables recomputed)
New script `code/scripts/analyze_round6_stats.py` recomputes ALL paired inference from the
released per-user artifacts (v3 primary + v4b C1b), deterministic seed 20260818:
1. **Joint user resampling across seeds**: unit of analysis is the seed-mean paired difference
   d_u = (1/5)Σ_s (m^A_{u,s} − m^B_{u,s}); the same user IDs are resampled, preserving
   cross-seed dependence (R1 stat item 2).
2. **Paired sign-flip permutation p-values with +1 correction**, B=10,000 — replaces the
   zero-count "<1/B" bootstrap sign-count p-values (R1 stat item 3). p ≥ 1/10001 always.
3. **Bootstrap 95% and 90% CIs** and **bootstrap CIs for d_z** on every reported contrast
   (R1 Medium: effect-size CIs).
4. **Wilcoxon signed-rank sensitivity** (Holm-adjusted) on every contrast (R3 requested).
5. **Legacy within-seed bootstrap** re-run at B=10,000 for the primary family, reported as a
   sensitivity column (Table `tab:robustness`) to show procedure robustness (R1 stat item 8).
6. **Friedman + Nemenyi–Holm omnibus** over all nine C1b families with users as blocks
   (Table `tab:friedman`; R2/R3 requested). Omnibus rejects (p<1e-30); all Nemenyi pairwise
   n.s. — reported transparently; paired contrasts remain primary.
7. **Minimum detectable effect + TOST power**: user-level MDE d_z = 0.036 (ML-1M) / 0.033
   (Amazon); seed-level MDE at 5 seeds = 1.25 (why seed-level tests are descriptive);
   TOST power at δ=0.001: Amazon 0.85, ML-1M only 0.34 (R1 stat items 6/9).
8. **Top-20 crossing rates** (practical significance): LOO vs uniform changes top-20 membership
   for 3.0% (ML-1M) / 1.4% (Amazon) of users; LOO vs Shapley 1.4% / 0.4% (R3 Phase-15 #6).
9. **Runtime medians + IQRs** from per-seed runtime.json added to the cost section (R1 Medium).
10. Holm families re-declared explicitly with their distinct questions (R1 stat item 8).

**Scientific consequence reported honestly:** under the corrected joint inference, ML-1M
Shapley-vs-LOO (NDCG@20) is non-significant in the permutation test (p=0.207 Holm) while
Wilcoxon rejects; Amazon favors LOO under all procedures. The ML-1M NDCG@20 equivalence CI
exceeds the −0.001 margin by 6.8e-5 (primary) / 1.4e-4 (C1b) → formal equivalence is now stated
as *established on Amazon-Book, marginal and underpowered on ML-1M*. The overall conclusion is
unchanged: no practically meaningful Shapley advantage; direction favors LOO; LOO is cheaper.

### A3. High — Mathematical/algorithmic corrections
- **Conditional coalition game formalized**: B_u = H_u \ P_u, value written v_u(S | B_u)
  (new Eq. for B_u; Principle 2 and Algorithm 2 updated; R1 issue #7).
- **P1/additivity reframed**: Principle 3 now states linearity proves *decomposition only*,
  never a performance ordering; matched controls isolate leakage empirically (R1 issue #4).
- **Efficiency residual de-promoted**: estimator-convergence text now states explicitly that
  complete permutation paths telescope for any M, so the residual is an implementation sanity
  check, NOT convergence evidence; convergence evidence = Spearman ρ vs M=256 reference +
  stability across two independent estimator seeds (R1 issue #5).
- **Algorithm 1**: effective budget m=min(k,|H_u|) drives q,r (short histories scale
  automatically); train-only vectors renamed x_j (distinct from native embeddings e);
  complexity O(k|H_u|d) with maintained minimum distances (R1 issue #10, R2 Alg review).
- **Algorithm 2**: P_u and B_u listed as explicit inputs; "frozen parameters, recomputed
  degrees" stated (R2 Alg review).
- **Algorithm 4**: z-scoring rewritten as vectorized across all candidates (z^b, z^a), no
  scalar z-score inside the loop (R1 Alg 4).
- **Principle 4** recast as a construct/model-alignment design recommendation, not a
  performance hypothesis, since the external kernel ablation outperforms native (R1 issue #12).
- Complexity paragraph already reports C_v with cache-reuse note; MC/LOO ratio described as
  nominal M-fold with measured ratios in the cost table.

### A4. High — λ-dependence and fairness
- **Oracle best-λ table added** (`tab:lambda_oracle`, from released sweep artifacts), explicitly
  labelled test-oracle upper bound — shows Shapley gains strongly at λ=0.4 on ML-1M while
  controls stay flat; states that the headline is a shared-λ=0.10 protocol statement (R1 #6, R2).
- **LOO λ-sweep**: released as `scripts/run_loo_lambda_sweep.py` — also implements proper
  validation-tuned λ selection (λ chosen by validation NDCG@20, test reported once) for
  uniform/additive-pref/LOO/Shapley. RUN PENDING.
- λ-dependence now mentioned in the abstract-level limitations paragraph.

### A5. Medium/Low editorial items
- Figure 2 regenerated: **HitRate@20** label (Recall@20 noted as numerically equal with one
  test item), ±1 SD bars, **distinct hatch patterns** for grayscale/colorblind readability
  (`scripts/plot_round6_figure2.py`; PNG+SVG in both packages).
- Cost-figure caption: Pareto claim explicitly restricted to the two plotted axes at shared
  λ=0.10.
- Keywords: "cost-effectiveness" replaced by "computational efficiency"; "graph explainability"
  and "graph neural networks" added.
- Naming: one sentence declares `loo-marginal` ≡ "CoalGameRec (LOO)" (same algorithm).
- Reproducibility: hardware/software stated (macOS-26.6-arm64, Apple Silicon MPS, Python
  3.12.13, torch 2.3.1; per-run manifest.json records device and deviations).
- Ethics: new paragraph on recommender-intervention risks (popularity amplification, long-tail
  exposure, preference-reinforcement, user autonomy) (R1 Phase 14).
- Related work expanded with 8 new references (below); Table-1/"first in literature" claims
  already absent from the condensed version.
- 2026 dates: correct — protocol frozen 2026-07-15, manuscript submitted August 2026 (current
  date 2026-08-18). No change needed.

### A6. New references added (all metadata verified against publisher pages)
amara2022graphframex (LoG/PMLR 198:44:1–23), li2024fragile (ICML 2024, PMLR 235:28551–28567),
tai2025redundancy (ICML 2025, PMLR 267:58169–58188), ho2024shapleyembedding (KBS 300:112244),
pereira2023distilnexplain (AISTATS 2023, PMLR 206:6199–6214), agarwal2022probing (AISTATS 2022,
PMLR 151:8969–8996), li2024shapleyreview (Auton. Intell. Syst. 4:2), wu2021sgl (SIGIR 2021),
plus the 6 previously-missing cited keys. Existing keys muschalik2024shapiq, markchom2025review,
cai2023lightgcl, yu2024xsimgcl, schnake2022higher, myerson1977graphs now actually cited.

## B. Released scripts — runs pending on the authors' Mac (results to be appended)

| Review demand | Script | Status |
|---|---|---|
| Second backbone (structurally different) | `scripts/run_second_backbone.py --backbone ngcf` (NGCF-style nonlinear aggregation, identical protocol; new `NGCF` class in `coalgamerec/models.py`) | RUN PENDING |
| LOO λ-sweep + validation-tuned λ | `scripts/run_loo_lambda_sweep.py` | RUN PENDING |
| Multi-seed design ablations | `scripts/run_design_ablations.py --seed {42..46}` (existing) | RUN PENDING |
| Multi-seed masked-forward faithfulness | `scripts/run_masked_forward_faithfulness.py --seed {42..46}` (existing) | RUN PENDING |
| Validation-negative sensitivity (50/100/500) | `scripts/run_negset_sensitivity.py` | RUN PENDING |
| Attribution stability + model randomization + perturbation | `scripts/run_randomization_sanity.py` | RUN PENDING |
| Corrected inference (this revision) | `scripts/analyze_round6_stats.py` | EXECUTED (artifacts in `results/journal_runs/round6_analysis/`) |

## C. Items deferred with explicit justification in the manuscript
- Human-subject explanation study, third dataset/domain, influence-function/counterfactual
  external baselines, peak-memory/serving-latency profiling: listed in Limitations as required
  extensions beyond this revision's scope.
- Seed-population hierarchical modeling: estimand stated as conditional on the five fitted
  models; seed-level MDE reported to justify why seed-level tests are descriptive.
