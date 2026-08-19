# Reviewer Round 7 — assessment of manuscript v20 (Discovery AI protocol)

Manuscript: "CoalGameRec: When Leave-One-Out Marginals Suffice for Shapley-Based Graph
Recommendation Attribution" (v20, integrating the corrected round-6 inference and the executed
v6 experiments). This review verifies every round-6 mandatory item against released artifacts.

## PHASE 1 — Overall assessment

**Recommendation: MINOR REVISION** (conditional on the pending runs listed in Phase 4).

| Criterion | Round-6 score | Round-7 score |
|---|---:|---:|
| Novelty | 5–7 | 7 |
| Technical correctness | 6 | 8 |
| Experimental quality | 6–7 | 8 |
| Scientific rigor | 6–8 | 9 |
| Writing quality | 4–6 | 7 |
| Organization | 3–5 | 7 |
| Reproducibility | 8 | 9 |
| Impact | 4–7 | 7 |
| References | 6–8 | 8 |
| **Overall** | **5–6.2** | **7.7** |

## PHASE 2 — Verification of round-6 CRITICAL items

1. **Table 11 caption/body mismatch + broken `??` references** — RESOLVED (v19). All table
   captions match bodies (`tab:c1_paired`, `tab:c1_shap`, `tab:c1_faith` separated); grep shows
   zero `??`; every `\ref` resolves to a `\label`; Figures 2 and 4 restored after the condensation.
   Verified against `paper_package/main.tex`.

2. **Bootstrap p-value methodology** — RESOLVED (v19, recomputed). All paired tables now use
   sign-flip permutation p-values with the +1 correction (B=10,000, seed 20260818) on joint
   seed-mean user differences; Wilcoxon and the legacy within-seed bootstrap are reported as
   sensitivity (`tab:robustness`). The honest consequence is reported rather than hidden:
   ML-1M Shapley-vs-LOO is procedure-dependent; ML-1M NDCG@20 equivalence is marginal and
   underpowered (TOST power 0.34), Amazon-Book equivalence is established. I specifically commend
   the authors for weakening their own equivalence claim when the corrected inference demanded it.

## PHASE 3 — Verification of round-6 HIGH items

3. **Second backbone** — RESOLVED (v20). NGCF under the identical frozen protocol, five seeds,
   both datasets, Shapley included (`*_ngcf_v6_second_backbone`). The full family ordering
   replicates: LOO > Shapley > valid-linear > matched controls > unreranked on both datasets.
   This was the single most important external-validity gap; it is now closed at the descriptive
   level. Remaining: paired per-user inference for this backbone (artifact push pending).

4. **Independently tuned λ** — RESOLVED (v20). Proper validation-based selection with one test
   evaluation (`tab:lambda_tuned`): tuning widens LOO's lead (+30.0% ML-1M, +23.9% Amazon over the
   best tuned non-game family). Together with the test-oracle table, this fully answers the
   "shared λ is unfair to Shapley/competitors" objection — if anything the direction reverses.

5. **LOO missing from the λ-sweep** — RESOLVED (v20). Dedicated five-seed sweep integrated into
   `tab:ablation_lambda` and `fig:lambda_sensitivity`; LOO is the strongest family at every λ>0.

6. **Conditional coalition semantics v_u(S|B_u)** — RESOLVED (v19): explicit B_u = H_u\P_u,
   conditional notation in Principle 2 and Algorithm 2.

7. **P1/additivity reframing** — RESOLVED (v19): linearity stated as decomposition only, never a
   performance ordering; matched controls isolate leakage empirically.

8. **Efficiency residual de-promoted** — RESOLVED (v19): explicitly identified as a telescoping
   identity, not convergence evidence; Spearman ρ vs M=256 + two estimator seeds are the evidence.

9. **Algorithm fixes** — RESOLVED (v19): vectorized z-score (Alg 4), P_u/B_u inputs (Alg 2),
   short-history scaling + x_j notation + O(k|H_u|d) (Alg 1), RNG/seeds clarified (Alg 3).

10. **Friedman–Nemenyi, MDE/power, effect-size CIs, top-20 crossing, runtime medians** —
    RESOLVED (v19): all present and correctly framed (omnibus significant, pairwise n.s.,
    paired contrasts remain primary).

11. **References** — RESOLVED (v19): all previously-undefined keys fixed; reviewer-suggested
    works added with verified metadata (GraphFramEx, fragile explanations, Tai redundancy,
    SEGE, Distill n' Explain, Probing, Shapley reviews, SGL, FastSHAP, PRINCE, counterfactuals,
    KG reasoning, influence functions, Cohen).

## PHASE 4 — Remaining requirements (what keeps this at Minor Revision, not Accept)

**R7-1 (required, artifact push).** The NGCF per-user metrics exist on the authors' machine but
were excluded from the repository by the old `**.gz` ignore rule (now removed). Push
`*_ngcf_v6_second_backbone/raw/seed_*/per_user_metrics.csv.gz` and
`raw/per_user_metrics_all.csv.gz`, then run the paired permutation/Wilcoxon/equivalence pipeline
(`analyze_round6_stats.py` pattern) for the NGCF study and add the contrast table. Without it the
second-backbone claim is descriptive only.

**R7-2 (required runs, cells 5–6).** Multi-seed design ablations and multi-seed masked-forward
faithfulness. Both scripts are released and multi-seed-aware; seed 42 exists. Complete at least
seeds 43–44 (5 preferred) and replace the single-seed captions with mean±SD + paired intervals
where applicable.

**R7-3 (required runs, cells 7–8).** Attribution stability / model-randomization / perturbation
sanity and validation-negative-set sensitivity (50/100/500). These are the last two faithfulness
and robustness diagnostics demanded in round 6.

**R7-4 (editorial, after R7-1..3).** Update the Limitations paragraph to strike the completed
items; add the NGCF contrast table next to `tab:second_backbone`; refresh the README/claim
documents to v20; recompile with `make` and re-verify zero `??`.

## PHASE 5 — Explicitly accepted as out of scope (documented in Limitations)
- Third dataset/domain, influence-function/counterfactual external baselines, human-subject
  explanation study, peak-memory/serving-latency profiling, seed-population hierarchical modeling.
  The manuscript's scoping is now internally consistent and I consider these legitimate future work.

## Decision
**Minor Revision.** Every critical and high item from round 6 is either verified resolved in v19/v20
or reduced to a concrete, scripted, pending execution (cells 5–8 + one artifact push). Once R7-1 to
R7-3 land and the tables are appended, this manuscript will have satisfied the full round-6
mandatory checklist; I will then reassess for acceptance.

**Confidence: 9/10** (all v19 claims verified against in-repo artifacts; v20 numbers verified
against the committed v6 run artifacts; pending items tracked explicitly).
