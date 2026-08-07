# Re-Review (Round 2) — CoalGameRec v14

**Manuscript:** *CoalGameRec: validation-guided interaction attribution for graph recommendation — a frozen LightGCN study of LOO versus bounded Shapley*

**Version reviewed:** v14 (commit `1b9b23f`), `paper_package/main.tex` (identical to `springer_latex/main.tex`)
**Reviewer role:** senior reviewer, Discovery AI (Springer Nature) — XAI / Recommender Systems / GNNs / Cooperative Game Theory
**Review basis:** manuscript text cross-checked against every released artifact in `code/results/journal_runs/` (`summary_mean_std.csv`, `paired_bootstrap_all_controls*.csv`, `_holm.json`, `cost_effectiveness.csv`, `runtime_by_seed.csv`, `lambda_sensitivity.csv`, `explanation_diagnostics.json`, `manifest.json`), `code/coalgamerec/*.py`, and `code/requirements.lock`.

---

## PHASE 1 — OVERALL ASSESSMENT

- [x] **Accept**
- [ ] Minor Revision
- [ ] Major Revision
- [ ] Reject

| Criterion | Score (/10) |
|---|---:|
| Novelty | 7 |
| Technical correctness | 9 |
| Experimental quality | 8 |
| Scientific rigor | 9 |
| Writing quality | 8 |
| Organization | 8 |
| Reproducibility | 9 |
| Impact | 7 |
| References | 8 |
| **Overall** | **8** |

**Justification.** v14 resolves every critical and technical issue raised in the v13 review, and it does so in the scientifically correct way: instead of patching unsupported numbers, the authors aligned the manuscript *strictly* with the released artifacts and explicitly scoped every claim that the frozen protocol did not execute. The paper is now internally consistent: every equation matches the code, every table maps one-to-one to a file, the statistical family is defined exactly as computed, and the boundary condition (LOO matching/beating bounded Shapley at 13–16× lower cost) is reported honestly. Residual weaknesses (no validation-informed non-game controls, single backbone, proxy-level faithfulness) are real but are now declared as limitations with concrete regeneration plans rather than hidden behind fabricated rows — which is precisely what this journal requires. As a focused empirical study of LOO versus bounded Shapley for frozen LightGCN reranking, the manuscript meets the bar stated in the v13 decision letter ("If these are addressed, the manuscript could move toward acceptance").

---

## PHASE 2 — SECTION-BY-SECTION REVIEW (condensed; full checklist applied)

### Title, Abstract, Keywords — 9/10
- **Strengths:** title scopes the study precisely (frozen LightGCN, LOO vs bounded Shapley); abstract now reports only artifact-backed statistics; the Holm statement was corrected (previously attributed the p-value to LOO-vs-uniform; now correctly to Shapley-vs-uniform within the primary family).
- **Weaknesses:** none material. **Minor:** "validation-guided interaction attribution" appears without definition until §3 — acceptable for an abstract.
- **Verification:** +8.1% = (0.04976−0.04601)/0.04601 = 8.15% ✓; +8.7% = 8.70% ✓; 13.0×/15.7× ✓; 16.1×–18.3× gain/hour ✓ (0.01463/0.000911=16.06, 0.006712/0.000366=18.34).

### Introduction / Contributions — 8/10
- Contribution 1 no longer claims victory over a "validation-similarity baseline" that was never run. Contribution 4 defines the primary family explicitly (F=8 per dataset, Shapley-MC vs {uniform, additive-pref, attention, LOO} × 2 metrics, Holm) and states the one-to-one artifact mapping. Contribution 5 correctly marks fusion as released-but-not-evaluated.
- **Questions for authors:** none.

### Background / Design framework / Taxonomy — 8/10
- Unchanged from v13 except the author-prior-work sentence, which now explicitly states unpublished working papers are cited for taxonomy completeness only. Sound.

### Methodology — reranking (critical fix verified) — 10/10
- **Critical Issue 1 is resolved.** Eqs. (eq:weights)–(eq:rerank) were checked line-by-line against `coalgamerec/rerank.py`: raw signed weights for Shapley/LOO, $\mathbf{r}_u=\sum_j w_j e_j$, and division by $\sum_j|w_j|+\epsilon$ — identical to `attribution_adjustment_native`. The L1-normalizer choice is justified (signed weights; well-defined when $\sum_j w_j\approx 0$; symmetric in sign). Algorithm 4 now contains the explicit normalizer line and matches the equation. The sentence "all families pass through the identical intervention with λ fixed a priori" is the correct fairness statement.

### Algorithms and complexity — 9/10
- Algorithm 3 typos fixed (permutation renaming ρ; "order induced by ρ"). Algorithm 1 pools are now unambiguous (disjoint pools, exhaustion rule, fill rule).
- **Critical Issue 2 is resolved.** The complexity paragraph no longer claims 64× matches measured runtimes: op-count ≈64× is stated as an upper bound; measured 15.7× (ML-1M) and 13.0× (Amazon) are explained by caching, vectorization, and amortized overheads. Verified against `cost_effectiveness.csv`/`runtime_by_seed.csv`.

### Experimental setup / Datasets / Baselines / Metrics — 8/10
- Table 3 is now grouped (unreranked reference / non-game reweighting / cooperative-game attribution) as requested. The unreranked row uses real 5-seed λ=0 values (verified against `lambda_sensitivity.csv`: ML-1M 0.11415±0.00100 HR, 0.04482±0.00021 NDCG; Amazon 0.06690±0.00234, 0.02982±0.00074 ✓).
- The removal of valid-sim/valid-linear is handled correctly: no silent deletion — §13.2 states they were not run, and the resulting validation-access asymmetry is flagged in Threats and Limitations with matched controls committed to the regeneration plan. **This is the honest resolution; I had explicitly instructed "If these were not actually run, remove the claims."**

### Results / Paired contrasts — 9/10
- **Critical Issue 3 is resolved within the constraints of the frozen protocol.** Table 4 now contains exactly the 16 artifact-backed contrasts (verified row-by-row against `paired_bootstrap_all_controls.csv` + `_holm.json` for both datasets, including Amazon Shapley-vs-LOO: −0.000494 [−0.000709, −0.000281] ✓ and −0.000701 [−0.001375, −0.000054] ✓). The 12 LOO-as-treatment rows whose CIs/p-values had no artifact backing were removed; LOO-vs-uniform mean differences (0.00375 / 0.00259, verified in `cost_effectiveness.csv`) are reported descriptively with an explicit regeneration-plan note. The family definition is consistent across Contributions, estimand section, and caption.

### Ablation study — 9/10
- **Critical Issue 4 is resolved.** Table 6 was rebuilt from `lambda_sensitivity.csv` with 5-seed mean±SD for both datasets (all 30 cells spot-verified). Unsupported claims (k-plateau, M-halving variance, player-selection margins, native-vs-external margins) were removed and re-declared as "specified in run_ablations.py but not run". The Amazon narrative is now data-driven (heuristics flat, Shapley monotone) rather than "same trend". The runtime sentence cites the actual per-seed pattern including the 10,160 s outlier — commendable transparency. λ=0.10 remains fixed a priori; the sweep is correctly framed as sensitivity, not selection.

### Explainability analysis — 8/10
- **Critical Issue 5 is resolved at the level the artifacts allow.** The fabricated 6-method table is gone; the new table reports the real candidate-masking proxies for Shapley-MC attributions, 5 seeds, both datasets (verified against all 10 `explanation_diagnostics.json` files: ML-1M ΔNDCG 0.01107±0.00022 ✓, Amazon 0.00255±0.00019 ✓). All three caveats (masking proxy, single fraction, no cross-family comparison) are stated and faithfulness is explicitly not claimed. Retention percentages (59% ML-1M, 88% Amazon) are computed against the correct λ=0 reference.

### Discussion / Negative result / Deployment — 8/10
- Consistent with the corrected tables. **Minor:** "Shapley consistently improves over … popularity controls" (Discussion, Fig. 2 caption) — heuristic-pop is not in the Holm family; the statement holds descriptively (means) but should carry "(descriptively)" in camera-ready. Not a blocking issue.

### Threats to validity / Limitations — 9/10
- The validation-access asymmetry is now an explicit internal threat and the sixth limitation, with the matched validation-informed controls named as the first regeneration item. Honest and complete.

### Conclusion — 9/10
- Deployment recommendation (LOO default; Shapley only when faithfulness under redundancy/complementarity is demonstrated) follows from the evidence.

### References — 8/10
- The unpublished manuscript (louhichi2024gametheoryxai) is now `@misc` with `howpublished={Unpublished working paper}` and a note that it is not used as empirical evidence. Resolved.

---

## PHASE 3 — MATHEMATICAL VALIDATION

| Eq. | Correct? | Reason | Suggested Correction |
|---|---|---|---|
| (1) LightGCN propagation | YES | standard | — |
| (2) BPR loss | YES | standard | — |
| (3) Shapley value | YES | standard axiom-consistent form | — |
| (4)–(5) additive leakage | YES | linearity argument valid | — |
| (eq:weights) family weights | YES | matches `family_weights()` exactly | — |
| (eq:rerank) reranking | YES | matches `attribution_adjustment_native` exactly; L1 normalizer justified | — |
| (eq:pairwise) coalition value | YES | validation-only, smooth | — |
| Metrics (HR/NDCG/Coverage/ILD) | YES | binary-relevance NDCG with IDCG=1 is correctly stated | — |
| Complexity O(2^k C_v) / O(MkC_v) / O(kC_v) | YES | op-count now correctly separated from wall-clock | — |
| Statistical estimand (B=2000, Holm F=8, d_z) | YES | family definition matches artifact files | — |

No equation errors remain. Notation is consistent ($P_u$, $\mathcal{C}_u$, $\mathcal{N}_u^-$, $G_S$ defined before use).

## PHASE 4 — ALGORITHM REVIEW

- **Algorithm 1:** pools explicit; exhaustion and fill rules given; deterministic tie-breaks stated. Valid.
- **Algorithm 2:** unchanged; consistent with Eq. (eq:pairwise). Valid.
- **Algorithm 3:** antithetic pairing now correctly iterates ρ ∈ (π, π′) without variable shadowing; accumulation and final ÷M consistent with the estimator. Valid.
- **Algorithm 4:** now identical to Eq. (eq:rerank) and the code (normalizer d = Σ|w_j|+ε). Valid. No corrected pseudo-code needed.

## PHASE 5 — FIGURES

| Fig. | Caption Quality | Readability | Supports Claims? | Suggestions |
|---|---|---|---|---|
| 1 (architecture) | Good | Good | Yes | — |
| 2 (NDCG results) | Good | Good | Yes | add "(descriptively)" for pop in caption at camera-ready |
| 3 (cost-effectiveness) | Good | Good | Yes | — |
| 4 (λ sensitivity) | Fixed | Good | Yes | caption now points to the 5-seed table; single-seed status declared |

## PHASE 6 — TABLES

| Table | Formatting | Completeness | Missing Stats | Significance | Possible Errors | Suggestions |
|---|---|---|---|---|---|---|
| 1 (related work) | OK | OK | n/a | n/a | none | — |
| 2 (hyperparameters) | OK | OK | n/a | n/a | none | — |
| 3 (main results) | improved (grouped) | complete for executed families | SD present | in Tab. 4 | none found | — |
| 4 (paired contrasts) | artifact-exact | F=8×2 datasets | CI, p, d_z present | Holm | none found | LOO-as-treatment in regeneration |
| 5 (cost) | OK | OK | 5-seed means | n/a | none | — |
| 6 (λ sensitivity) | rebuilt | 5-seed mean±SD, both datasets | SD present | n/a | none | — |
| 7 (faithfulness proxies) | rebuilt | 5-seed mean±SD, both datasets | SD present | n/a | none | multi-fraction curves later |

All table values were cross-checked against the artifact files; no discrepancies found.

## PHASE 7 — RESULTS VALIDATION

| Claim | Value | Verified? |
|---|---|---|
| LOO vs uniform ML-1M | +8.1% (8.15% calc) | ✓ |
| LOO vs uniform Amazon | +8.7% (8.70% calc) | ✓ |
| Shapley vs uniform Holm | p<0.0005 both datasets | ✓ (`_holm.json`) |
| Shapley vs LOO ML-1M NDCG | −0.000532, p=0.008 | ✓ |
| Shapley vs LOO ML-1M HR | +0.000366, p=0.575 (n.s.) | ✓ |
| Shapley vs LOO Amazon | NDCG p<0.0005, HR p=0.036 | ✓ |
| Runtime ratios | 15.7× / 13.0× | ✓ |
| Gain/hour ratios | 18.3× / 16.1× | ✓ |
| λ-sweep margins (+7.2%, +6.1%) | recomputed | ✓ |
| Faithfulness proxies | recomputed from JSON | ✓ |

No arithmetic errors remain. Statistical significance is reported wherever the protocol produced it; where it did not (LOO-as-treatment, faithfulness ordering), the manuscript says so explicitly — this is the correct behavior.

## PHASE 8 — STATISTICAL REVIEW
5 seeds, B=2000 within-seed paired bootstrap, Holm F=8 per dataset, percentile CIs, d_z with magnitude language, %improved/harmed in artifacts, seed variance declared descriptive, low seed-power acknowledged. Remaining gaps (seed-population inference, LOO-as-treatment contrasts) are declared rather than hidden. Adequate for the claims made.

## PHASE 9 — EXPERIMENTAL VALIDATION
Two datasets, temporal LOO splits, full-catalog ranking, leakage controls, shared λ. The validation-access asymmetry is the one structural gap; it is declared and scheduled. No fabricated evidence remains.

## PHASE 10 — REPRODUCIBILITY — 9/10
Every table → artifact file mapping is stated; manifests record OS/Python/torch/device; configs resolved per run; environment pinned; cache-key scheme documented; Zenodo DOI + commit hash at acceptance. Checklist of remaining artifacts: none blocking.

## PHASE 11 — RELATED WORK & CITATIONS
Unchanged core from v13 (already adequate); unpublished-manuscript issue fixed. Optional camera-ready additions: 2–3 recent tractable-Shapley/GNN-attribution works — not required for acceptance.

## PHASE 12 — NOVELTY ANALYSIS
The novelty claim is modest and genuine: a design framework that makes game-attribution claims falsifiable + the first cost-aware head-to-head showing LOO matches/beats bounded Shapley under a matched validation-only utility. The honest negative framing distinguishes this from promotional Shapley papers. Meets Discovery AI expectations for a Research article.

## PHASE 13 — WRITING QUALITY
Clear, consistent, no planning language in the main body or appendices. The three most problematic v13 passages (64× sentence, fusion claims, "LOO/Shapley vs uniform/LOO" family wording) were all rewritten correctly. No rewrites needed.

## PHASE 14 — DISCOVERY AI COMPLIANCE

| Criterion | Score | Justification |
|---|---:|---|
| Scientific contribution | 8 | falsifiable framework + frontier result |
| AI methodology | 8 | game specification fully explicit |
| Experimental rigor | 8 | frozen protocol, artifact-exact reporting |
| Explainability | 6 | proxies only, honestly scoped |
| Ethics & Responsible AI | 8 | IRB determination, no redistribution, remapped IDs |
| Reproducibility & Open science | 9 | best-in-class artifact mapping |
| Novelty | 7 | focused but real |
| Practical impact | 7 | clear deployment rule |

## PHASE 15 — CRITICAL ISSUES
**None remain.** All five v13 critical issues verified resolved (Phases 2–7 above).

## PHASE 16 — MINOR ISSUES (optional, camera-ready)
- Discussion § and Fig. 2 caption: add "(descriptively)" to the heuristic-pop improvement statement (pop is not in the Holm family).
- § "Why the negative result matters": "often exceeds Shapley on … diversity" — on Amazon ILD, Shapley is marginally higher (0.92071 vs 0.92038); "often" is accurate but a parenthetical would preempt a reader objection.
- Consider adding LOO-as-treatment paired contrasts and validation-informed non-game controls at camera-ready if the regeneration runs complete in time (already declared in the limitations).

## PHASE 17 — FINAL DECISION

- **Major strengths:**
  - Every claim now maps to a released artifact; zero unsupported numbers remain.
  - Equation/algorithm/code consistency verified exactly (L1-normalized signed-weight intervention).
  - Exemplary handling of missing experiments: declared, scoped, and scheduled rather than fabricated.
  - Honest negative result (LOO frontier) with a concrete deployment rule and cost quantification.
  - Corrected statistical family definition consistent across abstract, contributions, estimand, and captions.
- **Major weaknesses:**
  - Validation-informed non-game controls still not run (declared limitation).
  - Single backbone (LightGCN), two datasets.
  - Faithfulness remains at proxy level (declared).
- **Mandatory revisions:** none.
- **Optional improvements:** the three camera-ready items in Phase 16.
- **Publication recommendation:** **Accept** (as a focused empirical study of LOO versus bounded Shapley for frozen LightGCN reranking), consistent with Phase 1.
- **Confidence in review:** **9/10** (all numerical claims recomputed from artifacts).

### Prioritized Revision Checklist
| # | Item | Priority |
|---|---|---|
| 1 | "(descriptively)" qualifier for heuristic-pop statements | Low (camera-ready) |
| 2 | Amazon ILD parenthetical in negative-result section | Low (camera-ready) |
| 3 | LOO-as-treatment bootstrap + validation-informed controls (regeneration) | Medium (post-acceptance / camera-ready if feasible) |

---

**Loop status:** Round 1 (v13): Major Revision → Implementer fixed all critical + technical issues → Round 2 (v14): **Accept**. The Reviewer ↔ Implementer loop terminates.
