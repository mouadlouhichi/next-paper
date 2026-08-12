# Re-Review (Round 5) — CoalGameRec v18

**Manuscript:** *CoalGameRec: validation-guided interaction attribution for graph recommendation — a frozen LightGCN study of LOO versus bounded Shapley*

**Version reviewed:** v18 (commit `b109e65`)

**Review basis:** full re-inspection of `paper_package/main.tex` against every released artifact (v3 primary runs, v4b C1 runs with Shapley, design-ablation / convergence / masked-forward / redundancy artifacts).

---

## PHASE 1 — OVERALL ASSESSMENT

- [x] **Accept**
- [ ] Minor Revision
- [ ] Major Revision
- [ ] Reject

| Criterion | Round 4 | Round 5 |
|---|---:|---:|
| Novelty | 6 | 7 |
| Technical correctness | 6 | 9 |
| Experimental quality | 6 | 8.5 |
| Scientific rigor | 7 | 9 |
| Writing quality | 6 | 8 |
| Organization | 5 | 8 |
| Reproducibility | 7 | 9 |
| Impact | 6 | 7.5 |
| References | 5 | 7.5 |
| **Overall** | **6** | **8.3** |

**Justification.** Every mandatory revision from Round 4 has been executed rather than discussed, and the manuscript's claims now match its evidence exactly. The paper has completed the transformation the review demanded: it no longer claims more validation than the executed experiments provide, and the experiments now cover the design principles the framework asserts.

---

## PHASE 2 — MANDATORY-REVISION VERIFICATION (Round-4 checklist)

| # | Required revision | Status | Evidence |
|---|---|---|---|
| 1 | Correct Eq. (7) normalization mathematics | ✅ Done | divisor-free form; L1 divisor proven inert under candidate-wise z-scoring; stable z-score + zero-variance behavior defined |
| 2 | Correct LightGCN complexity | ✅ Done | $C_v=\mathcal{O}(L\|E_S\|d+\|\mathcal{N}_u^-\|d)$; op-count stated as upper bound, not equality |
| 3 | $k$-sensitivity | ✅ Executed | flat for $k\ge16$ on both datasets (tab:design_ablations) |
| 4 | $M$-budget convergence | ✅ Executed | efficiency residual $\le4.3\times10^{-10}$ all $M$; Spearman $0.81\to0.96$ vs $M=256$, passing $0.91$ at $M=64$ (tab:estimator_convergence) |
| 5 | Player-selection ablation | ✅ Executed | stratified ≈ similarity ≈ random, indistinguishable |
| 6 | Hard-vs-smooth utility ablation | ✅ Executed | smooth beats hard for both families on both datasets — first direct empirical support for the smooth-utility principle (P2 now tested, not merely asserted) |
| 7 | Native-vs-external intervention ablation | ✅ Executed | reported as an honest finding \emph{against} the alignment principle as a performance claim; native retained as the protocol intervention — exactly the transparency this journal expects |
| 8 | Redundancy/complementarity comparison | ✅ Executed | synthetic coverage game: LOO assigns zero to all four redundant players, double-counts the complementary pair, $-25\%$ efficiency gap vs Shapley residual $\sim10^{-15}$ (tab:redundancy_demo) — the controlled instance of the real-data efficiency gaps |
| 9 | True masked-forward faithfulness | ✅ Executed | true masked re-propagation (CPU, self-tested to $\sim10^{-10}$): insertion retains/exceeds unmasked under game families while uniform falls below; deletion degrades most under LOO (ML-1M) and Shapley (Amazon); faithfulness still bounded, as it should be |
| 10 | Equivalence instead of "n.s. = match" | ✅ Done | SESOI declared a priori; equivalence on all four contrasts; both NDCG intervals entirely on the LOO side |
| 11 | C1 scope gap (Shapley not re-run) | ✅ Closed | C1b re-runs Shapley under the identical matched environment; LOO significantly preferred on NDCG@20 on both datasets (ML-1M $p=0.007$, Amazon $p<0.0005$); LOO beats all six matched controls (ML-1M 12/12, Amazon 11/12 Holm) |
| 12 | Meta-review language / PRISMA identity | ✅ Done | all reviewer-facing drafting language removed; PRISMA claim removed; framework repositioned as proposed conceptual language tested on one configuration |
| 13 | References | ✅ Improved | 9 verified published references added; bibliography 1:1 with citations; unpublished working-paper entry resolved to its published version |

---

## PHASE 3 — NUMERICAL SPOT-CHECKS (recomputed)

- C1b ordering (NDCG@20): LOO $>$ Shapley $>$ valid-linear $>$ valid-sim $>$ heuristics on both datasets ✓
- ML-1M LOO vs valid-sim $+0.00190$ $(+3.98\%)$, vs valid-linear $+0.00126$ $(+2.61\%)$ ✓ arithmetic verified
- Amazon LOO vs valid-sim $+0.00194$, vs valid-linear $+0.00075$; the single n.s. contrast (HR vs valid-linear, $p=0.086$) is reported as such ✓
- Redundancy demo: Shapley sums to $1.000$ (residual $1.2\times10^{-15}$), LOO to $0.750$ ($-25\%$ gap) ✓
- Runtime ratios $13.0$–$15.7\times$ and gain/hour $16.1$–$18.3\times$ ✓

No arithmetic or table/text inconsistencies found.

---

## PHASE 4 — REMAINING (NON-BLOCKING) OBSERVATIONS

These do not block acceptance; they are camera-ready or future-work items, all of which the manuscript already declares:

1. A second graph backbone (e.g., LightGCL/XSimGCL) would test the external validity of the boundary result; declared as future work, appropriately scoped.
2. Multi-seed masked-forward curves and perturbation-stability / model-randomization controls remain future work; declared.
3. Effect sizes are small in absolute terms; the manuscript states this plainly and anchors interpretation on the equivalence analysis and cost ratios rather than on magnitude.
4. The kernel-vs-native finding invites a follow-up study of alignment-vs-accuracy trade-offs; noted in the ablation discussion.

---

## PHASE 5 — FINAL DECISION

- **Major strengths:**
  - A genuinely falsifiable boundary result: full coalition-context averaging (bounded Shapley) is not necessary for ranking improvement over the grand-coalition LOO marginal, established with matched controls, equivalence testing, and a re-run under identical conditions.
  - Every design principle of the framework is now either empirically tested (smooth utility, k-budget, selection, intervention) or honestly scoped.
  - The redundancy/complementarity mechanism is demonstrated on a controlled synthetic game and matches the real-data efficiency gaps.
  - Unusually candid reporting: the kernel-beats-native finding, the n.s. contrast, the small effect sizes, and the bounded faithfulness claim are all stated plainly.
  - Full artifact trail: every table maps to a released artifact; every analysis is script-reproducible.
- **Major weaknesses:** none blocking; residual items are declared future work.
- **Mandatory revisions:** none.
- **Publication recommendation:** **Accept.**
- **Confidence in review:** **9/10** (all headline numbers recomputed from artifacts; residual uncertainty only over the not-yet-executed future-work items, which the manuscript declares as such).

---

**Loop status:** Round 1 (v13): Major Revision → Round 2 (v14): Major Revision → Round 3 (v15): Major Revision → Round 4 (v16/v17): experiments executed → **Round 5 (v18): ACCEPT.** The Reviewer ↔ Implementer loop terminates.
