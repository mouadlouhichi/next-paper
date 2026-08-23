# Reviewer Round 10 — assessment of manuscript v24 (Discovery AI protocol)

Scope: verification of the round-9 deep-review demands after the v23 (wording/protocol/code)
and v24 (corrected-protocol results) revisions. All claims checked against in-repo artifacts.

## PHASE 1 — Round-9 required items: status

| # | Item | Status in v24 | Evidence |
|---|---|---|---|
| 1 | Temporal protocol coherence + candidate exclusion | **DONE (ML-1M)**, Amazon executing | `ml1m_lightgcn_v7_corrected_protocol` (user, MPS, 5 seeds, 58,747 s): ordering preserved; ML-1M Shapley–LOO NDCG@20 equivalence now **established** (90% CI [−0.000937,+0.000143] ⊂ ±0.001); HR@20 equivalent with small Shapley point estimate |
| 2 | Relabeling + 2×2 selection×valuation factorial | Relabeling DONE; factorial RELEASED | `run_selection_factorial.py` (execution queued) |
| 3 | Circular λ tuning | Scoped as exploratory; nested version RELEASED | `run_nested_lambda_tuning.py` (execution queued) |
| 4 | Mixed v3/v6 sweeps | Disclosed; matched sweep RELEASED | `run_matched_lambda_sweep.py` (execution queued) |
| 5 | Equivalence/sufficiency overclaims | DONE | Title softened; dataset-specific verdicts; v7 ML-1M equivalence now formally established |
| 6 | "Beats all controls" overstatement | DONE | Nuanced wording throughout |
| 7 | Intervention as central factor | Scoped; factorial RELEASED | `run_design_ablations.py` multi-seed (execution queued) |
| 8 | Estimator convergence | Scoped; v2 RELEASED | `run_convergence_v2.py` (execution queued) |
| 9 | Inferential justification | DONE | sign-flip naming + symmetry assumption; MDE/observed-power removed; conditional scope stated |
| 10 | Stronger matched baselines | RELEASED | `run_sequential_baselines.py` (kNN / updated-profile / frozen edge-update / recency); Amazon 3-seed execution running in sandbox |
| 11 | Controlled randomization | RELEASED | `run_controlled_randomization.py`; Amazon execution queued in sandbox chain |
| 12 | Negative-sampling spec + multi-draw | Spec DONE; multi-draw queued | §protocol timeline |
| 13 | Faithfulness usage spec | DONE | §faithfulness scope |
| 14 | Cost labeling | DONE | "gain over uniform per attribution hour" |
| 15 | Algorithm 1 short histories | DONE + unit-tested | `tests/test_select_players.py` passes |
| 16 | Provenance | DONE | Provenance subsection incl. v7 run IDs |

## PHASE 2 — New results verified this round

**v7 corrected protocol, ML-1M** (paired inference re-run in sandbox, B=10,000, seed 20260821):
- LOO beats all six comparators on NDCG@20 (Holm p ≤ 0.007), including both validation-informed
  controls; HR@20: 4/6 significant (the two validation-informed contrasts n.s.).
- Shapley beats uniform and valid-sim on both metrics; vs valid-linear NDCG n.s. (Holm 0.062);
  vs LOO n.s. under sign-flip (both metrics), Wilcoxon disagrees on NDCG (p=0.0009) — the same
  procedure divergence seen in every execution, reported transparently.
- Equivalence: NDCG@20 established (CI inside ±0.001, straddling zero → no directional claim);
  HR@20 equivalent within ±0.002 (point estimate +0.00106 Shapley, n.s.).
- Friedman omnibus rejects (p<1e-100), LOO/Shapley top ranks.
Interpretation: the candidate-exclusion correction strengthens, not weakens, the manuscript's
central boundary claim on ML-1M.

## PHASE 3 — Remaining work (the shortest list of any round)

Executing now (sandbox chain, auto-committed): v7 Amazon → controlled randomization Amazon →
sequential baselines Amazon (3 seeds).

Queued (scripts released; sandbox or author hardware): matched λ sweep (both datasets), nested
λ tuning, convergence v2, 2×2 selection factorial, multi-seed intervention factorial, ML-1M
sequential baselines + controlled randomization, multi-draw negative-set sensitivity.

## PHASE 4 — Decision

**Major Revision — final iteration.** All protocol-integrity and claim-scope items are resolved;
the residual list consists exclusively of executions of already-released scripts. The
recommendation will move to **Accept** when:
1. v7 Amazon completes and is appended (running),
2. the sequential-baseline and controlled-randomization results land for at least one dataset
   (Amazon running),
3. the matched λ sweep and nested tuning replace the exploratory λ tables (queued),
with the remaining factorials/convergence acceptable as clearly-labeled queued replications if
compute is the only blocker.

**Confidence: 9/10.**

---

## Round-11 addendum (2026-08-23, manuscript v25)

The authors delivered the **ML-1M matched single-execution λ sweep** (v8, user-executed on MPS,
5 seeds, per-user artifacts). Round-9 item #4 is therefore **CLOSED for ML-1M**:

- All eight families reranked on identical fitted models; at λ=0 every family equals the base
  model (0.04493) — the within-model comparison is exact.
- LOO is the best family at **every** λ>0: 0.04751 / 0.04968 / 0.05448 / 0.06246 at
  λ=0.05/0.10/0.20/0.40 (Shapley: 0.04697–0.05946; valid-linear: 0.04673–0.05706).
- The earlier impression that "Shapley overtakes LOO at large λ" is confirmed to have been a
  **mixed-execution artifact** (v3 Shapley vs v6 LOO); it does not hold within identical models.
- Paired per-user LOO−Shapley differences grow with λ and are significant at λ=0.05/0.20/0.40
  (sign-flip p=0.008/0.004/0.003; Wilcoxon significant at all four points).
- Manuscript v25: tab:ablation_lambda and Fig 4 rebuilt with the matched ML-1M block (Amazon
  block retains the mixed-execution caveat until its matched sweep lands); provenance updated.

Remaining for Accept (unchanged from Round 10): v7 Amazon (sandbox), sequential baselines +
controlled randomization (sandbox queue), Amazon matched sweep + nested tuning (released scripts).
