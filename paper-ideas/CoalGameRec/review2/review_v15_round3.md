# Re-Review (Round 3) — CoalGameRec v15

**Manuscript:** *CoalGameRec: validation-guided interaction attribution for graph recommendation — a frozen LightGCN study of LOO versus bounded Shapley*

**Version reviewed:** v15 (commit `54cc65a`), `paper_package/main.tex`
**Reviewer role:** senior reviewer, Discovery AI (Springer Nature)
**Review basis:** manuscript cross-checked against all v3 artifacts, both v4 C1 runs (`ml1m_lightgcn_v4_matched_controls`, `amazon_books_lightgcn_v4_matched_controls` — manifests, `run.log`, per-seed artifacts, all `tables/*.csv` + `_holm.json`), and the executable C1 infrastructure (`scripts/run_matched_controls.py`, `notebooks/CoalGameRec_C1_Run.ipynb`).

---

## PHASE 1 — OVERALL ASSESSMENT

- [x] **Accept**
- [ ] Minor Revision
- [ ] Major Revision
- [ ] Reject

| Criterion | v14 | v15 |
|---|---:|---:|
| Novelty | 7 | 7 |
| Technical correctness | 9 | 9 |
| Experimental quality | 8 | 9 |
| Scientific rigor | 9 | 10 |
| Writing quality | 8 | 8 |
| Organization | 8 | 9 |
| Reproducibility | 9 | 10 |
| Impact | 7 | 8 |
| References | 8 | 8 |
| **Overall** | **8** | **9** |

**Justification.** v15 resolves both remaining blockers from Round 2 with the strongest possible response: the authors did not merely re-insert the validation-informed baselines — they executed a full confirmatory re-run of the frozen protocol (C1) on both datasets with five seeds each, on the original hardware for ML-1M, with a passed fidelity check (uniform NDCG@20 = 0.04601, identical to the primary study), complete execution logs, and Holm-adjusted paired contrasts. The scientific integrity displayed across rounds — removing unsupported claims in v14, then generating the missing evidence in v15 rather than arguing around it — is exactly what this journal expects.

---

## PHASE 2 — ROUND-2 ISSUE VERIFICATION

### Critical Issue 1 — validation-informed non-game baselines: **RESOLVED**
- `valid-sim` and `valid-linear` are precisely defined (Eqs. in §C1), implemented in `coalgamerec/rerank.py`, share the a-priori λ=0.10, and were executed under the frozen protocol (5 seeds, both datasets).
- Table `tab:c1_main` groups families correctly (unreranked / non-game / validation-informed / game).
- Fidelity check is convincing: ML-1M C1 on the original v3 hardware reproduces uniform 0.04601±0.00022 (identical), unreranked 0.04493 vs 0.04482, LOO 0.04968 vs 0.04976. Amazon C1 (CPU) likewise within noise (LOO 0.03229±0.00082 vs 0.03237±0.00077).
- Result: LOO beats valid-sim on NDCG@20 and HitRate@20 on both datasets (Holm p<0.0005), and beats valid-linear on NDCG@20 on both datasets (ML-1M +0.00126 p<0.0005; Amazon +0.00076 p=0.001). The single non-significant contrast (Amazon HR vs valid-linear, p=0.077) is reported as such — no overclaiming.

### Critical Issue 2 — paired LOO statistics: **RESOLVED**
- Table 4b (`tab:paired_loo`): Holm F=10 per dataset from the primary v3 per-user artifacts; all LOO-vs-heuristic contrasts p<0.0005.
- Table `tab:c1_paired`: Holm F=12 per dataset from C1; ML-1M rejects 12/12, Amazon 11/12. Every row maps one-to-one to a released CSV + Holm JSON.
- The abstract's "Holm p≤0.004" is correctly scoped to NDCG@20 and matches the largest rejected p (ML-1M HR vs valid-linear, p=0.004). Verified.

### Critical Issue 3 — ablations not used as design justification: **RESOLVED (v14, stable in v15)**
- k=24 justification rests solely on the feasibility/complexity argument; unexecuted ablations are declared as such and no numbers are reported for them.

### Critical Issue 4 — XAI evidence: **SUBSTANTIALLY RESOLVED**
- Table `tab:c1_faith` adds deletion/insertion curves over fractions {0.05, 0.10, 0.20, 0.30} with uniform and seeded-random controls, five seeds, both datasets. Verified from `faithfulness_curves_all.csv`: LOO's deletion drop strictly exceeds uniform and random at every fraction on both datasets; the ordering is monotone in fraction; insertion retention favors LOO.
- The candidate-masking caveat and the no-faithfulness-claim scope are retained. Masked-forward evaluation and perturbation stability remain declared future work with executable scripts — acceptable given the paper's explicit scope.

---

## PHASE 7 — RESULTS VALIDATION (spot checks recomputed)

| Claim | Source | Verified? |
|---|---|---|
| ML-1M LOO vs valid-sim NDCG +3.98% | (0.04968−0.04778)/0.04778 = 3.98% | ✓ |
| ML-1M LOO vs valid-linear +2.61% | (0.04968−0.04842)/0.04842 = 2.60% | ✓ (rounding) |
| Amazon LOO vs valid-sim +6.26% | (0.03229−0.03039)/0.03039 = 6.25% | ✓ |
| Amazon LOO vs valid-linear +2.39% | (0.03229−0.03154)/0.03154 = 2.38% | ✓ (rounding) |
| ML-1M C1 runtime | run.log stage sums ≈ 14,067 s manifest | ✓ |
| Holm rejection counts | 12/12 ML-1M, 11/12 Amazon vs `_holm.json` | ✓ |
| Faithfulness ordering (all 8 fraction×dataset cells) | LOO > max(uniform, random) | ✓ |

No arithmetic errors found. No claim without an artifact.

## PHASE 10 — REPRODUCIBILITY — 10/10
The C1 study is reproducible end-to-end from the repository: one script + one notebook, archived splits, pinned hyperparameters, per-run manifests (hardware, torch, python, deviations), complete execution logs, and per-seed artifacts. The ML-1M run was executed on the original v3 machine (macOS arm64, MPS, torch 2.3.1) per its manifest, and the Amazon run's CPU deviation is recorded. This is exemplary.

## PHASE 15 — CRITICAL ISSUES
**None remain.**

## PHASE 16 — MINOR ISSUES (optional, camera-ready)
- Consider adding a one-line note in §C1 that the Amazon C1 models were trained on CPU while the Amazon v3 models trained on MPS, alongside the already-recorded manifest deviation (currently only stated as "hardware" in the deviations list — sufficient, but explicitness costs nothing).
- The Discussion could echo the C1 outcome once (it currently lives in Results/Threats/Limitations/Abstract only) — optional.

## PHASE 17 — FINAL DECISION

- **Major strengths:**
  - The validation-access confound — the central methodological objection — is closed with executed, Holm-tested, artifact-backed evidence on both datasets.
  - Confirmatory fidelity: C1 reproduces the primary study within seed noise on the original hardware.
  - Honest reporting throughout: the one non-significant contrast and all deviations are declared.
  - Full reproducibility chain (script + notebook + logs + manifests).
- **Major weaknesses:** none blocking; residual scope limits (masking-proxy faithfulness, no C1 Shapley re-run, unexecuted k/M ablations) are declared as limitations/future work rather than overclaimed.
- **Mandatory revisions:** none.
- **Optional improvements:** the two camera-ready items in Phase 16.
- **Publication recommendation:** **Accept**, consistent with Phase 1.
- **Confidence in review:** **9/10** (all headline numbers recomputed from the released artifacts).

### Prioritized Revision Checklist
| # | Item | Priority |
|---|---|---|
| 1 | Explicit CPU-vs-MPS note for Amazon C1 in §C1 text | Low (camera-ready) |
| 2 | One C1 sentence in Discussion | Low (camera-ready) |

---

**Loop status:** Round 1 (v13): Major Revision → Round 2 (v14): Major Revision → Round 3 (v15): **Accept**. The Reviewer ↔ Implementer loop terminates.
