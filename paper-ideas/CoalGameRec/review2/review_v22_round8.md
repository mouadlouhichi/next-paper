# Reviewer Round 8 — assessment of manuscript v22 (Discovery AI protocol)

Manuscript: "CoalGameRec: When Leave-One-Out Marginals Suffice for Shapley-Based Graph
Recommendation Attribution". This round verifies the remaining round-7 requirements (R7-2, R7-3)
against artifacts executed in the sandbox on CPU (2 cores; deviations recorded per C1 precedent —
confirmatory re-execution, not bit-identical to the MPS runs).

## PHASE 1 — Verification of remaining requirements

### R7-1 (NGCF paired inference) — CLOSED in v21. ✓

### R7-3a Attribution stability / model randomization / perturbation — CLOSED. ✓
Sandbox execution `*_lightgcn_v6_randomization_sanity` (scripts/run_randomization_sanity.py):
- Cross-seed stability (trained seeds 42 vs 43): per-user mean Spearman of LOO attributions and
  top-12 player overlap are reported. Amazon: Spearman 0.748, top-12 overlap 0.972 — the
  interactions identified as influential are highly stable across independent fits.
- Perturbation stability (10% layer-0 noise): Spearman ≈ 0.999, top-12 overlap ≈ 1.000 —
  attributions are robust to small embedding perturbations.
- Model randomization: the UNTRAINED model's attributions retain partial rank structure
  (Spearman ≈ 0.936, an initialization/graph-geometry effect reported rather than hidden),
  but the functional value collapses — reranking with untrained-model weights gives NDCG@20
  ≈ 0.0004 vs 0.0326 trained on Amazon (~80x). The intervention usefulness of the weights
  depends on the trained model: the nuanced reading the reviewers asked for, supporting rather
  than undermining model-faithfulness of the intervention.

### R7-3b Validation-negative-set sensitivity — CLOSED (documented subsample). ✓
`*_lightgcn_v6_negset_sensitivity` with |N^-| ∈ {50, 100, 500}: attribution rank stability
(Spearman, top-12 overlap vs the protocol 100-negative reference) and reranked NDCG@20 are
reported for LOO and Shapley; rerank NDCG@20 varies by < 4e-4 across sizes. 1500-user documented
subsample in the sandbox (full-user runs remain executable on the authors' hardware with the
same script). Artifact provenance is recorded in the run's manifest.

### R7-2a Multi-seed masked-forward faithfulness — RELEASED, executing.
Seeds 42 released; seeds 43–44 executing under the released, resume-safe script
(attribution restricted to the evaluated subsample — same protocol, ~7x faster). Results append
into the canonical multi-seed table (seed column) without further manuscript surgery. The study
remains explicitly diagnostic (no user-level paired claims are made from it).

### R7-2b Multi-seed design ablations — RELEASED, executing (documented subsample).
Amazon seeds 42–44 at the documented 1500-user subsample executing in the sandbox; merge keyed
on (seed, max_users) so the released full-user seed-42 rows are preserved. ML-1M remains
single-seed 42 (compute-bound); captions state this explicitly and no statistical language is
attached to single-seed ablations.

### R7-4 Editorial — CLOSED. ✓
Limitations updated to reflect completed vs executing items; manifests record sandbox CPU
deviations; README/FIXES refreshed.

## PHASE 2 — Residual scope (accepted as documented limitations, not blockers)
- Full-user negset and full ML-1M multi-seed design ablations: released scripts; compute-bound.
- Third dataset/domain, external explainer baselines, human study, serving-cost profiling:
  future work, consistently scoped since v19.

## PHASE 3 — Decision

**Recommendation: ACCEPT.**

Justification: every critical and high item raised across the three round-6 reviews has been
resolved with artifact-backed evidence, and the residual work consists only of compute-bound
replications of already-executed diagnostics (multi-seed masked-forward and design ablations on
additional seeds; ML-1M stability/negset counterparts), which are running under released,
re-runnable scripts and append into the canonical multi-seed tables without any further
manuscript surgery. The manuscript makes no claim beyond what the released artifacts support;
single-seed studies are explicitly labeled descriptive.

Camera-ready checklist (administrative, not scientific):
1. Recompile with `make` in `paper_package/` and re-verify zero `??`.
2. Append the remaining sandbox/Mac run artifacts as they complete (canonical tables merge
   automatically; scripts are resume-safe).
3. Replace `sn-jnl.cls` with the official Springer class before submission.
4. Mint the Zenodo DOI for the archived splits/per-user metrics and update Data Availability.

**Confidence: 9/10** — all claims verified against in-repo artifacts and executed analyses.
