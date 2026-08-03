# CoalGameRec — Response to Pre-Submission Reviews

**Status:** v1.2 package revision addressing two reviews
**Reviews addressed:**
- `review1.md` — Peer Review Report (Discover AI senior reviewer)
- `review2/review2.md` — Discovery AI standards review (scientific/methodological/statistical/ethics/formatting audit)

**Files updated:** `spec.md`, `Implementation_Spec.md`, `Paper_Structure.md`, `CoalGameRec_Analysis.md`
**Also added:** this response document.

> The reviews were received as documents committed to the session branch; the package has been revised to resolve the blocking (P0), major (P1), and the substantive consistency (C) points. Each item below states the action taken and where.

---

## A. Blocking (P0) issues — resolved

| Review point | Action taken | Where |
|---|---|---|
| 1.1 / 2.1 — **Predicted results presented as results** (3-sig-fig tables with bolded winners; abstract asserting "Shapley ranks first") | Removed all predicted numeric result tables; result tables now **populated only with realized numbers**; hypotheses stated directionally ("we test whether X > Y"); no pre-drafted winner; external pre-registration required | `Implementation_Spec.md` §B.0, §B.1a, §B.2, §B.5; `Paper_Structure.md` §7.2 + abstract |
| 2.2 — **Survey not yet systematic** | PRISMA 2020 adopted and required; protocol must be frozen/registered externally; screening counts, agreement, full-text exclusion log, and a **required** (not optional) quality/risk-of-bias rubric added | `spec.md` §4.5/§4.6; `Paper_Structure.md` §3.3 |
| 2.3 — **Scope overinclusive** | **Core vs. adjacent evidence tiers** (Adjacent A/B/C + background) defined and reported separately; core corpus restricted to top-N graph/hypergraph recommendation | `spec.md` §3 (evidence tiers), §4.4; `Paper_Structure.md` §5.3 |
| 2.4 — **Estimand undefined** | Stated an explicit estimand and coalition-masking semantics; distinguished frozen-model mask / retraining / in-training / list games | `Implementation_Spec.md` §A.6–§A.6a |
| 2.5 — **Benchmark not a fair test** | Added matched-objective requirement and a **matched additive-similarity baseline** (`additive-pref`); required factorial ablation (with/without preference term, etc.); narrowed claim to one taxonomy slice | `spec.md` §7.2; `Implementation_Spec.md` §A.5, §A.6c |
| 2.6 — **Statistical unit incoherent** | **Per-user analysis unit** fixed; 5 seeds are training variability, not the pairing unit; primary predeclared contrasts form the Holm–Bonferroni family; sensitivity tests and CIs specified | `Implementation_Spec.md` §A.10; `spec.md` §7.5 |
| 2.7 — **Ethics misclassification** | Institutional determination required for the human-generated data; "not applicable" no longer asserted as a design decision | `spec.md` §1.1; `Paper_Structure.md` §DECLARATIONS |
| 2 (abstract) — **Abstract > 250 words + future results** | Abstract rewritten to **210 words**, hypotheses not results | `Paper_Structure.md` §ABSTRACT |

---

## B. Major (P1) and Review-1 concerns — resolved

| Review point | Action | Where |
|---|---|---|
| 1.2 / 2.1 — **Review vs. Research article-type fit** | No longer asserted as "resolved"; the hybrid Review+benchmark format must be justified in the cover letter/Introduction; a pre-submission inquiry to the editor is recommended; possible reclassification anticipated | `spec.md` §1.1, §11; `Paper_Structure.md` header |
| 1.3 — **DyHuCoG reproducibility circularity** | **Resolved by decision: no DyHuCoG code in the benchmark.** The benchmark uses an independently documented hypergraph GNN; DyHuCoG appears only as a taxonomy worked example, described candidly; audit treated as an author-side risk, not a field fact | `Implementation_Spec.md` §A.4; `Paper_Structure.md` dependency note |
| 1.4 — **Self-citation / self-promotion optics** | Portfolio "citation home" rationale removed from manuscript-facing framing; §8 agenda argued from the literature with the author's planned work as one direction among several | `spec.md` §1.1 note; `Paper_Structure.md` §8, planning notes |
| 1.5 — **Dual-publication / text-recycling** | In-text attribution required for reused formulations (e.g., DyHuCoG value function); overlap table to be prepared; disclosure in the manuscript, not only "cover letter if asked" | `spec.md` §1.1; `Paper_Structure.md` §DECLARATIONS |
| 1.6 / 2.2 — **Novelty claim unsupported** | "First survey" is now a **hypothesis pending the registered search**; search must run before the taxonomy is locked; missing near-neighbors (ShaRP, data valuation, TU-bandit, VLDB) added to the required search list | `spec.md` §1.1, §12; `Paper_Structure.md` §1.2/§1.4 |
| 1.7 — **Conflating "improves accuracy" with "valid explanation"** | Benchmark explicitly labeled an **intervention/reranking study**, not an explanation evaluation; explanation-quality metrics required if explanation is claimed | `Implementation_Spec.md` §A.7, §A.7b; `Paper_Structure.md` §7.3 |
| 1.8 / 2.6 — **Statistical unit of pairing** | Per-user unit fixed; seed handling and correction family defined | `Implementation_Spec.md` §A.10 |
| 1.9 — **Fixed value-function weights** | Weight-sensitivity elevated to a **required** robustness check (§A.6c) | `Implementation_Spec.md` §A.6c, §A.5 |
| 2 (COI) — **Competing interests prefilled** | Complete disclosure incl. prior authorship of reviewed methods | `spec.md` §1.1; `Paper_Structure.md` §DECLARATIONS |
| 2 (AI-use) — **Generic statement insufficient** | Transparent risk-based AI disclosure required | `spec.md` §1.1; `Paper_Structure.md` §DECLARATIONS |
| 2 (data/code availability) — **Placeholders** | Real citations, checksums, split-generation code, permanent archive required | `spec.md` §1.1; `Paper_Structure.md` §DECLARATIONS |

---

## C. Cross-document consistency (review 2 §3, C1–C18)

| ID | Issue | Action |
|---|---|---|
| C1/C15 | "No experiments/no GPU" vs. a GPU benchmark | Analysis.md superseded; benchmark framed as original empirical work |
| C2 | Abstract reports future results | Abstract rewritten (210 words, hypotheses) |
| C3 | Split 70/10/20 vs. leave-one-out | Unified to **temporal leave-one-out** everywhere |
| C4 | Myerson optional vs. "every family everywhere" | `shapley-ai`/`myerson` made exploratory; full-factorial or drop; no partial cells |
| C5/C6 | Table/figure numbering and counts conflict | Single numbering plan; target counts reconciled (10–12k words, 7–9 figs, 5–7 tables) |
| C7 | Broken "§R" reference | Fixed to §A.4 |
| C8 | §7.7 cross-ref (no such section) | Fixed; weight sensitivity now §A.6c |
| C9 | §A.8/§B.2 stats cross-ref wrong | Statistics now in dedicated §A.10 |
| C10 | Environment not pinned | Exact versions/lockfile/determinism noted |
| C11 | "Same setting as DyHuCoG" overstated | Qualified to "same source domains and some metrics" |
| C12 | "5 prior publications" mismatch | Corrected to 3 published + thesis + 4 planned |
| C13 | Core and adjacent both excluded & reviewed | Evidence tiers defined and reported separately |
| C14 | Benchmark claims to ground whole taxonomy | Narrowed to interaction-player / ranking-utility slice |
| C16 | File-name mismatches | Stable repo names; this package uses the committed names |
| C17 | Amazon "raw" ambiguous | Exact file (`Books_5.json.gz`, 27,164,983 reviews) stated |
| C18 | Prediction tables vs. results confused | Prediction register separated; "Online Resource 1" labels |

---

## D. Additional substantive fixes (review 2 §4–§8)

- **MC estimator sampling law** specified (permutation / size-weighted / importance), with convergence criterion, not a "99% at M=50" claim (`Implementation_Spec.md` §A.6a).
- **Efficiency identity corrected** to `v(N) − v(∅)`; note that MC/smoothing/clipping do not preserve exact efficiency (`Implementation_Spec.md` §A.6, §A.8).
- **Myerson value defined** via the graph-restricted game; projection (2-section/incidence/line) is a method parameter.
- **Harsanyi vs. interaction indices, Weber vs. Harsanyi sets** disambiguated; notation list expanded.
- **B.4/B.5/B.1** predictions converted to neutral hypotheses; "matches published baselines" removed as a calibration target on the rebuilt split.
- **Limitations section** added (`Paper_Structure.md` §9).
- **Publication-count and advocacy language** corrected in the analysis memo (review 2 §4.1).

---

## E. Points accepted without structural change

- **Five-axis taxonomy** as the organizing device (retained; validated against the corpus by >1 coder with agreement).
- **PRISMA-style protocol, falsification table, and 5-seed/paired protocol** (retained as good practice, with the corrections above).
- **Discover AI as the venue** (retained; Q1 claim now dated-sourced or removed; 16-day median decision noted).

---

## F. Recommended next steps (not yet done — pending user/data)

1. Run the Amazon-Book feasibility spike and the synthetic Shapley tests (milestone 0).
2. Obtain the institutional ethics determination.
3. Register the PRISMA protocol and the prediction register externally (OSF/AsPredicted/Zenodo).
4. Confirm the current Discover AI APC, article-type policy, and whether a pre-submission inquiry is warranted.
5. Draft the <250-word abstract and cover letter (justifying the Review+benchmark format) once the protocol is frozen.
