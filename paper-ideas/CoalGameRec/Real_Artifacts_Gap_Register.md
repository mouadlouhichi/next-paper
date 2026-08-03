# CoalGameRec — Real Artifact Gap Register

**Purpose:** This file separates issues that cannot be fixed by wording edits from artifacts that must be produced, archived, validated, or approved before preregistration, benchmark execution, or journal submission.

**Current status:** planning/specification files are synchronized after the latest re-check, but the project is **not ready to preregister or run** until the blocking real artifacts below exist.

---

## 1. Blocking artifacts before benchmark preregistration

| Artifact | Why it is required | Minimum content / acceptance criteria | Status |
|---|---|---|---|
| **Pinned HCCF port** | The primary backbone is HCCF Option B; a planned port is not a reproducible backbone. | Fork URL, exact fork commit, license confirmation, documented diff from `akaxlh/HCCF`, deterministic inference/training settings, exact integration point for post-hoc reranking. | Missing |
| **`PORT.md` for HCCF** | Reviewers must know what changed relative to the official implementation. | File-level and equation-level changes, data-interface changes, removed/retained contrastive components, inference-time masking semantics, validation protocol and result. | Missing |
| **Environment lockfile / container** | Broad package ranges are not reproducible and may change HCCF behavior. | Exact Python, PyTorch, CUDA, driver, NumPy/SciPy/Pandas/scikit-learn versions; OS; GPU; lockfile or container digest; deterministic-kernel flags and tolerances. | Missing |
| **HCCF validation artifact** | The selected primary backbone must be validated before confirmatory use. | Official validation dataset/protocol, split, seed count, metrics, tolerance, validation logs, pass/fail decision, and fallback trigger if failed. | Missing |
| **Executable benchmark config** | The preregistration must point to one frozen configuration, not prose. | Dataset files, preprocessing choices, split rules, HCCF params, LightGCN params, seeds, negative sampling, MC `M=128`, reranking `λ_attr=0.10`, sensitivity grid, primary contrasts. | Missing |
| **Synthetic/unit test suite** | The reranking and Shapley implementation must be proven non-degenerate before real runs. | Efficiency, symmetry, dummy player, empty-mask, split reproducibility, deterministic backbone, mask locality, and nonzero unseen-item reranking tests all passing. | Missing |
| **Amazon-Book feasibility spike** | The custom Amazon sample may collapse under train-period 5-core filtering. | Fixed sample seed, final users/items/interactions/density, removals by step, candidate set sizes, hash of surviving users/items/splits. | Missing |
| **External preregistration record** | A local Markdown file is not preregistration. | OSF/AsPredicted/Zenodo or equivalent immutable timestamp; commit hash; protocol; primary hypotheses; analysis plan; deviation policy. | Missing |
| **Institutional ethics determination** | MovieLens/Amazon data are human-generated; public data does not automatically mean no ethics review. | Committee/body name, reference/exemption/approval decision, fields used, identifier handling, raw text/demographic handling, redistribution policy. | Missing |

---

## 2. Artifacts required before confirmatory benchmark execution

| Artifact | Why it is required | Minimum content / acceptance criteria | Status |
|---|---|---|---|
| **Frozen data splits** | Prevents leakage and post-hoc split changes. | Train/validation/test files, original line-index tie keys, config hash, dataset stats, train-period-only 5-core verification. | Missing |
| **Dataset checksums and access log** | Public-source datasets and custom Amazon processing must be auditable. | Source URLs, access dates, file names, checksums, license/terms notes, final processed graph hash. | Missing |
| **Raw result schema** | Ensures realized results cannot be confused with predictions. | Per-user metrics, per-seed summaries, method labels, config hash, run IDs, failure flags, runtime/memory logs. | Missing |
| **Primary analysis script** | Confirmatory inference must be reproducible and fixed. | Seed-clustered bootstrap implementation, Holm correction for 4 HCCF tests, effect sizes, per-seed outputs, sensitivity-analysis scripts. | Missing |
| **Shapley MC convergence report** | `M=128` must be documented, not assumed. | Exact-vs-MC checks for `|N_u|≤8`, synthetic analytic games, convergence curves for `M={16,32,64,128}`, pilot `M=512` reference. | Missing |

---

## 3. Artifacts required before manuscript submission

| Artifact | Why it is required | Minimum content / acceptance criteria | Status |
|---|---|---|---|
| **Registered PRISMA 2020 review protocol** | The survey's novelty and corpus claims require a reproducible systematic review. | Database-specific queries, search dates, dedup rules, inclusion/exclusion, evidence tiers, quality rubric, scope-change rule. | Missing |
| **Search exports and PRISMA flow** | The “no prior review meets scope” claim depends on actual search evidence. | Source counts, dedup count, title/abstract exclusions, full-text exclusions with reasons, core/adjacent counts, final update. | Missing |
| **Extraction/codebook artifact** | The taxonomy must be independently auditable. | Per-paper fields, allowed values, NR/NA/unclear coding, core/adjacent labels, game semantics, evaluation protocol, reproducibility flags. | Missing |
| **Two-coder agreement and adjudication log** | Taxonomy and quality claims need coding reliability. | Coder IDs/roles, Cohen’s kappa or agreement statistic, disagreement resolution notes, final reconciled sheet. | Missing |
| **Quality/risk-of-bias table** | Critical synthesis needs quality qualification. | 0/1/2 domain scores with anchors; NR/NA/unclear separated; no arbitrary total score unless justified. | Missing |
| **Complete DOI-checked bibliography** | Current files name methods without full citation records. | Foundational game theory, PRISMA, HCCF/HGNN/LightGCN/BPR, datasets, metrics, all included/adjacent papers, near-neighbor surveys. | Missing |
| **Realized result tables and figures** | Result tables must contain only measured values. | HCCF confirmatory NDCG@K and Recall@K/HitRate@K tables, secondary/supplementary LightGCN results, coverage/ILD/cost, prediction-vs-realized deviation report. | Missing |
| **Code/data archive** | Discover AI requires availability statements that support claims. | Public repository or archive DOI, configs, lockfile/container, frozen splits or split-generation scripts, raw outputs, tests, table emitters. | Missing |
| **Manuscript declarations** | Required submission metadata cannot remain placeholders. | Funding, CRediT, competing interests including prior authorship, data availability, code availability, ethics, AI-use disclosure, overlap/text-recycling statement. | Missing |
| **Final Springer formatting/accessibility package** | Planning Markdown is not submission-ready. | Springer template, <250-word abstract verified, square-bracket citations, accessible figures/captions, sequential numbering, cover letter. | Missing |

---

## 4. Go/no-go gates

1. **Do not preregister** until the HCCF port, lockfile/container, validation protocol/result, benchmark config, ethics determination, and preregistration text are complete.
2. **Do not run confirmatory data processing** until ethics determination and external preregistration are complete.
3. **Do not run confirmatory benchmark comparisons** until synthetic/unit tests and HCCF validation pass, or the preregistered HGNN fallback is triggered and documented.
4. **Do not draft result claims** until realized tables are generated from raw outputs by script.
5. **Do not submit** until PRISMA artifacts, extraction/quality coding, bibliography, code/data archive, declarations, and formatting checks are complete.

---

## 5. Current wording status

The planning files now correctly describe these artifacts as **planned/missing** rather than completed. In particular:

- HCCF is selected but not yet validated.
- The case study is planned for external preregistration, not preregistered.
- Ethics determination is required but not yet obtained.
- PRISMA corpus and screening artifacts are required but not yet produced.
- Result tables must remain realized-only and are currently empty/planned.
