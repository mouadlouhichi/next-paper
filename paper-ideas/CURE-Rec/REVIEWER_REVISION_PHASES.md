# CURE-Rec reviewer-revision phases

This document separates completed evidence from remaining work. No manuscript claim
should be upgraded until the corresponding phase has completed, been archived, and
been inserted into the Springer manuscript.

## Completed evidence

- exact six-player CURE-Sim game, scenario Shapley regions, interactions, and robust-game Shapley values;
- controlled oracle regimes and misspecification negative control;
- 20-seed behavioural study, OAT study, and 25-configuration LHS study;
- shared-candidate MovieLens popularity/BPR/SASRec evaluation and fixed-seed replication;
- held-out selector studies, independent selector replication, two utility-weight studies, and strict provider-threshold study;
- exact versus sampled robust-Shapley comparison;
- stochastic click-feedback CRN diagnostic.

## Phase A — aggregate and report completed reviewer evidence

**Notebook:** `code/notebooks/08_phase_a_reviewer_assets.ipynb`

**Purpose:** aggregate the completed selector studies without rerunning CURE-Sim.

**Output:** `runs/reviewer-phase-a-assets/reviewer_table_selector_holdout.csv`.

**Required manuscript additions:**

1. selector-baseline table, explicitly reporting ties when exact CURE, singleton, robust-Shapley, greedy, and nominal selectors choose the same portfolio;
2. independent held-out replication table;
3. utility-weight and strict-threshold sensitivity table;
4. robust Shapley numeric table and a worked explanation card;
5. constraint-margin and interaction-matrix figures.

## Phase B — decision and variance ablations

**Status:** implementation required before execution.

1. **Maximin versus mean selection:** compare the selected masks, robust utility, constraint margins, and held-out feasibility over disjoint selection/evaluation seeds.
2. **Hard constraints versus penalty scalarization:** define a predeclared penalty grid; report constraint violation, utility, and repair rate rather than only final utility.
3. **CRN paired-difference study:** measure variance of a fixed pairwise coalition difference with common shocks versus independent shocks. Comparing selected-policy seed SDs alone is insufficient.
4. **Component ablations:** remove repair semantics, interaction reporting, robust-game attribution, and exact search separately; report what changes in decisions and diagnostics.
5. **Utility/threshold design:** predeclare all weight and threshold vectors. Treat changes in utility scale as objective sensitivity, not as performance gains.

## Phase C — scalability and attribution approximation

**Status:** implementation required before execution.

1. support restricted player libraries of size 6, 8, and 10;
2. record wall-clock time, memory, coalition count, and Shapley efficiency gap;
3. compare exact Shapley against sampled Shapley budgets 32, 128, 512, and 2048;
4. report MAE, sign agreement, rank correlation, and runtime versus exact attribution;
5. archive every run configuration and seed.

## Phase D — external statistical evaluation

**Status:** implementation required before execution.

1. export per-user rank/hit/NDCG outcomes for popularity, BPR, and SASRec;
2. calculate paired user bootstrap confidence intervals;
3. run paired permutation or Wilcoxon tests, effect sizes, and Holm correction;
4. retain model-seed statistics separately from user-level uncertainty;
5. add a second chronological dataset with explicit download/local-path configuration; do not convert it into causal-policy evidence.

## Phase E — manuscript and release

1. insert only completed-run tables and figures into `paper/cure-rec-springer/cure-rec.tex`;
2. keep causal-oracle, behavioural, and external-ranking claims separate;
3. rebuild using `paper/cure-rec-springer/build_clean.sh`;
4. verify `Table ??`, `???`, and unresolved metric notation are absent;
5. archive new revision artifacts with checksums;
6. create a tagged release and Zenodo/OSF record before submission.

## Execution rule

Run one expensive action at a time. After each completed action: inspect output,
archive it, commit it, and push it before enabling the next action.
