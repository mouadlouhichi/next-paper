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

## Phase A — detailed interpretation and manuscript rules

The completed selector studies show that the deterministic selectors often agree:
exact CURE maximin, best singleton, robust-Shapley-1, budgeted robust-Shapley,
greedy robust, and nominal selection all choose `repeat_cap` in the tested full
configuration. This is not a failure and must not be rewritten as a superiority
claim. The correct claim is that the exact planner recovers the same simple policy
when the decision problem has a clear dominant feasible intervention, while exposing
when random portfolios and the grand coalition are poor or infeasible.

The strict provider-threshold study must also be reported as a limitation: a policy
selected on calibration seeds can fail the strict constraint on some unseen stochastic
evaluation seeds. This is evidence for held-out portfolio evaluation, not evidence
that the threshold is universally satisfied.

The robust-game attribution should be described precisely:

```math
v^{rob}(S)=\min_{m\in\mathcal{M}}\Delta V_m(S),
```

and `robust_phi` is the exact Shapley allocation of this characteristic function.
Scenario Shapley regions answer a separate sensitivity question. Neither object alone
fully explains a hard-constrained decision; the decision card must also report the
binding constraint margins.

## Phase B — detailed design

### B1. Maximin versus mean-utility

For each selection seed, construct both:

```math
S_{maximin}=\arg\max_{S\in\mathcal{F}}\min_m\Delta V_m(S),
```

and:

```math
S_{mean}=\arg\max_{S\in\mathcal{F}}\frac{1}{|\mathcal{M}|}\sum_m\Delta V_m(S).
```

Freeze each selected mask and evaluate both on disjoint evaluation seeds. Report:
selected mask frequency, held-out lower utility, held-out mean utility, feasibility
rate, provider margin, fatigue margin, and paired difference. A tie is a valid result.

### B2. Hard constraints versus scalar penalty

Keep the hard-feasible selector as the reference. Predeclare penalty coefficients
before running the comparator and define:

```math
J_{pen}(S)=v^{rob}(S)-\lambda_r[r_{min}-r(S)]_+-\lambda_d[d(S)-d_{max}]_+-\lambda_f[f(S)-f_{max}]_+-\lambda_c[c(S)-B]_+.
```

Report both utility and violation rate. A penalty selector must not be called better
solely because it has higher unconstrained utility while violating a declared limit.

### B3. CRN analysis

The valid CRN endpoint is the variance of a *paired coalition difference*, not merely
the SD of a selected policy across seeds. Predeclare one or more coalition pairs,
for example base versus repeat-cap and repeat-cap versus repeat-cap-plus-tail. For
each seed compute the difference under shared shocks and under independent shocks.
Report the variance ratio and a confidence interval. The click-feedback variant is a
separate stochastic simulator variant and must be labelled as such.

### B4. Component ablations

Evaluate one component change at a time: remove repair semantics, replace exact
search with greedy search, remove the hard feasibility filter, remove robust-game
attribution from the explanation card, and remove common random numbers. The output
should identify which masks, constraint margins, and held-out outcomes change.

## Phase C — detailed scaling protocol

A larger player library must be explicitly declared. Do not create artificial duplicate
players merely to reach a larger n. Each added player needs a distinct operational
transformation, cost, eligibility rule, and collision semantics. For every n report:

```text
coalition count = 2^n
scenario count
seed count
runtime per game
peak memory
exact Shapley time
interaction time
Shapley efficiency gap
```

For sampled Shapley, repeat the estimator with independent permutation seeds. Report
mean absolute error, maximum error, sign agreement, rank correlation, and runtime
against the exact value. Exact versus sampled comparisons are meaningful only where
the exact table is available.

## Phase D — detailed external protocol

External ranking remains descriptive. For every evaluated user save the target rank,
hit, NDCG contribution, candidate count, and cold-target flag for every model. The
statistical comparison must be paired by user because each model ranks the same user
and candidate vector. Report bootstrap confidence intervals for metric differences,
paired permutation or Wilcoxon tests, effect sizes, and Holm-corrected p-values.

A second dataset must have real timestamps and sufficient positive history. Its loader,
license, preprocessing, positive threshold, user filtering, split, warm/cold handling,
and model search budget must be documented. Do not label a second ratings dataset as
causal evidence without logged exposures and propensities.

## Release checklist

Before submission, create a release containing:

```text
source commit hash
requirements and package versions
configuration YAML files
seed lists
raw small summary tables
all manuscript figures
SHA256SUMS.txt
REPRODUCE.md
revision experiment manifests
manuscript PDF and TeX source
```

Use a tag and archive DOI rather than only a moving working branch.

## Execution rule

Run one expensive action at a time. After each completed action: inspect output,
archive it, commit it, and push it before enabling the next action. Never use an
unfinished or stale kernel result in the manuscript.

## Phase E/F — second-round reviewer revision (2026-08-18)

New evidence produced for the Major-Revision round, archived under
`code/results/reviewer_phase_assets/`:

1. `objective_constraint_sweeps/` — utility-weight sensitivity grid (14 predeclared
   weight vectors) and constraint-frontier phase diagram (135 combinations of
   B, r_min, d_max, f_max), recomputed offline from the archived seed-42 exact game
   (no new rollouts). Scripts: `code/scripts_review/phase_e_offline_sweeps.py`.
2. `divergent_selector_holdout/` — predeclared screening of the baseline and the
   archived LHS design points whose seed-level decisions were not always repeat-cap,
   followed by the full held-out selector protocol (selection seeds 42–46, disjoint
   evaluation seeds 200–219) on the first two divergent configurations in design
   order (`lhs-012` and `lhs-009`). Both held-out studies are complete; summary CSVs
   are checksummed in `SHA256SUMS.txt`. Scripts: `code/scripts_review/phase_e_divergence.py`,
   `phase_e_focus.py`, `phase_e_resume_holdout.py`, `finalize_holdout.py`.

Manuscript claims that depend on these assets are limited to simulator-conditional
decision sensitivity and held-out seed generalization. No external causal claim is
upgraded. The MovieLens-1M user-level paired inference remains blocked on re-running
the audited pipeline with per-user export (dataset download unavailable in the
current environment); the analysis plan is pre-specified in the manuscript
limitations instead.

### Addendum — semi-real integration and integrated scalability (2026-08-19)

Two further Phase-9 reviewer items are now implemented and executed:

3. `semireal_integration/` — a learned BPR-MF ranker (trained on 14,400
   simulator-logged impressions) deployed as the CURE-Sim base policy, with the
   exact intervention game re-run on top of it; the decision (repeat cap, repair
   mode) and robust attributions are consistent with the hand-crafted base game.
   Script: `code/scripts_review/phase_f_semireal.py` (`cure_rec/semireal.py`).
4. `integrated_scalability/players_8/` — the four extended operators
   (session_length_cap, freshness_quota, provider_cooldown, category_coverage_quota)
   implemented as real slate transformations in `cure_rec/interventions.py`; the
   exact integrated n=8 game ran (7,505 s, 114 MB, zero efficiency gap, decision
   identical to n=6, selection regret 0). n=10 runs through the closure notebook.
   Script: `code/scripts_review/phase_g_integrated_scalability.py`.

Runs that require data downloads or long compute (MovieLens-1M per-user paired
inference, Amazon second-domain audit, n=10 integrated game) are gathered in
`code/notebooks/13_reviewer_closure_runs.ipynb` for execution on a machine with
network access; the notebook emits manuscript-ready tables.

### Addendum — MovieLens-1M user-level paired inference executed (2026-08-21)

Run A of the closure notebook was executed on a machine with data access
(commit `40cc0d3`): the frozen BPR and SASRec configurations were re-trained for
seeds 42-46, per-user hit/NDCG rows exported (1,000 warm users x 11 model-seeds),
and the paired user-level analysis computed. Results: all four model x metric
comparisons significant after within-family Holm correction (p <= 1.5e-10 on the
five-seed per-user means), all bootstrap CIs exclude zero, permutation tests
p <= 0.0007; effect sizes small-to-moderate (dz 0.099-0.280). Integrated into
Table 9 of the manuscript (Section 5.4).

A bug was found and fixed during integration: the exact sign/McNemar test in
`paired_user_statistics` computed only the lower binomial tail, which reports
p = 1.0 for any positive effect. It now doubles the smaller of the two tails
(two-sided exact sign test). The MovieLens-25M second-dataset results are
unaffected (negative effects were already in the correct tail); the ML-1M
p-values were recomputed from the archived per-user metrics (bootstrap CIs
unchanged to 1e-16). `phase_d_ml1m_paired.py` additionally gained resume support
(skips retraining when per-user metrics are already archived) and per-family
Holm correction.

## Phase H — calibration-robust planning (third-round reviewer item 2, 2026-09-02)

The second-round manuscript left held-out feasibility of `0.70` (`lhs-012`) and
`0.47` (`lhs-009`) as an acknowledged limitation, with the seed-robust extension
described but explicitly "not evaluated in the present artifact". That extension
is now implemented and executed.

**Implementation.** `cure_rec/seed_robust.py` enlarges the planner's uncertainty
set from scenarios to scenarios x calibration seeds `Z_cal = {42,...,46}`:

- `calibration_robust_values` — `min_{m,zeta} Delta V_{m,zeta}(S)`;
- `seed_scenario_metrics` — constraint margins pooled over seeds plus per-seed
  feasibility indicators;
- `select_calibration_robust_portfolio(rule="minimax")` — every constraint must
  hold for every (scenario, seed) pair;
- `select_calibration_robust_portfolio(rule="chance", alpha)` — empirical chance
  constraint `(1/K) sum_zeta 1_F(S;zeta) >= 1 - alpha`, ranked by the
  seed-averaged worst-scenario improvement;
- `heldout_evaluation` — scores one frozen portfolio on unseen seeds; the
  initialization seed is applied to a private settings copy because `CureSim`
  seeds its generator from `settings.run.seed`.

Improvement / repair / abstention / certified-infeasibility semantics are
unchanged. With `|Z_cal| = 1` the minimax rule reproduces the shipped planner
exactly (same mask, value agreement `6.9e-17`).

**Driver.** `code/scripts_review/phase_h_seed_robust.py <lhs-012|lhs-009>`.
Calibration games are reconstructed from the archived exact coalition tables
(`divergent_selector_holdout/<point>/selection/.../tables/coalition_values.csv`),
so the new planners see precisely the rollouts that produced the published
numbers; held-out rows are fresh rollouts on seeds 200-219.

**Verification performed.**

1. The published per-seed protocol is reproduced exactly from archived tables:
   `lhs-012` -> `-0.089270` / feasibility `0.70`; `lhs-009` -> `+0.198583` /
   `0.47`, matching `heldout_selector_summary.csv` to the printed precision.
2. Fresh held-out rollouts agree with the archived `events.jsonl` values to
   `9.7e-17` (lhs-012, 140 overlapping mask x seed rows) and `8.3e-17`
   (lhs-009). The only initial discrepancy was the base-coalition row, whose
   logged `improvement` field stores raw utility because
   `run_scenario_game` applies `replace(improvement=0.0)` after logging.
3. Live rollouts through `evaluate_seed_masks` reproduce the reconstructed
   seed-42 game to `9.7e-17` across all 256 coalition cells and select the same
   mask, confirming the reconstruction and the live path are interchangeable.

**Result.** Pooling over calibration seeds converts an unstable multi-portfolio
decision into one frozen portfolio and raises held-out feasibility from `0.70`
to `1.00` (`lhs-012`) and from `0.47` to `1.00` (`lhs-009`), at a measured
utility cost of `0.031746` and `0.053897` respectively (paired `d_z = -137.1`
and `-81.2` over 20 evaluation seeds). The chance-constrained rule with
`alpha = 0.2` is intermediate on `lhs-009` (`0.55` feasibility, `0.008973`
cost), matching its declared conservatism. Reported in the manuscript as
Section 5.4 (`sec:seed-robust`) and Table `tab:seed-robust`.

**Assets.** `code/results/reviewer_phase_assets/seed_robust_planner/{lhs-012,lhs-009}/`
(`calibration_selections.csv`, `scenario_only_per_seed_selections.csv`,
`heldout_planner_evaluations.csv`, `heldout_planner_summary.csv`,
`archived_crosscheck.csv`, `revision_manifest.json`); checksums regenerated with
`scripts_review/hash_revision_assets.py` (349 entries).
