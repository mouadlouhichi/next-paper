# Review-5 experiment guide (run on your machine)

Reviewer round 5 (Discover Artificial Intelligence, Major Revision) requires
several new experiments that cannot be computed from the existing release.
Everything below runs with the existing `actionshap` package; push the result
JSONs to `paper-ideas/ActionShap/code/results/review5/` and I will integrate
them into the manuscript.

Priority P0 = mandatory revision item; P1 = strongly requested; P2 = optional.

## P0-1. LIME mask-design ablation (mandatory #6)

Reviewer: short Amazon histories (median n_u = 5 → 32 unique coalitions) get
512 seeded Bernoulli rows with heavy duplication; must test alternatives.

For both datasets, ItemKNN primary condition, the first 200 cohort users:
- design A (current): full + empty + LOO + seeded Bernoulli(0.5) rows up to 512
- design B (unique): sample WITHOUT replacement over distinct masks
- design C (enumerate): all 2^n masks for users with n_u <= 9 (fall back to B)
Report per design: bounded AIA (mean + user-bootstrap CI), deletion AIA,
signed alignment, wall time. Also sweep ridge lambda {0.1, 1, 10} under
design A (kappa sweep already exists in review3 curves).

Suggested entry point: add `--lime-mask-ablation` to
`scripts/run_review3_experiments.py` (reuse `lime_attribution` with a
mask-design argument), write `results/review5/lime_mask_ablation.json`.

## P0-2. SASRec recommendation-quality gate (mandatory #1 support)

The SASRec replication currently proves attribution transfer, but the reviewer
needs the sequential model established as *competitive*: report NDCG@10 /
HR@10 / MRR vs the popularity baseline on both datasets (same candidate
protocol as Table 3), and confirm both masking gates pass on all 5 seeds.
Write `results/review5/sasrec_quality_gates.json`.

## P0-3. Full-catalogue at 1,000 users (mandatory #11)

Current full-catalogue subset is 250 users with as few as 5 active oracles in
one cell. Run Amazon Digital Music full-unseen-catalogue, target-margin,
ItemKNN, all five methods, 1,000 users, 5 seeds (Amazon histories are short,
so this is the cheaper dataset). Report: AIA components, paired differences
with user-bootstrap CIs, active-oracle counts, decision metrics.
Write `results/review5/full_catalogue_1000.json`.

## P0-4. Convergence figure with per-user quantiles (mandatory #4)

Regenerate the convergence panel from the existing convergence study: plot
median + 5th/25th percentile of user-level rank correlation and action
Jaccard vs M_pair, plus the fraction of users satisfying BOTH thresholds
(rank >= 0.95, Jaccard >= 0.80) at each M_pair; mark the exact-reference
subset where available. Update `figures/convergence.pdf`.

## P1-5. Variance components for the stochastic profile model (stats #4)

Factorial repetitions: (a) one frozen profile model x 5 attribution seeds;
(b) 5 model-initialization seeds x fixed attribution budget. Estimate
variance components (model vs explainer) for bounded AIA and report as a
small table. Write `results/review5/variance_components.json`.

## P1-6. Exact-Shapley per-user error distribution (mandatory #3 extension)

Already have max-abs-error and rank rho for enumerable users (review3). Add:
per-user error quantiles (p50/p90/p95), top-2 action Jaccard between exact
and MC-selected actions, and sign-error rate. Extend
`results/review5/exact_shapley_distribution.json` (reuse the review3 exact
enumeration code path).

## P2 (optional, strengthens but not required)

- LightGCN or BERT4Rec as a second competitive architecture (same gates).
- Regenerated hardware benchmarks (processor model, RAM, peak RSS) for the
  timing table.
- 2-3 qualitative per-user case studies (profile items, attributions,
  deletion vs bounded effects, selected pair vs oracle pair, rank before/after).

## Push-back contract

After each run, commit the JSON(s) under `code/results/review5/` and ping me;
I will (1) validate them against the matrices schema, (2) generate the tables
and body text, (3) re-run the cross-table validator and compile check, and
(4) update the review response document.
