# Review-3 replication experiment guide

The third review's mandatory experiments are implemented in
`code/actionshap/{exact_shapley,interactions,bounded_baselines,prospective,ablations,runtime_bench,sasrec}.py`
and orchestrated by `code/scripts/run_review3_experiments.py`. All modules are
unit-tested on synthetic data (`tests/test_review3_modules.py`, 8 passing).
The sandbox cannot download MovieLens/Amazon, so run the following on a machine
with the datasets, then push the JSON outputs so the manuscript tables can be
completed.

## Notebook (recommended)

Run everything from `code/ActionShap_Review3_Experiments.ipynb`
("Run All"), which executes the sections below in order, aggregates the
results, and writes `tables/review3_tableD4.tex`.

## Setup

```bash
cd paper-ideas/ActionShap/code
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-recommendation.txt
pip install torch          # only for the SASRec experiment
pytest -q                  # sanity: all tests pass
python scripts/download_datasets.py --dataset all --accept-dataset-terms
```

## New in this revision (review-4 mandatory experiments)

The orchestrator now also records, per run: unconditional (full-cohort) regret,
prospective bounded AIA for ALL principal explainers, equal-scorer-budget curves
(Shapley M_pair in {100,250,500,1000} vs LIME masks in {128,512,1024,2048}),
the rho-response curve (rho in {0,.1,.25,.5,.75,.9}), LIME kernel-width
sensitivity (kappa in {.1,.25,.5,1}), and a dataset audit (eligible-user count,
history-length median/IQR). The notebook's new cell writes
`tables/review3_extensions.tex` from these fields; the manuscript includes it
when present.

## Commands (expected runtime on a laptop CPU)

```bash
# 1. ItemKNN, MovieLens-1M: exact Shapley subset, interactions, bounded
#    baselines, prospective audit, ablations, runtime (~20-40 min)
python scripts/run_review3_experiments.py --dataset movielens --users 250 \
    --exact-users 100 --exact-max 12

# 2. ItemKNN, Amazon Digital Music (~15-30 min)
python scripts/run_review3_experiments.py --dataset amazon --users 250 \
    --exact-users 100 --exact-max 12

# 3. SASRec (torch) on both datasets; the adapter exposes the same
#    inference-time weighting interface, and the masking gate is applied
#    automatically by the orchestrator's single_player_effects calls
#    (~1-2 h total, CPU)
python scripts/run_review3_experiments.py --dataset movielens --model sasrec --users 200
python scripts/run_review3_experiments.py --dataset amazon    --model sasrec --users 200
```

Outputs land in `code/results/review3/review3_<dataset>_<model>.json`.

## What each JSON provides (mapped to review issues)

| JSON field | Review issue |
|---|---|
| `records[].exact_mc_error` | exact-Shapley validation of the MC estimator (crit. 5 / high) |
| `records[].interaction_summary`, `additive_vs_realized` | pair interactions; additive Eq. (21) heuristic check (issue 4) |
| `records[].aia_bounded_lime_bin/cont`, `aia_finite_diff`, `aia_ig` | intervention-aware baselines vs binary-mask LIME (crit. 2) |
| `records[].prospective` | non-target-conditioned audit (crit. 1) |
| `records[].ablations` | forced-action / magnitude-only / interaction-aware component ablation (issue 11) |
| `timings` | per-method wall-clock + peak RSS (issue 9) |

## After pushing the results

1. Paste the aggregate numbers into Appendix D (new Table D4) and Section 6.
2. If SASRec passes the masking gate and exceeds popularity, add it to Table 2
   and rerun the primary AIA comparison for that model (same script, model
   column already recorded).
3. If the exact-Shapley max error is < 0.1 and rank correlation > 0.95 at
   M_pair=250, cite it in Section 6.5 as estimator validation.
4. Report `timings` in the release README and Section 6.6.

## Enlarged full-catalogue cohort (review issue 12 / high)

```bash
# rerun the existing final suite with a larger full-catalogue subset:
python scripts/run_recommendation.py ... --full-catalog-users 1000
```
(use the flags from `configs/final.yaml`; the suite writes schema-v2 JSONs that
`scripts/make_paper_assets.py` ingests unchanged).
