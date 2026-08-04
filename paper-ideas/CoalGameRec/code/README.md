# CoalGameRec executable implementation

This folder contains a runnable implementation for the CoalGameRec empirical case-study pipeline. It supports two modes:

1. **Notebook smoke/feasibility mode** for a Mac M4 Pro / 48GB RAM.
2. **Journal-style batch mode** that produces per-seed, per-user, summary, sensitivity, and bootstrap-analysis artifacts from a YAML configuration.

> Scope note: the currently implemented trainable backbone is a local BPR-MF prototype. The Q1 confirmatory design still requires replacing/augmenting this with the validated HCCF port (`PORT.md`, fork commit, lockfile/container, official-code validation report, deterministic inference tests, ethics determination, and external preregistration). The batch runner is structured to generate paper-grade artifacts once that validated HCCF adapter is added.

## Install on macOS

```bash
cd paper-ideas/CoalGameRec/code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch on Apple Silicon should use `mps` automatically when available.

## Notebook

```bash
jupyter notebook notebooks/CoalGameRec_Mac_M4_Pro_Run.ipynb
```

The notebook automatically downloads MovieLens-1M and includes an optional Amazon Reviews 2018 `Books_5.json.gz` loader if you provide the file locally.

## Journal-style batch run

Run the full five-seed MovieLens local pipeline:

```bash
python scripts/run_q1_pipeline.py --config configs/q1_mac_ml1m.yaml
python scripts/analyze_q1_results.py --run-dir results/journal_runs/ml1m_mac_journal_v1
```

Main outputs:

```text
results/journal_runs/<run_id>/
├── config.resolved.json
├── dataset_stats.json
├── item_vectors_report.json
├── manifest.json
├── splits/
├── raw/
│   ├── per_user_metrics_all.csv
│   └── seed_<seed>/
│       ├── summary_by_family.csv
│       ├── per_user_metrics.csv
│       ├── lambda_sensitivity.csv
│       ├── shapley_shape_report.json
│       └── runtime.json
└── tables/
    ├── summary_by_seed_family.csv
    ├── summary_mean_std.csv
    ├── paired_bootstrap_contrasts.csv
    └── holm_primary.json
```

## What is implemented

- MovieLens-1M downloader/loader.
- Optional Amazon Books 2018 JSONL gzip loader.
- Rating-to-positive conversion (`rating >= 4`).
- Temporal leave-one-out split: second-last positive validation, last positive test.
- Train-period iterative 5-core filtering.
- Train-only sparse item-user vectors `x_i` with leakage fingerprinting.
- Frozen BPR-MF backbone for local runs.
- Full-catalogue base-score caching.
- Family-specific post-hoc weights:
  - `uniform`
  - `additive-pref`
  - fixed post-hoc `attention`
  - `heuristic-pop`
  - `shapley-mc`
- Validation-only Shapley coalition values.
- Common kernel reranking operator.
- HitRate@K, NDCG@K, coverage, and ILD.
- Reranking-strength sensitivity.
- Per-user outputs suitable for paired/bootstrap analysis.
- Conditional user-population bootstrap analysis and Holm correction.

## Data and generated artifacts are ignored

Raw datasets and large raw dumps should not be committed. They are covered by `.gitignore`:

```text
data/raw/
data/processed/
results/raw/
results/checkpoints/
results/models/
__pycache__/
*.pyc
```

Keep only curated small summaries/manifests when needed.

## Runtime reality and recommended settings

Unbounded interaction-level Shapley on every MovieLens user can take days on a laptop because the cost scales roughly with:

```text
users × history_length × M_permutations × full-catalogue value evaluations
```

The Mac journal-style config therefore uses a **bounded-player estimator**:

```yaml
attribution:
  m_permutations: 32
  max_players_per_user: 24
  player_selection: stratified
```

This computes Shapley over the 24 training interactions most similar to the user's profile and assigns zero Shapley weight to non-selected interactions for that run. The rule is deterministic and recorded in the config. For an unbounded/HPC run, set:

```yaml
max_players_per_user: null
m_permutations: 128
```

The code checkpoints Shapley estimates per seed at:

```text
raw/seed_<seed>/shapley_checkpoint.npz
```

If a run stops, rerunning the same config resumes from the checkpoint.

For Q1 manuscript claims, report exactly which estimator was preregistered: unbounded full-history Shapley, or the bounded-player estimator with its deterministic selection rule and sensitivity analysis.

## Prospective redesign after MovieLens prototype review

The first five-seed BPR-MF MovieLens run showed very small Shapley-vs-uniform gains and a stronger fixed-attention mean. The prospective configuration has therefore been changed before any new confirmatory run:

- primary coalition utility is now `pairwise_logsigmoid` on validation positives versus fixed validation negatives;
- the additive preference term is removed from the primary Shapley game (`lambda_pref: 0.0`) to avoid similarity-decomposition degeneracy;
- bounded-player selection is now `stratified` rather than pure similarity;
- permutation sampling uses antithetic pairs;
- the primary intervention is `native` embedding aggregation where the backbone exposes item embeddings;
- `loo-marginal` is added as a secondary control to test Shapley averaging against simple leave-one-out ablation;
- lightweight explanation diagnostics are emitted per seed (`explanation_diagnostics.json`).

The previous run in `results/journal_runs/ml1m_mac_journal_v1` remains a pilot. New prospective runs should use:

```bash
python scripts/run_q1_pipeline.py --config configs/q1_mac_ml1m.yaml
python scripts/analyze_q1_results.py --run-dir results/journal_runs/ml1m_mac_journal_v2_prospective
```

For Q1 claims, do not mix the old pilot (`v1`) and the redesigned prospective run (`v2`) as if they came from the same protocol.
