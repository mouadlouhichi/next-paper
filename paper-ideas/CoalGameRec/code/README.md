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

## Strong-baseline paired analyses

After a run finishes, generate all reviewer-critical paired contrasts in one file:

```bash
python scripts/analyze_q1_results.py \
  --run-dir results/journal_runs/ml1m_mac_journal_v2_prospective \
  --treatment shapley-mc \
  --controls uniform additive-pref attention loo-marginal \
  --output-prefix paired_bootstrap_all_controls
```

This writes:

```text
tables/paired_bootstrap_all_controls.csv
tables/paired_bootstrap_all_controls_by_seed.csv
tables/paired_bootstrap_all_controls_holm.json
```

The table includes mean paired difference, 95% conditional user bootstrap CI,
user-conditional descriptive `d_z`, median difference, and proportions of users
improved/harmed/unchanged. The Shapley-vs-LOO contrast is the key test of whether
coalition-context averaging adds value beyond simple leave-one-out marginal
importance.

## Q1 main-claim readiness additions

The repository now contains the missing infrastructure needed to move beyond the
BPR-MF MovieLens pilot:

1. **Graph backbone:** `backbone.name: lightgcn` is implemented and can be run via `configs/q1_lightgcn_ml1m.yaml`.
2. **Second dataset template:** `configs/q1_lightgcn_amazon_template.yaml` defines the custom temporal Amazon-Book run; set `books_5_json_gz` to your local file.
3. **Strong baseline inference:** `analyze_q1_results.py --controls uniform additive-pref attention loo-marginal` produces paired CIs for all reviewer-critical contrasts.
4. **Cost/effectiveness:** `scripts/cost_effectiveness.py` summarizes runtime, attribution cost, and NDCG gain per attribution hour.

Recommended next commands:

```bash
# 1) Untouched graph-backbone MovieLens run
python scripts/run_q1_pipeline.py --config configs/q1_lightgcn_ml1m.yaml
python scripts/analyze_q1_results.py \
  --run-dir results/journal_runs/ml1m_lightgcn_v3_prospective \
  --treatment shapley-mc \
  --controls uniform additive-pref attention loo-marginal \
  --output-prefix paired_bootstrap_all_controls
python scripts/cost_effectiveness.py \
  --run-dir results/journal_runs/ml1m_lightgcn_v3_prospective

# 2) Then Amazon after the MovieLens LightGCN gate is acceptable
python scripts/run_q1_pipeline.py --config configs/q1_lightgcn_amazon_template.yaml
```

A Q1 main empirical claim should not be made until the LightGCN/HCCF and Amazon
runs are complete, paired strong-baseline contrasts are positive or properly
qualified, and ethics/preregistration artifacts exist.

## Checkpoint safety warning

Older versions wrote attribution checkpoints as `shapley_checkpoint.npz` and
`loo_checkpoint.npz`. Those files can become stale if the backbone implementation,
base scores, dataset split, or attribution config changes. The pipeline now writes
base-score-aware checkpoint names:

```text
shapley_checkpoint_<cache_tag>.npz
loo_checkpoint_<cache_tag>.npz
attribution_cache_manifest.json
```

If you rerun an old output directory after changing model code, delete or ignore
old untagged checkpoint files. Runs where `shapley_seconds` or `loo_seconds` is
near zero after a backbone change should be considered invalid and rerun with the
new cache-tagged checkpointing.

## Amazon Books data preparation

The Amazon config expects the UCSD Amazon Reviews 2018 5-core Books file:

```text
Books_5.json.gz
```

It is not committed to the repository. Prepare it with either:

```bash
# Option A: helper script
python scripts/prepare_amazon_books.py --dest data/raw/Books_5.json.gz

# Option B: resumable curl
mkdir -p data/raw
curl -L -C - -o data/raw/Books_5.json.gz \
  https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Books_5.json.gz
```

If the file is elsewhere, set:

```bash
export AMAZON_BOOKS_5=/absolute/path/to/Books_5.json.gz
python scripts/run_q1_pipeline.py --config configs/q1_lightgcn_amazon_template.yaml
```

The pipeline now raises a detailed error with these instructions if the file is missing.
