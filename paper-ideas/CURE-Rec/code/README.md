# CURE-Rec implementation scaffold

This directory contains the first runnable implementation shape for **CURE-Rec**: a causal, uncertainty-aware cooperative intervention layer that sits above a base recommender.

It is intentionally **CPU-first** and designed to run on an Apple Silicon Mac with 48 GB RAM. The first executable milestone is CURE-Sim, a fully disclosed sequential recommendation simulator with six intervention players and exact enumeration of all `2^6 = 64` coalitions.

## What is implemented in this milestone

- Typed YAML configuration using Pydantic.
- A structured observability layer:
  - human-readable `run.log`;
  - machine-readable `events.jsonl`;
  - run manifest with configuration and platform metadata;
  - per-coalition metrics, timings, constraints, and policy manifests.
- CURE-Sim with public user profiles, hidden preferences, exposure, popularity feedback, fatigue, provider exposure, and deterministic seeds.
- A documented history-aware base policy.
- Six composable policy interventions:
  - `repeat_cap`;
  - `explore_slot`;
  - `tail_slot`;
  - `diversify`;
  - `novel_slot`;
  - `provider_balance`.
- Exact collision resolution for the three slot-injection interventions.
- Exact coalition sweep, Shapley values, pairwise Grabisch–Roubens interactions, model/scenario regions, feasibility-aware semivalue sensitivity, and direct robust improvement selection.
- A generic CSV loader and logging audit for local recommendation logs.
- A notebook that runs the complete quickstart, including data loading, exact game evaluation, logs, plots, and artifact inspection.

## Deliberate non-goals of this milestone

This scaffold does **not** claim that a generic logged recommendation dataset yields valid long-horizon causal effects. Real-data causal claims remain gated by the audit in `cure_rec.data.audit_interactions` and the CURE-Rec implementation specification.

The current simulator supports multiple plausible scenarios as a computational ambiguity approximation. The paper-level `Γ` sensitivity design remains explicitly scoped to audited stochastic policy-mixture assignments; this scaffold records the inputs required for that extension rather than falsely treating arbitrary model ensembles as partial identification.

## Setup on macOS / Apple Silicon

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Recommended for the Adam/item-bias/hard-negative BPR baseline on Apple Silicon.
python -m pip install -e '.[dev,torch]'
```

Run the quick synthetic experiment:

```bash
cure-rec simulate --config configs/curesim_quickstart.yaml
```

Load and audit public/local interaction data explicitly:

```bash
# Download only after you explicitly opt in; standardize to a local CSV.
cure-rec load-data --dataset movielens_1m --source data/raw/ml-1m --download \
  --output data/processed/movielens_1m_interactions.csv

# Coat is loaded only with explicit download consent.
cure-rec load-data --dataset coat --source data/raw/coat --download \
  --output data/processed/coat_interactions.csv

# Yahoo! R3 must be downloaded manually according to its access terms.
cure-rec load-data --dataset yahoo_r3 --source data/raw/yahoo-r3 \
  --output data/processed/yahoo_r3_interactions.csv

# Generic local interaction file.
cure-rec load-data --dataset csv --source /path/to/interactions.csv
```

Run the full evidence-first workflow from the command line:

```bash
# Fetch/load -> audit -> popularity + BPR-MF baseline analysis -> CURE-Sim causal game -> all assets
cure-rec full-run \
  --config configs/curesim_quickstart.yaml \
  --dataset movielens_1m \
  --source data/raw/movielens_1m \
  --download
```

Or run only the external-data model analysis:

```bash
cure-rec analyze-data \
  --dataset movielens_1m \
  --source data/raw/movielens_1m \
  --bpr-updates 50000

# Staged validation-only Torch BPR search; requires .[dev,torch].
cure-rec search-bpr \
  --dataset movielens_1m \
  --source data/raw/movielens_1m \
  --output-root runs/bpr-search-movielens \
  --stage-epochs 40 \
  --final-epochs 200

# SASRec is enabled only after the BPR evaluation protocol and seed replication
# have been accepted. Search selects by validation NDCG@10 and does not rank test
# targets. It requires the optional torch install.
cure-rec search-sasrec \
  --dataset movielens_1m \
  --source data/raw/movielens_1m \
  --output-root runs/sasrec-search-movielens \
  --stage-epochs 30 \
  --final-epochs 120

# This is the first held-out-test evaluation after SASRec validation selection.
cure-rec final-sasrec-audit \
  --dataset movielens_1m \
  --source data/raw/movielens_1m \
  --search-root runs/sasrec-search-movielens \
  --output-root runs/final-sasrec-audit-movielens \
  --seed 42

# Replicate the frozen selection; this never retunes SASRec or uses test metrics
# for configuration choice.
cure-rec final-sasrec-seeds \
  --dataset movielens_1m \
  --source data/raw/movielens_1m \
  --search-root runs/sasrec-search-movielens \
  --output-root runs/final-sasrec-seed-replication \
  --seeds 42,43,44,45,46

# Controlled known-structure benchmark; run before large behavioural sweeps.
cure-rec regimes \
  --config configs/curesim_quickstart.yaml

# Paired stabilization sweep; common random numbers are shared within each seed.
cure-rec sweep \
  --config configs/curesim_full.yaml \
  --seeds 42,43,44,45,46

# Pre-specified behavioural sensitivity. This runs exact games and can take
# hours under the full configuration; start with a guarded notebook smoke run.
cure-rec calibrate \
  --config configs/curesim_full.yaml \
  --seeds 42,43,44,45,46 \
  --design oat

# Resume an interrupted calibration directory. Completed child seed runs are
# recovered from their manifests/tables; only missing seeds are evaluated.
cure-rec calibrate \
  --config configs/curesim_full.yaml \
  --seeds 42,43,44,45,46 \
  --design oat \
  --resume-dir runs/calibration-oat-20260805T184133Z

# Joint robustness probe. It adds one baseline plus the requested number of
# Latin-hypercube settings, each evaluated for every listed seed.
cure-rec calibrate \
  --config configs/curesim_full.yaml \
  --seeds 42,43,44,45,46 \
  --design lhs \
  --lhs-samples 24

# Regenerate aggregate figures and feasibility tables from a completed expensive sweep.
cure-rec postprocess-sweep \
  --config configs/curesim_full.yaml \
  --run-dir runs/seed-sweep-<timestamp>

# Reviewer Phase A: select on disjoint seeds and evaluate frozen portfolios on held-out seeds.
# The output contains selection_choices.csv, heldout_selector_evaluations.csv,
# heldout_selector_summary.csv, and revision_manifest.json.
cure-rec reviewer-holdout \
  --config configs/curesim_full.yaml \
  --output-root runs/reviewer-revision \
  --selection-seeds 42,43,44,45,46 \
  --evaluation-seeds 200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219

# Aggregate multiple completed Phase-A runs without rerunning simulations.
cure-rec reviewer-aggregate \
  --run-dirs runs/reviewer-revision/run-a,runs/reviewer-revision/run-b \
  --output-dir runs/reviewer-phase-a-assets
```

`calibrate` is a sensitivity analysis of stated CURE-Sim assumptions, **not** a
fit to MovieLens or an empirical causal calibration. OAT varies fatigue strength,
repeat threshold, horizon, provider threshold, provider-balance strength, delayed
novelty preference drift, and exploration cost. The joint LHS design samples the
same assumptions together. Both preserve seed-level decisions, base feasibility,
planner mode, exact attributions, and interactions.

Every loader runs the conservative audit and labels the strongest claim supported by the available fields. MovieLens, Coat, and Yahoo! R3 are not automatically promoted to long-horizon policy-evaluation evidence.

The notebook additionally has a guarded master `RUN_ALL_VARIATIONS` cell that executes every self-contained variation. Set both:

```python
RUN_ALL_VARIATIONS = True
CONFIRM_RUN_ALL = "RUN_ALL"
```

only when you intentionally want the several-hour complete experiment plan.

The simulation command prints the run directory. Inspect:

```bash
cat runs/<run-id>/run.log
head -n 5 runs/<run-id>/logs/events.jsonl
open runs/<run-id>/figures/coalition_improvements.png
```

Run tests:

```bash
pytest
```

Launch the notebook:

```bash
jupyter lab notebooks/00_cure_rec_quickstart.ipynb
```

## Project layout

```text
cure_rec/
├── config.py          # typed YAML configuration and hashes
├── observability.py   # JSONL events, run manifests, artifact writing
├── data.py            # CURE-Sim loading plus generic log audit
├── simulator.py       # disclosed sequential CURE-Sim environment
├── policies.py        # documented history-aware base policy
├── interventions.py   # six policy operators and exact collision allocation
├── game.py            # 64 coalition sweep, exact Shapley, interactions, regions
├── planner.py         # robust improvement and abstention planner
├── calibration.py     # OAT and Latin-hypercube CURE-Sim sensitivity designs
├── sasrec.py          # optional Torch sequential external-ranking comparator
├── sasrec_search.py   # validation-only SASRec selection, audit, and replication
├── pipeline.py        # complete end-to-end run
└── cli.py             # `cure-rec simulate`
```

Generated data and artifacts stay outside Git under `data/` and `runs/`.

## Second-round reviewer revision scripts

`scripts_review/` contains the experiment drivers added for the second revision
round; their archived outputs live under `results/reviewer_phase_assets/`:

- `phase_e_offline_sweeps.py` — utility-weight sensitivity grid and constraint
  frontier phase diagram, recomputed offline from one archived exact game.
- `phase_e_divergence.py` / `phase_e_focus.py` — predeclared screening for
  configurations in which the substantive selectors disagree, followed by the
  held-out selector protocol (selection seeds 42–46, evaluation seeds 200–219).

These scripts are simulator-conditional decision experiments; they do not create
external causal evidence.

## Observability contract

Every run writes an immutable manifest, including:

- config hash and source version;
- platform/Python metadata;
- random seeds;
- intervention parameters and canonical composition order;
- coalition masks and active intervention names;
- collision/no-candidate statistics;
- per-model coalition values and constraint metrics;
- selected portfolio and robust-improvement decision.

The JSONL event stream is intentionally simple so it can be inspected with `jq`, Pandas, or a text editor without an external tracking platform.
