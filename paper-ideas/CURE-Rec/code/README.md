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
```

Run the quick synthetic experiment:

```bash
cure-rec simulate --config configs/curesim_quickstart.yaml
```

The command prints the run directory. Inspect:

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
├── pipeline.py        # complete end-to-end run
└── cli.py             # `cure-rec simulate`
```

Generated data and artifacts stay outside Git under `data/` and `runs/`.

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
