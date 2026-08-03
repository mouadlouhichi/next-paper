# CoalGameRec Mac-local implementation

This folder contains a runnable local implementation/prototype for the CoalGameRec empirical case-study pipeline, with a notebook targeted at a Mac M4 Pro / 48GB RAM machine.

## Notebook

Open:

```text
paper-ideas/CoalGameRec/code/notebooks/CoalGameRec_Mac_M4_Pro_Run.ipynb
```

Run from the notebook directory. The notebook automatically downloads MovieLens-1M and includes an optional loader for Amazon Reviews 2018 `Books_5.json.gz` if you provide the file locally.

## Install

Recommended on macOS:

```bash
cd paper-ideas/CoalGameRec/code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook notebooks/CoalGameRec_Mac_M4_Pro_Run.ipynb
```

PyTorch on Apple Silicon should use `mps` automatically when available.

## What is implemented

- MovieLens-1M downloader/loader.
- Optional Amazon Books 2018 JSONL gzip loader.
- Rating-to-positive conversion (`rating >= 4`).
- Temporal leave-one-out split: second-last positive validation, last positive test.
- Train-period iterative 5-core filtering.
- Train-only sparse item-user vectors `x_i`.
- Frozen BPR-MF backbone for local Mac runs.
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

## Important scope note

This is a **local executable prototype**, not the validated HCCF confirmatory implementation. The project documents still require the real HCCF port artifacts before preregistration:

- HCCF fork/commit and `PORT.md`;
- exact environment lockfile/container;
- official-code validation rerun and tolerance report;
- deterministic inference tests;
- ethics determination;
- external preregistration.

Use this notebook to verify data processing, attribution/reranking logic, and Mac feasibility before producing the preregistration artifacts.
