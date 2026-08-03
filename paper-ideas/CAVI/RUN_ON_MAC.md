# Running the CAVI stack on your Mac M4 (48 GB RAM)

Everything below runs on macOS with Apple Silicon. Your 48 GB RAM is plenty — the
datasets are small (~1–26 MB) and the models are lightweight.

**TL;DR:** make a venv → install deps → run tests → run scripts → open the notebook.

---

## 1. Clone the repo (if not already on disk)

```bash
cd ~/Desktop/personal/phd/next-paper
git fetch origin
git reset --hard origin/arena/019fc350-next-paper   # or: git pull origin arena/019fc350-next-paper
```

Make sure you're on the right branch:

```bash
git checkout arena/019fc350-next-paper
git log --oneline -3
# expect: 5ea707c Q1 experiment suite ...  /  1bf1de3 add resource (skip)  /  018d3ec new results
```

---

## 2. Create a Python virtual environment (recommended)

Your Mac M4's system Python is fine, but a venv keeps things clean. Apple Silicon
uses **arm64**; numpy/scipy/pandas ship native arm64 wheels, so no Rosetta needed.

```bash
cd ~/Desktop/personal/phd/next-paper/paper-ideas/CAVI

# create venv
python3 -m venv .venv

# activate it (for THIS terminal session)
source .venv/bin/activate

# upgrade pip + install deps
python -m pip install --upgrade pip
pip install numpy scipy pandas pytest nbformat nbconvert jupyter ipykernel matplotlib
```

> **Every time you open a new terminal**, re-activate: `source .venv/bin/activate`.

Check it works:

```bash
python -c "import numpy, scipy, pandas; print('OK', numpy.__version__, scipy.__version__)"
```

---

## 3. Verify the data is present

The three datasets should already be tracked in the repo (commit `598ad32`):

```bash
ls -la paper-ideas/CAVI/gate/data/          # ml1m_ratings.dat, ml1m_items.dat
ls -la paper-ideas/CAVI/data/amazon-book/   # train.txt, test.txt, ...
ls -la paper-ideas/CAVI/data/yelp2018/      # train.txt, test.txt, ...
```

If `gate/data/` is missing the ML-1M files, the notebook's `ensure_ml1m()` will
auto-download them from grouplens (internet needed). Amazon-Book / Yelp2018 are
fetched via a GitHub clone fallback if the local files are absent.

---

## 4. Run the test suite (fast, should pass)

```bash
cd ~/Desktop/personal/phd/next-paper/paper-ideas/CAVI
python -m pytest tests/ -q
# expect: 22 passed (or more)
```

---

## 5. Run the experiments (small → large)

All scripts live in `paper-ideas/CAVI/scripts/` and take `--users`, `--seeds`,
`--candidates` args. **Start small** to confirm they run, then scale up.

### 5a. Synthetic theory validation (seconds)

```bash
python scripts/run_synthetic_validation.py
```

### 5b. ML-1M end-to-end pipeline

```bash
python scripts/run_ml1m_experiment.py --users 50 --nmax 8 --seed 7
```

### 5c. Cross-dataset (Amazon-Book, Yelp2018)

```bash
python scripts/run_cross_dataset.py --dataset yelp2018 --users 40 --seed 7
python scripts/run_cross_dataset.py --dataset amazon-book --users 40 --seed 7
```

### 5d. Paper-A divergence study

```bash
python scripts/run_paperA_divergence.py --users 25 --nmax 8 --seed 7
```

### 5e. Q1-style benchmarks

There are **three** Q1 experiment scripts. They are the empirical backbone, and
they differ in what they measure:

| Script | What it measures | Runtime (ML-1M) |
|---|---|---|
| `run_q1_experiment.py` | NDCG/Recall/MRR vs baselines (v1) | ~6 min / 200 users |
| `run_q1_v2.py` | non-degenerate forward CAV reweighting (v2) | ~4 min / 200 users |
| `run_q1_v3_action.py` | actionable-recourse: which interaction to amplify | ~4 min / 200 users |

```bash
# small smoke first
python scripts/run_q1_experiment.py --users 30 --train-users 400 --seeds 7 --candidates 60
python scripts/run_q1_v2.py        --users 30 --train-users 400 --seeds 7 --candidates 60
python scripts/run_q1_v3_action.py --users 30 --train-users 400 --seeds 7 --candidates 60

# then scale up (3 seeds, more users) — still CPU-only, a few minutes each
python scripts/run_q1_experiment.py --users 200 --train-users 1500 --seeds 7 42 123 --candidates 100
python scripts/run_q1_v2.py        --users 200 --train-users 1500 --seeds 7 42 123 --candidates 100
python scripts/run_q1_v3_action.py --users 200 --train-users 1500 --seeds 7 42 123 --candidates 100
```

Results are written to `paper-ideas/CAVI/results/*.json`.

---

## 6. Run the Jupyter notebook

```bash
cd ~/Desktop/personal/phd/next-paper/paper-ideas/CAVI
source .venv/bin/activate
jupyter notebook
```

Open `CAVI_walkthrough.ipynb`, then **Kernel → Restart Kernel & Run All**. It
loads all three datasets, validates the theory, runs the forward game + CAV,
recourse, divergence study, OPE gate, and cross-dataset sections.

> If you edited any `cavi/*.py` code, restart the kernel before re-running (the
> setup cell already `importlib.reload`s the package, so a re-run usually suffices).

---

## 7. Optional: PyTorch / GNN backbone (for a real Q1 empirical run)

The current scripts use **BPR item factors** (CPU-only, no torch). The proposal's
stronger backbone is a **graph/hypergraph GNN** (LightGCN / DyHuCoG), which is
what a Q1 empirical paper would need to beat strong baselines *significantly*.
Your Mac M4's GPU (MPS) can run PyTorch. To add it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only (simplest)
# or for Apple MPS acceleration:
# pip install torch
# check: python -c "import torch; print(torch.backends.mps.is_available())"
```

Then you could implement a LightGCN-style encoder and plug it in where
`bpr_item_factors` currently sits (see the `recommender.py` interface). This is
the step that lets the forward/backward Shapley produce *large, significant*
ranking shifts — the current BPR backbone is the known bottleneck (interaction
reweighting moves rankings too little to reach significance).

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: numpy` | Activate venv: `source .venv/bin/activate`, `pip install -r requirements.txt` |
| `UnicodeDecodeError` on items | Already fixed in `cavi/data.py` (encoding-robust loader); **pull latest + restart kernel** |
| `ValueError: max() ... empty` | Means `load_items` returned 0 — check the file is the `::`-separated grouplens format (auto-detected now); pull latest |
| Kernel runs old code | Restart the kernel; setup cell does `importlib.reload` |
| `git push rejected` | The sandbox occasionally resets local refs; `git fetch origin && git reset --hard origin/arena/019fc350-next-paper` |
| Data download blocked | Place the `.dat`/`.txt` files manually (they're tracked in the repo) |

---

## 9. Where results go / what's what

- `paper-ideas/CAVI/results/synthetic_validation.json` — theory ground-truth recovery
- `paper-ideas/CAVI/results/ml1m_experiment.json` — end-to-end ML-1M
- `paper-ideas/CAVI/results/cross_*.json` — Amazon-Book / Yelp2018
- `paper-ideas/CAVI/results/paperA_divergence.json` — forward-vs-backward divergence
- `paper-ideas/CAVI/results/q1_v2_experiment.json` — non-degenerate CAV benchmark
- `paper-ideas/CAVI/results/q1_v3_action.json` — actionable-recourse benchmark

**Recommended order:** tests → synthetic → notebook → cross-dataset → q1 scripts.
Start small (`--users 30`), confirm, then scale.
