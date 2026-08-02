# Running the recommendation-only ActionShap pipeline

This guide runs the revised recommendation-only implementation. The pipeline is deterministic given the same data, configuration, and seed. The code has not been executed in this repository; run the commands locally and inspect the diagnostics before treating any output as a paper result.

## 1. Environment

From the repository root:

```bash
cd paper-ideas/ActionShap/code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS, install OpenMP first if the dependency installation requires it:

```bash
brew install libomp
```

If the full legacy dependency set is inconvenient, the recommendation implementation requires at minimum:

```bash
python -m pip install numpy pandas scipy scikit-learn tqdm
```

## 2. Prepare MovieLens-1M

Download MovieLens-1M from GroupLens and extract it locally. The required file is:

```text
ml-1m/ratings.dat
```

Do not edit the file. The loader:

- keeps ratings `>= 4`;
- sorts by `(timestamp, original_record_index)`;
- uses the last positive interaction as test;
- uses the second-last as validation;
- uses the remainder as training history;
- excludes users with fewer than three positive interactions.

## 3. Run the masking-sensitivity gate first

Open the single all-in-one notebook from the repository root:

```bash
cd /path/to/next-paper
jupyter notebook paper-ideas/ActionShap/code/ActionShap_All.ipynb
```

Use **Run All**. The notebook runs the masking gate first, then the primary recommendation experiments, a Monte Carlo convergence study, a 100-user full-catalogue robustness subset, and finally generates all paper assets under `paper/figures/`, `paper/tables/`, `paper/data/`, and `paper/manifests/`.

Run all cells. The blocking checks are:

- the history-conditioned profile model changes the top-10 list for at least 50% of sampled users;
- mean absolute NDCG change is at least `1e-3`;
- the static BPR-MF control is insensitive to profile masking;
- empty and full coalitions are deterministic;
- the Monte Carlo prefix walk has only numerical efficiency error;
- the AIA permutation null is near zero;
- leave-one-out is exactly the `B=1` oracle;
- the `B=2` joint comparison is non-degenerate.

If the masking gate fails, stop. Do not compensate by increasing the number of Monte Carlo permutations.

## 4. Run a small smoke experiment

From `paper-ideas/ActionShap/code`:

```bash
python scripts/run_recommendation.py \
  --ratings /absolute/path/to/ml-1m/ratings.dat \
  --output results/raw/movielens_smoke.json \
  --max-users 100 \
  --oracle-users 10 \
  --epochs 2 \
  --permutations 25 \
  --candidate-k 200
```

This is only a pipeline check. Do not use it for the paper.

Inspect:

```bash
cat results/raw/movielens_smoke.json
```

Confirm that:

- candidate recall is reported;
- evaluated users have their test item in the fixed candidate set;
- AIA values are finite for non-constant users;
- AIA null means are near zero;
- `B=1` is not presented as the main method comparison;
- joint `B=2` effects and regrets are present;
- efficiency error is near floating-point precision but is not interpreted as convergence evidence.

## 5. Run the primary experiment

The primary scientific comparison is joint intervention budget `B=2` with feasible downweighting `rho=0.5`. Deletion `rho=0` is retained only as the faithfulness diagnostic; `B=1` leave-one-out is an oracle for that deletion condition and is not the main comparison.

```bash
python scripts/run_recommendation.py \
  --ratings /absolute/path/to/ml-1m/ratings.dat \
  --output results/raw/movielens_actionshap_seed42.json \
  --n-max 50 \
  --candidate-k 200 \
  --embedding-dim 64 \
  --epochs 10 \
  --permutations 250 \
  --seed 42 \
  --oracle-users 100
```

Repeat with at least five seeds:

```text
42, 43, 44, 45, 46
```

Use identical data splits and candidate-generation settings for every seed. The current runner writes per-user results and aggregate summaries; do not average raw coalition evaluations as though they were independent users.

## 6. Convergence experiment

The required convergence comparison is against a high-permutation reference, not against efficiency error. Use the `convergence_table` function from `actionshap.convergence` on a saved representative user game or add a small driver script locally.

Required permutation values:

```text
25, 50, 100, 250, 500, 1000
```

Required checks:

- mean rank correlation with the 1000-permutation reference;
- standard deviation across at least five seeds;
- top-1 and top-2 action agreement;
- efficiency error as a numerical-stability diagnostic only.

Choose the smallest permutation count satisfying the predeclared convergence thresholds in `ActionShap_Recommendation_Spec.md`.

## 7. Interpreting the output

The output JSON contains:

- dataset size and candidate recall;
- per-method AIA;
- within-user AIA null means;
- joint `B=2` intervention effects;
- joint intervention regret on the exhaustive-oracle subset;
- per-user selected actions;
- prefix-walk efficiency errors.

The methods are:

- `shapley_mc`: Monte Carlo Shapley attribution;
- `loo_oracle`: single-player deletion oracle, not a fair B=1 competitor;
- `lime`: binary local surrogate over history masks.

The current implementation uses profile masking/downweighting at inference time and does not retrain after each intervention, as required by the specification.

## 8. Required reporting before paper claims

Before writing conclusions, add or verify:

1. five-seed paired confidence intervals;
2. within-user AIA permutation p-values;
3. candidate recall and the number of excluded test users;
4. `n_max` sensitivity for `20`, `50`, and `100`;
5. candidate-size sensitivity if candidate recall is low;
6. `B=1` oracle sanity results;
7. `B=2` main comparison;
8. restricted exhaustive-versus-greedy validation for `B=3` if `B=3` is reported;
9. synthetic redundancy validation;
10. leakage audit confirming that validation/test interactions never enter training histories or attributions.

Do not claim that ActionShap is causal. The paper evaluates declared offline interventions under a frozen model and fixed candidate set.

## 9. Common failure modes

### All Shapley values are zero

The model is probably static with respect to history. Re-run the masking gate. BPR-MF and LightGCN user embeddings cannot be post-hoc masked unless the model explicitly recomputes the user representation from the retained history.

### Candidate recall is low

Increase `--candidate-k` only after reporting the original value. Do not insert the test item artificially, because that invalidates the recall diagnostic.

### AIA is NaN

A user's attribution or intervention-effect vector is constant. Record how many users are affected; do not silently replace NaN with zero.

### Efficiency error is zero

This is expected for the prefix-walk estimator because marginal contributions telescope. It does not demonstrate Monte Carlo convergence.

### Leave-one-out is perfect

This is expected at `B=1`. It is the diagnostic oracle. The scientific comparison must use joint interventions at `B=2` or higher.

### Results change between runs

Check the seed, stable temporal tie-break, candidate tie-break, item-index mapping, and fixed candidate-set cache.
