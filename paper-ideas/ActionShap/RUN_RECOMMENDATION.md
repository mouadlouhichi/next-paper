# Running the corrected recommendation-only ActionShap pipeline

This guide implements revision 4 of `ActionShap_Recommendation_Spec.md`.
Schema-v1 pilot outputs are archived under `paper/legacy_pilot/` and cannot be
used as final-paper evidence.

## 1. Environment

From `paper-ideas/ActionShap/code`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-recommendation.lock
pytest -q
```

`requirements-recommendation.lock` is the exact tested environment. Use
`requirements-recommendation.txt` only when a platform cannot resolve a locked
wheel, and record the resolved versions in the result manifest.

```bash
# compatibility fallback only
python -m pip install -r requirements-recommendation.txt
```

## 2. Run-all notebook

Open `ActionShap_All.ipynb` from the repository root or this `code/` directory
and choose **Run All**. Its configuration cell defaults to the complete final
workflow: dependency installation, terms-confirmed downloads, data audit,
85 scientific commands, asset generation, manuscript checks, and release
packaging. Set the booleans in that cell only when intentionally reusing data or
existing raw outputs.

## 3. Data

The canonical notebook performs both downloads and preparation automatically.
From the command line, the equivalent one-command setup is:

```bash
python scripts/download_datasets.py --dataset all --accept-dataset-terms
```

Pass the acceptance flag only after reviewing the GroupLens and Amazon source
terms and citation requirements. Existing valid files are reused; `--force`
redownloads and rebuilds them.

### MovieLens-1M

Download MovieLens-1M from GroupLens and place the unedited file at:

```text
code/data/ml-1m/ratings.dat
```

The loader keeps ratings `>=4`, sorts each user by
`(timestamp, original_record_index)`, and reserves the final two positive events
for validation and test.

### Amazon Digital Music secondary dataset

Download the unmodified `Digital_Music_5.json.gz` file from the Amazon Review
Data (2018) release, retain its license/readme, and build the analysis CSV:

```bash
python scripts/prepare_amazon_digital_music.py \
  --input /absolute/path/to/Digital_Music_5.json.gz \
  --output data/amazon-digital-music/interactions.csv
```

The builder retains ratings `>=4`, resolves duplicate user-item reviews by the
latest `(timestamp, original_record_index)`, reapplies an iterative 5-core after
thresholding, and writes a SHA-256 provenance sidecar. Do not edit the generated
CSV. The final asset validator requires MovieLens and Amazon Digital Music.

## 4. Smoke test

A smoke test is explicitly ineligible for paper claims. It may skip the
200-user gate only to verify plumbing:

```bash
python scripts/run_recommendation.py \
  --ratings data/ml-1m/ratings.dat \
  --output results/raw/smoke_seed42.json \
  --max-users 25 \
  --oracle-users 25 \
  --n-max 20 \
  --permutations 10 \
  --null-draws 50 \
  --lime-samples 64 \
  --gate-evaluation-size 200 \
  --epochs 1 \
  --skip-gate
```

The output status must be `smoke_only`; the paper asset generator rejects it.

## 5. Final suite

Inspect `configs/final.yaml`, supply both data files, and run:

```bash
python scripts/run_final_suite.py --config configs/final.yaml
```

Use `--dry-run` to inspect every command without executing it.

The suite performs the operations in a fixed order:

1. independent convergence studies for every dataset--model pair using an
   `M=1000` reference;
2. automatic selection of the minimum usable Monte Carlo budget;
3. five seeded sampled-ranking runs for the profile and ItemKNN models;
4. five seeded full-unseen-catalogue robustness runs;
5. predeclared history-cap, intervention-strength, candidate-size, and budget sensitivities;
6. schema validation, distinct-user hierarchical inference, figures, tables,
   and provenance manifests.

The primary game uses the most recent 20 interactions and ItemKNN; the latent
profile aggregator is architecture robustness. Candidate, user, and tie-break
seeds are fixed independently of experiment randomness, so every seed sees the
same users and candidates; full-catalogue and sensitivity cohorts are matched
subsets of the primary 1,000-user cohort.

## 6. Blocking real-data masking gate

Every paper-eligible run masks one interaction for at least 200 real users. It
must satisfy:

- top-10 changes for at least 50% of users;
- mean absolute NDCG@10 change is at least `1e-3`;
- the frozen-score control changes exactly zero outputs.

A failed primary-ItemKNN gate stops the run. A declared robustness-model or
history-length sensitivity may continue only so the failure is reported as a
non-responsiveness boundary. `--skip-gate` marks the result `smoke_only` and is
not a way around the primary requirement.

## 7. Candidate semantics

The primary evaluation set contains the temporal target plus uniformly sampled
negatives after excluding the user's **complete pre-test history**, including
validation. Target coverage is one by construction. This is sampled ranking,
not retrieval, and no candidate-recall claim is made.

The robustness mode uses the full unseen catalogue plus the target. Seen
training or validation items are never inserted as negatives.

## 8. Utility and interventions

The primary attribution utility is continuous target margin, selected by the
archived convergence preflight. NDCG@10 remains the operational action outcome:
every action receives separate target-margin and NDCG effects, exact oracles,
and regrets. An NDCG-attribution sensitivity is run at `M=1000` even when its
rank/action convergence thresholds fail. Target-margin values must never be
called NDCG.

The feasible action is interaction downweighting at `rho=0.5`. Signed
attributions predict downweight benefit as `-phi`. The action space includes:

- no action;
- all single-player actions;
- all pairs at the primary budget `B=2`.

The exact oracle evaluates this complete space for every primary user. A method
abstains when it predicts no positive-benefit action. Leave-one-out is labelled
an oracle only for the `B=1`, `rho=0` deletion identity.

## 9. Statistical reporting

`make_paper_assets.py` averages repeated seeds within each distinct user before
inference. It generates:

- user-bootstrap confidence intervals;
- paired user-level sign-permutation tests with plus-one p-values;
- Holm--Bonferroni corrections;
- paired Cohen effect sizes;
- success and abstention rates;
- missing constant-vector AIA counts;
- undefined normalized-regret counts;
- attribution rank stability across seeds.

Five seeds over 1,000 users are never described as 5,000 independent users.

## 10. Final validation

Inspect:

```text
paper/final/manifests/validation_report.json
```

Only `status: PASS` permits numerical manuscript claims. Then run
`python scripts/validate_manuscript.py --require-final` after replacing every
result placeholder. The validators block or flag:

- fewer than five common seeds;
- different users or candidate seeds across runs;
- failed real-data gates;
- fewer than the required users;
- missing dataset/model robustness;
- missing or inadequate convergence;
- a primary permutation count below the selected value;
- smoke or legacy schemas.

## 11. Raw-result provenance

Raw JSON is intentionally excluded from ordinary Git history because complete
per-user attribution records can be large. The final manifest records
repository-relative source paths, byte sizes, and SHA-256 hashes. Package raw
files for the archival release and publish the archive DOI with the manuscript;
do not commit machine-specific absolute paths.

## 12. Common failures

### AIA is missing

The relevant intervention-effect vector is constant for that user. This is most
common for cross-utility NDCG AIA and the NDCG-attribution sensitivity. Report
the missing count; do not replace the correlation with zero.

### A selected action is empty

This is a valid abstention. The old pipeline incorrectly forced exactly two
actions.

### Normalized regret is undefined

The oracle found no positive action. Report the count and retain unnormalized
regret; do not divide by zero.

### Efficiency error is nearly zero

This follows from prefix-walk telescoping and says nothing about convergence.
Use the independent rank/action convergence study.

### Final asset validation fails

Do not edit the report. Complete or rerun the missing experiment group.
