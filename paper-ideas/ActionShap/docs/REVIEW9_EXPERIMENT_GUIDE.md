# Review-9 (KBS-style peer review of the ACM TORS manuscript) experiment guide — runs on your machine

The paper-side revisions are already integrated on the branch:

* **Critical #1 (construct):** the normalized intervention is now named and derived
  as *relative profile reweighting* (coefficient-change derivation, uniform-scale
  invariance, and the rho-factorization that explains the flat Amazon ItemKNN
  rho-response). A **fixed-denominator pure-suppression** scorer
  (`FixedDenominatorItemKNN`) is implemented and unit-tested for the ablation below.
* **Critical #3:** the Amazon full-catalogue negative Shapley AIA is promoted into
  the abstract, robustness section, and conclusion.
* **Critical #6:** "executable/actionable" is tightened to "simulator-executable"
  throughout the claim sites.
* **High #7/#15:** the BPR objective is rewritten as a per-triple loss with exact
  gradients + clipping semantics; the collision-prone integer-offset random-control
  seed is replaced by a tuple `SeedSequence` derivation (also applied to LIME masks,
  the MC Shapley stream, and the within-user null stream), and the manuscript text
  now describes it.
* **High #10/#17/#18:** every flagged statistic is relabeled by utility and
  denominator (TOST n=1000, gap-vs-regret uses target-margin normalized regret over
  positive-oracle users, S15/S22/S25/S28 captions corrected); the MDE generator bug
  (pooled ItemKNN+profile models) is fixed so S17 now reports 0.014/0.051 matching
  the main text; the confirmatory multiplicity map and the 0.0010 permutation-floor
  are documented.
* **High #14:** the Spearman-AIA invariance claim is corrected; a direct
  utility-matched bounded-AIA vs decision-quality association is computed from the
  existing matrices and added to the hypothesis-adjudication section.
* **Phase 11:** eight missing references added (Yeh 2019, Slack 2020, Kumar 2020,
  Jeyakumar 2020, Balog & Radlinski 2020, Gedikli 2014, Tintarev & Masthoff 2015,
  GREASE/Chen 2025) and cited.

The remaining **mandatory** items need compute on the datasets (MovieLens-1M,
Amazon Digital Music) on your machine. Everything below uses the new runner
`scripts/run_review9_experiments.py` (added on this branch, smoke-tested). Run from
`code/`.

```bash
cd /Users/<you>/.../next-paper/paper-ideas/ActionShap/code
source ../.venv/bin/activate   # or the env you normally use
```

---

## R9-1. Fixed-denominator (pure suppression) vs normalized reweighting — Critical #1

Directly tests whether the deletion-vs-bounded contrast is driven by the
normalization (profile-mass reallocation) rather than suppression.

```bash
python scripts/run_review9_experiments.py fixed-denominator --dataset movielens --users 1000 --permutations 250
python scripts/run_review9_experiments.py fixed-denominator --dataset amazon    --users 1000 --permutations 250
```
Writes `results/review9/fixed_denominator_<dataset>.json`: per-user bounded/deletion
AIA for Shapley/LIME/LOO under both scorers. **Expected key question to answer:** does
Shapley's bounded AIA gap vs LIME/LOO persist under fixed-denominator suppression?
Full cohort at M_pair=250 ≈ a few minutes per dataset.

## R9-2. Utility-matched factorial (attribution utility x outcome utility) — Critical #4

Isolates utility mismatch from nonadditivity in the H2 adjudication.

```bash
python scripts/run_review9_experiments.py utility-factorial --dataset movielens --users 1000 --permutations 250
python scripts/run_review9_experiments.py utility-factorial --dataset amazon    --users 1000 --permutations 250
```
Writes `results/review9/utility_factorial_<dataset>.json`: 2x2 cells
(target_margin/ndcg attribution x target_margin/ndcg outcome) with matched vs cross
AIA, realized effect, and regret. The exact-B2 oracle is re-evaluated per outcome
utility, so this is the heavier cell (~10-20 min per dataset at 1000 users).

## R9-3. Prospective (non-target-conditioned) full-cohort audit — Critical #5

Upgrades the prospective audit from the 126-227 selected users to the full cohort.

```bash
python scripts/run_review9_experiments.py prospective --dataset movielens --users 1000 --permutations 250
python scripts/run_review9_experiments.py prospective --dataset amazon    --users 1000 --permutations 250
```
Writes `results/review9/prospective_<dataset>.json`: bounded AIA for Shapley/LIME/LOO
auditing the model's own top-1 recommendation (no held-out target), plus coverage
(what fraction of users' generated top-1 lies in the candidate set) and whether the
prospective target equals the held-out target. This becomes a co-primary table.

## R9-4. Candidate-set resamples — High #8

Quantifies dependence on the single frozen candidate sample.

```bash
python scripts/run_review9_experiments.py candidate-redraw --dataset movielens --users 200 --permutations 125 --redraws 20
python scripts/run_review9_experiments.py candidate-redraw --dataset amazon    --users 200 --permutations 125 --redraws 20
```
Writes `results/review9/candidate_redraw_<dataset>.json`: 20 independent candidate
resamples x per-user Shapley/LIME/LOO bounded AIA, with between-resample variability.
Kept at 200 users / M_pair=125 for tractability (~10-25 min per dataset). Raise
`--users`/`--permutations` if you have time.

## R9-5. Stratified within-user nulls — High #12

Tests whether the free-shuffle AIA null is robust to recency/popularity structure.

```bash
python scripts/run_review9_experiments.py stratified-null --dataset movielens --users 500 --permutations 125 --r-null 1000
python scripts/run_review9_experiments.py stratified-null --dataset amazon    --users 500 --permutations 125 --r-null 1000
```
Writes `results/review9/stratified_null_<dataset>.json`: observed vs null AIA under
free, recency-block, and popularity-block shuffles with plus-one p-values.

## R9-6. Compute-matched budget-response curves — High #13

Replaces the mislabeled "equal-scorer-budget" table with genuinely matched
scorer-call points (Shapley prefix states vs LIME masks at the same call count).

```bash
python scripts/run_review9_experiments.py compute-matched --dataset movielens --users 200 --mpair-grid 25 50 100 250
python scripts/run_review9_experiments.py compute-matched --dataset amazon    --users 200 --mpair-grid 25 50 100 250
```
Writes `results/review9/compute_matched_<dataset>.json`.

## R9-7. Hardware + repeated timing — reproducibility

```bash
python scripts/run_review9_experiments.py hardware --dataset movielens --timing-repeats 3 --timing-users 20
```
Writes `results/review9/hardware_<dataset>.json`: platform/processor/cpu count, peak
RSS, and repeated per-method timing medians. (Run once per dataset; MovieLens is
enough if you prefer.)

---

## After running

Push the JSONs under `code/results/review9/` and ping me ("see last commit"). I will
then:
1. integrate them into the main manuscript and supplement as new tables/sections,
2. add the fixed-denominator, utility-factorial, prospective-full, candidate-redraw,
   stratified-null, and compute-matched results to the hypothesis-adjudication and
   robustness narrative,
3. reconcile any claim the new numbers change (e.g., whether the Shapley-vs-LIME gap
   survives pure suppression).

## Optional (if compute allows)

* **Competitive neural/graph recommender passing the quality gate (Critical #2).** This
  is the single hardest outstanding item and is an *engineering* constraint, not a
  tuning one: SASRec and LightGCN both underperform popularity because the protocol's
  inference-time history-weighting interface breaks the trained model. A real fix needs
  a model trained to be robust to bounded weighting at inference (e.g., history-dropout
  / weight-augmented training, or a scorer whose user vector is a learned weighted
  aggregation). This is a separate workstream; say the word and I'll draft the
  training-side design.
* **Immutable artifact.** The cover letter and manuscript still carry a placeholder
  artifact URL. Deposit the code + release matrices + raw JSONs to Zenodo/OSF/Figshare
  and send me the DOI/URL so I can fill the placeholder before submission.
