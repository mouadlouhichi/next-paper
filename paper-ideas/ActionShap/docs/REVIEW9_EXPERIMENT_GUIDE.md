# Review-9 (KBS-style peer review of the ACM TORS manuscript) experiment guide — runs on your machine

> **Preferred path: the notebook.** `../notebooks/REVIEW9_REPLICATION_RUNS.ipynb`
> (rebuilt by `code/scripts/make_review9_notebook.py`) does everything below in order:
> checks the data paths, pilots all seven experiments at the real budget to measure cost on
> your machine, launches a resumable detached queue into `code/results/review9/`, then
> regenerates the tables, syncs `\resultmanifeststamp` in both documents and runs every
> validator. The per-experiment sections below document what each run is *for* and what it
> must contain; the notebook is how to run it. If you prefer the shell, the commands are
> identical to what the queue emits (`code/results/_review9_scratch/run_review9_queue.sh`).

> **Status after the follow-up revision round.** `scripts/run_review9_experiments.py`
> now also accepts `--dataset gowalla`, and the Gowalla copy of the data is in the
> repository, so the following have already been executed and integrated without
> waiting for the datasets machine:
>
> * **R9-1 fixed-denominator ablation** on Gowalla (600 sampled users,
>   `M_pair=250`, rho=0.5) -> `code/results/review9/fixed_denominator_gowalla.json`,
>   rendered as Supplementary Table `tab:r9-fixed-denominator`.
> * **R9-2 utility factorial** and **R9-5 stratified nulls** on Gowalla (250 users),
>   when those runs complete; the MovieLens/Amazon instances below are still needed
>   for the primary cohort.
>
> The items that need no new runs at all were closed from the frozen release
> matrices instead (`code/scripts/make_review9_stats.py` -> Supplementary
> Section S11): inclusion flow + all-user sensitivity (#10), the confirmatory
> multiplicity map with raw exceedance counts plus the regenerated Table S4 (#18),
> the studentized robustness of the sign-flip tests (#15), the
> attribution-utility x outcome-utility factorial on the primary cohort (#4), and
> the Monte Carlo propagation into the alignment statistic (#9).
>
> **One non-experiment action is required before submission:** the two committed
> PDFs predate the revised sources. Rebuild with `make -C <repo root> pdf`
> (needs `texlive` + `latexmk`) and then `make check`, which fails while the PDFs
> lag behind their sources.

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
# already run in-repo on the third dataset (results are committed):
python scripts/run_review9_experiments.py fixed-denominator --dataset gowalla   --users 600  --permutations 250
# still needed for the primary cohort:
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

Push the JSONs under `code/results/review9/`, then regenerate and validate:

```bash
make -C <repo root> stats    # tables (from the matrices + review9 JSONs) + manifest hash
make -C <repo root> pdf      # rebuild both PDFs against the new tables
make -C <repo root> check    # validators + full test suite
make -C <repo root> artifact # build the deposit archive + its sha256 (see below)
```

`make stats` runs `make_review9_stats.py`, which writes every review-9 table to
BOTH table mirrors byte-identically, regenerates Table S4 from
`release/paired_tests.csv`, regenerates `tables/review5_validation.tex` into both
mirrors, and refreshes `code/results/manifest.json`; both documents must then quote
the new `\resultmanifeststamp` (the test suite fails otherwise). Cell 8 of the
notebook does the stamp sync for you.

### Running it on a workstation instead of the sandbox

The notebook's cost model is measured, not assumed: cell 5 runs all seven experiments
at the *real* permutation budget on a 12-user cohort, prints minutes per 1000 users,
and cell 6 rewrites its estimates from those numbers (the `hardware` job is
cohort-independent and is kept as absolute minutes). Reference point from the 2-core
sandbox, at $M_{\mathrm{pair}}=250$:

| experiment | min per 1000 users | what it is per |
|---|---|---|
| `fixed-denominator` | ~53 | cohort |
| `prospective` | ~27 | cohort |
| `stratified-null` | ~25 | cohort |
| `utility-factorial` | ~44 | cohort |
| `compute-matched` | ~65 | *grid point* (two points $\approx 1+4\times$ cost) |
| `candidate-redraw` | ~50 | *redraw* (six redraws $\approx 6\times$) |
| `hardware` | ~0.3 | fixed |

Read the pilot line for your own machine before deciding on `USERS` and `SCALE`: the
per-user work is dominated by the permutation walk, so a many-core machine mainly wins
through BLAS in the scorer, not through parallelism in this code (runs are serialised
per user by design, since the published numbers must be reproducible from one seed).
Two practical knobs: set `ONLY = ["compute-matched", "candidate-redraw"]` to queue only
the runs the sandbox could not finish, and leave `SCALE = 1.0` for the publishable
cohorts (`SCALE = 0.3` is a legitimate pre-flight check, but the resulting table rows
must be labelled with their smaller $n$). The queue is resumable: a job whose output
JSON already exists is skipped, so it is safe to stop and restart it.

### Building the deposit (issues 16/17)

`make artifact` calls `code/scripts/package_results.py`, which tarballs the
schema-v2 raw runs under `raw/` **and** the generated half of the artifact --
`results/review9/*.json`, `results/manifest.json`, both review-9 table mirrors, the
generators and validators, the integrity test, this notebook and the two planning
documents -- under the matching directory names, writes
`release_checksums.json` (path, bytes, sha256 of every member) inside the archive,
and leaves `<archive>.sha256` next to it. Quote that hash in the cover letter and in
the data-availability statements. `--allow-no-raw` builds the review-9 addendum alone
(for a versioned deposit where the raw archive already has a DOI); `--no-derived`
reproduces the old raw-only behaviour. `results/release/*.tar.gz` is ignored by the
manifest stamp, so building an archive never invalidates the documents.

Completed after the runs land:
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
## Before the submission button: `make ready`

`make check` verifies the sources; it can pass while the PDFs a reviewer will read are the
previous build, which is exactly what happened after review 9. `make ready` answers the
submission question from the artefacts: one compiled copy per document at its canonical
path, built no earlier than its sources, page 1 anonymity matching the class options, the
result-manifest stamp and the review-9 panels present in the text, no placeholder
sentence left in either PDF, and a deposit whose every recorded checksum recomputes. It
exits non-zero and prints the blockers, so run it after `make pdf` and `make artifact` and
before uploading anything. Install `pypdf` (`pip install pypdf`) in the interpreter that runs the gate,
or it refuses to pass rather than quietly skipping the checks that matter most.
## Compiling the review copy on Overleaf: `make overleaf`

No TeX distribution is available in the sandbox (`apt` and CTAN are unreachable, so the
build stays an owner-machine step), so the submission is compiled on Overleaf from the
zip `make overleaf` writes to `code/results/release/build/actionshap-overleaf.zip`
(git-ignored). It is generated rather than hand-picked: the packer walks the same include
graph TeX does (`\safeinput`, `\includegraphics` with `\graphicspath`, `\bibliography`,
`\bibliographystyle`), refuses to pack when a reference does not resolve, then re-resolves
every reference *inside the zip* before returning --- a missing asset in a review build
shows up as "[Missing table asset]", which `\safeinput` downgrades to a warning, so the
only way to catch it is to check the graph. The compiled PDFs and the `.bbl` are excluded
deliberately: an old build beside the sources looks exactly like a new one, and a
present-but-stale `.bbl` lets latexmk skip BibTeX (the repository copy trails the current
text by ten citations). `README-OVERLEAF.txt` inside the zip records the commit and the
result-manifest stamp the project was generated from, so a compiled PDF can be tied to a
state of the tree; the test suite pins the self-containedness check.
