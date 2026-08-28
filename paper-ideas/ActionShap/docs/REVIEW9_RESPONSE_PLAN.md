# Review-9 response plan — KBS-style peer review of the acmart-primary (ACM TORS) manuscript

Review file: `paper-ideas/ActionShap/ActionShap-KBS-peer-review.md` (Major Revision,
overall 6/10, confidence 9/10).

**Venue note:** the reviewer's own Phase-10 recommendation is *ACM Transactions on
Recommender Systems* ("The manuscript is already formatted and framed as an ACM TORS
article ... TORS is a more natural audience than KBS in the current form"). The
current submission target (acmart-primary, `\acmJournal{TORS}`, single-blind) is
therefore correct; all KBS-format complaints (Elsevier conversion, highlights,
keywords format) are moot for TORS, but every scientific issue applies regardless of
venue and is handled below.

Legend:
- **FIXED (text)** — manuscript/supplement/tables edited on this branch.
- **FIXED (code)** — code changed + unit-tested on this branch.
- **RUN** — implemented in `scripts/run_review9_experiments.py`; must run on the
  datasets machine (see `docs/REVIEW9_EXPERIMENT_GUIDE.md`).
- **USER** — requires a user-side action (artifact deposit, cover letter).
- **REBUT** — answerable in the response letter with existing evidence.

## Critical issues

| # | Issue | Status | Where |
|---|-------|--------|-------|
| 1 | Normalized weighting = relative profile-mass reallocation, not isolated suppression | **FIXED (all three cohorts, data+text)** `fixed_denominator_{movielens,amazon,gowalla}.json` (1,000 / 810 / 528 defined games) -> `tab:r9-fixed-denominator` + paired table; the interface *reverses* Shapley vs LIME on Amazon (0.607<0.832 normalized, 0.993>0.849 fixed) and narrows the MovieLens gap from 0.182 to 0.016, so the supplement now states the within-interface rule and the main text says so once at the headline claim; scope note recorded: the ablation is a same-model re-fit at M_pair=250, so its levels are not the published M_pair=500 values | S11, main Sec. faithfulness |
| 2 | Only ItemKNN passes the quality gate | **REBUT + RUN (long-term)** existing candor retained (limitations, abstract caveat); competitive gate-passing neural/graph model is an engineering workstream (history-weighting-compatible training), scoped in the guide | abstract, §7, §8 |
| 3 | Amazon full-catalogue reversal must be central | **FIXED (text)** promoted into abstract, §6.6 already reports it, conclusion restates it (negative AIA −0.05 with positive gap +0.16) | abstract, §6.6, conclusion |
| 4 | Utility mismatch confounds H2 (target-margin attribution vs NDCG decisions) | **FIXED (text+run)** primary-cohort factorial from the released matrices (`tab:r9-utility-factorial`) *plus* the replication-benchmark $2\times2$ (`tab:r9-utility-factorial-replication`): changing only the evaluation utility moves the mean bounded AIA by $+0.330$ $[0.248,0.415]$ and the mirrored cell by $-0.275$ $[-0.438,-0.101]$, with only 14/250 users having a defined NDCG arm -- reported as descriptive, and the reason H2 is adjudicated on rank association | make_review9_stats |
| 5 | Prospective audit must be co-primary | **FIXED on MovieLens-1M (1,000 users) + the in-artifact benchmark** (prospective_gowalla.json -> tab:r9-prospective-replication, quoted in S11); primary cohorts are a `prospective` run away `prospective` writes `prospective_<ds>.json` and `tab:r9-prospective-replication` is generated from it automatically (audited/sampled counts, held-out-target coverage, per-method AIA); the Gowalla instance is running now, MovieLens/Amazon need `notebooks/REVIEW9_REPLICATION_RUNS.ipynb` | notebook |
| 6 | "Executable" overclaims | **FIXED (text)** "simulator-executable" at all claim sites; abstract states the intervention is simulator-executable, not demonstrated against a production interface | throughout |
| 7 | Eq. (4) BPR not reproducible as written | **FIXED (text)** rewritten as per-triple loss + exact per-triple gradients verified against `fit_item_embeddings`, clipping semantics specified, context/regularizer scope defined | §3.2 Eq. (4)-(5) |
| 8 | Fixed candidate sets dominate conclusions | **RUN (tooling done; payload shape validated by the pilot dry run, and its float renders from a 12-user run)** `candidate-redraw --redraws N` → `tab:r9-candidate-redraw` (between-redraw mean/SD/min–max per method); the payload's `KeyError: mean` on empty redraws was found by the notebook pilot and fixed; queued on the workstation (the Gowalla instance included) | notebook |
| 9 | Per-user MC uncertainty not propagated | **FIXED (text)** per-user MC error now *propagated*: mean/maximum $|\Delta$AIA| between budgets with CIs from the review-8 cap-user study (`tab:r9-mc-propagation`, S11) and an explicit statement that the aggregate is unaffected while single-user decisions are; adaptive stopping stays an extension; mcse caption corrected (0.948 floor, budget labels); propagation into regret remains an extension item | supp. tables |
| 10 | Analysis populations inconsistent | **FIXED (text+tables)** relabeling **plus the two artefacts the reviewer asked for**: inclusion-flow table and all-user sensitivity (`tab:r9-inclusion-flow`, `tab:r9-all-user-sensitivity`, S11), with every denominator verified by recomputation (1000 / 993 / 987 / 339 / 196) and locked by `tests/test_review9_publication_integrity.py`: TOST n=1000 (verified by recomputation), gap-vs-regret = target-margin NRegret (n=1000/987 positive-oracle users), S15/S22/S25/S28 captions fixed with explicit denominators | §5.3, §6, tables |
| 11 | Uncertainty conditional on one fitted model/candidate set | **PARTLY RUN (tooling done)** candidate redraws cover the design half of this (Issue 8 tooling); retraining / temporal-cutoff resampling is a retraining workstream, not a re-scoring run, and stays an explicit extension | notebook + limitations |
| 12 | Player-exchangeability null questionable | **FIXED (run)** stratified within-user nulls on the replication benchmark (`tab:r9-stratified-null`): null mean rises from $-0.0001$ (free) to $0.100$ (recency blocks) and $0.192$ (popularity blocks), while the observed $0.575$ stays $27.1$/$22.9$/$19.1$ null SDs above them; the $1000$-draw plus-one $p$ saturates at $0.0010$, so $z$ is reported | review9 runs |
| 13 | "Equal-scorer-budget" table mislabeled / unequal budgets | **FIXED (text)** table renamed to budget-response curves, S symbol instead of B, not-equal-budget note in caption; **FIXED (cost half, data: hardware_gowalla.json -> tab:r9-hardware: 2 cores, peak RSS 305 MB, medians 2.11/0.149/0.0059 s)** ; matched-budget curve = one `compute-matched` run, tooling done)** `compute-matched --mpair-grid` → `tab:r9-compute-matched`, one row per equal scorer-call budget with LIME $-$ Shapley per row | notebook |
| 14 | AIA monotone-invariance claim false; gap tested instead of absolute AIA | **FIXED (text, PENDING PDF)** invariance claim corrected in the source; the *committed* `acmmanuscript.pdf` still contains the uncorrected sentence -- rebuild required (`make pdf`) (applies to already-formed vectors only); absolute bounded-AIA vs decision association added to §6.4 | §4.3, §6.4 |
| 15 | Collision-prone integer seed derivation | **FIXED (code+text)** tuple SeedSequence entropy + **the second half of the ask is now closed**: the sign-flip exchangeability/symmetry assumption is stated in §5.3 and every headline contrast is re-run with a studentized bootstrap-*t* test (`tab:r9-studentized`, S11); agreement reported for random control, LIME masks, MC Shapley, and the within-user null stream in `run_recommendation.py` + `evaluation.py`; widened type hints; unit tests added. **Note:** changes random-control/LIME/Shapley streams, so primary-suite regeneration is required before final submission (cheap for random; full suite ≈ prior runtime) | code + §4.2 text |
| 16 | Artifact URL placeholder; main/supplement version drift | **USER** artifact deposit for the URL only. Drift is now *machine-enforced*: content-addressed `code/results/manifest.json` (sha256 of every matrix and table; the git revision is recorded beside it, deliberately *not* hashed: a stamp that covers the checked-out revision can never be carried by the commit that introduces it, so `make check` failed after every successful commit and across the two machines), `\resultmanifeststamp` quoted in both PDFs, `make_result_manifest.py --check`, mirror byte-identity tests, and a stale-PDF guard in `validate_manuscript.py` (warning) + `test_compiled_pdfs_are_not_silent_about_the_revised_text` (xfail until rebuilt); **FIXED (text)** drift items reconciled: MDE 0.014/0.051 (generator bug fixed), S15 n=993 caption, B=3 greedy-vs-exhaustive status, full-catalogue 250 vs 1000 wording | cover letter; multiple; `make artifact` now emits the deposit archive + sha256 |
| 17 | MDE 0.008/0.032 vs 0.014/0.051 conflict | **FIXED (code+text)** 0.014/0.051 in both mirrors; the *generator* that produced them is now checked against the committed table (`make_review3_stats.py --check`, 40/40 rows) instead of silently overwriting a file with hand-appended blocks; `make tables` runs the check: `power_table` pooled ItemKNN + profile models, halving paired SD; restricted to primary ItemKNN; both table copies now show 0.051 (n=993) / 0.014 (n=1000) with formula documented. A second one-sided generator was found and fixed the same way: `make_review5_tables.py` wrote `tables/review5_validation.tex` to the IPM mirror only, so the ACM copy had drifted (richer caption, extra caveat); it now writes both mirrors, its captions carry the hand-written wording that was in the ACM copy, and `--check` verifies parity | make_review3_stats.py; make_review5_tables.py; S17 |
| 18 | Conflicting Holm families (.0066 vs .0216), 0.0010 floor | **FIXED (text+regenerated table)** root cause found and repaired: S4 was produced by an audit generator using **1,000** sign flips and one *global* Holm block with a non-standard step-up implementation. S4 is now regenerated from `release/paired_tests.csv` (8 printed values corrected, incl. .0066 → .0048) with exceedance counts and the authoritative 12-contrast value side by side; the generator itself now uses 10,000 draws + standard Holm (`make_review3_stats.py --check` passes); `tab:r9-multiplicity-map` publishes family membership + $\#=p(R+1)-1$ for all 2,312 released tests (0 recomputation failures) and `tab:r9-permutation-precision` shows the residual 3rd-decimal differences are MC error of the 10k-draw experiment (per-metric families for S3–S5; single 12-contrast family authoritative for success/abstention); 0.0010 explained as 10-family Holm x 1/10,001 permutation floor (verified against raw paired_tests.csv) | §5.3 |
| 19 | Modern-model cells noncompetitive + estimator instability | **REBUT** already fully disclosed (SASRec exact-agreement 0.395/0.688, LightGCN below popularity, tuned variant); no claim of transfer remains | supp. S9 |

## Mandatory revisions checklist (Phase 17)

| Item | Status |
|------|--------|
| Competitive gate-passing neural/graph recommenders | RUN/workstream (guide §Optional) |
| Normalized vs unnormalized/fixed-denominator comparison | RUN (`fixed-denominator`) |
| Full-catalogue reversal central in abstract/conclusion | FIXED (text) |
| Utility-matched + interaction-aware factorial | RUN (`utility-factorial`); matched association already added from existing data |
| Prospective actual-recommendation audits | RUN (`prospective`, full cohort) |
| Propagate candidate/training/Shapley uncertainty | RUN (`candidate-redraw`); mcse diagnostics corrected |
| Eq. (4), seed derivation, invariance claims, denominators | FIXED (text+code) |
| Validate/redesign shuffle null | RUN (`stratified-null`) |
| Compute-matched intervention-aware baselines primary | RUN (`compute-matched`); path-matched baselines already co-discussed in §6.6/S9 |
| Exact-subset validation extended to effects/regret | extension item (guide) |
| Immutable reproducibility artifact | USER |
| Full complexity/runtime/memory incl. hardware | RUN (`hardware`) + existing S10 |
| Moderate actionability terminology | FIXED (text) |

## Phase 11 related-work additions — FIXED (text)

Added and cited: Yeh et al. 2019 (fidelity/sensitivity), Slack et al. 2020 (fooling
LIME/SHAP), Kumar et al. 2020 (Shapley-explanation problems), Jeyakumar et al. 2020
(human evaluation of explanation media), Balog & Radlinski 2020 (conflicting
explanation goals), Gedikli et al. 2014 (explanation-type comparison), Tintarev &
Masthoff 2015 (quality criteria framework), Chen et al. 2025 GREASE (GNN
counterfactual explanations, ACM TORS). Previously unused keys
`jannach2016purpose` and `verma2022counterfactual` are now cited.

## Minor items addressed (selection)

- Table 7 (convergence): ItemKNN capitalization + final M_pair=500 floor rows added.
- Table 9 (-0.000): full-catalogue paired differences printed to 6 decimals with CIs.
- S19/S22/S25/S28 caption corrections (equal-budget rename, 400-vs-200 cohorts,
  n=810 intersection explained + Pearson agreement added, 0.948 min correlation).
- "full catalogue" hyphenation standardized; citation brackets are standard natbib
  output (source has no literal-bracket citations).
- Direction-accuracy sign(0) prose corrected (zero predictions with nonzero effects
  count as mismatches); Prec@k coverage caveat added.
- BPR gradients Eq. added; clipping semantics stated.
- validate_manuscript.py re-pointed at acmart-primary (was failing on stale
  paper-v3 path); all tests + validator pass.

## Known remaining placeholders

- Cover letter / supplement: `[INSERT PERMANENT OR REVIEWER-VIEW OSF/ZENODO URL...]`
  and the preprint statement — fill after the artifact deposit (USER action).
- Both committed PDFs still predate the revised sources: no LaTeX toolchain exists in
  this workspace, so run `make -C . pdf` (then `make check`) before submission. The
  test suite fails while `code/results/manifest.json` and the two
  `\resultmanifeststamp` macros disagree, and xfails (not fails) while the PDFs lag.

## Round 10 (follow-up revision pass, in this workspace)

Everything below is generated, mirrored to both `tables/` copies, and covered by
`code/tests/test_review9_publication_integrity.py` (14 tests):

* `scripts/make_review3_stats.py`: permutation draws 1,000 → 10,000, the running-minimum
  "Holm" replaced by the standard step-down adjustment, and a `--check` mode. The root
  `Makefile`'s `tables` target now *verifies* rather than rewrites
  `review3_statistics.tex`, because that file carries 19 hand-appended rows.
  `--check` reports PASS on all 40 published rows, i.e. the table was already right and
  only the generator was stale.
* `scripts/make_review9_stats.py` additionally emits
  `tables/review9_benchmark_replications.tex` (fixed-denominator levels + paired
  contrasts + utility-factorial replication + stratified nulls) from the Gowalla runs,
  and regenerates Table S4 from `release/paired_tests.csv`.
* Structural guards added while fixing a real defect this pass introduced: the S4
  longtable preamble declared 11 columns for 12-cell rows, three review-9 tables had
  off-by-one colspecs, and one render block had lost a level of backslash escaping
  (`\toprule` had become a tab character). The new test asserts no control characters,
  matched environments, the house `@{}…@{}` preamble, per-row cell counts against the
  declared columns, unique labels and resolvable cross-references for every generated
  asset — it is the substitute for the LaTeX build that cannot run here.
* `make -C . stats|tables|manifest|check` are now path-independent and work from any
  directory; `code/results/review9/*.json` joined the manifest scope.

* **Handoff notebook for the runs that need the datasets machine**:
  `notebooks/REVIEW9_REPLICATION_RUNS.ipynb`, authored by
  `code/scripts/make_review9_notebook.py`. It autodetects the repo, checks the data paths,
  pilots all seven experiments at the real `M_pair` budget (measuring per-cohort cost on the
  user's own machine), launches a resumable detached queue that writes straight into
  `code/results/review9/`, and finishes with a rebuild cell that runs `make stats`, syncs
  `\resultmanifeststamp` in both documents, checks the mirrors byte-for-byte (review-9 tables,
  S3b, and now `review5_validation.tex`) and runs the validators. Step 1 calls the root
  `make stats` rather than the review-9 generator alone, so a run collected through the notebook
  refreshes every generated table and the manifest in the order the documents expect, and fails
  loudly (`SystemExit`) instead of validating a half-regenerated tree. Scratch (pilot JSONs, queue log and script) lives in
  `code/results/_review9_scratch/`, which is git-ignored *because* the manifest hashes
  `code/results/review9` recursively.
* **`make_review9_stats.py` now consumes every run type**, so the notebook needs no follow-up
  editing to publish a result: `prospective_*.json` → `tab:r9-prospective-replication`,
  `candidate_redraw_*.json` → `tab:r9-candidate-redraw`, `compute_matched_*.json` →
  `tab:r9-compute-matched`, `hardware_*.json` → `tab:r9-hardware`, all in
  `tables/review9_benchmark_replications.tex`. Each returns nothing when its run is absent, so
  no `[Missing table asset]` placeholder can appear for work that has not been done yet.
* **The pilot paid for itself immediately**: it crashed on `candidate-redraw`
  (`KeyError: mean` when a redraw yields no defined AIA on a small cohort), which is a bug in
  `run_review9_experiments.py`, not in the notebook; the payload now reads the per-redraw means
  defensively and records `redraws_with_data`.
* `validate_manuscript.py` additionally errors when a `\safeinput` asset is missing from
  *either* document (previously only the main paper was checked) and warns about generated
  tables that no document inputs, which is how a result can exist in source and be invisible in
  the PDFs. The orphan scan now reads all four entry points (ACM main + supplement, IPM main +
  supplement) rather than the two ACM ones, because a table the IPM document inputs is not an
  orphan. Five review-3-era tables (`actionability_gap_robustness`, `appendix_contract`,
  `attribution_stability`, `protocol_audit`, `sensitivity_results`) are input by *no* document:
  they stay on disk, hashed in the manifest, as provenance for numbers quoted in prose, and the
  warning is expected output rather than a defect.
* **A partial collection no longer renders nothing.** `benchmark_replication_tables`
  returned early when the fixed-denominator run was absent, so six other benchmark floats
  were silently withheld even though their own runs had finished. The independent floats
  are now appended through `_benchmark_extras` on both paths, verified by regenerating the
  committed tables byte-identically and by pointing `AES_REVIEW9_RESULTS` at a
  two-payload directory (which then renders exactly the two floats it owns). `--dry-run`
  also stopped writing `review9_statistics.json`/`review9_multiplicity_map.csv` into the
  directory it was only *checking* -- a stray derived file in a scratch dir is how a
  scratch run gets mistaken for a deposited one.
* **A generator that owns only one mirror is a drift machine.** `make_review5_tables.py` wrote
  `actionshap-ipm/tables/review5_validation.tex` while `acmart-primary/tables/` kept a hand-enriched
  copy; running the pipeline therefore silently *reverted* the ACM caption (losing the "rows are not
  paired / not comparable to the 0.827 primary cohort" caveat) and left the two documents typesetting
  different text for the same label. The generator now writes both mirrors byte-identically, has
  absorbed the hand-written caption wording, and supports `--check` for parity. Its `--check` branch
  initially shadowed the module-level `json` import with a local one (`UnboundLocalError` in
  `json.load` three lines away) -- a reminder that a `def` touched in one branch needs a full-file
  syntax + run check, not just an `ast.parse` of the edit.
* **Fixing the cross-machine part also removed a self-reference.** `manifest_stamp` used to
  hash the checked-out revision along with the files, so the *commit* changed the stamp the
  documents quote: re-running `make check` immediately after committing reported drift with
  nothing edited, which is how a freshness check loses its readers. The stamp now covers file
  contents only (revision recorded beside it), and `make check` stays green across commits --
  verified by re-running it after the commit that introduced this change.
* **The content hash quoted in both PDFs was machine-dependent, and their own
  replication run proved it.** Re-generating the derived payloads on the authors'
  workstation (macOS, Accelerate, pandas 2.3.3) and in the review sandbox (Linux,
  OpenBLAS, pandas 3.0.5) changed the 16th significant digit of 2,214 stored floats --
  four $p$-values printed identically while the hash the documents quote moved, which
  trains a reader to ignore the very check that is supposed to catch an undocumented
  change. Two sources, two fixes: derived payloads are now serialised at 12 significant
  digits (the tables print 3-5), and `review9_multiplicity_map.csv` is written by hand
  instead of through `DataFrame.to_csv`, because pandas 2 emits `1.0` where pandas 3
  emits `1` for the same `%.12g` value. Verified by replaying both machines' committed
  payloads through the new serializer (byte-identical, 388,514 B) and by regenerating
  under pandas 2.3.3 and 3.0.5 (byte-identical); `test_derived_payloads_do_not_depend_on_blas_last_digits`
  keeps it that way. Raw run outputs stay at full precision on purpose: they are the
  deposited record, and a change there is a fact about the run.
* **`make pdf` reported "no toolchain" on a machine that has one.** A venv-launched
  kernel does not inherit MacTeX's `/Library/TeX/texbin`, so the collect cell skipped the
  rebuild that the stale-PDF guard is waiting on. The root Makefile now probes the usual
  install locations and prepends the first one it finds, the notebook does the same for its
  own report, and every subprocess plus the `make` calls now pass the *kernel's*
  interpreter (`PY = sys.executable`) rather than a bare `python3`, so the validators,
  the generators and the queue all run under the numpy the user actually installed.
  The environment check also probes scipy/scikit-learn/matplotlib/pytest, since the test
  suite imports them and a missing one used to surface as an unrelated collection error.
* Their run also validated the queue's bookkeeping end to end on a third platform:
  21 jobs (7 experiments x 3 datasets) were enumerated, the five Gowalla outputs already
  in the repository were reported `exists` and skipped rather than re-run, and the pilot's
  measured costs (13-27 min per 1000 users, against 25-65 in the sandbox) replaced the
  reference estimates before launch.
* **`make artifact`** (new root target) builds the deposit: raw schema-v2 runs plus the generated
  half of the paper (review-9 outputs, manifest, both table mirrors, generators, validators, the
  integrity test, the notebook, the two planning docs), with a `release_checksums.json` inside and
  a `.sha256` beside it. This closes the *tooling* half of #16; the deposit itself and the DOI are
  still the user's action, and `results/release/` is git-ignored so rebuilding an archive cannot
  dirty the tree or invalidate the manifest stamp.

State after this pass: #5 is **FIXED (data+text) on the benchmark that ships in the
artifact** -- `prospective_gowalla.json` (600 users, 528 with a defined score, 12.0\%
top-$1$ coverage of the held-out item) typesets `tab:r9-prospective-replication` and S11
quotes it, including the reason no paired contrast against held-out conditioning is
defined. #13's *cost* half is likewise closed with data (`hardware_gowalla.json` ->
`tab:r9-hardware`: peak RSS, CPU count, library versions, 5-repeat per-method medians),
and the supplement's earlier "peak memory was not serialized" limitation is now scoped
to the archived runs instead of covering everything. What remains for #8 and #13 is the
equal-budget curve and the candidate-set redraw on the primary cohorts: both are
single-command runs, both tables are already wired into the generator, and they are
being run on a workstation with 48 GB rather than in the 2-core sandbox (the sandbox
attempt was stopped rather than left to burn an hour for a 60-user curve). #2 and #19 need a retrained, tuned neural/graph
recommender, i.e. a separate engineering workstream rather than a re-scoring run; #16 needs the
artifact deposit (USER) and the PDF rebuild (`make pdf`, no LaTeX toolchain in this workspace).

## Round 11 (workstation runs start landing: issues 1 and 5 on the primary cohorts)

`5b5b6d5`/`f389f18` brought the first three primary-cohort payloads from the 12-core
workstation: `fixed_denominator_movielens.json`, `fixed_denominator_amazon.json` (both
1,000 users) and `prospective_movielens.json`. Collected here with the usual
`make stats` -> stamp -> validators path (65 manifest entries, stamp `22d42b58733e`,
124 tests green).

* **Issue 1 got a real answer, and it is not the comfortable one.** The reviewer's claim
  was that a bounded effect under the released interface is partly a normalization
  artifact. On MovieLens the Shapley--LIME separation collapses from $0.182$ to $0.016$
  when the deleted mass is suppressed instead of reallocated; on Amazon it *reverses*
  (Shapley $0.607$ vs LIME $0.832$ normalized, $0.993$ vs $0.849$ fixed). The tables and
  S11 now report this, the main text concedes it in one sentence at the headline claim,
  and the effect-scale reading (mean $|\Delta|$ of order $7$-$9\times10^{-4}$) is given so
  that a $0.99$ bounded AIA cannot be misread as strong evidence.
* **Scope honesty added while writing it:** the ablation fits its own ItemKNN
  ($200$ neighbours, cohort-only histories, $M_{\mathrm{pair}}=250$), so its normalized
  column differs from the published primary values ($0.744$ vs $0.779$ on MovieLens,
  $0.607$ vs $0.414$ on Amazon). S11 says this outright and reads only the paired
  within-replication comparison; an earlier sentence claiming the primary cohorts needed
  released per-user intervention files was wrong (the runner re-fits from the raw
  interaction files) and is gone.
* **Issue 5 on the primary cohort:** $1{,}000$ MovieLens users, all with a defined game,
  prospective-target Shapley $0.800$ / LIME $0.939$ / LOO $0.981$ / signed $0.932$ against
  held-out conditioning $0.744$ / $0.926$ / $0.974$ / $0.906$; coverage of the held-out
  item is $11.8\%$, so the two audits mostly score different events and no paired contrast
  between them is claimed.
* Still queued on the workstation: `prospective amazon`, both `stratified-null`, both
  `utility-factorial`, all three `compute-matched`, all three `candidate-redraw`, both
  `hardware` (13 of 21). Cell 5's re-measured pilot is now honest about their cost
  ($158$ min per 1000 users for compute-matched, $131$ for candidate-redraw with 10 redraws).
## Round 12 (submission hygiene: preprint decision and double-blind anonymity)

The authors will not post a preprint, so the cover letter now says so in the
indicative ("no preprint of this work exists, none is planned, and the manuscript is
not under consideration at any other venue") in both the `.tex` and `.md` variants, and
the artifact paragraph describes the deposit that `make artifact` actually builds
(code, runners, per-user JSONs, generated tables, validators, integrity tests, the run
notebook, matrices, a per-member checksum manifest and the archive `.sha256`) instead of
carrying a bracketed TODO; only the DOI/URL slot remains, since it cannot exist before
the deposit is minted. The manuscript's own availability section was rewritten the same
way and no longer ends in "must be inserted here before submission".

That rewrite surfaced a desk-reject risk none of the review-9 checks covered: the
document class carried `manuscript,screen,review` but **not** `anonymous`, so the review
PDF printed the author block, affiliations, emails and ORCIDs on page 1 --- and even
after adding `anonymous`, the running head (`\shortauthors`) and the CRediT paragraph in
the acknowledgments kept naming the authors, which `anonymous` cannot hide because they
are body text. Both documents now derive their running head and contribution paragraph
from the class option through a single `\ifreviewcopy` switch (drop `anonymous` at
camera-ready and the names come back by themselves), and `validate_manuscript.py` fails
if `anonymous` is set while any name from an `\author{}` block still occurs after
`\maketitle`, with the conditional suggested as the remedy. The rule was written against
the un-anonymized file first and confirmed to fire on it, so it is not decoration.
## Round 13 (is it submittable? the answer now comes from the files)

`build pdf` landed as a commit adding `acmmanuscript (1).pdf` next to the manuscript, and
that file is byte-identical to the one already in `acmart-primary/` (sha256
`6740fc54...`, `/CreationDate D:20260827085839Z` --- the same build both documents had
before the anonymity work). Nothing had been recompiled: page 1 of both PDFs still prints
the three authors, neither contains the manifest stamp the sources quote, the supplement
still carries the sentence the sources deleted ("... must be inserted here before
submission"), and the review-9 fixed-denominator panels are not in it. Every source-level
check in the repository passed all of that, which is the argument for checking the
artefacts instead of the intent.

`make ready` (code/scripts/check_submission.py) is now that check. It runs three groups:
the sources, through `validate_manuscript.py` rather than a second copy of its rules; the
compiled PDFs, for exactly one copy per document at the canonical path, a build date
behind no source, a page 1 whose anonymity matches what the class options say, the frozen
manifest stamp present in the text, no placeholder sentence left, and the review-9 panels
present in the supplement; and the deposit, recomputing every hash in
`release_checksums.json` from the archive bytes and checking that the archived manifest is
the stamp the documents quote. It exits non-zero with the blockers named, and
`code/tests/test_submission_readiness_gate.py` pins that it can also answer "ready", so a
gate cannot quietly degrade into a refusal machine. Today it reports nine blockers, all
of them the two stale PDFs plus the duplicate, and one action fixes eight of them.

Two follow-ups from the same round. The cover letters no longer carry the open `[DOI/URL]`
bracket at all: the artifact paragraph now says the archive ships with the submission as
reviewer-visible material and that a permanent identifier will be minted at revision, which
is true whether or not a handle exists yet. And the gate cannot be exercised in this
sandbox --- `apt` and CTAN are unreachable, only PyPI egress is allowed, so there is no TeX
Live to build with --- which is why `make ready` treats a missing text extractor as a
blocker instead of a skip: the only way to pass the PDF group is to have actually read a
build on a machine that can produce one.

Round 13 follow-up: the submission is compiled on Overleaf, so `make overleaf` generates the
project from the sources (include graph resolved, then re-resolved inside the zip) instead
of shipping a hand-picked folder; the compiled PDFs and the lagging `.bbl` are excluded by
design, and the README in the zip records the commit and manifest stamp so a downloaded PDF
can be tied to a state of the tree. Both cover letters are also free of placeholder text
now. `make ready` still reports eight blockers, all of them the two stale builds: one
Overleaf compile of each document, dropped on the canonical paths, clears them.
