# Round-9 deep review → v23 implementer response

Date: 2026-08-20. Manuscript v23 (`paper_package/main.tex`). Status legend:
**DONE** = in v23 now; **EXECUTING** = running in the sandbox (artifacts appended as they land);
**RELEASED** = script released, run on sandbox/author hardware; **SCOPED** = wording fix + explicit scope.

## Required major fixes

### 1. Temporal/deployment meaning of the validation item
- **DONE (wording):** new §"Protocol timeline and information flow" (`subsec:protocol_timeline`)
  with Table `tab:information_flow` listing exactly which interactions feed graph training,
  player selection, coalition utility, λ selection, candidate exclusion, and evaluation.
  The sequential reading is stated explicitly (calibration event t−1 is context available to all
  methods; test event t is unknown).
- **DONE (code):** corrected protocol implemented — `exclude_by_user` support in
  `coalgamerec/metrics.py` (mask_seen/per_user_hit_ndcg/evaluate) and `coalgamerec/rerank.py`
  (`zscore_candidates`, candidate-aware `rerank_user_scores`, `rerank_all`, `valid_sim_scores_all`,
  `valid_linear_scores_all`): candidates become I \ (H_train ∪ {i_val}) and z-scores are computed
  over that set only. `scripts/run_protocol_v7.py` re-runs the full nine-family matched protocol
  under this correction with everything else byte-identical (same script internals, seeds, k, M,
  negatives, λ).
- **EXECUTING:** Amazon v7 (this sandbox); ML-1M v7 queued. Headline tables will be re-reported
  under the corrected protocol when these land; until then the manuscript states the v1–v22
  candidate-set deviation explicitly (§6.1 and Table `tab:information_flow`).
- Not chosen: relabeling as "transductive label-guided reranking" — the sequential reading with
  candidate exclusion is implemented instead (the reviewer's preferred option).

### 2. False "no validation access" characterization
- **DONE:** all table group headings relabeled "no direct calibration weighting; shared player
  set"; §6.1 rewritten to state that similarity representations are train-only while player
  selection uses the calibration positive; C1 introduction rephrased (matches *direct calibration
  weighting*, not "validation access"); main-results text distinguishes the shared-player-set
  comparisons from the matched C1 ones.
- **RELEASED:** the 2×2 factorial (player selection calibration-guided vs profile-only ×
  valuation non-game vs LOO/Shapley) is scriptable with existing selection strategies
  (`stratified` with/without calibration target vs `similarity`); a runner is queued behind the
  v7 runs (`run_selection_factorial.py`, next commit).

### 3. Circularity of the validation-tuned λ experiment
- **DONE (scoping):** Table `tab:lambda_tuned` is now labeled EXPLORATORY with the circularity
  stated in the text and caption (the calibration item both constructs the signal and is the
  tuning target; valid-linear is self-similar to the target). All "tuning widens LOO's lead"
  confirmatory language removed.
- **RELEASED:** nested tuning script (signal from event t−2, λ tuned predicting event t−1, final
  signal from event t−1 predicting t), with the documented approximation that event t−2 remains in
  the frozen training graph (per-user retraining is infeasible; stated in the script header).
  Will replace the exploratory table.

### 4. Mixed v3/v6 λ sweeps
- **DONE (disclosure):** text + caption of `tab:ablation_lambda`/Fig 4 now state that LOO rows
  come from the v6 execution while other families come from v3, that base models differ (visible
  in unequal λ=0 values), and that cross-family curves are not yet a single-execution comparison.
  The incorrect limitation sentence ("Shapley exceeds LOO at larger reranking weights on ML-1M")
  is removed and replaced by an accurate mixed-evidence statement.
- **RELEASED:** matched single-execution sweep script (all families incl. LOO and Shapley on
  identical fitted models, per-user paired differences at each λ) — queued after v7.

### 5. Equivalence/sufficiency overclaims
- **DONE:** title softened to "When Leave-One-Out Marginals Can Match Shapley for
  Validation-Guided Graph Reranking"; abstract rewritten to the reviewer's safer form; the
  three-part dataset-specific conclusion adopted verbatim in §equivalence (Amazon: equivalence
  supported, interval favors LOO; ML-1M LightGCN: not established, estimate favors LOO; ML-1M
  NGCF: LOO advantage possibly beyond the negligible band); discussion sentence replaced;
  "equivalent or better" removed from contributions; observed-power/analytic-MDE removed.

### 6. Shapley-beats-all-controls overstatement
- **DONE:** abstract and main-results text now distinguish (a) Shapley vs uniform/profile
  heuristics, (b) Shapley vs validation-informed controls (does not consistently beat
  valid-linear once matched), (c) LOO vs validation-informed controls. The reviewer's defensible
  sentence is adopted almost verbatim.

### 7. Intervention as a central factor
- **SCOPED (v23):** every headline is explicitly scoped to the native intervention in the
  abstract/protocol section; the design-ablations text already reports the external-kernel
  reversal on Amazon single-seed.
- **RELEASED/QUEUED:** multi-seed native-vs-kernel factorial with paired inference
  (`run_intervention_factorial.py`, next commit); until it lands, the family ordering is stated
  as unverified under the external intervention.

### 8. Shapley estimator convergence
- **SCOPED (v23):** limitations text now states M=64 is validated only on a 1,000-user subset
  against a noisy M=256 reference, and that "LOO beats M=64 MC Shapley" is what the evidence
  supports, not "LOO matches the Shapley value".
- **RELEASED/QUEUED:** `run_convergence_v2.py` (next commit): independent M=1024 reference on a
  subset, exact Shapley for users with |P_u|≤8 (already the code's exact branch), attribution
  error/sign/top-k agreement and downstream NDCG/HR vs M∈{16,…,256}, both datasets, two training
  seeds, user-level uncertainty.

### 9. Inferential justification
- **DONE:** renamed to *paired sign-flip test* with the symmetry assumption stated; no longer
  called a permutation/randomization test; conditional-on-fitted-graphs scope and the
  graph-linked-user dependence limitation stated explicitly; analytic MDE and observed TOST power
  removed (CI-first policy stated); multiplicity structure consolidated in text (families declared
  a priori with their distinct questions); frozen-protocol evidence pointed at in-repo timestamped
  artifacts rather than external preregistration (Provenance subsection).
- **RELEASED/QUEUED:** simulation-based power under the actual zero-inflated paired metric
  distribution and the declared multiplicity plan (small script, next commit).

### 10. Stronger matched baselines
- **RELEASED/QUEUED:** `run_sequential_baselines.py` (next commit) implementing: (1) last-item
  kNN/item-item similarity from the calibration event; (2) updated-profile baseline (calibration
  item added to the profile representation); (3) frozen-graph edge update (add the calibration
  edge, re-propagate the frozen backbone, no attribution); (4) recency-weighted history. All use
  the corrected candidate exclusions; leak-free λ tuning follows the nested script.
- The 2×2 selection controls of item 2 are part of the same queue.

### 11. Randomization/faithfulness diagnostics
- **DONE (interpretation):** the ρ=0.936 trained-vs-untrained rank correlation is now presented
  as a warning sign (initialization/graph-geometry dominated ordering) rather than a resolved
  test; the reranking-collapse argument is explicitly labeled insufficient because the whole
  untrained model changed, not only the weights.
- **RELEASED/QUEUED:** `run_controlled_randomization.py` (next commit): trained base scorer and
  intervention embeddings held fixed; only the attribution weights are swapped — weights from the
  untrained model, shuffled weights, distribution-matched random weights, selection-only weights;
  user-level distributions with CIs; top-12 overlap reported against the random-overlap baseline.
- **DONE (noise-test definition):** "10% layer-0 noise" defined in §stability (zero-mean noise at
  10% of the mean |E0| magnitude); the near-perfect stability is acknowledged and the controlled
  study above will determine its meaning.

### 12. Validation-negative sampling
- **DONE:** full sampler specification added to §protocol timeline: uniform without replacement
  over catalog minus the user's training items and the calibration target (test item never
  blocked); deterministic per-user draw (estimation-seed + 10^5 + u); fixed across coalitions and
  families within a run; identical across training seeds for a given estimation seed.
- **RELEASED/QUEUED:** multi-seed, all-user, multi-draw negative-set sensitivity (existing
  `run_negset_sensitivity.py` + a draws loop, queued); "stable" is already tempered to
  "moderately rank-stable with similar aggregate NDCG" in v22/v23 wording.

### 13. Faithfulness intervention reproducibility
- **DONE:** §faithfulness scope now defines: ranking by absolute attribution magnitude over the
  full history H_u; fractions relative to |H_u|; deletion retains background B_u; insertion keeps
  only the top fraction (background removed); ties by item index; unmasked full-history run as
  reference; signed weights used through magnitude only, positive/negative separation part of the
  released multi-seed extension. AUC summaries + multi-seed curves are in the queued multi-seed
  masked-forward runs.

### 14. Cost reporting
- **DONE:** Table 6 caption renamed to "NDCG gain **over uniform reweighting** per attribution
  hour"; one-time offline scope stated; outlier effect on the mean noted with medians/IQR in text.
- **RELEASED/QUEUED:** per-seed paired runtime ratios, coalition-evaluation counts, seconds per
  evaluation, and peak-memory profiling join the next cost artifact (queued with the factorial
  runs).

### 15. Algorithm 1 for very short histories
- **DONE:** pseudocode rewritten to match the implementation exactly: if |H_u| ≤ k the bounded
  game covers the full history and selection is inactive (so no stratum is ever initialized from
  an empty set); otherwise three strata with explicit farthest-point seeding from P1∪P2, which is
  non-empty because |H_u| > k ≥ 3. The malformed Ensure line is fixed. A unit test for
  |H_u|∈{1,2,3,24,25} ships with the script set (`tests/test_select_players.py`, next commit).

### 16. Reproducibility/auditability
- **DONE:** new "Provenance and auditability" declaration: public repository URL + branch,
  per-run immutable commit hashes in manifests, table-level run-ID mapping (v3/v4b/v6/v7 IDs),
  SHA256 fingerprints for item vectors + split meta, deviations recorded per execution,
  environment lock + regeneration scripts under `code/`. The word "pre-specified/frozen" is now
  tied to in-repo timestamped artifacts, not external preregistration (stated plainly).

## Important reporting/consistency fixes
- A1–A8: all applied in v23 (see items 1–6, 9; contribution 3 renamed to "paired sign-flip tests
  with joint user bootstrap intervals"; conclusion qualified re valid-linear).
- B (dataset construction): §6.1 now gives the filter/split order (5-core on training-period
  interactions before val/test selection), explains 6,015 vs 6,040 ML-1M users, and states that
  val/test items can be absent from dense subgraphs. "Global eligible catalog" = catalog minus
  training items (minus calibration item under v7).
- C (representations): new notation table (`tab:notation`) defines x_j, e_j, e_j^(0), external
  kernel, and which stage uses which.
- D (signed attributions): z-score direction-dependence stated in §reranking; fraction-negative,
  LOO–Shapley weight correlation, and positive-only/absolute variants ship with the v7 re-run
  artifacts (queued).
- E (omnibus): Friedman kept but moved to "supplementary" framing with an explanation of the
  reject-but-no-pair pattern (heavy ties + conservative Nemenyi at nine families).

## Pending compute (queued in order)
1. v7 corrected protocol: Amazon (EXECUTING), ML-1M.
2. Matched single-execution λ sweep (all families, paired per-user per λ).
3. Nested λ tuning (leak-free).
4. Sequential baselines (kNN, updated profile, frozen edge update, recency).
5. 2×2 selection × valuation factorial.
6. Multi-seed native-vs-kernel intervention factorial.
7. Convergence v2 (M≤1024, exact references, downstream metrics).
8. Controlled randomization (weights-only swaps).
9. Multi-seed/multi-draw negative-set sensitivity.
10. Simulation-based power under the actual metric distribution.
