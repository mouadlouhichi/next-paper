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
| 1 | Normalized weighting = relative profile-mass reallocation, not isolated suppression | **FIXED (text)** new derivation paragraph (coefficient change, uniform-scale invariance, rho-factorization explaining the flat Amazon 0.708 rho-response) + rename "relative profile reweighting"; **RUN** fixed-denominator pure-suppression ablation (`fixed-denominator` subcommand, `FixedDenominatorItemKNN` implemented + tested) | acmmanuscript §3.2; run_review9 |
| 2 | Only ItemKNN passes the quality gate | **REBUT + RUN (long-term)** existing candor retained (limitations, abstract caveat); competitive gate-passing neural/graph model is an engineering workstream (history-weighting-compatible training), scoped in the guide | abstract, §7, §8 |
| 3 | Amazon full-catalogue reversal must be central | **FIXED (text)** promoted into abstract, §6.6 already reports it, conclusion restates it (negative AIA −0.05 with positive gap +0.16) | abstract, §6.6, conclusion |
| 4 | Utility mismatch confounds H2 (target-margin attribution vs NDCG decisions) | **FIXED (text)** direct utility-matched association analysis computed from existing matrices and added to §6.4; **RUN** full 2x2 attribution x outcome factorial (`utility-factorial`) | §6.4; run_review9 |
| 5 | Prospective audit must be co-primary | **RUN** full-cohort prospective audit (`prospective`); existing 250-user prospective results remain in S9 | run_review9 |
| 6 | "Executable" overclaims | **FIXED (text)** "simulator-executable" at all claim sites; abstract states the intervention is simulator-executable, not demonstrated against a production interface | throughout |
| 7 | Eq. (4) BPR not reproducible as written | **FIXED (text)** rewritten as per-triple loss + exact per-triple gradients verified against `fit_item_embeddings`, clipping semantics specified, context/regularizer scope defined | §3.2 Eq. (4)-(5) |
| 8 | Fixed candidate sets dominate conclusions | **RUN** 20 independent candidate resamples with between-resample variability (`candidate-redraw`) | run_review9 |
| 9 | Per-user MC uncertainty not propagated | **REBUT (partially addressed)** S21/S24/S28 diagnostics already present; mcse caption corrected (0.948 floor, budget labels); propagation into regret remains an extension item | supp. tables |
| 10 | Analysis populations inconsistent | **FIXED (text)** every flagged statistic relabeled: TOST n=1000 (verified by recomputation), gap-vs-regret = target-margin NRegret (n=1000/987 positive-oracle users), S15/S22/S25/S28 captions fixed with explicit denominators | §5.3, §6, tables |
| 11 | Uncertainty conditional on one fitted model/candidates | **RUN** candidate redraws (R9-4); retraining/temporal-cutoff resampling noted as extension in guide | run_review9 |
| 12 | Player-exchangeability null questionable | **RUN** recency- and popularity-stratified within-user nulls (`stratified-null`) + free-shuffle retained as declared default | run_review9 |
| 13 | "Equal-scorer-budget" table mislabeled / unequal budgets | **FIXED (text)** table renamed to budget-response curves, S symbol instead of B, not-equal-budget note in caption; **RUN** genuinely matched scorer-call points (`compute-matched`) | S9 table; run_review9 |
| 14 | AIA monotone-invariance claim false; gap tested instead of absolute AIA | **FIXED (text)** invariance claim corrected (applies to already-formed vectors only); absolute bounded-AIA vs decision association added to §6.4 | §4.3, §6.4 |
| 15 | Collision-prone integer seed derivation | **FIXED (code)** tuple SeedSequence entropy for random control, LIME masks, MC Shapley, and the within-user null stream in `run_recommendation.py` + `evaluation.py`; widened type hints; unit tests added. **Note:** changes random-control/LIME/Shapley streams, so primary-suite regeneration is required before final submission (cheap for random; full suite ≈ prior runtime) | code + §4.2 text |
| 16 | Artifact URL placeholder; main/supplement version drift | **USER** artifact deposit (Zenodo/OSF) then fill placeholders; **FIXED (text)** drift items reconciled: MDE 0.014/0.051 (generator bug fixed), S15 n=993 caption, B=3 greedy-vs-exhaustive status, full-catalogue 250 vs 1000 wording | cover letter; multiple |
| 17 | MDE 0.008/0.032 vs 0.014/0.051 conflict | **FIXED (code+text)** root cause found: `power_table` pooled ItemKNN + profile models, halving paired SD; restricted to primary ItemKNN; both table copies now show 0.051 (n=993) / 0.014 (n=1000) with formula documented | make_review3_stats.py; S17 |
| 18 | Conflicting Holm families (.0066 vs .0216), 0.0010 floor | **FIXED (text)** confirmatory multiplicity map declared (per-metric families for S3–S5; single 12-contrast family authoritative for success/abstention); 0.0010 explained as 10-family Holm x 1/10,001 permutation floor (verified against raw paired_tests.csv) | §5.3 |
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
  and the preprint statement — fill after artifact deposit (USER).
- Review-9 results JSONs not yet produced — tables for R9-1..R9-7 will be integrated
  after the runs (see REVIEW9_EXPERIMENT_GUIDE.md).
