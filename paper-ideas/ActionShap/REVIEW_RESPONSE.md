# ActionShap revision and rejection-risk response

This revision addresses the rejection-risk issues identified in the revision-4 audit. The canonical manuscript is `paper/paper.tex`; the Springer Nature submission source is `overleaf-springer/paper.tex` and uses the `sn-jnl` journal class in `overleaf-springer/template/`.

## Blocking issues resolved

- **Scope and claim inflation:** the manuscript is recommendation-only and no longer claims a cross-domain or causal contribution.
- **Non-responsive models:** ItemKNN and profile aggregation consume the retained history at scoring time; the real-data masking gate and inert static control are required.
- **Data leakage:** models are fit on complete training histories, while the released scoring game deliberately evaluates only the retained player window; validation and complete pre-test histories are excluded from negatives.
- **Metric conflation:** target margin is the attribution utility; NDCG, Recall, and MRR are separate operational outcomes. NDCG-attribution convergence failure is reported as a limitation, not hidden.
- **Circular B=1 comparison:** LOO is retained as the deletion identity/oracle; the scientific comparison is exact joint action selection for all actions up to B=2.
- **Forced harmful actions:** signed benefit selection includes no action and permits abstention.
- **Chance calibration and inference:** within-user permutation nulls, random control, paired user-level inference, seed averaging, effect sizes, and Holm correction are required.
- **Overclaiming the Actionability Gap:** the corrected results explicitly state that the random control can also have a positive gap. Gap advantage is not treated as absolute action quality or Shapley superiority.
- **Template and declarations:** the Springer Nature `sn-jnl` source is provided, placeholders were removed, and funding, availability, ethics, competing-interest, and contribution statements are explicit.

## Remaining limitations disclosed rather than repaired by narrative

The profile model is a robustness boundary, full-catalogue NDCG correlations are sparse, and NDCG attribution does not meet the frozen convergence threshold at M=1000. These are retained in the Results and Limitations. No additional experiment is claimed until raw, provenance-complete data are available; the tracked final assets are accepted only through `validation_report.json`.

## Second-review corrections implemented

- Replaced the three-method gap narrative with a five-method component report. The manuscript now reports deletion AIA, bounded AIA, and their difference for Shapley, LIME, LOO, greedy, and random.
- Recast the Actionability Gap as a descriptive singleton perturbation-sensitivity statistic. It is no longer presented as proof of validity, robustness, or superiority.
- Removed LOO from positive-gap competitor claims and formalized its deletion-AIA ceiling: for valid nonconstant users, deletion AIA is exactly one and the gap is non-positive.
- Removed `B=1` and `B=3` from the gap figure, gap count, and gap comparisons. Those budgets remain only in joint-action effect, success, abstention, harm, and regret analyses.
- Corrected the Amazon full-catalogue distinction: bounded target-margin AIA is `-0.119`; the Actionability Gap is `+0.1689`.
- Replaced the invalid “22 of 22” headline with component-wise reporting and an explicit random-control comparison. Conditions are described as dependent repeated estimands, not independent confirmations.
- Added `aia_components.tex/csv` and `intervention_outcomes.tex/csv`, including confidence intervals and valid/missing-user counts.
- Redesigned Figure 2 as three panels: deletion AIA, bounded AIA, and their difference.
- Added explicit normalized-regret definition, epsilon threshold, conditional averaging rule, and unbounded-regret behavior.
- Added executable pseudocode covering attribution, bounded actions and the exact B<=2 oracle, within-user inference/nulls, and convergence selection.
- Added `scripts/validate_review_contract.py`, which asserts gap algebra, method completeness, budget exclusion, table existence, normalized-regret documentation, and forbidden headline claims.
- Added `ActionShap_Final_Experiment.ipynb` as the single fixed end-to-end runner.

## Round-10 corrections implemented (prose, labels, and provenance pinning)

This batch closes the correctness and cross-reference items that can be verified from the
repository; nothing in it required a new experiment, and no claim below is asserted without a
file that already exists in the release.

- **Eq. (4) prefactor.** The published identity omitted a factor `(n_u-1)/n_u`; the score shift a
  singleton downweighting induces is
  `(n_u-1)(1-rho) / (n_u (n_u-1+rho)) * (b - s_p)`. Corrected in the manuscript and pinned by an
  algebraic unit test that recomputes the shift from the weight definition for several cohort sizes
  and intervention strengths.
- **Renamed the "Actionability Gap" label.** The protocol defines bounded AIA, deletion AIA and
  their difference; "actionability" was never defined. Row labels, the figure title, and the table
  caption now read "bounded-minus-deletion AIA difference", while the released CSV keys keep their
  historical `actionability_gap` name (a data key is not a claim). A validator guard rejects the
  old label reappearing in a table.
- **Monte-Carlo error claim.** The sentence asserting a population mean "cannot be an estimation
  artifact" from `0.0013` was replaced by the deterministic bound actually available: the mean
  absolute per-user shift (`0.042` MovieLens, `0.026` Amazon) bounds the aggregate shift, no
  `sqrt(n)` improvement is claimed, and the abstention rule is described for what it is (a
  numerical-zero test, not an uncertainty criterion).
- **Utility switching is dataset-dependent, not uniformly insignificant.** The NDCG-utility
  association is unresolved on MovieLens (rho=0.085, p=0.1135, n=349) but still significant on
  Amazon (rho=0.165, p=0.0117, n=233); the text now says so.
- **Prospective panel denominator.** The released Gowalla audit samples and audits 600 users but
  only 528 have a defined score for all four estimators. The table now prints a `Defined n` column,
  states the 528/600 denominator, and distinguishes candidate-set *containment* of the held-out
  item from top-1 *equality*.
- **Two inference records, each labelled.** The per-metric Holm column and the predeclared
  12-contrast success/abstention column are corrected within different families, over different
  draw counts and different success estimands. The false reconciliation sentence was deleted; each
  caption now names the released record it is built from, and a new validator recomputes
  `p=(1+#)/(R+1)` and the Holm step-down from those records and refuses any table that publishes a
  family without naming its source.
- **Success estimand pinned at its definition.** Success is the indicator of the seed-averaged
  realized effect; the supplement's complete outcome matrix now states that `n` is the user count
  (the resampling unit), that its entries average binary per-user-and-seed indicators, and that the
  two poolings are not interchangeable (0.461 versus 0.149 for Shapley on the primary MovieLens
  NDCG comparison).
- **Cross-document pointers.** The declared supplement range is corrected to S1--S11 in both the
  manuscript and the supplement, the three stale pointers (Table S16, Table S25, Section S7) are
  re-anchored at section level, and `validate_prose_references.py` now enforces that every section
  pointer exists, that the declared range equals the real section count, and that a supplementary
  *table* pointer is only legal when the owning section title declares that float number.
- **Budget language.** The construct-validity ablations share `M_pair=250`; the primary analysis is
  frozen at 500, and no caption may call 250 "the primary budget" (generator text plus a test).
- **Interface conditionality is now a contribution.** The bounded statistic is measured against a
  reweighting interface; the contribution list and the model paragraph state that the sign of the
  effect and the cross-method ordering move when the denominator is frozen, and that a bounded AIA
  near 1 is saturation of a ratio rather than absolute evidence of suppression.
- **Availability.** The paragraph names the deposited archive and states plainly that the manifest
  stamp is a build-time macro recorded in `code/results/manifest.json` inside the archive, so the
  verifiable chain is archive bytes to manifest entry.

Still owed by the authors: the 13 queued cohort-scale runs on the owner's machine, the PDF rebuild
(`make pdf`) after which the `make ready` blockers clear and the marked test can be un-xfailed, and
no reformatting for another venue was attempted here.
