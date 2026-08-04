# ActionShap revision and rejection-risk response

This revision addresses the rejection-risk issues identified in the revision-4 audit. The canonical manuscript is `paper/paper.tex`; the Springer Nature submission source is `overleaf-springer/paper.tex` and uses the `sn-jnl` journal class in `overleaf-springer/template/`.

## Blocking issues resolved

- **Scope and claim inflation:** the manuscript is recommendation-only and no longer claims a cross-domain or causal contribution.
- **Non-responsive models:** ItemKNN and profile aggregation consume the retained history at scoring time; the real-data masking gate and inert static control are required.
- **Data leakage:** models use complete training histories; players are truncated only for attribution; validation and complete pre-test histories are excluded from negatives.
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
