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
