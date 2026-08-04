# ActionShap schema-v2 results summary

## Headline finding

Across all **11 predeclared target-margin ItemKNN conditions**,
Shapley Actionability Gap is positive across the predeclared ItemKNN conditions, while LIME and LOO are negative in those conditions. Shapley's paired gap advantage over each local baseline is Holm-significant in **22/22** predeclared comparisons (`p <= .001`, paired `d_z = 0.19--0.65`). The random control is not a negative control for this derived gap: it is positive in several conditions, so the gap is interpreted only together with absolute AIA, signed direction, and decision metrics.

This is an intervention-robustness result, not universal Shapley superiority.

## Absolute target-margin AIA

| Dataset | Shapley | LIME | LOO |
|---|---:|---:|---:|
| MovieLens | 0.120 | -0.080 | -0.080 |
| Amazon Digital Music | 0.120 | -0.080 | -0.080 |

Local baselines retain higher absolute AIA. On MovieLens, they also have a small
NDCG decision advantage; on Amazon, primary decision differences are not
significant. The gap is therefore a relative perturbation-sensitivity measure,
not evidence of universal Shapley superiority. Intervention robustness, absolute alignment, and NDCG utility are
therefore separate axes.

## Validity boundaries

- Primary ItemKNN exceeds item popularity on both datasets.
- The profile model is a negative robustness boundary on Amazon.
- NDCG attribution remains an unconverged stress test at `M=1000`.
- NDCG AIA and normalized regret require explicit valid/missing-user counts.
- Full-catalogue NDCG subsets are descriptive because very few users are active.

## Safe claim

> Deletion faithfulness systematically understates Shapley's alignment under
> feasible bounded intervention. Shapley alone improves across all predeclared
> ItemKNN conditions, even though local methods remain stronger in absolute
> alignment and sometimes downstream NDCG decision quality.
