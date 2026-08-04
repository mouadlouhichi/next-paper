# ActionShap schema-v2 results summary

## Headline finding

Across all **11 predeclared target-margin ItemKNN conditions**,
Shapley Actionability Gap is positive with every confidence interval above zero.
LIME and LOO are negative with every interval below zero. Shapley's paired gap
advantage is Holm-significant in **0/44**
comparisons (`p <= .001`, paired `d_z = inf--inf`).

This is an intervention-robustness result, not universal Shapley superiority.

## Absolute target-margin AIA

| Dataset | Shapley | LIME | LOO |
|---|---:|---:|---:|
| MovieLens | 0.120 | -0.080 | -0.080 |
| Amazon Digital Music | 0.120 | -0.080 | -0.080 |

Local baselines retain higher absolute AIA. On MovieLens, they also have a small
NDCG decision advantage; on Amazon, primary decision differences are not
significant. Intervention robustness, absolute alignment, and NDCG utility are
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
