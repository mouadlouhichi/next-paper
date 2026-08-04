# ActionShap schema-v2 results summary

## Headline finding

Across **9 distinct singleton target-margin ItemKNN conditions**
(budgets are excluded because they do not enter singleton AIA), Shapley's
bounded AIA changed relative to deletion. This change was not unique: the
random control was also positive in displayed conditions and greedy was positive
in some. The Actionability Gap is therefore a descriptive perturbation-
sensitivity statistic, not standalone evidence of explanation validity.

LOO is reported as the deletion oracle only. For every nonconstant valid user,
its deletion AIA is exactly one, so its gap cannot be positive. LOO is excluded
from Shapley gap-competitor claims and from the headline comparison count.

## Required component reporting

Every method and condition is reported with deletion AIA, bounded AIA, their
difference, valid-user counts, confidence intervals, and null-adjusted context in
`tables/aia_components.tex` and `tables/aia_permutation_null.tex`. Operational
action quality is reported separately in `tables/intervention_outcomes.tex`:
NDCG effect, success, harm/abstention, and normalized regret.

## Safe claim

> Under the declared bounded-downweighting policy, Shapley's target-margin
> alignment changed relative to deletion across the evaluated ItemKNN
> configurations. A positive change was not unique to Shapley: the random
> control also produced positive gaps, and greedy did so in several conditions.
> The gap must therefore be interpreted jointly with absolute bounded AIA,
> signed alignment, null-adjusted comparisons, and decision regret.
