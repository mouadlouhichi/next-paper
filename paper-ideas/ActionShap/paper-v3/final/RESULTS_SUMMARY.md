# ActionShap schema-v2 results summary

## Current claim boundary

The canonical manuscript is `paper-v3/actionshap.tex`; older revision notes that
refer to “11 conditions” or “22/22 comparisons” are historical and are not the
current headline claim. The current component report uses nine singleton
ItemKNN target-margin conditions: primary and full-catalogue conditions for both
datasets plus the five MovieLens candidate/history/rho sensitivities. Budget
conditions are excluded because budgets change joint decisions, not the
singleton AIA estimand.

## Headline finding

Shapley has a positive bounded-minus-deletion AIA change across the nine current
singleton ItemKNN conditions. This change is not unique evidence of validity:
the random control is near its within-user null in the primary conditions, but
random and greedy can show positive descriptive gaps in some robustness
conditions. Shapley still has lower absolute bounded AIA than LIME and LOO in
the primary sampled conditions and does not consistently select the best NDCG
action.

For valid nonconstant users, LOO is an exact deletion diagnostic. The asset
builder validates `phi_LOO[p] = -Delta_deletion[p]` with tolerance `1e-12` and
reports deletion AIA as exactly one by definition. Its gap is therefore the
bounded AIA minus one; LOO is not treated as a positive-gap competitor.

## Required component reporting

Every method and non-budget condition has deletion AIA, bounded AIA, their
difference, valid-user counts, confidence intervals, and null-adjusted context
in `tables/aia_components.csv` and `tables/aia_permutation_null.csv`.
Operational action quality is reported separately in
`tables/intervention_outcomes.csv`: NDCG effect, success, harm/abstention, and
normalized regret. The `.tex` files are compact publication views; the CSV/JSON
files are the complete audit release. Pointwise AIA and gap rows are removed
from budget-one and budget-three summary exports.

## Safe claim

> Under the declared bounded-downweighting policy, Shapley’s target-margin
> alignment changed favorably relative to deletion across the evaluated
> ItemKNN singleton configurations. A positive change is not a standalone
> validity score: local methods retain higher absolute alignment, the random
> control calibrates chance, and joint NDCG action quality is dataset-dependent.

## Validation notes

The release manifest reports PASS for schema-v2 provenance and protocol checks.
The profile model remains robustness-only because it is weaker than popularity
and one MovieLens masking gate fails. NDCG attribution remains an unconverged
stress test at the maximum declared Monte Carlo budget and is not used as the
primary attribution utility.
