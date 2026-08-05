# Design preflight

These files document milestone-zero design decisions made before the corrected
final experiment. They are diagnostics, not headline results and are excluded
from `paper-v3/final/` aggregation.

`movielens_masking_gate.json` records the real MovieLens gate that selected a
20-interaction primary history window. A 50-interaction average profile failed
the predeclared sensitivity threshold. On the fixed 1,000-user preflight, all
five ItemKNN gates passed and ItemKNN exceeded item popularity. Four of five
latent-profile gates passed at 20 interactions, while one seed remained below
the NDCG-change threshold and the model underperformed popularity. ItemKNN
therefore became primary and profile aggregation became an explicitly bounded
architecture-robustness condition.
Longer windows remain robustness conditions and any failed gate is reported as
a non-responsiveness boundary. A separate 100-user convergence preflight selected
continuous target margin for the attribution game; NDCG remains the operational
outcome and a deliberately reported utility stress test.
