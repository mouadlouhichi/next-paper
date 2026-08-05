# Legacy pilot assets — not final-paper evidence

These assets are preserved for provenance only. They were generated on
2026-08-02 from the schema-v1 pilot and must not be cited as corrected results.

Blocking reasons:

- the manuscript labelled the utility as NDCG while the runner used a sigmoid
  target-margin utility;
- the sampled-negative pool could include earlier observed items;
- model, candidate, user, and Monte Carlo seeds were confounded;
- benefit-seeking actions ignored attribution sign and forced exactly two
  interventions, with no abstention;
- regret covered 20 distinct users even though seed-user rows were reported as
  independent users;
- the masking gate shown in the notebook used synthetic rather than MovieLens
  data;
- required controls, convergence criteria, secondary data/model robustness,
  and hierarchical inference were incomplete.

The files remain unchanged so earlier numbers can be traced, but the canonical
manuscript reads only from `paper/final/`.
