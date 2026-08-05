# ActionShap revision 4 — Q1 audit resolution (historical audit trail)

> The 11-condition/22-comparison paragraph below is superseded by the current schema-v2 `paper-v3/final/RESULTS_SUMMARY.md`. It is retained only to document the revision history; do not use it as a current claim.

This document records how the schema-v1 pilot was corrected. The complete
schema-v2 matrix has now run and passed the frozen validator; numerical assets
live under `paper/final/`, while this file remains the design audit trail.

## Blocking scientific corrections

| Pilot issue | Revision 4 resolution |
|---|---|
| Target-margin outcomes labelled as NDCG | Target margin is now the explicitly labelled attribution utility selected by convergence preflight; NDCG is the operational outcome. Both receive separate effects, exact oracles, regrets, convergence studies, tables, and labels. |
| Truncated histories used to train the model | Models train on complete training histories; `n_max` limits attribution players only. |
| Positive item included in its own BPR context | The profile model uses a leave-one-out context for every sampled positive. |
| Earlier positives and validation could be sampled as negatives | Candidate construction excludes the complete pre-test history, including validation. |
| “Full catalogue” included seen items | Robustness uses the full unseen catalogue plus the temporal target. |
| Candidate, user, and model randomness were confounded | Candidate, user, tie, model, and attribution seeds are explicit and independently frozen. |
| Per-user tie handling depended on candidate order/item ID | One seeded catalogue-wide priority permutation is reused everywhere. |
| Synthetic gate presented as final validity | Every primary/full-catalogue run executes a blocking gate on at least 200 real users using one fixed 200-item gate set. The archived preflight selected `n_max=20`; longer-window sensitivity failures are reported as boundaries. |
| Static model failure could look like a null result | The gate includes an exactly inert frozen-score control and rejects a non-responsive dynamic model. |
| Unweighted ridge masks labelled LIME | LIME now uses deterministic local neighbours, random masks, a locality kernel, and weighted ridge regression. |
| Missing modern/search and random controls | Methods include MC Shapley, LIME, LOO, greedy sequential counterfactual search, and random attribution. |
| Absolute attribution used to claim beneficial actions | Magnitude AIA is separated from signed alignment. Downweight benefit is `-phi`; action selection uses sign. |
| Exactly two actions were forced | The feasible space contains no action, all singletons, and all pairs. Methods and oracle may abstain. |
| B=2 regret covered 20 users | Batched exact enumeration supports every primary user. Greedy approximation is used only above B=2 and is labelled a lower-bound oracle. |
| LOO called an oracle outside deletion B=1 | The implementation and tables call it LOO; oracle language is restricted to the algebraic B=1 deletion identity. |
| Efficiency error interpreted as convergence | Prefix-walk efficiency is numerical only. Independent M=1000 references determine M from aggregate rank and action-set agreement, while valid-user coverage is reported separately. |
| AIA had only per-user null means | Final assets construct a within-user, within-seed aggregate null with a 95th percentile and plus-one p-value. |
| Seed-user rows treated as independent users | Seeds are averaged within each distinct user before bootstrap or paired sign permutation inference. |
| Literal p=0 values | Plus-one permutation p-values have a finite floor recorded in the manifest. |
| Missing effect sizes/multiplicity | Paired Cohen’s dz and Holm correction are generated per experiment and metric family. |
| No recommendation quality audit | Target rank, NDCG, Recall, and MRR are compared with item popularity. The preflight promoted ItemKNN to primary because it exceeded popularity; the weaker latent profile remains visible as robustness. |
| One model and one dataset | The final contract requires MovieLens-1M, timestamped Amazon Digital Music with raw-source provenance, primary ItemKNN, and a latent profile robustness model. |
| Missing robustness matrix | Five-seed conditions cover full unseen catalogue, history caps, rho, candidate sizes, budgets, and an NDCG-attribution utility stress test. |
| Machine-specific source paths | Final manifests use repository-relative paths and SHA-256 hashes; raw results can be packaged into a content-addressed archive. |
| Stale pilot claims remained in the paper | Pilot assets are preserved under `paper/legacy_pilot/`; the canonical manuscript reads only `paper/final/`. |

## Validated headline result

Historical pilot headline (retired): the current release does not use the 11-condition/22-comparison claim. The schema-v2 paper-v3 assets instead report nine singleton conditions, all five methods, the LOO identity, null calibration, and decision outcomes. A positive Actionability Gap remains descriptive and is not evidence of universal method superiority; local baselines retain higher absolute AIA and slightly better MovieLens NDCG decisions.

## Claim boundary

The primary game is a **retrospective target-conditioned audit**. The temporal
target defines the labelled instance being explained. Explainers cannot inspect
measured intervention effects or oracle actions, and no test-driven tuning is
allowed. This does not constitute a prospective policy with access to future
feedback and does not identify causal effects in the world.

## Final acceptance gate

`code/scripts/make_paper_assets.py` returns a non-zero status unless all of the
following hold:

- schema-v2, paper-eligible runs only;
- two datasets and two history-conditioned models;
- five common seeds with identical users and candidate policy;
- at least 1,000 primary users or every eligible user;
- fixed 200-user real-data gates;
- independent convergence and an admissible primary M;
- exact B<=2 oracles for every primary/full-catalogue user;
- every required method and sensitivity condition;
- complete player/effect vectors and content-addressed provenance.

Only `paper/final/manifests/validation_report.json` with `status: PASS` permits
numerical claims. The tracked schema-v2 report now passes with zero errors and
zero warnings; robustness limitations remain explicit notes.
