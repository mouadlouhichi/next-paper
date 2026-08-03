# ActionShap — Recommendation-Only Paper Structure

**Canonical manuscript plan, revision 4.** This file supersedes the earlier cross-domain ActionShap structure. The older clustering/air-quality version is retained only as historical material and must not be used for the recommendation-only paper. Numerical schema-v1 pilot results are also historical: only `paper/final/` assets generated from the corrected two-dataset, two-model, real-gate protocol may enter the manuscript.

**Target:** recommender-systems/XAI venue; Discover Artificial Intelligence is a suitable target if the final experiments support the claims.

## Working title

**ActionShap: Intervention-Grounded Evaluation of Actionable Explanations in Recommendation**

## Central claim

Recommendation explanations are usually evaluated by deletion-based faithfulness, although deployed systems act through feasible modifications. ActionShap is a protocol for evaluating an explanation by whether its ranking predicts the effects of feasible recommendation interventions.

ActionShap does **not** assume that Shapley, LIME, attention, or any other method is superior. They are interchangeable explanation methods supplied to the same evaluation protocol. Monte Carlo Shapley is an estimator used when the user-specific player set is large; it is not the paper's scientific endpoint.

## Abstract — working draft

Recommendation explanations are commonly evaluated with deletion-based faithfulness measures, although recommendation systems are used through feasible changes to user profiles, item exposure, or interaction influence. A factor can therefore be faithful under deletion while being infeasible or ineffective under an operational intervention. We introduce **ActionShap**, an intervention-grounded protocol for evaluating actionable explanations in recommendation. For each user, ActionShap defines a player set of retained historical interactions, a fixed sampled evaluation set, a declared intervention policy, and a ranking-quality utility. It measures whether an explanation ranking predicts the realized effect of feasible interventions using Attribution–Intervention Alignment, top-k intervention precision, intervention regret, intervention success, and attribution stability. The protocol supports arbitrary attribution methods; Monte Carlo Shapley is included as a cooperative attribution baseline because the user-specific player set is too large for exact enumeration. We evaluate ActionShap on temporally split recommendation data using fixed sampled evaluation sets, leakage-safe intervention budgets, and a full-catalogue robustness subset. The primary analysis compares faithfulness-based rankings with intervention-grounded rankings and tests whether explanation methods select useful joint interventions at budget two. We report method differences with paired uncertainty and within-user permutation nulls rather than assuming Shapley superiority. The result is an auditable evaluation instrument for asking whether a recommendation explanation predicts what happens when a system operator or user performs a feasible change.

**Keywords:** explainable recommendation; actionability; intervention; faithfulness; Shapley value; LIME; counterfactual evaluation; recommender systems

## Contributions

1. **Operational actionability protocol.** A recommendation-specific definition of feasible intervention and realized intervention effect under a frozen model and fixed candidate set.
2. **Evaluation metrics.** AIA, top-k intervention precision, intervention regret, intervention success, and stability, with a within-user permutation null.
3. **Faithfulness–actionability analysis.** A controlled comparison showing when deletion/faithfulness and feasible intervention produce different method rankings.
4. **Budgeted decision evaluation.** A primary joint-intervention experiment at \(B=2\); \(B=1\) is explicitly reported only as the leave-one-out oracle sanity check.
5. **Reproducible artifact.** One notebook wrapper over tracked scripts, deterministic temporal splits, fixed target-plus-unseen-negative evaluation sets, frozen intervention policy, independent convergence analysis, hierarchical statistics, and content-addressed generated assets.

## Research questions

### RQ1 — Can actionability be measured reproducibly?

Does the protocol produce deterministic, non-degenerate utility values, valid intervention effects, a calibrated chance null, and stable conclusions across seeds?

### RQ2 — Faithfulness versus actionability

Do deletion-based faithfulness metrics agree with feasible-intervention metrics? Under what conditions do their rankings diverge?

### RQ3 — Intervention decision quality

Which explanation methods select the most effective feasible joint interventions under the same budget, and which minimize regret against the feasible oracle?

### RQ4 — Robustness

Are the conclusions stable across users, random seeds, history lengths, candidate-set sizes, intervention strengths, and Monte Carlo sample counts?

## Experimental object

For user \(u\), the player set is the most recent \(n_{\max}\) training interactions:

\[
P_u=\{p_{u1},\ldots,p_{un_u}\}.
\]

The primary model is history-conditioned ItemKNN with \(n_{\max}=20\); a latent profile aggregator is architecture robustness. Both consume retained history at inference time. Static user embeddings are not attributed because masking their history does not change their output.

The primary attribution characteristic function is continuous target margin,

\[
v_u^{\mathrm{attr}}(S)=\sigma\!\left(s_y(S)-\operatorname{mean}(\operatorname{TopL}_{-y}s_i(S))\right),
\]

while \(q_u(S)=\operatorname{NDCG@K}(f_u^S,y_u)\) is the operational action outcome. Effects, exact oracles, and regrets are stored separately for both utilities. They are computed on a fixed target-plus-unseen-negative sampled evaluation set; a separate subset uses the full unseen catalogue. Negatives exclude the complete pre-test history, candidate and tie seeds are independent of experiment randomness, and the empty coalition uses a zero profile with one catalogue-wide seeded tie-break.

The primary feasible intervention is bounded interaction downweighting:

\[
w_p\leftarrow \rho w_p,
\qquad \rho\in\{0,0.25,0.5\}.
\]

The primary scientific budget is \(B=2\). Signed attributions predict downweight benefit as \(-\phi\); a method may choose no action, one action, or two actions. Magnitude-only rankings are reported separately from beneficial decisions. The \(B\le2\) intervention oracle exhaustively evaluates no action, every singleton, and every pair for every primary user; greedy approximation is reserved for optional budgets above two.

## Methods compared

- Monte Carlo Shapley;
- LIME local surrogate;
- permutation/leave-one-out attribution;
- greedy sequential-deletion counterfactual search;
- random attribution negative control;
- attention or gradient attribution only when the evaluated model exposes that signal.

No method is declared best in advance. All methods use the same users, player factors, candidate sets, intervention strengths, budgets, and evaluation outcomes.

## Primary result tables

### Table 1 — Protocol and data audit

Dataset counts, temporal split, candidate size, target coverage and evaluation-set size, history cap, model, intervention strengths, budget, and number of users.

### Table 2 — Faithfulness versus actionability

For every method:

- deletion/faithfulness metric;
- AIA;
- AIA permutation-null mean and 95th percentile;
- paired confidence interval;
- user count.

### Table 3 — Joint intervention decisions

For \(B=2\):

- top-2 intervention precision;
- intervention success rate;
- realized NDCG change;
- intervention regret;
- oracle coverage.

### Table 4 — Robustness

History length, candidate size, intervention strength, Monte Carlo sample count, seed stability, and model sensitivity.

### Table 5 — Negative controls and validity checks

Static model inertness, empty/full coalition checks, synthetic additive game, symmetry, efficiency diagnostic, and AIA null calibration.

## Primary figures

1. **Protocol validity:** masking sensitivity and static inertness; appendix or implementation-validation section, not a headline result.
2. **Faithfulness versus actionability:** paired method-level faithfulness and AIA values.
3. **AIA against chance:** observed AIA with within-user null intervals.
4. **Joint decision quality:** intervention regret and success at \(B=2\).
5. **Robustness:** convergence and seed stability.
6. **Case study:** one user with the original recommendation, explanation ranking, selected intervention, oracle intervention, and realized ranking changes.

Figure titles should be descriptive, not implementation labels such as “spec 7.1.1”. Captions must state the unit of analysis, number of users, uncertainty method, and whether the result is diagnostic or primary.

## Statistical analysis

Use distinct-user paired comparisons. Average repeated experiment seeds within the same user before primary inference or use an explicitly hierarchical model. Report user-bootstrap confidence intervals, plus-one paired permutation or Wilcoxon/sign tests, corrected multiplicity, effect sizes, and valid/missing user counts. Do not treat coalition evaluations or repeated seed--user rows as independent observations.

A method ranking is only considered meaningful if it is supported by:

- a predeclared metric;
- paired uncertainty across the same users;
- a within-user chance null where appropriate;
- and stability across seeds.

A result in which LIME or another baseline outperforms Shapley is reportable and must not be hidden. The conclusion follows the data.

## Q1-level quality requirements

Before submission, the paper must satisfy all of the following:

- precise novelty claim limited to intervention-grounded evaluation;
- no unsupported claim that Shapley is universally superior;
- public code and fixed configuration;
- deterministic temporal split and candidate set;
- explicit intervention semantics and limits of offline evaluation;
- leakage audit;
- at least five seeds for final comparisons;
- convergence analysis for Monte Carlo estimates;
- paired statistical tests and effect sizes;
- confidence intervals on all headline comparisons;
- negative controls and synthetic correctness tests;
- ablations for history cap, candidate size, intervention strength, and budget;
- limitations covering model scope, offline utility, profile intervention semantics, target coverage and evaluation-set size, and generalization;
- no causal language unless causal assumptions are explicitly defended;
- generated assets linked to the exact input result files through a manifest.

## Paper organization

1. Introduction and motivation
2. Related work: recommendation explanation, faithfulness, counterfactual evaluation, recourse, actionability, and cooperative attribution
3. ActionShap protocol and formal definitions
4. Experimental protocol and validity checks
5. Results: faithfulness/actionability divergence and joint intervention decisions
6. Robustness, case studies, limitations, and threats to validity
7. Conclusion

The paper should lead with the evaluation gap, not with the Shapley estimator.
