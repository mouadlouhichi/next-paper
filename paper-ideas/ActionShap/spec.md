# ActionShap Recommendation-Only Specification

**Status:** design specification
**Scope:** replacement specification for the current cross-domain ActionShap proposal
**Target:** recommendation-only research paper and reproducible implementation
**Primary venue:** Discover Artificial Intelligence, or a recommender-systems/XAI venue of comparable scope

---

## 1. Purpose of this change

The original ActionShap proposal combines clustering, tabular intervention, air quality, wine quality, and recommendation. This specification replaces that scope with a focused recommendation paper.

The revised paper will answer one question:

> **Do recommendation explanations identify factors whose feasible modification produces a predictable improvement or change in recommendation quality?**

The paper must not claim to evaluate actionability across clustering and recommendation. It must not contain Wine Quality, Beijing Air Quality, TreeSHAP for clustering, or a cross-domain unification claim.

The revised paper is deliberately different from SignalShap:

| Paper | Players | Main question | Shapley computation |
|---|---|---|---|
| SignalShap | System-level recommendation sources | Which source creates ranking value? | Exact, five players |
| ActionShap | User-specific recommendation factors | Do explanations predict feasible intervention effects? | Monte Carlo, many players |

SignalShap explains **where system value comes from**. ActionShap evaluates **whether an explanation tells us what can be changed effectively**.

---

## 2. Revised working title

### Primary title

**ActionShap: Intervention-Grounded Evaluation of Actionable Explanations in Recommendation**

### Alternative titles

- **From Attribution to Intervention: Evaluating Actionable Explanations for Recommender Systems**
- **Do Recommendation Explanations Predict What Happens When We Act?**
- **Monte Carlo ActionShap: Feasibility-Constrained Attribution Evaluation for Recommendation**

---

## 3. Revised scientific contribution

ActionShap is an evaluation framework, not primarily a new recommender architecture. Given a trained recommender, an explanation method, and a declared intervention policy, ActionShap measures whether highly attributed recommendation factors predict the outcome of feasible interventions.

The contribution consists of four parts:

1. **Recommendation-specific actionability definition** based on feasible intervention, intervention effect, intervention cost, and attribution stability.
2. **Monte Carlo Shapley evaluation protocol** for large user-specific player sets.
3. **Attribution–Intervention Alignment (AIA)** and decision-level metrics that compare attribution rankings with realized intervention rankings.
4. **A reproducible intervention benchmark** using fixed candidate sets, frozen models, declared intervention budgets, convergence tests, and leakage-safe evaluation.

The paper must not claim that ActionShap discovers causal effects. It measures intervention effects under the declared recommender and intervention simulator. Causal language is allowed only when the intervention semantics and assumptions justify it.

---

## 4. Research questions and hypotheses

### RQ1 — Evaluation validity
Do attribution rankings agree with the ranking of feasible intervention effects?

**Primary metric:** Spearman correlation between absolute attribution and absolute intervention effect.

### RQ2 — Method comparison
Do cooperative-game attributions align better with feasible interventions than LIME, attention, gradient, and permutation baselines?

**Hypothesis:** Monte Carlo Shapley will have higher AIA and lower intervention regret than non-cooperative baselines, but this is an empirical hypothesis and must not be stated as a result before experiments.

### RQ3 — Faithfulness versus actionability
Do standard deletion/faithfulness metrics rank explanation methods in the same order as intervention-grounded metrics?

**Hypothesis:** the rankings will diverge when deletion is infeasible, non-local, or outside the candidate intervention budget.

### RQ4 — Stability and convergence
How many Monte Carlo permutations are required for stable intervention decisions?

Report both attribution convergence and top-k intervention convergence. The latter is more important for practical use.

### RQ5 — Decision quality
Does selecting an intervention using ActionShap produce lower regret than selecting an intervention using competing attribution methods?

---

## 5. Scope decisions

### Included

- Recommendation only.
- One trained recommender family in the main experiment, plus one additional architecture for robustness if feasible.
- MovieLens-1M as the primary dataset.
- One sparse dataset as a secondary dataset, preferably Amazon-Book rebuilt from raw data or another dataset with reliable timestamps and metadata.
- User-specific explanations.
- Monte Carlo Shapley estimation.
- Feasible, bounded, system-executable interventions.
- Offline ranking evaluation with frozen models and fixed candidate sets.

### Excluded from the revised paper

- Clustering and tabular prediction.
- Wine Quality and Beijing Air Quality.
- Claims of a general framework spanning unsupervised learning and recommendation.
- Human-subject experiments.
- Open-ended user recourse.
- Uncontrolled online A/B testing.
- Provider fairness as a primary objective.
- A new hypergraph recommender architecture.
- Exact Shapley values for the recommendation player set.

---

## 6. Player definition

The primary player set is the user's **historical interaction factors**, not all users, items, and contexts in the dataset.

For user \(u\), let the retained history be:

\[
P_u = \{p_{u1}, \ldots, p_{un_u}\},
\]

where each player is one observed user–item interaction, optionally carrying timestamp and interaction strength.

To control computation and make interventions comparable:

- retain at most the most recent \(n_{\max}=50\) interactions per user in the primary experiment;
- use the same history window for all attribution methods;
- report a sensitivity analysis for \(n_{\max}\in\{20,50,100\}\);
- never include validation or test interactions in the player set.

A player is therefore a recommendation factor that can be masked or downweighted in the user's profile. This is a user-level explanation, not a global source attribution.

### Optional secondary player regime

If implementation time allows, add candidate-item players as a separate experiment. Do not combine interaction players and candidate-item players in one attribution vector. The two regimes answer different questions and must have separate metrics and figures.

---

## 7. Recommendation model and evaluation protocol

### 7.1 Main model

Use a frozen, reproducible recommender. The recommended main model is one of:

- BPR-MF;
- LightGCN, if the implementation is already stable; or
- the existing recommendation model only after its masking behavior is documented and tested.

The main paper must not depend on unspecified DyHuCoG equations or undocumented hypergraph construction. If DyHuCoG is used, freeze the exact implementation, publish the graph construction, and include a simpler model as the primary reproducibility reference.

### 7.2 Data split

Use a temporal split:

- training: all interactions before the validation cutoff;
- validation: the next interaction per eligible user;
- test: the final held-out interaction per eligible user.

Tie handling must be deterministic and reported. All model training, hyperparameter tuning, candidate generation, and attribution-budget selection must be completed without using test interactions.

### 7.3 Fixed candidate set

Generate a candidate set \(C_u\) once from the frozen full model and reuse it for:

- the original recommendation;
- every coalition evaluation;
- every intervention evaluation;
- every attribution method.

This prevents candidate-retrieval changes from being mistaken for attribution or intervention effects.

Report candidate recall before reporting ActionShap results. If the held-out item is absent from \(C_u\), that user cannot contribute to a ranking-improvement claim and must be handled consistently across all methods.

### 7.4 Utility function

The primary characteristic function is the user's ranking utility on the fixed candidate set:

\[
v_u(S)=
\operatorname{NDCG@K}\left(f_u^{S}, y_u\right)
\]

where \(f_u^{S}\) is the frozen recommender evaluated with only the interaction players in coalition \(S\) active, and \(y_u\) is the held-out target.

The main paper uses one primary utility, NDCG@10 or NDCG@20. Recall, MRR, coverage, and diversity are secondary outcomes and must not be silently mixed into the Shapley value.

For a positive recommendation objective, define the intervention effect as:

\[
\Delta_u(a)=v_u(\operatorname{do}(a))-v_u(P_u).
\]

For interventions intended to remove harmful evidence, also report the signed effect and absolute effect separately. The direction of the action must be declared before results are inspected.

---

## 8. Feasible interventions

The primary intervention is **bounded interaction downweighting**.

For player \(p\), define:

\[
\operatorname{do}(p;\rho):
\quad w_p \leftarrow \rho w_p,
\qquad \rho\in\{0,0.25,0.5\}.
\]

This represents suppressing or discounting the influence of a historical interaction. The model is not retrained after each intervention; the intervention is applied at the declared input/profile level. If the selected recommender cannot support this operation faithfully, use interaction masking as the primary intervention and document the difference.

Each intervention must specify:

- factor being changed;
- allowable values;
- intervention cost;
- whether it is available to the user, system operator, or both;
- whether it is applied at inference or requires retraining;
- expected direction of the objective;
- and all implementation details.

### Feasibility and cost

Avoid subjective author-assigned modifiability scores in the primary experiment. Feasibility is determined by the intervention policy:

\[
m_u(p)=
\begin{cases}
1,&p\text{ has a permitted intervention under the declared policy},\\
0,&\text{otherwise}.
\end{cases}
\]

If all retained interaction players are technically mutable, the feasibility term is constant and must not be presented as a source of differentiation. In that case, actionability is measured by intervention effect, intervention cost, and stability—not by multiplying by an artificial modifiability label.

### Intervention budget

Declare a per-user budget \(B\), such as one or three interventions. The selected action is the feasible action with the highest predicted benefit under that budget. The same budget must be used for every explanation method.

---

## 9. Monte Carlo Shapley estimator

For each user and player \(p\), sample \(M\) random permutations of \(P_u\). For permutation \(\pi\), let \(S_{p,\pi}\) be the set of players preceding \(p\). Estimate:

\[
\hat{\phi}_{u,p}=
\frac{1}{M}
\sum_{m=1}^{M}
\left[
 v_u(S_{p,\pi_m}\cup\{p\})-v_u(S_{p,\pi_m})
\right].
\]

Use paired marginal evaluations so that the two coalition values for a marginal contribution share the same cached data and evaluation path.

### Required estimator settings

- Primary \(M\): 250 or 500 permutations, selected before final test evaluation.
- Convergence sweep: \(M\in\{25,50,100,250,500,1000\}\), subject to runtime.
- At least five independent random seeds for the convergence experiment.
- Antithetic permutations where possible: evaluate both a permutation and its reverse.
- Cache coalition/profile states to avoid duplicate model evaluations.
- Report standard error and confidence intervals for aggregate attribution statistics.

### Efficiency diagnostic

Monte Carlo estimates need not satisfy efficiency exactly. Report:

\[
\epsilon_{u}^{\mathrm{eff}}=
\left|
\sum_{p\in P_u}\hat\phi_{u,p} -
[v_u(P_u)-v_u(\varnothing)]
\right|.
\]

Report the mean, median, 95th percentile, and maximum efficiency error. Do not silently normalize attributions to force efficiency. A normalized version may be included as a sensitivity analysis only.

### Convergence criterion

Define the minimum usable \(M\) before final testing as the smallest value for which both conditions hold:

1. mean top-1 intervention agreement with the \(M=1000\) reference is at least 0.95; and
2. mean Spearman correlation with the reference is at least 0.95.

If these thresholds are not reached, report the instability rather than increasing \(M\) until a desired result appears.

---

## 10. Attribution baselines

At minimum compare:

1. Monte Carlo Shapley;
2. permutation importance;
3. LIME or a local surrogate;
4. gradient-based attribution, if the model is differentiable;
5. attention weights, only if the model has an attention mechanism;
6. random ranking as a negative control.

All methods must receive the same input factors, candidate set, evaluation users, and intervention budget.

The paper must distinguish:

- attribution computation;
- intervention selection;
- intervention execution;
- and outcome measurement.

No method may use the test outcome when producing its attribution or selecting its intervention.

---

## 11. Metrics

### 11.1 Attribution–Intervention Alignment

Primary metric:

\[
\operatorname{AIA}_{u}(g)=
\operatorname{Spearman}\left(
|\phi_{u,p}^{g}|,
|\Delta_u(p)|
\right).
\]

Report mean user-level AIA with bootstrap confidence intervals. Report Kendall's \(\tau\) as a robustness metric.

### 11.2 Modifiability-restricted AIA

If some factors are infeasible under the declared intervention policy, compute AIA only over feasible factors. Report the number of feasible factors and do not compare correlations across different cardinalities without a matched-cardinality control.

### 11.3 Top-k intervention precision

For \(k\in\{1,3,5\}\), measure whether the method's top-k factors contain one of the true top-k feasible intervention factors.

### 11.4 Intervention regret

Let \(a_g\) be the action selected using method \(g\), and \(a^*\) the best feasible action under the declared budget:

\[
\operatorname{Regret}_u(g)=
\Delta_u(a^*)-\Delta_u(a_g).
\]

Normalize only when the denominator is nonzero, and report the fraction of users for whom normalization is undefined.

### 11.5 Stability

Measure rank correlation of attributions across:

- Monte Carlo seeds;
- model seeds, if the model is stochastic;
- and small perturbations of the user history.

### 11.6 Standard recommendation metrics

Report NDCG@K, Recall@K, MRR, and candidate recall. These evaluate recommendation quality; they are not automatically explanation-quality metrics.

---

## 12. Actionability score

Do not use the original score that multiplies attribution by author-elicited modifiability labels.

For recommendation-only ActionShap, use two clearly separated quantities:

### Factor-level realized utility

\[
A_{u,p}=c_u(p)\cdot |\Delta_u(p)|\cdot s_{u,p},
\]

where:

- \(c_u(p)\) is the declared intervention cost normalized to \([0,1]\);
- \(|\Delta_u(p)|\) is the measured feasible intervention effect;
- \(s_{u,p}\) is attribution stability.

This is a descriptive factor score, not a proof that the explanation is correct.

### Method-level actionability

The primary method-level actionability measures are:

- AIA;
- top-k intervention precision;
- normalized intervention regret;
- intervention success rate;
- and stability.

A method should not receive a high actionability claim solely because its score includes the measured intervention effect. The method-level metrics must evaluate whether the attribution predicted that effect before it was measured.

---

## 13. Statistical analysis

Use user-level paired comparisons because every method is evaluated on the same users.

Required analyses:

- paired bootstrap confidence intervals;
- Wilcoxon signed-rank or paired permutation tests for method comparisons;
- Holm–Bonferroni correction across primary pairwise comparisons;
- effect sizes, not only p-values;
- five independent seeds where stochastic training or Monte Carlo estimation is involved.

The statistical unit is the user, not an individual coalition evaluation. Do not treat thousands of coalition evaluations from the same user as independent observations.

---

## 14. Required implementation changes

The current `ActionShap/code/` is primarily a static clustering prototype. It must be reorganized as follows:

```text
paper-ideas/ActionShap/code/
├── configs/
│   ├── movielens.yaml
│   └── sparse_dataset.yaml
├── actionshap/
│   ├── data.py                 # temporal split and user histories
│   ├── candidates.py           # fixed candidate sets and recall
│   ├── models/
│   │   ├── base.py
│   │   ├── bpr.py               # primary frozen recommender
│   │   └── lightgcn.py          # optional robustness model
│   ├── players.py               # interaction-player construction
│   ├── coalition.py             # profile masking/downweighting
│   ├── utility.py               # NDCG/Recall utility functions
│   ├── attribution.py           # MC Shapley and baselines
│   ├── intervention.py          # declared feasible interventions
│   ├── actionability.py         # AIA, regret, top-k precision
│   ├── convergence.py           # M sweep and rank stability
│   ├── stats.py
│   └── reporting.py
├── scripts/
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── build_candidates.py
│   ├── run_attribution.py
│   ├── run_interventions.py
│   ├── run_convergence.py
│   └── make_tables.py
├── tests/
└── results/
    ├── raw/
    ├── tables/
    └── figures/
```

### Files to archive, not delete

The following static-clustering files should be moved to an archive directory or clearly marked as legacy:

- `models/static.py`;
- `annotations/`;
- `data/raw/winequality-white.csv`;
- `scripts/freeze_annotation.py`;
- `scripts/run_static.py`;
- `docs/clustering_spec.md`;
- `results/raw/wine_static.json`.

Do not include the static code or wine data in the recommendation experiment package.

### Existing modules to rewrite

- `data.py`: recommendation datasets, temporal split, history truncation.
- `attribution.py`: Monte Carlo Shapley over user-specific interaction players.
- `intervention.py`: profile masking/downweighting and budgeted action selection.
- `metrics.py`: recommendation and intervention metrics.
- `rerank.py`: remove unless candidate-item interventions are explicitly added.
- `decomposition.py`: replace static H1/H2/H3 misalignment switches with convergence and error decomposition.

### New tests

1. Empty coalition utility is defined and deterministic.
2. Full coalition reproduces the original frozen recommendation.
3. Masking one interaction changes only the intended user profile.
4. Reversing a permutation produces valid marginal contributions.
5. Synthetic additive utility gives the expected Shapley ranking.
6. Identical players receive statistically indistinguishable estimates.
7. Increasing \(M\) reduces standard error on synthetic games.
8. Efficiency error is reported and not silently corrected.
9. No test or validation interactions enter the player set.
10. Fixed candidate sets are identical across methods and interventions.
11. A zero-budget intervention produces zero intervention effect.
12. Intervention selection obeys the declared budget.

---

## 15. Revised paper structure

```text
Abstract
1. Introduction
   1.1 Recommendation explanations and actionability
   1.2 Faithfulness versus feasible intervention
   1.3 Research questions and contributions
2. Related Work
   2.1 Explainable recommendation
   2.2 Shapley and Monte Carlo attribution
   2.3 Counterfactual and intervention-based recommendation
   2.4 Explanation evaluation
   2.5 Positioning against SignalShap
3. Problem Formulation
   3.1 User-specific recommendation factors
   3.2 Feasible intervention policy
   3.3 Recommendation utility
   3.4 Actionability and intervention regret
4. ActionShap
   4.1 Interaction-player game
   4.2 Monte Carlo Shapley estimator
   4.3 Variance reduction and convergence
   4.4 Feasible interventions
   4.5 AIA and decision-level evaluation
5. Experimental Protocol
   5.1 Datasets and temporal splits
   5.2 Frozen recommender and fixed candidates
   5.3 Attribution baselines
   5.4 Intervention budgets
   5.5 Statistical methodology
6. Results
   6.1 Candidate recall and recommendation quality
   6.2 Attribution–Intervention Alignment
   6.3 Faithfulness versus actionability
   6.4 Intervention precision and regret
   6.5 Monte Carlo convergence
   6.6 Stability and robustness
   6.7 Case studies
7. Limitations
8. Conclusion
Appendix A: Intervention and data protocol
Appendix B: Convergence and efficiency diagnostics
Appendix C: Hyperparameters and reproducibility
```

### Claims that must be removed from the old manuscript

Remove or rewrite all claims that ActionShap:

- unifies static clustering and dynamic recommendation;
- evaluates four datasets including Wine Quality and Beijing Air Quality;
- requires no human/domain specification at all;
- uses exact Shapley values for recommendation players;
- proves a cross-domain actionability theorem;
- compares TreeSHAP for clustering with recommendation attention;
- or reranks explanations across unrelated task types.

### Claims that may remain after revision

- Faithfulness and feasible intervention are different evaluation targets.
- Recommendation actionability can be evaluated without human-subject experiments when interventions are operationally defined and simulated offline.
- Monte Carlo Shapley is appropriate for large user-specific player sets.
- Attribution quality should be evaluated by intervention decision quality, not only deletion metrics.

---

## 16. Minimum acceptance criteria

The project is ready for paper writing only when all conditions below are met:

1. The primary dataset pipeline is deterministic and leakage-safe.
2. Candidate recall is reported and adequate for the chosen claim.
3. The full coalition and empty coalition are validated.
4. MC Shapley convergence is documented.
5. At least three attribution baselines run on the same users and factors.
6. Intervention semantics and budgets are frozen before final test evaluation.
7. AIA, top-k precision, regret, and stability are implemented with tests.
8. Efficiency error is reported for MC estimates.
9. At least one negative control and one synthetic validation game pass.
10. Results are reported with paired confidence intervals and corrected statistical tests.
11. No conclusion depends only on the composite Actionability Score.
12. The final manuscript distinguishes recommendation-factor attribution from source-level SignalShap attribution.

---

## 17. Recommended build order

1. Rewrite the data loader and freeze the temporal split.
2. Train and validate the baseline recommender.
3. Build fixed candidate sets and report candidate recall.
4. Implement profile masking for the empty, singleton, and full coalitions.
5. Implement exact Shapley on tiny synthetic games for validation only.
6. Implement Monte Carlo Shapley for real user histories.
7. Implement the intervention simulator and budgeted selection.
8. Add AIA, top-k precision, regret, and stability metrics.
9. Run the Monte Carlo convergence study.
10. Add attribution baselines.
11. Run the primary experiment on MovieLens-1M.
12. Add the sparse dataset only after the primary pipeline is stable.
13. Generate figures and tables from raw result files.
14. Rewrite the manuscript around the results; do not pre-fill headline numbers.

---

## 18. Final design decision

The revised ActionShap paper uses:

- recommendation only;
- interaction-level user-specific players;
- Monte Carlo Shapley values;
- bounded, declared profile interventions;
- fixed candidate sets;
- intervention-grounded metrics;
- and convergence/stability diagnostics.

This is the version that should be ranked first: it has a clearer novelty claim than SignalShap while remaining substantially more feasible than the original cross-domain ActionShap proposal.
