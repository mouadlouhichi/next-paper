# ActionShap Recommendation-Only Specification

**Status:** design specification, revision 4
**Scope:** replacement specification for the current cross-domain ActionShap proposal

> **Revision 4 correction contract (supersedes conflicting text below).** The Q1 audit found that the schema-v1 pilot could not support manuscript claims. Revision 4 therefore requires: (i) target margin as the explicitly labelled primary attribution utility after convergence preflight, with NDCG retained as the operational action outcome and a separate NDCG-utility sensitivity; (ii) model fitting on complete training histories, while the primary inference-time profile deliberately uses the retained player window only (older training context is not added as fixed scoring context); (iii) sampled negatives and full-catalogue candidates excluding the complete pre-test history, including validation; (iv) user, candidate, and tie seeds fixed independently of model seeds; (v) signed benefit selection for downweighting, an explicit no-action option, and all action sizes up to the budget; (vi) exact `B<=2` oracles for every primary user; (vii) real-data rather than synthetic masking gates; (viii) random control, genuine locally weighted LIME, direction metrics, recommendation quality, and success/abstention reporting; (ix) independent `M=1000` convergence references with rank and action agreement; (x) repeated seeds averaged within each distinct user before inference; and (xi) final evidence from two timestamped datasets and two history-conditioned models. Schema-v1 assets are archived under `paper/legacy_pilot/` and are not evidence for the revised manuscript. The executable contract is `code/configs/final.yaml` plus `code/scripts/run_final_suite.py`; `code/ActionShap_All.ipynb` is the end-to-end download, audit, execution, validation, and packaging interface over those tracked scripts.

> **Revision 2 changelog.** Seven corrections applied to the original draft, all before implementation:
> 1. **§7.1 — the recommended models were incompatible with the player definition.** BPR-MF and LightGCN hold static user embeddings, so profile masking cannot move their scores and the whole game would have degenerated silently. Replaced with history-conditioned model families and a mandatory gate (§7.1.1).
> 2. **§7.4 — the empty coalition was undefined** for a profile-aggregation model. Pinned to a zero profile vector with a fixed, seeded tie-break.
> 3. **§9 — the efficiency diagnostic is vacuous** under prefix-walk permutation sampling, where efficiency holds exactly by telescoping. Corrected, with the two valid resolutions stated.
> 4. **§11.1 — AIA had no null distribution.** Added a required within-user permutation null, motivated by the random-attribution result of 0.518 in the earlier experiments.
> 5. **§11.4 — \(a^*\) was undefined for budgets above one.** Pinned \(B=1\) as the diagnostic oracle, \(B=2\) as the primary joint-action comparison, and greedy as the defined fallback above \(B=2\), with a validation of the greedy gap.
> 6. **§7.2 / §14 — the tie-break rule was promised but not stated, and the repo path was wrong.** Both fixed.
> 7. **§11.1 — RQ2 was circular at \(B=1\).** Under single-player masking the measured intervention effect *is* leave-one-out with the sign flipped, so the ablation baseline scores a perfect AIA by algebra. Leave-one-out is redemoted to an oracle and the method comparison is moved to joint interventions at \(B\ge 2\).
>
> A runnable gate notebook covering items 1, 3, 4, 5 and 7 lives at `ActionShap/notebooks/00_gate_masking_sensitivity.ipynb`.
> 8. **Narrative correction — ActionShap evaluates actionability rather than proving Shapley superiority.** Method comparisons are secondary; AIA, intervention precision, regret, and faithfulness–actionability divergence are the primary results. The canonical executable is `ActionShap/code/ActionShap_All.ipynb`.
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

**ActionShap: Beyond Deletion Faithfulness toward Intervention-Robust Recommendation Explanations**

### Alternative titles

- **From Attribution to Intervention: Evaluating Actionable Explanations for Recommender Systems**
- **Do Recommendation Explanations Predict What Happens When We Act?**
- **Evaluating Recommendation Explanations by Feasible Intervention Outcomes**

---

## 3. Revised scientific contribution

ActionShap is an evaluation framework, not primarily a new recommender architecture. Given a trained recommender, an explanation method, and a declared intervention policy, ActionShap measures whether highly attributed recommendation factors predict the outcome of feasible interventions.

**Primary claim discipline.** ActionShap makes no a priori claim that Monte Carlo Shapley, LIME, or any other explainer is superior. The framework is the contribution; the explainers are interchangeable inputs to the evaluation protocol. Any method ranking is an empirical result and must be reported with paired uncertainty and a within-user null.

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

**Primary metric:** Spearman correlation between absolute attribution and absolute target-margin intervention effect, with cross-utility NDCG AIA reported separately.

### RQ2 — Faithfulness versus actionability
Do deletion-based faithfulness metrics agree with feasible-intervention metrics, and under what conditions do their rankings diverge?

**Hypothesis:** the rankings will diverge when deletion is infeasible, non-local, or outside the declared intervention budget. This is the central scientific hypothesis; it does not assume that any particular attribution method is best.

### RQ3 — Decision quality
Do explanation rankings lead to different intervention choices, and which methods minimize NDCG regret under the same feasible action budget?

This is an open comparison. Monte Carlo Shapley, weighted LIME, leave-one-out, greedy counterfactual search, and random attribution are evaluated under the same protocol; no superiority result is assumed.

### RQ4 — Generality and stability
Are the actionability conclusions stable across random seeds, user histories, model variants, intervention budgets, and Monte Carlo sample counts?

Monte Carlo convergence is a reproducibility analysis, not the paper's scientific contribution.

---

## 5. Scope decisions

### Included

- Recommendation only.
- One trained recommender family in the main experiment, plus one additional architecture for robustness if feasible.
- MovieLens-1M as the primary dataset.
- Amazon Digital Music 5-core rebuilt from the timestamped Amazon Review Data (2018) source as the sparse secondary dataset, with source/output SHA-256 provenance.
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

- retain at most the most recent \(n_{\max}=20\) interactions per user in the primary experiment, selected by the archived real-data masking preflight;
- use the same history window for all attribution methods;
- report sensitivity analyses for \(n_{\max}\in\{50,100\}\), including any masking-gate failure as a non-responsiveness boundary;
- never include validation or test interactions in the player set.

A player is therefore a recommendation factor that can be masked or downweighted in the user's profile. This is a user-level explanation, not a global source attribution.

### Optional secondary player regime

If implementation time allows, add candidate-item players as a separate experiment. Do not combine interaction players and candidate-item players in one attribution vector. The two regimes answer different questions and must have separate metrics and figures.

---

## 7. Recommendation model and evaluation protocol

### 7.1 Main model

> #### The model must be history-conditioned. This constraint is not negotiable and it disqualifies the obvious choices.
>
> The player set is the user's retained interactions (§6) and the intervention is applied at the profile level **without retraining** (§8). For `v_u(S)` to vary with `S` at all, the recommender's score must be a *function of which interactions are retained, evaluated at inference time*.
>
> **BPR-MF does not satisfy this.** Its user representation is a free parameter `P[u]` fixed during training; the score `P[u]·Q[i]` does not read the profile at inference. Masking or downweighting a historical interaction afterwards changes nothing. **LightGCN does not satisfy it either** — its user embedding is produced by message passing during training and baked in, so recomputing it from a retained-interaction subset would amount to re-running propagation, i.e. retraining, which §8 forbids.
>
> Building on either would produce `v_u(S)` constant in `S`, every `\(\hat\phi_{u,p}\approx 0\)`, an efficiency error of zero for the wrong reason, and an AIA that is a rank correlation over a constant vector. **None of this crashes.** It presents as uniformly flat results that can be mistaken for a genuine null finding, which is the worst possible failure mode for an evaluation paper.

Use a frozen, reproducible recommender whose scoring function consumes the retained history. The corrected experiment uses:

1. **Item-based neighbourhood CF (primary).** Scores are an explicit weighted mean of frozen item--item cosine similarities over retained history items. Masking removes one term and bounded downweighting scales it. The real MovieLens preflight showed strong masking sensitivity and recommendation quality above item popularity, making this the defensible primary model.
2. **Profile-aggregation model (architecture robustness).** The user representation is computed at scoring time from the retained history, e.g. \(p_u(S)=\big(\sum_{h\in S} w_h q_h\big)/\big(\sum_{h\in S} w_h\big)\) and \(f_u^{S}(i)=p_u(S)^\top q_i\). Item embeddings are trained once on each user's **complete training history** with a leave-one-out BPR objective (the sampled positive is excluded from the context that predicts it) and then frozen. The \(n_{\max}\) cap defines both the inference-time profile boundary and the attribution players; longer-window conditions are declared profile-window sensitivities. Any quality shortfall relative to popularity is disclosed as a robustness limitation rather than hidden.
3. **Sequential model (future extension).** SASRec or GRU4Rec can consume a masked sequence, but removal shifts positions and bounded downweighting is not naturally equivalent. Such a model requires a separate, explicit intervention semantics and is outside the frozen revision-4 matrix.

**BPR-MF and LightGCN may appear only as recommendation-quality reference points** in §6.1, never as the model under attribution, unless they first pass the masking-sensitivity gate below and the mechanism by which they do so is documented.

#### 7.1.1 Masking-sensitivity gate (run before any other implementation work)

Before building the attribution layer, verify on a sample of at least 200 **real users from each primary dataset--model experiment** that profile masking actually moves the output. Use one fixed 200-item sampled-ranking gate set, independent of candidate-size and full-catalogue robustness conditions, so target sparsity does not redefine model sensitivity. Primary-model sampled and full-catalogue runs are blocked by failure. A declared architecture-robustness model or history-length sensitivity may fail the gate and continue only to quantify where the player game becomes non-responsive; that failure is reported as a robustness boundary, not hidden. A synthetic gate is a unit test only and cannot satisfy this acceptance criterion:

- mask one uniformly chosen history item per user, rescore the fixed candidate set, and record the change;
- **gate 1:** the top-10 list changes for at least 50% of sampled users;
- **gate 2:** mean \(|\Delta \mathrm{NDCG@10}|\) over sampled users is at least \(10^{-3}\);
- **gate 3:** the same test on a static-embedding model returns exactly zero change, confirming the test itself has power.

If gates 1 and 2 fail in a primary or full-catalogue run, the declared player game is operationally non-responsive at that history cap and no amount of Monte Carlo sampling will fix it. Stop the primary run. A predeclared longer-history sensitivity may continue only to quantify and report that boundary.

The main paper must not depend on unspecified DyHuCoG equations or undocumented hypergraph construction. If DyHuCoG is used, freeze the exact implementation, publish the graph construction, include a simpler model as the primary reproducibility reference, and run the gate above on it first.

### 7.2 Data split

Use a temporal split:

- training: all interactions before the validation cutoff;
- validation: the next interaction per eligible user;
- test: the final held-out interaction per eligible user.

Tie handling must be deterministic and reported. **Specify the rule concretely rather than promising one:** sort each user's interactions by `(timestamp, original_record_index)` and take the last as test and the second-last as validation. This matters more than it looks. MovieLens timestamps are second-resolution and ties are rare, but any Amazon-derived secondary dataset carries day-resolution timestamps, where a user reviewing several items on one day produces ties routinely — and an unstable sort there silently changes which item is the test target between runs, which shows up later as irreproducible attributions rather than as a data bug. All model training, hyperparameter tuning, candidate generation, and attribution-budget selection must be completed without using test interactions.

### 7.3 Fixed candidate set

Build a fixed **sampled evaluation set** \(E_u\) once per user by including the held-out target and sampling a declared number of unseen negatives uniformly after excluding the user's **complete pre-test history, including validation**, not merely the truncated attribution window. This is not retrieval and must not be described as candidate recall. Freeze the candidate seed and global catalogue-wide tie-break independently of model and attribution randomness, so all five experiment seeds see identical users and evaluation items. Reuse the same \(E_u\) for:

- the original recommendation;
- every coalition evaluation;
- every intervention evaluation;
- every attribution method.

Target coverage is therefore exactly one by construction. Report the evaluation-set size, negative-sampling seed, and the fact that the primary results are sampled-ranking results. As a robustness check, evaluate a smaller user subset against the full unseen catalogue.

### 7.4 Utility function

The primary attribution game uses a continuous target-margin utility on the fixed candidate set:

\[
v_u^{\mathrm{attr}}(S)=\sigma\!\left(s_{u,y_u}^{S}-\frac{1}{L}\sum_{i\in\operatorname{TopL}_{-y_u}}s_{u,i}^{S}\right),
\]

where \(f_u^{S}\) is the frozen recommender evaluated with only coalition \(S\) active and \(y_u\) is the held-out target. A real-data convergence preflight showed that discrete NDCG coalition values left Monte Carlo ranks and action sets unstable even at \(M=1000\), whereas target margin achieved stable rank estimates. This is a design decision, not permission to call target-margin effects NDCG.

The operational outcome remains

\[
q_u(S)=\operatorname{NDCG@K}\left(f_u^{S},y_u\right).
\]

Every selected action is therefore scored in both target-margin and NDCG units, and regret is computed against separate exact oracles for each utility. Primary attribution alignment uses target-margin intervention effects; cross-utility AIA and decision regret use NDCG effects. A predeclared utility sensitivity runs the attribution game itself with NDCG and reports its missing/unstable cases.

**Define the empty coalition explicitly.** Both models return zero scores when \(S=\varnothing\). Thus \(v_u^{\mathrm{attr}}(\varnothing)=0.5\), while NDCG depends on one catalogue-wide seeded tie priority reused for every user and method. Report both null values and never rely on an implementation-specific argsort.

Recall, MRR, coverage, and diversity are recommendation-quality outcomes and must not be silently mixed into the attribution game.

Define utility-specific intervention effects:

\[
\Delta_u^{\mathrm{attr}}(a)=v_u^{\mathrm{attr}}(\operatorname{do}(a))-v_u^{\mathrm{attr}}(P_u),
\qquad
\Delta_u^{\mathrm{NDCG}}(a)=q_u(\operatorname{do}(a))-q_u(P_u).
\]

For interventions intended to remove harmful evidence, report signed and absolute effects separately. The direction of the action must be declared before results are inspected.

---

## 8. Feasible interventions

The primary intervention is **bounded interaction downweighting**, deliberately separated from deletion-based faithfulness.

For player \(p\), define:

\[
\operatorname{do}(p;\rho):
\quad w_p \leftarrow \rho w_p,
\qquad \rho\in\{0,0.25,0.5\}.
\]

Use \(\rho=0\) only for the deletion/faithfulness diagnostic. The primary feasible-action analysis uses \(\rho=0.5\) (with \(\rho=0.25\) as a sensitivity condition), so actionability is not algebraically identical to leave-one-out deletion. This separation is essential: if feasible action and faithfulness use the same deletion intervention, the paper cannot measure a faithfulness–actionability gap.

The intervention represents suppressing or discounting the influence of a historical interaction. The model is not retrained after each intervention; the intervention is applied at the declared input/profile level. If the selected recommender cannot support this operation faithfully, use interaction masking as the primary intervention and document the difference.

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

Use two budgets with different roles:

- **B=1:** diagnostic sanity check only. Leave-one-out is the exact oracle for single-player masking and must not be used as the headline method comparison.
- **B=2:** primary scientific comparison. This is the smallest budget at which interaction and redundancy can distinguish coalition-aware attribution from leave-one-out.
- **B=3:** optional robustness analysis, using the greedy oracle defined in §11.4.

The feasible space contains the no-action option and every action size **up to** the budget, \(\mathcal A_{u,B}=\{A:|A|\le B\}\). No method or oracle may be forced to perform a harmful action. Magnitude prediction and beneficial action selection are separate: for signed attributions under a fixed downweight intervention, predicted benefit is \(-\phi_{u,p}\), and only positive predicted benefits are selected. Absolute attribution may be used for AIA or a separately labelled change-magnitude diagnostic, but not to claim that an intervention improves utility. The same budget, intervention strengths, abstention rule, and signed action-selection rule must be used for every explanation method.

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

- Primary \(M\): a conservative floor of 500 permutations, increased if the independent convergence study selects more.
- Convergence sweep: \(M\in\{25,50,100,250,500,1000\}\), subject to runtime.
- At least five independent random seeds for the convergence experiment.
- Antithetic permutations where possible: evaluate both a permutation and its reverse.
- Cache coalition/profile states to avoid duplicate model evaluations.
- Report standard error and confidence intervals for aggregate attribution statistics.

### Efficiency diagnostic

\[
\epsilon_{u}^{\mathrm{eff}}=
\left|
\sum_{p\in P_u}\hat\phi_{u,p} -
[v_u(P_u)-v_u(\varnothing)]
\right|.
\]

Report the mean, median, 95th percentile, and maximum. Do not silently normalize attributions to force efficiency; a normalized version may appear as a sensitivity analysis only.

> **Know what this number can and cannot tell you — it depends on the estimator, and under the recommended one it is vacuous.** If each sampled permutation is evaluated by a **prefix walk** — order the players, add them one at a time, and take consecutive differences of \(v_u\) along the way — the marginal contributions telescope, so \(\sum_p \hat\phi_{u,p} = v_u(P_u)-v_u(\varnothing)\) holds **exactly for every single permutation**, and therefore exactly in the mean. The measured \(\epsilon^{\mathrm{eff}}_u\) will be floating-point noise on the order of \(10^{-16}\) no matter how few permutations you draw and no matter how badly the estimate has converged.
>
> That is not a validation of anything. Reporting it as evidence of estimate quality, and listing it in the acceptance criteria as though it certified convergence, would be reporting an identity as though it were a result — precisely the kind of thing a careful reviewer catches.
>
> Resolve it one of two ways, and state which:
> - **Keep the prefix walk** (recommended: it costs \(n_u+1\) evaluations per permutation rather than \(2n_u\), so it is both cheaper and lower-variance). Then say plainly that efficiency holds by construction, demote \(\epsilon^{\mathrm{eff}}\) to a numerical-stability check on the caching and arithmetic, and rely on §9's convergence criterion — not on efficiency — as the evidence that \(M\) is large enough.
> - **Use independent paired marginal sampling per player**, where each player's marginals are drawn from separately sampled coalitions. Then efficiency genuinely does not hold in finite samples and \(\epsilon^{\mathrm{eff}}\) becomes an informative convergence diagnostic, at roughly double the evaluation cost and higher variance.
>
> Do not use the prefix walk and then present its efficiency error as a convergence result.

### Convergence criterion

Define the minimum usable \(M\) before final testing as the smallest value for which both conditions hold against an **independently seeded** \(M=1000\) reference:

1. mean Spearman correlation with the reference is at least 0.95; and
2. mean Jaccard overlap of the signed \(B=2\) action set is at least 0.80.

Rank correlation must be defined for at least 95% of users. Exact top-1 agreement and the fraction of users meeting both thresholds are reported diagnostics, not selection rules, because near-tied factors make exact identity brittle. Constant games are counted and reported rather than assigned a correlation. The primary experiment may not use fewer permutations than the selected value. If these thresholds are not reached, report the instability rather than increasing \(M\) until a desired result appears.

---

## 10. Attribution baselines

At minimum compare:

1. Monte Carlo Shapley;
2. permutation importance;
3. genuinely locally weighted LIME over binary history masks (an unweighted global ridge mask model must be labelled as such, not as LIME);
4. greedy sequential-deletion counterfactual search, recomputed after each selection and never given bounded-intervention or oracle outcomes;
5. gradient-based attribution only for a differentiable utility (it is not defined for discrete NDCG and must not be silently computed on a different target);
6. attention weights, only if the model has an attention mechanism;
7. random ranking as a negative control.

All methods must receive the same input factors, candidate set, evaluation users, and intervention budget.

The paper must distinguish:

- attribution computation;
- intervention selection;
- intervention execution;
- and outcome measurement.

The primary experiment is a **retrospective, target-conditioned audit**: the held-out target defines the ranking utility being explained, just as a labelled test instance defines the object of an explanation. This is not a prospective deployment policy. No method may inspect measured intervention effects, oracle actions, or aggregate test results when producing its attribution or selecting its intervention, and no hyperparameter or policy may be changed after final test evaluation. Any prospective-action claim requires a separate validation-derived policy whose test-time explanation target is available without future feedback.

---

## 11. Metrics

### 11.1 Attribution–Intervention Alignment

Primary metric:

\[
\operatorname{AIA}^{\mathrm{attr}}_{u}(g)=
\operatorname{Spearman}\left(
|\phi_{u,p}^{g}|,
|\Delta_u^{\mathrm{attr}}(p)|
\right).
\]

Also report cross-utility \(\operatorname{Spearman}(|\phi|,|\Delta^{\mathrm{NDCG}}|)\), signed alignment \(\operatorname{Spearman}(-\phi,\Delta^z)\), and direction accuracy for each utility \(z\). Report mean distinct-user AIA with bootstrap confidence intervals and missing constant-vector counts. Kendall's \(\tau\) is a robustness metric.

#### Required: a permutation null for AIA

An AIA value is not interpretable on its own, and the earlier cross-domain ActionShap experiments demonstrated exactly why — a *random* attribution scored 0.518 on the alignment metric, which is meaningless without knowing what the metric returns under chance for that player-set size and effect distribution. Listing random ranking among the baselines (§10) is necessary but not sufficient, because it yields a single number with no dispersion.

For every method, construct the primary null by **shuffling \(|\Delta_u^{\mathrm{attr}}(p)|\) across the players within each user and seed**, recomputing AIA, and repeating at least 1,000 times. Report:

- the observed mean AIA against the null distribution's mean and 95th percentile;
- a permutation \(p\)-value per method;
- and the null's own mean, which should sit near zero — **if it does not, the metric is mis-specified and must be fixed before any method comparison is reported.**

Shuffle within user, never across users: player-set cardinality \(n_u\) varies, and Spearman's null distribution depends on it, so a pooled shuffle would mix nulls of different widths and produce a biased reference.

#### Joint-action attribution rule

For budgets \(B\ge 2\), individual attributions must be converted into a score for a joint intervention set before an action is selected. Magnitude-only change prediction may use

\[
\widehat{\Phi}^{\mathrm{mag}}_{u,g}(A)=\sum_{p\in A}|\widehat{\phi}_{u,p}^{g}|,
\qquad |A|\le B,
\]

but it is not the beneficial-action rule. Under the primary fixed downweighting intervention, the prespecified predicted benefit of a signed attribution is

\[
\widehat{\Phi}^{\mathrm{benefit}}_{u,g}(A)=-\sum_{p\in A}\widehat{\phi}_{u,p}^{g}.
\]

The method selects the positive-benefit action with largest score, or \(A=\varnothing\) when no action has positive predicted benefit. The primary action is

\[
\widehat{A}_{u,g}=\operatorname{argmax}_{A\in\mathcal{A}_{u,B}}\widehat{\Phi}^{\mathrm{benefit}}_{u,g}(A),
\qquad \mathcal A_{u,B}=\{A:|A|\le B\}.
\]

The action score and tie-break must be fixed before test evaluation. For the primary \(B=2\) experiment, enumerate no action, all singletons, and all pairs on the retained history; for \(B=3\), use the greedy procedure in §11.4 unless restricted exhaustive validation is being run. Report magnitude AIA, signed alignment, direction accuracy, success, and abstention separately.

This set-level rule is required because a per-player AIA alone does not evaluate whether a method chooses a good *joint* intervention.

#### The single-player intervention makes leave-one-out the ground truth. Do not run RQ2 at \(B=1\).

Work the algebra before designing the comparison. The deletion/faithfulness diagnostic at \(\rho=0\) is masking one player, so the measured effect is

\[
\Delta_u(p)=v_u(P_u\setminus\{p\})-v_u(P_u),
\]

while leave-one-out attribution is \(\mathrm{LOO}_u(p)=v_u(P_u)-v_u(P_u\setminus\{p\})\). These are the same number with opposite sign, so \(|\mathrm{LOO}_u(p)| = |\Delta_u(p)|\) identically and

\[
\operatorname{AIA}_u(\mathrm{LOO}) = \operatorname{Spearman}\big(|\mathrm{LOO}_u|, |\Delta_u|\big) = 1.0
\]

for every user, by construction and not by merit. Top-1 intervention precision and regret for leave-one-out are likewise perfect. **RQ2 as posed would therefore be answered "the ablation baseline wins, decisively" before a single model is trained**, and the finding would be an identity rather than a result.

Two consequences for the design:

1. **Report leave-one-out at \(B=1\) as the oracle, not as a competitor.** It *is* the ground-truth ranking under a single-player masking intervention. Label it as such in the table, and use it as the upper bound against which the other methods, Shapley included, are measured. A method approaching oracle performance without evaluating \(n_u\) counterfactuals is the interesting claim; beating the oracle is impossible.
2. **Site the real comparison at \(B\ge 2\).** Leave-one-out assumes the joint effect of masking several players is the sum of their individual effects. That assumption fails exactly when players interact — two history items supporting the same recommendation are mutually redundant, so removing either alone does little and removing both is decisive. This is the same redundancy failure the group's source-attribution work formalizes, transplanted from sources to interactions, and it is where differences between explanation methods can be measured without assuming which method should win. Make the \(B=2\) and \(B=3\) joint-intervention comparison the headline of RQ2, and report \(B=1\) as the sanity check that everything is wired correctly.

If the \(B\ge 2\) comparison shows no separation, that is a reportable negative result about this player set — but the paper must not fall back on the \(B=1\) numbers to manufacture one.

### 11.2 Modifiability-restricted AIA

If some factors are infeasible under the declared intervention policy, compute AIA only over feasible factors. Report the number of feasible factors and do not compare correlations across different cardinalities without a matched-cardinality control.

### 11.3 Top-k intervention precision

For \(k\in\{1,3,5\}\), measure whether the method's top-k factors contain one of the true top-k feasible intervention factors.

### 11.4 Intervention regret

Let \(a_g\) be the action selected from method \(g\)'s target-margin attribution, and \(a^{*,z}\) the best feasible action for utility \(z\in\{\mathrm{attr},\mathrm{NDCG}\}\):

\[
\operatorname{Regret}_u^z(g)=
\Delta_u^z(a^{*,z})-\Delta_u^z(a_g).
\]

NDCG regret is the headline decision result. Target-margin regret checks consistency with the attribution game. Normalize only when the corresponding oracle effect is positive, and report the undefined fraction.

#### How \(a^*\) is computed — pin this down before implementing

\(a^{*}\) is an oracle over the feasible action space, and its cost depends entirely on the budget. With the primary \(n_{\max}=20\) players and three diagnostic intervention strengths \(\rho\in\{0,0.25,0.5\}\):

| Budget | Action space | Evaluations for exhaustive \(a^*\) | Verdict |
|---|---|---|---|
| \(B=1\) | one player, one \(\rho\) | \(20\times 3=60\) | exhaustive, trivially affordable |
| \(B=3\) | three players, each with a \(\rho\) | \(\binom{20}{3}\times 3^3 = 30{,}780\) | avoid in the main all-user sweep; validate greedy on a restricted subset |

**Use \(B=1\) deletion only as the algebraic diagnostic oracle.** It makes the single-player reference exact, but it cannot support the main deletion-method comparison because leave-one-out is identical to the measured intervention effect. Use \(B=2\) as the primary experiment; with the primary retained history capped at 20, enumerate no action, every singleton, and every pair for **every primary user**, using the same action space for every method. This is the exact \(B\le2\) oracle, not a greedy approximation.

For \(B>2\), \(a^*\) is defined by **greedy forward selection with early stopping**: pick the best improving single action, fix it, re-measure remaining actions against that modified profile, and stop when no addition improves utility or the budget is reached. State that this is a greedy oracle rather than a true optimum, so reported regret is a lower bound on regret against the exact optimum. Validate the approximation on a restricted setting small enough for exhaustive search and report the gap. Every method's selected action must be scored under the same budget and abstention policy.

### 11.5 Stability

Measure rank correlation of attributions across:

- Monte Carlo seeds;
- model seeds, if the model is stochastic;
- and small perturbations of the user history.

### 11.6 Standard recommendation metrics

Report NDCG@K, Recall@K, MRR, target coverage (=1 by construction), and evaluation-set size. These evaluate recommendation quality; they are not automatically explanation-quality metrics.

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

The statistical unit is the **distinct user**, not an individual coalition evaluation and not a seed--user row. When the same users are evaluated under five experiment seeds, average seeds within user before primary inference or use an explicitly hierarchical bootstrap/mixed model. Do not report 500 users under five seeds as \(n=2500\). Sign-permutation p-values use the plus-one correction and are bounded below by \(1/(R+1)\), never printed as zero. Report paired effect sizes, missing constant-vector correlations, undefined normalized regrets, success, and abstention.

---

## 14. Required implementation changes

The current `ActionShap/code/` is primarily a static clustering prototype. It must be reorganized as follows:

```text
ActionShap/code/
├── configs/
│   └── final.yaml             # frozen two-dataset/two-model experiment matrix
├── actionshap/
│   ├── recommendation_data.py   # generic temporal split and deterministic user sampling
│   ├── candidates.py            # sampled and full-unseen sets plus global tie priorities
│   ├── models/
│   │   ├── itemknn.py           # primary history-conditioned neighbourhood model
│   │   └── profile.py           # leave-one-out-trained architecture robustness model
│   ├── recommendation.py        # UserGame, utilities, cached MC Shapley, action rules
│   ├── baselines.py             # LOO, weighted LIME, greedy CF, random
│   ├── evaluation.py            # effects, batched exact oracle, AIA, nulls, gates
│   ├── convergence.py           # independent M=1000 rank/action convergence
│   └── stats.py                 # distinct-user hierarchical comparisons
├── scripts/
│   ├── download_datasets.py
│   ├── prepare_amazon_digital_music.py
│   ├── run_recommendation.py
│   ├── run_convergence.py
│   ├── run_final_suite.py
│   ├── make_paper_assets.py
│   ├── validate_manuscript.py
│   └── package_results.py
├── tests/
└── results/
    ├── raw/                      # ignored schema-v2 JSON
    └── release/                  # content-addressed archival package
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

0. **Masking sensitivity.** Masking a history item changes `v_u` for a majority of sampled users, and the same test on a static-embedding model returns exactly zero. This is the gate of §7.1.1 and it must be the first test written, because every other test below can pass on a model for which the whole game is degenerate.
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

0. **The masking-sensitivity gate of §7.1.1 passes on the primary model.** Nothing below is meaningful until it does.
1. The primary dataset pipeline is deterministic and leakage-safe.
2. Candidate recall is reported and adequate for the chosen claim.
3. The full coalition and empty coalition are validated, with the empty-coalition tie-break fixed and reported.
4. MC Shapley convergence is documented against the §9 criterion, **not** against efficiency error.
5. At least three attribution baselines run on the same users and factors.
6. Intervention semantics and budgets are frozen before final test evaluation, with \(B=1\) treated as a diagnostic oracle and \(B=2\) used for the primary joint-action comparison.
7. AIA, top-k precision, regret, and stability are implemented with tests.
8. Efficiency error is reported, with an explicit statement of whether the estimator makes it exact by construction.
9. At least one negative control and one synthetic validation game pass, the AIA permutation null of §11.1 centres near zero, and the joint-action aggregation rule is tested on synthetic interactions.
10. Results are reported with paired confidence intervals and corrected statistical tests.
11. No conclusion depends only on the composite Actionability Score.
12. The final manuscript distinguishes recommendation-factor attribution from source-level SignalShap attribution.
13. NDCG and target-margin outputs have separate schema fields, labels, tables, and figures.
14. Complete pre-test histories, not truncated player windows, define unseen candidates; the full-catalogue check excludes seen items.
15. The primary action space includes no action and all sizes up to \(B=2\), with exact oracles for every primary user.
16. At least 1,000 distinct primary users (or all eligible users), two timestamped datasets, two history-conditioned models, five common seeds, and a random control are present.
17. The real-data gate passes for every primary-ItemKNN dataset condition; any robustness-model failure is explicitly retained. The primary \(M\) meets independent rank/action convergence criteria.
18. The asset manifest uses repository-relative paths and hashes, and schema-v1 pilot assets are excluded from final tables.
19. Target rank, NDCG, Recall, and MRR are reported against item popularity; the primary ItemKNN model does not underperform popularity on NDCG or Recall. Profile-model shortfalls are retained as an explicit robustness limitation.

---

## 17. Recommended build order

0. Archive schema-v1 assets and freeze revision 4 before inspecting corrected final outcomes.
1. Build deterministic MovieLens and Amazon Digital Music temporal datasets with source hashes.
2. Fit ItemKNN and the leave-one-out-trained profile robustness model on complete training histories.
3. Run the real 200-user gate and quality preflight; freeze ItemKNN, \(n_{\max}=20\), and target-margin attribution utility.
4. Build sampled and full-unseen candidate sets from complete pre-test histories with independent seeds.
5. Validate empty/full coalitions, exact tiny-game Shapley, symmetry, redundancy, abstention, and dual-utility oracles.
6. Run independent \(M=1000\) convergence studies before any final attribution run.
7. Execute the frozen two-dataset, two-model, five-seed primary and full-catalogue matrix.
8. Execute the predeclared history, rho, candidate, budget, and NDCG-utility sensitivities.
9. Generate distinct-user statistics, tables, figures, and content-addressed manifests only from schema-v2 raw results.
10. Write numerical conclusions only after `paper/final/manifests/validation_report.json` says PASS.

---

## 18. Final design decision

The revised ActionShap paper uses:

- recommendation only;
- interaction-level user-specific players capped at 20 in the primary game;
- primary ItemKNN plus latent-profile architecture robustness;
- target-margin attribution with separately evaluated NDCG outcomes and oracles;
- Monte Carlo Shapley as one interchangeable explainer;
- bounded, signed, abstention-aware profile interventions;
- fixed sampled and full-unseen candidate sets;
- distinct-user intervention-grounded metrics;
- and independent rank/action convergence plus stability diagnostics.

This is the version that should be ranked first: it measures recommendation actionability directly, treats explanation methods as exchangeable inputs, and keeps Monte Carlo Shapley as an estimator rather than the claimed scientific endpoint.
