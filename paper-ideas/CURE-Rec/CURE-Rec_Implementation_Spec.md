# CURE-Rec — Technical Implementation Specification and Registered Evaluation Plan

**Companion to:** `CURE-Rec_Paper_Structure.md`  
**Status:** pre-implementation design specification  
**Purpose:** This file defines what must be built, what must be measured, and what claims are prohibited until the relevant causal and computational checks pass.

> **Read this first.** CURE-Rec is not implemented by adding a Shapley regularizer to a recommender. It is a two-layer system: a fixed, reproducible base recommendation policy produces candidate slates; an intervention layer evaluates small portfolios of explicit policy transformations under a causal response model and causal ambiguity set. The proposed method selects a portfolio using **direct robust coalition value**. Shapley regions explain and audit that decision; they are not an excuse to replace robust policy evaluation with a sum of individual scores.

---

# PART A — IMPLEMENTATION

## A.1 Design decisions fixed before coding

| Decision | Fixed choice | Reason |
|---|---|---|
| Domain | Recommender systems | Preserves the research program’s application focus while changing the scientific object from entity attribution to policy intervention. |
| Main task | Finite-horizon recommendation-policy adaptation | Makes delayed satisfaction, fatigue, exposure, and provider outcomes meaningful. |
| Players | Six composable recommendation-policy transformations | Keeps the game exact: \(2^6=64\) coalitions. |
| Coalition value | Discounted long-term causal utility minus intervention cost | Measures outcome of a policy change rather than predictive importance. |
| Causal uncertainty | Explicit \(\Gamma\)-sensitivity ambiguity set plus confidence region | Avoids calling model ensembles “partial identification” without assumptions. |
| Portfolio selection | Exact direct maximin enumeration | Prevents invalid addition of lower Shapley values and avoids unproved greedy claims. |
| Attribution | Exact Shapley and interaction values per plausible model | No permutation sampling error at six players. |
| Primary evidence | CURE-Sim sequential SCM | Necessary for oracle causal credit and intervention ground truth. |
| Secondary evidence | Randomized/propensity-logged recommendation data after audit | Supports only the short-horizon claims justified by the log. |
| Base recommender | One documented baseline plus one robustness baseline | CURE-Rec is a policy layer, not an architecture paper. |
| Retraining feedback | Excluded from main horizon | Full training-loop dynamics are a separate causal object; add only as an appendix simulation stress test. |

## A.2 Proposed repository layout

This is a planned layout. Do not create empty modules merely to satisfy the tree; build in the order in §B.7.

```text
CURE-Rec/
├── requirements.txt
├── README.md
├── configs/
│   ├── curesim_base.yaml
│   ├── curesim_confounded.yaml
│   ├── curesim_policy_shift.yaml
│   ├── coat.yaml
│   ├── yahoo_r3.yaml
│   └── interventions.yaml
├── cure_rec/
│   ├── __init__.py
│   ├── config.py                 # typed config loading + config hash
│   ├── seed.py                   # all RNGs, deterministic run manifest
│   ├── data/
│   │   ├── schema.py             # logged-slate schema and validation types
│   │   ├── loaders.py            # CURE-Sim, Coat, Yahoo R3, audited logs
│   │   ├── audit.py              # propensity, timing, support, overlap audit
│   │   └── splits.py             # train/validation/test temporal cohorts
│   ├── policy/
│   │   ├── base.py               # BasePolicy protocol
│   │   ├── bpr.py                # primary reproducible base policy
│   │   ├── lightgcn.py           # optional robustness policy
│   │   ├── candidates.py         # candidate generation and availability filters
│   │   └── slates.py             # canonical slate representation
│   ├── interventions/
│   │   ├── base.py               # Intervention protocol
│   │   ├── library.py            # six fixed intervention definitions
│   │   ├── compose.py            # canonical composition g_S
│   │   └── constraints.py        # budget/capacity/relevance/provider constraints
│   ├── environment/
│   │   ├── state.py              # platform state S_t
│   │   ├── curesim.py            # oracle sequential SCM environment
│   │   ├── dynamics.py           # user/item/provider transition mechanisms
│   │   └── rollout.py            # policy rollout interface
│   ├── causal/
│   │   ├── response.py           # learned response/world-model interface
│   │   ├── ope.py                # sequential IPS, DR, diagnostics
│   │   ├── sensitivity.py        # Gamma ambiguity and adversarial weights
│   │   ├── ambiguity.py          # model-consistent M_Gamma interface
│   │   └── calibration.py        # held-out policy calibration checks
│   ├── game/
│   │   ├── coalitions.py         # all 64 coalition masks and cache keys
│   │   ├── value.py              # V_M(S), confidence/outer bounds
│   │   ├── shapley.py            # exact intervention Shapley values
│   │   ├── interaction.py        # exact pair interaction indices
│   │   └── regions.py            # model-consistent and outer attribution bounds
│   ├── planner/
│   │   ├── robust.py             # direct maximin enumeration
│   │   ├── heuristics.py         # individual, greedy, lower-Shapley baselines
│   │   └── certificate.py        # selection/explanation/abstention report
│   ├── metrics/
│   │   ├── recommendation.py     # NDCG, Recall, MRR as diagnostics
│   │   ├── causal.py             # attribution MAE, coverage, robust regret
│   │   ├── ecosystem.py          # fatigue, concentration, provider exposure
│   │   └── stats.py              # CIs, paired tests, Holm correction
│   └── report/
│       ├── tables.py             # manuscript table emitters
│       ├── figures.py            # manuscript figure emitters
│       └── cards.py              # intervention explanation cards
├── scripts/
│   ├── 00_audit_log.py
│   ├── 01_train_base_policy.py
│   ├── 02_build_curesim_logs.py
│   ├── 03_fit_response_models.py
│   ├── 04_run_exact_game.py
│   ├── 05_select_portfolios.py
│   ├── 06_run_ablations.py
│   ├── 07_emit_report.py
│   └── run_all.py
├── tests/
├── results/
│   ├── manifests/
│   ├── raw/
│   ├── coalitions/
│   ├── tables/
│   └── figures/
└── docs/
    ├── causal_assumptions.md
    ├── dataset_audits.md
    └── intervention_cards.md
```

## A.3 Environment and dependency policy

Initial environment:

```text
python >= 3.11, < 3.13
numpy
scipy
pandas
scikit-learn
pydantic or attrs            # typed configs/schemas
pyyaml
pytest
hypothesis                   # property-based tests
matplotlib / seaborn
networkx                     # causal and intervention dependency graphs
cvxpy                        # only if ambiguity extremization requires it
pytorch                      # only for LightGCN/SASRec or neural response model
```

Implementation principles:

- Keep CURE-Sim, coalition enumeration, and exact Shapley computation NumPy-first and CPU-capable.
- Do not require a GPU for the game itself.
- Isolate optional PyTorch policies/world models so synthetic theory experiments run without them.
- Every run writes a config hash, data hash, commit hash, seed, hardware summary, and causal-assumption manifest.

## A.4 Data contract and mandatory logging audit

No real dataset may enter a causal result table before passing `00_audit_log.py`.

### Required log fields

| Field | Required for | Failure if absent |
|---|---|---|
| user/cohort ID | repeated measures and trajectories | cannot separate users or cohorts |
| timestamp/session order | sequential estimands | no long-horizon claim |
| full displayed slate or factorized action record | policy intervention evaluation | no valid slate-policy value |
| position/exposure indicator | exposure-bias modeling | cannot distinguish shown from merely available |
| logging propensity or randomized assignment | IPS/DR/OPE | only descriptive/model-based claims |
| candidate availability | overlap/support test | transformed slates may be impossible |
| response timing | treatment/outcome ordering | post-treatment leakage risk |
| item/provider identifiers | ecosystem/provider outcomes | no provider-exposure claim |
| outcome definition | policy utility | no causal objective |

### Audit outputs

The audit script must write:

1. a data dictionary;
2. a timing DAG showing what is pre-treatment, treatment, mediator, and outcome;
3. propensity distribution and extreme-weight diagnostics;
4. empirical support/overlap report for every intervention operator;
5. effective sample size after IPS weighting;
6. a permitted-claims label: `descriptive`, `short-horizon OPE`, `sensitivity-bounded`, or `not causally evaluable`.

**Hard rule:** MovieLens, Amazon Reviews, MIND, or arbitrary implicit-feedback datasets are not assumed to satisfy this contract. They can be used for semi-synthetic or descriptive robustness experiments, not presented as long-term causal exposure evidence without a successful audit.

## A.5 CURE-Sim: primary sequential SCM

### A.5.1 Why it is required

CURE-Sim is the primary scientific benchmark because it provides:

- true policy-intervention values;
- true Shapley values and interactions;
- controlled hidden confounding;
- known long-horizon feedback dynamics;
- a way to test whether ambiguity intervals cover the truth;
- oracle robust portfolios.

A real recommendation log alone generally cannot provide all of those.

### A.5.2 State

At time \(t\), represent platform state as:

\[
\mathcal S_t=(Z_t,F_t,E_t,Q_t,A_t),
\]

where:

| Variable | Meaning |
|---|---|
| \(Z_t\) | user-item latent affinity state |
| \(F_t\) | user fatigue and repeated-exposure state |
| \(E_t\) | user-item exposure history |
| \(Q_t\) | item/provider popularity and aggregate exposure state |
| \(A_t\) | candidate availability and catalogue state |

The deployed base policy creates slate \(L_t\), users respond \(Y_t\), and the environment transitions:

\[
\mathcal S_{t+1}=F_{M^\star}(\mathcal S_t,L_t,Y_t,U_t).
\]

### A.5.3 Structural mechanisms

Implement configurable mechanisms, all with disclosed equations and deterministic seeds:

1. **Preference:** latent affinity drives a component of click/satisfaction probability.
2. **Exposure:** displayed position changes response probability independently of preference.
3. **Fatigue:** repeated exposures increase short-term click decay and long-term dissatisfaction after a threshold.
4. **Novelty:** novel items may lower immediate click probability but improve delayed satisfaction for novelty-seeking users.
5. **Popularity feedback:** exposure increases item popularity, which affects future base-policy scores.
6. **Provider competition:** exposure is allocated among provider groups, yielding a platform-level disparity statistic.
7. **Unobserved confounding:** an unobserved user/context variable influences both logged policy assignment and response.
8. **Policy shift:** testing logs are generated under a policy differing from the response-model training policy.

### A.5.4 Regimes

At minimum generate six named regimes:

| Regime | Intended causal property |
|---|---|
| `additive` | interventions add independently |
| `complementary` | exploration + repeat cap has positive interaction |
| `redundant` | novelty + long-tail slot overlap in value |
| `antagonistic` | provider balancing conflicts with immediate relevance |
| `delayed` | fatigue mitigation has delayed long-term benefit |
| `confounded_shift` | hidden policy-assignment confounding and test-policy shift |

Every regime must expose an oracle `rollout(policy, seed)` and an exhaustive `oracle_values()` method over all 64 coalitions.

### A.5.5 Ground-truth outputs

For each environment seed, persist:

```text
oracle_values.json             # V_M*(S) for all 64 coalitions
oracle_shapley.json            # phi_i(M*)
oracle_interactions.json       # I_ij(M*)
oracle_feasible_portfolios.json
oracle_robust_portfolio.json   # for declared stress-set construction
logged_trajectories.parquet
logged_propensities.parquet
manifest.yaml
```

## A.6 Base recommendation policies

### Primary policy

Use a reproducible, documented base ranker. Recommended first implementation:

- BPR-MF or a simple session-aware score model for CURE-Sim;
- candidate set size and negative sampling fixed in config;
- policy score must be queryable so intervention operators can transform the slate.

### Robustness policy

Add exactly one standard robustness policy after the synthetic pipeline is correct:

- LightGCN **or** SASRec, using a maintained implementation and frozen hyperparameters.

### Policy API

```python
class BasePolicy(Protocol):
    def score(self, state: PlatformState, candidates: np.ndarray) -> np.ndarray: ...
    def recommend(self, state: PlatformState, k: int) -> Slate: ...
    def metadata(self) -> dict: ...
```

The CURE-Rec game must never depend on internal embeddings or base-model architecture. It sees only state, candidate availability, and ranked scores/slates.

## A.7 Fixed intervention library

### A.7.1 Six intervention players

| ID | Name | Transformation | Cost / capacity accounting |
|---|---|---|---|
| `repeat_cap` | Repeated-exposure cap | Exclude or penalize items shown \(\ge r\) times in a rolling window. | No reserved slot; possible relevance cost. |
| `explore_slot` | Exploration slot | Place one uncertainty-guided candidate in one eligible position. | One slot. |
| `tail_slot` | Long-tail slot | Place one item below a popularity quantile in one eligible position. | One slot. |
| `diversify` | Category diversity | Re-rank to meet declared category-distance target. | Soft score/relevance cost. |
| `novel_slot` | User-conditional novelty | Place one item beyond a history-similarity threshold. | One slot. |
| `provider_balance` | Bounded provider balancing | Apply a bounded re-ranking correction against aggregate provider exposure gap. | Provider-disparity constraint interaction. |

### A.7.2 Canonical composition

Operators must compose deterministically in this order:

```text
1. repeat_cap
2. eligibility and availability filters
3. explore_slot / tail_slot / novel_slot candidate injection
4. diversify re-ranking
5. provider_balance re-ranking
```

The final slate is a function only of the coalition membership mask, not of the order in which players enter a Shapley permutation.

```python
def apply_coalition(base_policy: BasePolicy, state: PlatformState, mask: CoalitionMask) -> Slate:
    slate = base_policy.recommend(state, k=config.k)
    for operator in CANONICAL_ORDER:
        if mask.includes(operator):
            slate = operator.apply(slate, state)
    return slate
```

### A.7.3 Coalition semantics versus portfolio constraints

All six operators must be evaluable in every coalition. Do not skip an intervention inside a Shapley permutation because of a budget or slot constraint.

- **Game:** evaluate all \(2^6\) policy transformations with the canonical operator definition.
- **Selection:** apply budget, capacity, relevance, provider, and safety restrictions to determine feasible portfolios.

If an operator family becomes genuinely mutually exclusive, encode it as a single categorical policy component or adopt a formally restricted game. Do not silently remove it from coalition evaluations.

## A.8 Utility and constraints

### A.8.1 Value function

For model \(M\), coalition \(S\), and horizon \(H\):

\[
V_M(S)=
\mathbb E_{P_M^{\pi_S}}
\left[
\sum_{h=0}^{H-1}\gamma^h
\left(
\alpha_{\mathrm{sat}}R^{\mathrm{sat}}_{t+h}
+\alpha_{\mathrm{ret}}R^{\mathrm{ret}}_{t+h}
-\alpha_{\mathrm{fat}}R^{\mathrm{fatigue}}_{t+h}
\right)
\right]
-\lambda_cC(S).
\]

Use one declared main utility. Do not put every desirable recommender metric into one scalar reward simply because it is available.

### A.8.2 Hard constraints

A portfolio belongs to \(\mathcal F_{\mathrm{safe}}\) only if it satisfies:

\[
C(S)\le B,
\]

\[
\inf_{M\in\mathfrak M_\Gamma}\Delta_{\mathrm{rel},M}(S)\ge -\epsilon_{\mathrm{rel}},
\]

\[
\sup_{M\in\mathfrak M_\Gamma}D_{\mathrm{provider},M}(S)\le\epsilon_{\mathrm{prov}},
\]

\[
\sup_{M\in\mathfrak M_\Gamma}\Pr_M(\mathrm{unsafe\;slate}\mid\pi_S)\le\delta.
\]

`NDCG` and `Recall` remain reporting diagnostics. Do not use ordinary held-out NDCG as a causal constraint unless it has a valid counterfactual estimator in that setting.

## A.9 Causal estimators and ambiguity construction

### A.9.1 Evidence modes

| Mode | Coalition value evaluator | Permitted use |
|---|---|---|
| `oracle` | true CURE-Sim SCM rollout | benchmark truth and method validation |
| `world_model` | learned sequential response model rollout | acceleration/ablation only unless calibrated and assumptions stated |
| `sequential_dr` | doubly robust off-policy estimator | audited logged-policy datasets with support |
| `sensitivity_dr` | DR estimator under bounded adversarial reweighting | partial-identification/sensitivity analysis |

### A.9.2 Sensitivity set

The formal target is:

\[
\mathfrak M_{\Gamma,r}=\{M:M\models\mathcal K,\;P_M\in\mathcal C_r,\;\operatorname{Conf}(M)\le\Gamma\}.
\]

Operationally, implement a finite approximation through candidate causal response models and bounded adversarial weights. `Gamma=1` is the sequential-ignorability reference. For \(\Gamma>1\), vary the allowed policy-assignment odds/density ratio according to a documented sensitivity model.

**Mandatory documentation:** exact sensitivity-model equation, variables treated as observed history, variables permitted to be unobserved, normalization of weights, extrema solver, and proof/argument that the implementation approximates the declared set.

### A.9.3 No unsupported interval language

- A posterior ensemble creates **model uncertainty intervals**.
- A bootstrap creates **sampling intervals**.
- A declared causal sensitivity set can create **partial-identification bounds**.

The report layer must label these separately. Only the third may be called a partially identified causal Shapley region.

### A.9.4 Calibration checks

For each response model/log:

- held-out logged-policy reward calibration;
- propensity calibration and clipping rate;
- sequential effective sample size;
- positivity/overlap diagnostics by intervention;
- one-step and multi-step rollout calibration in CURE-Sim;
- sensitivity of extrema to the number of candidate/adversarial models.

## A.10 Exact game computation

### A.10.1 Coalition enumeration

```python
ALL_COALITIONS = tuple(range(1 << 6))  # 0..63
```

Cache key:

```text
{environment_or_dataset_hash}/{base_policy_hash}/{ambiguity_model_id}/
{coalition_mask}/{horizon}/{seed}
```

### A.10.2 Coalition values

```python
def value(model: CausalModel, coalition: int, horizon: int, n_rollouts: int) -> ValueEstimate:
    # 1. Build pi_S with canonical operator composition.
    # 2. Evaluate by oracle rollout, calibrated world-model rollout, or sequential DR.
    # 3. Return mean, standard error, support diagnostics, constraint outcomes, and trace ID.
    ...
```

### A.10.3 Exact Shapley values

For every model representative \(M\):

\[
\phi_i(M)=\sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(n-|S|-1)!}{n!}
[V_M(S\cup\{i\})-V_M(S)].
\]

All 64 values must be stored so the paper can publish complete coalition tables.

### A.10.4 Exact interactions

With six players, compute every pairwise Shapley interaction exactly. Do not estimate interactions only after seeing attractive results.

## A.11 Model-consistent regions and outer bounds

### A.11.1 Model-consistent regions

For all retained causal models:

```python
lower_phi[i] = min(phi_by_model[m][i] for m in model_set)
upper_phi[i] = max(phi_by_model[m][i] for m in model_set)
```

This is model-consistent only relative to a computational approximation that adequately represents \(\mathfrak M_\Gamma\). Store the optimizing model IDs for every bound.

### A.11.2 Outer bounds

Given coalition intervals \([\underline V(S),\overline V(S)]\), compute:

\[
\underline\phi_i^{\mathrm{out}}=
\sum_Sw(S)[\underline V(S\cup\{i\})-\overline V(S)],
\]

\[
\overline\phi_i^{\mathrm{out}}=
\sum_Sw(S)[\overline V(S\cup\{i\})-\underline V(S)].
\]

Report the width ratio:

\[
\frac{\mathrm{width}(\Phi_i^{\mathrm{out}})}{\mathrm{width}(\Phi_i^{\mathrm{model-consistent}})}.
\]

The outer method is a computational baseline, not the preferred interpretation.

### A.11.3 Never do this

```python
# Invalid as a certified portfolio lower bound:
portfolio_score = sum(lower_phi[i] for i in S)
```

The lower endpoints for different interventions may come from different causal models. Use direct robust coalition value for selection.

## A.12 Robust portfolio selection

With six players, use exhaustive selection:

```python
def select_robust_portfolio(values_by_model, constraints) -> SelectionResult:
    candidates = []
    for S in ALL_COALITIONS:
        if violates_deterministic_cost_or_capacity(S, constraints):
            continue
        lower_value = min(values_by_model[m][S].utility for m in values_by_model)
        if not satisfies_robust_constraints(values_by_model, S, constraints):
            continue
        candidates.append((lower_value, S))
    return max(candidates, key=lambda x: x[0])
```

### Required baselines

1. no intervention;
2. best individual nominal intervention;
3. best individual lower-bound intervention;
4. greedy nominal coalition utility;
5. greedy lower-bound coalition utility;
6. lower-Shapley-score heuristic;
7. direct robust CURE-Rec selection;
8. oracle direct selection on CURE-Sim.

The lower-Shapley heuristic is included precisely to show why explanation should not be confused with robust optimization.

## A.13 Explanation card

For each selected portfolio, emit a JSON and human-readable card:

```text
Base policy: <hash/name>
Horizon / gamma: <H, gamma>
Ambiguity model: <Gamma, graph assumptions, support status>
Selected portfolio: [repeat_cap, explore_slot, diversify]
Robust utility interval: [lower, upper]
Constraint status: relevance / provider exposure / cost / safety
Per-intervention Shapley regions:
  repeat_cap: [x, y]
  explore_slot: [x, y]
  diversify: [x, y]
Key interaction regions:
  repeat_cap × explore_slot: [x, y]
Rejected/deferred interventions and reason:
  tail_slot: interval crosses zero / infeasible capacity / harmful interaction
```

This output is the practical artifact that demonstrates the proposed explanation is decision-facing rather than merely visual.

## A.14 Test suite

Tests are publication safeguards, not optional engineering polish.

### Game correctness

1. **Empty coalition:** \(\pi_\emptyset=\pi_0\) and \(V_M(\emptyset)\) is evaluated rather than hard-coded incorrectly.
2. **Efficiency:** for every fixed model \(M\), \(\sum_i\phi_i(M)=V_M(N)-V_M(\emptyset)\) to numerical tolerance.
3. **Symmetry:** two identical intervention operators receive identical Shapley value.
4. **Dummy:** a no-op intervention has zero Shapley value and zero interactions.
5. **Interaction sanity:** additive synthetic game has zero pair interactions; known complementary pair has positive interaction.
6. **Canonical composition:** all calls for a coalition mask produce exactly the same slate independent of Shapley evaluation order.

### Robustness and bounds

7. **Model-consistent interval:** lower/upper bound model IDs reproduce their reported value.
8. **Outer-bound validity:** every fixed-model Shapley value lies inside outer bounds on synthetic cases.
9. **Sign certificate:** synthetic model sets with all-positive/all-negative attribution satisfy certificate logic.
10. **No invalid lower-bound summation:** planner test asserts it chooses by direct lower coalition utility, not a Shapley sum.
11. **Nested contraction:** synthetic nested ambiguity sets cannot increase coordinate interval widths.

### Causal/logging diagnostics

12. **Timing:** response/outcome fields do not precede logged slate treatment.
13. **Support:** transformed policy probabilities outside support are flagged and block causal OPE claims.
14. **Propensity:** zero/near-zero propensities fail the audit unless a documented alternative estimator is used.
15. **Rollout:** CURE-Sim oracle response model predicts one-step transition probabilities within tolerance.

## A.15 Runtime budget

Exact coalition enumeration is intentionally modest. The dominant cost is causal evaluation, not Shapley arithmetic.

| Stage | CURE-Sim, six interventions | Logged-data setting |
|---|---:|---:|
| Train base policy | minutes | policy/dataset dependent |
| Build all 64 transformed policies | seconds | seconds–minutes |
| One causal model: coalition values | minutes with cached rollouts | OPE/data dependent |
| Exact Shapley + interactions | milliseconds | milliseconds |
| \(L\) ambiguity representatives | linear in \(L\) | linear in \(L\), parallelizable |
| Direct robust selection | milliseconds | milliseconds |

Scaling beyond 10–12 interventions is a separate algorithmic problem. The first paper should demonstrate controlled scalability but not promise exactness for an unrestricted intervention taxonomy.

---

# PART B — REGISTERED EVALUATION PLAN

> These are pre-implementation claims and falsification conditions, not results. The manuscript must preserve unfavorable outcomes rather than silently changing its story.

## B.1 Registered primary comparisons

| Question | Proposed method | Required comparator | Success condition |
|---|---|---|---|
| Causal credit | CURE-Rec model-consistent Shapley | predictive/observational attribution, point causal Shapley | lower attribution MAE or higher rank/sign accuracy on oracle SCM |
| Uncertainty | CURE-Rec partial-ID region | point estimate, bootstrap/model ensemble interval | nominal or conservative coverage with reasonable width under declared \(\Gamma\) |
| Portfolio value | direct robust enumeration | independent treatment ranking, nominal greedy, lower-Shapley heuristic | lower robust regret / lower harmful-policy rate in confounded-shift regimes |
| Feedback dynamics | selected CURE-Rec portfolio | base policy and immediate-reward portfolio | constraint satisfaction with better or comparable long-term utility |
| Model agnosticism | same intervention layer | two documented base policies | directionally stable portfolio/credit conclusions under both policies |

## B.2 Expected qualitative outcomes

These are directional expectations, not fabricated target numbers.

| Regime | Expected behavior if the method is working |
|---|---|
| Additive | Shapley, independent effects, and direct robust selection largely agree. |
| Complementary | Independent-effect ranking misses a beneficial pair; interaction region is positive; direct portfolio wins. |
| Redundant | Two policy interventions divide credit; leave-one-out can understate their joint importance. |
| Antagonistic | Interaction region is negative; direct robust selector avoids the pair despite positive individual effects. |
| Delayed fatigue | Immediate-click ranking overuses repeated exposure; long-horizon policy value favors repeat cap or novelty/diversity. |
| Hidden confounding | Point model overstates at least some intervention signs; CURE-Rec intervals widen as \(\Gamma\) grows. |
| Policy shift | Nominal policy selection degrades; lower-value selection sacrifices some mean return for lower harmful-policy frequency. |

## B.3 Falsification conditions and response

| If this happens | Meaning | Required response |
|---|---|---|
| Interventions are effectively additive in all environments | Cooperative portfolio claim untested | add/repair CURE-Sim complementary and antagonistic regimes before drafting claims |
| Point estimator has equal coverage and lower width under hidden confounding | ambiguity construction is not representing uncertainty correctly | audit sensitivity model; do not claim partial-ID advantage |
| Lower-Shapley heuristic equals direct robust selection everywhere | Shapley layer may be decorative | report this; narrow contribution to explanation or add interaction/constraint regimes that test the distinction |
| Robust selection has worse worst-case utility | implementation/theory mismatch | stop and audit model-consistency, constraint handling, and oracle calculation |
| Real logs fail support/propensity audit | no causal OPE evidence | use data only descriptively or remove it; do not patch with unlabelled model simulation |
| CURE-Sim only shows gains under one base policy | architecture dependency | report it and weaken model-agnostic claim |
| Intervals stay narrow as \(\Gamma\) rises | sensitivity extremization is broken | test against analytic small cases; do not interpret result |
| Provider constraint destroys all portfolio utility | genuine trade-off or bad utility scale | report Pareto frontier; do not hide constraint failure by changing threshold post hoc |

## B.4 Main ablations

1. **Observational versus interventional value:** replace \(V_M(S)\) with predictive/observational policy score.
2. **Point model versus ambiguity set:** set \(\Gamma=1\) and one response model.
3. **Model-consistent region versus outer bound:** measure tightness and selection consequences.
4. **Direct robust selection versus lower-Shapley selection:** establish correct division of labor.
5. **Interactions removed:** explain changes in complementary/antagonistic regimes.
6. **Short horizon versus long horizon:** quantify delayed fatigue/retention effect.
7. **Exposure state removed:** test whether feedback-loop modeling matters.
8. **Provider constraint removed:** display utility/disparity frontier rather than claiming one universally best outcome.
9. **Base-policy swap:** primary base recommender versus robustness base recommender.
10. **Ambiguity radius sweep:** \(\Gamma\in\{1,1.25,1.5,2\}\) plus any domain-calibrated values.

## B.5 Metrics

### Causal credit metrics — CURE-Sim only

\[
\operatorname{MAE}_\phi=
\frac{1}{n}\sum_i|\widehat\phi_i-\phi_i^\star|.
\]

Also report Spearman correlation, Kendall correlation, sign accuracy, top-\(k\) precision, and interaction recovery.

### Interval metrics — CURE-Sim primarily

\[
\operatorname{Coverage}=
\frac{1}{n}\sum_i\mathbf 1\{\phi_i^\star\in[\underline\phi_i,\overline\phi_i]\}.
\]

Also report mean width, width–coverage curve, false robust-positive rate, false robust-negative rate, and model-consistent versus outer-bound width ratio.

### Decision metrics

\[
\operatorname{RobustRegret}=
\underline V(S_{\mathrm{oracle}}^\star)-\underline V(\widehat S).
\]

Report expected utility, lower utility, harmful-policy rate, feasibility rate, cost, relevance loss, provider disparity, fatigue, popularity concentration, and catalogue coverage.

### Recommendation diagnostics

Report NDCG@\(K\), Recall@\(K\), MRR, and Hit Rate as ranking diagnostics. They are not automatically causal policy metrics.

## B.6 Statistical protocol

- Use at least five environment/base-policy seeds.
- Pair methods on identical environment seeds, cohorts, and trajectories.
- Report mean ± standard deviation and 95% bootstrap confidence intervals where appropriate.
- Use paired tests over independent environment/cohort units; apply Holm–Bonferroni within each result family.
- Do not treat every user interaction as independent if platform state or provider exposure creates interference.
- For OPE, report confidence intervals, effective sample size, maximum importance weight, and clipping sensitivity.

## B.7 Build order and gates

| Step | Deliverable | Gate before proceeding |
|---|---|---|
| 0 | Formal causal assumptions and ambiguity-set note | can write down exactly what \(\Gamma\) means and what is observable |
| 1 | CURE-Sim one-step environment | trajectories visually/analytically match structural equations |
| 2 | Six intervention operators | canonical composition and operator tests pass |
| 3 | Oracle 64-coalition evaluator | all exact Shapley efficiency/symmetry/dummy tests pass |
| 4 | Complementary/redundant/antagonistic CURE-Sim regimes | oracle interactions have expected signs |
| 5 | Direct robust enumerator | equals brute-force oracle solution on small synthetic cases |
| 6 | \(\Gamma\)-sensitivity ambiguity approximation | interval-width and coverage tests pass on known environments |
| 7 | Primary base-policy integration | intervention layer works without reading model internals |
| 8 | Real-log audit | data is labeled with permitted causal claim level |
| 9 | OPE/response-model experiments | support and calibration diagnostics are reportable |
| 10 | Second base policy, ablations, report emitters | full reproducible manuscript tables generated |

## B.8 Artifact checklist

Before submission, release or archive:

- CURE-Sim generator and all structural equations;
- exact oracle outputs for benchmark seeds;
- intervention-library YAML with canonical order and thresholds;
- every coalition value table;
- causal/logging audits for every real dataset;
- data preprocessing scripts and hashes;
- sensitivity-model specification;
- base-policy configurations;
- seeds and run manifests;
- evaluation code and figure/table scripts;
- limitations statement differentiating oracle, model-based, OPE, and sensitivity-bounded evidence.

---

# PART C — MANUSCRIPT CLAIM CHECKLIST

Use this checklist during drafting and review.

| Proposed claim | Allowed only if |
|---|---|
| “Partially identified Shapley region” | ambiguity set, sensitivity assumptions, and extrema construction are stated; interval validity is tested on CURE-Sim |
| “Causal intervention contribution” | policy intervention semantics, timing, support, and assumptions are explicit |
| “Long-term feedback-loop mitigation” | demonstrated in CURE-Sim or an appropriately logged long-horizon platform setting |
| “Robust portfolio selection” | direct maximin coalition value, not summed interval endpoints, determines selection |
| “Model-agnostic” | results use at least two documented base policies or the claim is weakened to architecture-independent interface |
| “Provider-exposure fairness” | provider groups, exposure measure, constraint threshold, and interference scope are declared |
| “Safe intervention” | phrase is replaced by a narrower ambiguity-set-conditional certificate unless a separate safety proof exists |
| “Exact Shapley” | all six-player coalition values are evaluated exactly for each retained causal model |

---

# PART D — FUTURE EXTENSIONS, EXPLICITLY OUT OF SCOPE

1. **Active CURE-Rec:** choose experiments that reduce attribution uncertainty near the robust selection boundary.
2. **Continual CURE-Rec:** update ambiguity sets under drift with regime-change detection; interval contraction is not assumed under shift.
3. **Policy-component games:** include feedback requests, human review, threshold changes, and model retraining operations as structured players.
4. **Federated CURE-Rec:** privacy-preserving causal policy intervention attribution across platforms.
5. **Structured games:** Owen/Myerson values for hierarchical or mutually exclusive intervention families.
6. **Multi-sided welfare:** richer user/provider/platform welfare negotiation rather than one provider-exposure constraint.
