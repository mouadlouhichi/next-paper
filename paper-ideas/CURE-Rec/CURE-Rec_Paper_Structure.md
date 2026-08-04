# CURE-Rec — Full Paper Structure, TOC & Embedded Content

**Working method name:** **CURE-Rec** — *Causal and Uncertainty-aware Recommendation Intervention Games*  
**Primary title:** *CURE-Rec: Partially Identified Cooperative Intervention Games for Robust Long-Term Recommendation*  
**Target venues:** ACM RecSys, WWW, WSDM, KDD, or ICDM. A substantially expanded journal version can target IEEE TKDE.  
**Article type:** Research article  
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹  
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco  
**Corresponding author:** mouad_louhichi@um5.ac.ma

**Target:** ≈9,000–11,000 words; 8–10 figures; 8–10 tables; code, a synthetic benchmark generator, and complete configurations released.

> **The paper in one sentence.** CURE-Rec treats *feasible changes to a recommendation policy*—not users, items, features, or interactions—as cooperative-game players, assigns each intervention a **partially identified causal Shapley region** over long-term platform utility, and uses direct robust policy evaluation to select an intervention portfolio that can be explained, certified, or deferred.

> **Scope lock.** This is a recommender-systems and causal-decision paper, not a new base recommendation architecture. The primary setting is a finite-horizon platform policy with a small, fixed library of composable re-ranking/exposure interventions. Deployment selects **robust improvement over the currently deployed base policy**, not raw model utility. The first paper does **not** jointly learn a recommender, discover an unrestricted causal graph, perform active experiment design, model arbitrary model-retraining interventions, or solve every fairness notion. Those are sequels, not hidden dependencies.

---

## Working titles

- **Primary (selected):** *CURE-Rec: Partially Identified Cooperative Intervention Games for Robust Long-Term Recommendation*
- Alt 1: *From Attribution to Action: Robust Causal Shapley Games for Recommendation Policy Interventions*
- Alt 2: *Which Recommendation Policy Should Change? Partially Identified Shapley Values for Long-Term Exposure Control*
- Alt 3 (journal): *Partially Identified Shapley Games for Robust Intervention Selection in Recommender Systems*

## One-paragraph thesis (the spine)

Recommenders alter the data on which they later learn: a policy determines exposure, exposure shapes interaction, interaction changes user state and catalogue popularity, and those changes affect future recommendations. Yet current systems choose interventions such as exploration, long-tail promotion, diversity injection, repeated-item suppression, and provider balancing largely from immediate predictive gains or point-estimated policy effects. This is unsafe because these interventions can be redundant, antagonistic, delayed in effect, and causally ambiguous under exposure bias and unmeasured confounding. CURE-Rec formulates a **sequential cooperative intervention game** whose players are feasible recommendation-policy transformations and whose value is long-term platform welfare under a causal model. Rather than asserting a single causal world, it maintains an explicit ambiguity set of plausible recommendation-response models and computes **model-consistent Shapley regions** for every intervention. The platform selects a portfolio by direct robust optimization of coalition utility, while Shapley regions explain the selected plan, certify robustly positive or harmful interventions, and identify interventions on which the evidence is insufficient. Thus the contribution is not a new ranker: it is a decision-facing causal attribution layer for recommendation ecosystems.

## Research questions and falsifiable hypotheses

| RQ | Question | Hypothesis | Link to prior thesis work |
|---|---|---|---|
| **RQ1** | Can long-term recommendation-policy changes be posed as a cooperative game whose players are feasible interventions? | Exact intervention Shapley values recover oracle policy credit in a controlled sequential SCM. | Moves from feature/entity attribution to intervention attribution. |
| **RQ2** | Can causal ambiguity be carried through the game as model-consistent Shapley regions rather than point explanations? | Regions achieve nominal coverage in controlled sensitivity experiments; interval width grows appropriately under hidden confounding and policy shift. | Makes uncertainty a property of the game, not an error bar appended to SHAP. |
| **RQ3** | Do robustly selected portfolios outperform independent intervention ranking when interventions interact? | Direct maximin portfolio selection has lower robust regret and lower harmful-policy rate than individual-effect, nominal, and predictive baselines. | Extends attribution-guided learning into attribution-guided policy control. |
| **RQ4** | Does robust causal policy control reduce long-run feedback harms while retaining useful recommendation quality? | It reduces fatigue/popularity concentration and respects relevance and provider-exposure constraints in the synthetic ecosystem; real logs support only the bounded, short-horizon claims their logging permits. | Extends DyHuCoG’s diversity/context utility to causal, longitudinal outcomes. |

---

# TABLE OF CONTENTS

```text
Abstract / Keywords
1. Introduction
   1.1 Background: recommendation changes its own data-generating process
   1.2 The intervention-attribution gap
   1.3 Why policy effects are not enough: interacting interventions and causal ambiguity
   1.4 From DyHuCoG to CURE-Rec
   1.5 Contributions
   1.6 Organization
2. Related Work
   2.1 Causal recommendation and exposure bias
   2.2 Long-term recommendation, bandits, and offline reinforcement learning
   2.3 Fair exposure, popularity feedback, and recommender ecosystems
   2.4 Shapley values, causal Shapley, and restricted cooperative games
   2.5 Partial identification and robust sequential decision-making
   2.6 Positioning and differentiation
3. Preliminaries and Problem Formulation
   3.1 Platform-level sequential recommendation environment
   3.2 Base recommendation policy and policy transformations
   3.3 Intervention players and canonical coalition composition
   3.4 Causal assumptions, logged-policy requirements, and ambiguity sets
   3.5 Long-term policy-improvement estimand, robust constraints, and abstention
4. CURE-Rec Framework
   4.1 Overview
   4.2 Intervention library and collision semantics
   4.3 Coalition policy semantics and composition-order sensitivity
   4.4 Sequential causal response model
   4.5 Partially identified coalition values
   4.6 Statistical uncertainty and estimated regions
   4.7 Exact intervention Shapley values and model-consistent regions
   4.8 Interaction regions
   4.9 Robust portfolio selection
   4.10 Explanation, certification, and abstention
   4.11 Computational complexity
5. Theoretical Analysis
   5.1 Identified sets of cooperative games
   5.2 Existence of model-consistent Shapley regions
   5.3 Valid coalition-wise outer bounds
   5.4 Robust attribution-sign certificates
   5.5 Simulation and exact-game estimation error
   5.6 Robust selection regret and feasibility under value-estimation error
   5.7 Nested-evidence interval contraction
6. CURE-Sim: A Ground-Truth Sequential Recommendation Benchmark
   6.1 Motivation and design goals
   6.2 User, item, provider, and platform state dynamics
   6.3 Exposure, fatigue, popularity, and confounding mechanisms
   6.4 Intervention families and interaction regimes
   6.5 Oracle values, oracle Shapley regions, and release protocol
7. Experimental Setup
   7.1 Datasets and evidence hierarchy
   7.2 Base recommender policies
   7.3 Causal estimators and ambiguity-set construction
   7.4 Baselines
   7.5 Metrics
   7.6 Reproducibility and statistical protocol
8. Results and Discussion
   8.1 RQ1 — causal credit recovery on CURE-Sim
   8.2 RQ2 — interval validity, width, and sensitivity to ambiguity
   8.3 RQ3 — robust intervention portfolio quality
   8.4 RQ4 — long-term feedback, relevance, and provider-exposure trade-offs
   8.5 Real logged-data results: bounded claims and off-policy diagnostics
   8.6 Ablations
   8.7 Efficiency and scalability
   8.8 Case study: explain a selected intervention portfolio
   8.9 Limitations
9. Conclusion and Future Work
Declarations
References
Appendix A — Proofs
Appendix B — CURE-Sim specification
Appendix C — Causal/logging audit templates
Appendix D — Hyperparameters and additional results
Notation list
```

---

# ABSTRACT (draft, ≈230 words)

Recommender systems influence the interaction data from which they subsequently learn: a ranking policy determines exposure, exposure changes user feedback and item popularity, and these changes affect future recommendations. Platforms routinely intervene in this loop through diversity injection, exploration, repeated-exposure suppression, long-tail promotion, and provider-exposure balancing, but lack a principled account of which interventions causally improve long-term outcomes, which interventions work together, and when the evidence is too uncertain to act. We introduce **CURE-Rec**, a framework that formulates recommendation-policy adaptation as a **partially identified cooperative intervention game**. Its players are feasible policy transformations rather than users, items, or model features; a coalition induces a transformed recommendation policy; and its value is discounted long-term platform utility under the corresponding intervention. To avoid unsupported causal precision, CURE-Rec maintains a causal ambiguity set compatible with logged-policy evidence, domain constraints, and a declared confounding-sensitivity model. It computes model-consistent lower and upper Shapley contributions for each intervention, as well as interaction regions for candidate pairs. The selected portfolio maximizes robust coalition utility directly under relevance, provider-exposure, capacity, and cost constraints; Shapley regions explain the decision, certify interventions with consistently positive or negative average marginal value, and flag interventions requiring more evidence. We establish existence and outer-bound results for partially identified Shapley regions, a robust sign certificate, and finite-simulation error bounds. A new sequential recommendation benchmark with oracle causal dynamics evaluates credit recovery, interval coverage, coalition regret, and long-term feedback effects, complemented by randomized or propensity-logged recommendation data for the claims those logs can support. *(Insert headline results after experiments.)*

**Keywords:** recommender systems · causal inference · Shapley value · cooperative game theory · partial identification · long-term recommendation · exposure bias · robust policy optimization · provider exposure

---

# 1. INTRODUCTION

## 1.1 Background: recommendation changes its own data-generating process

Open with the central feedback loop:

\[
\text{policy} \rightarrow \text{exposure} \rightarrow \text{response} \rightarrow \text{user and catalogue state} \rightarrow \text{future policy outcomes}.
\]

Immediate click-through rate, NDCG, and conversion are not complete policy objectives. Repeated exposure can create fatigue, popularity concentration can reduce discovery, and an apparently useful intervention can harm retention or provider exposure over a longer horizon. Establish that causal recommendation, counterfactual evaluation, bandits, and long-term RL all address parts of this problem, but none directly allocates the **joint causal value of a portfolio of recommendation-policy interventions**.

## 1.2 The intervention-attribution gap

Explain the difference between four questions:

1. *Why was this item ranked?* — item/feature explanation.
2. *Which interaction helped train the model?* — data valuation.
3. *Which user/item/context mattered inside the model?* — DyHuCoG-style entity attribution.
4. *Which deliberate policy change should the platform apply?* — CURE-Rec.

The fourth is the paper’s object. A high predicted relevance score or high feature SHAP value does not imply that increasing exposure, adding diversity, or suppressing repetition will improve future welfare.

## 1.3 Why policy effects are not enough

A single-intervention average treatment effect is inadequate because interventions may be:

- complementary: exploration is more useful when repeated-item suppression creates room for novel candidates;
- redundant: novelty injection and long-tail promotion may consume the same scarce slate capacity;
- antagonistic: aggressive provider balancing may conflict with a relevance constraint;
- uncertain: hidden exposure confounding and policy shift can reverse an apparently positive effect.

State the critical methodological point: point estimates from one fitted response model are not evidence of certain causal actionability.

## 1.4 From DyHuCoG to CURE-Rec

State the progression precisely.

- **DyHuCoG** uses Shapley estimates during hypergraph recommendation learning. Its players are recommendation entities/interactions and its utility is ranking, diversity, context, and preference consistency.
- **CURE-Rec** operates above a fixed recommender. Its players are deployable policy transformations and its utility is long-term causal platform welfare under uncertainty.

Suggested sentence:

> DyHuCoG learns which entities should influence a recommendation representation; CURE-Rec determines which changes to the recommendation process should be deployed, rejected, or deferred.

Do **not** claim that CURE-Rec reimplements or depends on DyHuCoG. The primary implementation must use reproducible standard base policies; DyHuCoG is an optional conceptual comparison only.

## 1.5 Contributions

1. **A new recommendation-policy game.** We formalize a sequential cooperative intervention game whose players are feasible recommendation-policy transformations and whose coalitions induce long-term recommendation policies.
2. **Partially identified causal attribution.** We define model-consistent Shapley regions over an explicit ambiguity set, rather than treating bootstrap dispersion or ensemble disagreement as a causal interval by default.
3. **Direct robust selection with cooperative explanation.** We select intervention portfolios by robust coalition utility under operational constraints, then use Shapley and interaction regions to explain, certify, screen, or defer interventions.
4. **Theory.** We establish conditions for existence of Shapley regions, valid outer bounds, robust attribution-sign certificates, finite-rollout error propagation, and interval contraction under nested stationary evidence updates.
5. **CURE-Sim.** We release a controlled sequential recommendation environment with known user, item, provider, and exposure-feedback dynamics, enabling oracle coalition values and causal Shapley evaluation.
6. **Empirical evaluation.** We test causal credit, uncertainty calibration, robust portfolio utility, feedback-loop mitigation, and model-agnostic operation using synthetic ground truth plus randomized/propensity-logged recommendation evidence.

## 1.6 Organization

One concise roadmap paragraph.

---

# 2. RELATED WORK

Target ≈1,500 words. Every subsection ends with a one-sentence differentiation statement.

## 2.1 Causal recommendation and exposure bias

Cover propensity scoring, inverse propensity scoring, doubly robust evaluation, counterfactual learning to rank, selection bias, exposure bias, and unobserved-confounding concerns. Explain that the paper is not another debiasing objective. Its question is how to allocate and select **interacting policy interventions** after defining a causal policy value.

## 2.2 Long-term recommendation, bandits, and offline reinforcement learning

Cover contextual bandits, slate bandits, sequential recommendation, offline RL, and model-based user simulators. Distinguish policy optimization from cooperative policy-intervention attribution. Existing long-term models can serve as response/policy estimators; they do not eliminate the need for intervention-level credit allocation.

## 2.3 Fair exposure, popularity feedback, and recommender ecosystems

Cover long-tail promotion, provider fairness, feedback loops, popularity concentration, repeated exposure, and fatigue. Do not claim that fairness is solved. In CURE-Rec, provider exposure is one constrained outcome in a multi-stakeholder policy setting.

## 2.4 Shapley values, causal Shapley, and restricted games

Cover SHAP, Data Shapley, causal Shapley, do-Shapley, asymmetric/ordered Shapley, and Shapley interaction indices. State the distinction:

> Existing causal Shapley methods attribute variables or causal mechanisms in a fixed prediction/estimand; CURE-Rec attributes feasible recommendation-policy transformations by their long-horizon, model-uncertain coalition utility.

## 2.5 Partial identification and robust sequential decision-making

Cover sensitivity analysis for hidden confounding, robust policy evaluation, ambiguity sets, and partially identified treatment effects. Define the paper’s novelty narrowly: it propagates partial causal identification through a cooperative-game operator and connects the resulting action-credit regions to robust intervention selection.

## 2.6 Positioning and differentiation

**Table 1 — Positioning against representative work.**

| Method family | Main object | Players | Long horizon | Causal uncertainty set | Shapley allocation | Portfolio selection | Ecosystem constraints |
|---|---|---:|---:|---:|---:|---:|---:|
| Exposure-debiased recommendation | Ranking loss | items/interactions | sometimes | limited | no | no | sometimes |
| Bandit/offline-RL recommendation | policy | actions/slates | yes | sometimes | no | yes | sometimes |
| Fair-exposure recommendation | re-ranking rule | item/provider groups | sometimes | rarely | no | sometimes | yes |
| Causal SHAP / do-Shapley | explanatory estimand | features/variables | no | usually point-identified | yes | no | no |
| DyHuCoG (prior work) | recommendation learning | users/items/contexts | dynamic training | no | yes | implicit in training | diversity/context |
| **CURE-Rec** | policy intervention | feasible policy transformations | **yes** | **yes** | **yes** | **yes, direct robust value** | **yes, constrained** |

**Novelty discipline.** The paper must use “to our knowledge” and claim novelty only for the *combination* of: (i) recommendation-policy interventions as players, (ii) long-horizon causal coalition value, (iii) model-consistent partial-identification regions, and (iv) robust portfolio selection. Perform a final systematic literature audit before submission.

---

# 3. PRELIMINARIES AND PROBLEM FORMULATION

## 3.1 Platform-level sequential recommendation environment

Use a platform state rather than a solely individual user state to make ecosystem claims coherent:

\[
\mathcal S_t = \left(\{H_{u,t}\}_{u\in\mathcal U}, E_t, Q_t, \theta_t\right),
\]

where:

- \(H_{u,t}\): observed user histories and response state;
- \(E_t\): cumulative exposure/repetition state;
- \(Q_t\): aggregate catalogue/provider state, including popularity and exposure concentration;
- \(\theta_t\): deployed policy state or model parameters.

The baseline policy \(\pi_0\) produces slates \(L_t\). A causal model \(M\) induces:

\[
\mathcal S_{t+1} = F_M(\mathcal S_t, L_t, Y_t, \varepsilon_t), \qquad L_t=\pi(\mathcal S_t).
\]

When utility includes provider exposure, popularity concentration, or other aggregate outcomes, the intervention unit is a **cohort/platform epoch**, not an isolated user. CURE-Sim therefore applies \(do(\pi=\pi_S)\) to a simulated cohort and evaluates aggregate outcomes over that cohort. Individual-level real-log analyses are not described as identifying arbitrary cross-user spillovers.

**MVP boundary:** \(\theta_t\) is fixed during an evaluation horizon. CURE-Sim may include a controlled retraining-feedback variant in an appendix, but the core method does not claim to model arbitrary online retraining dynamics.

## 3.2 Base policy and policy transformations

The base recommender supplies candidates and base scores. CURE-Rec is model-agnostic with respect to \(\pi_0\). In the first implementation, use a reproducible standard sequential/base ranker and one robustness base ranker—not a new architecture.

A coalition transforms the policy:

\[
\pi_S = g_S(\pi_0), \qquad S\subseteq\mathcal I.
\]

To make \(g_S\) well-defined, policy operators are composed in a **fixed canonical order**. The Shapley arrival ordering is used only to evaluate marginal coalition additions; it does not change the semantics of the final transformed policy. Every result stores the canonical-order version, intervention parameters, deterministic tie-breaking rule, and base-policy hash.

**Composition-order robustness.** The primary analysis pre-registers one canonical order. A required robustness analysis evaluates a small, semantically defensible set \(\mathcal O\) of alternative orders and reports \(\phi_i^{(o)}\), selected portfolios, and coalition values for every \(o\in\mathcal O\). The main causal ambiguity set is not silently widened by order choice; order sensitivity is reported as an independent operational-semantics threat. If it changes conclusions materially, the paper must describe policy-order dependence rather than attribute the effect to an intervention alone.

## 3.3 Intervention players and canonical composition

Define a compact intervention universe \(\mathcal I=\{I_1,\ldots,I_6\}\). Each player is:

\[
I_i=(g_i,\mathcal E_i,c_i),
\]

where \(g_i\) is a deterministic/stochastic policy transformation, \(\mathcal E_i\) is the eligibility rule, and \(c_i\) is operating or slate-capacity cost.

**Primary library (fixed before experiments):**

| ID | Intervention | Operational definition |
|---|---|---|
| \(I_{\mathrm{rep}}\) | repeated-exposure cap | Exclude/penalize items shown at least \(r\) times in the trailing window. |
| \(I_{\mathrm{exp}}\) | exploration slot | Replace one eligible slate position using uncertainty-guided exploration. |
| \(I_{\mathrm{tail}}\) | long-tail slot | Reserve one eligible position for an item below a declared popularity quantile. |
| \(I_{\mathrm{div}}\) | category-diversity operator | Re-rank to satisfy a minimum pairwise category-distance target. |
| \(I_{\mathrm{nov}}\) | user-conditional novelty slot | Add an item whose novelty relative to history exceeds a threshold. |
| \(I_{\mathrm{prov}}\) | bounded provider-exposure balance | Apply a bounded re-ranking correction toward a declared provider-exposure target. |

Canonical policy composition: repeated-exposure cap → eligibility filtering → exploration/long-tail/novelty candidate injection → diversity re-ranking → provider-exposure balancing. The exact order is pre-registered and applies to every coalition.

## 3.4 Causal assumptions, logs, and ambiguity set

State the causal requirements rather than hiding them.

1. **Logged-policy support:** slate/action propensities, or a defensible factorized approximation, are available for all offline policy claims.
2. **Sequential overlap:** candidate policies do not assign material probability to actions unsupported by the log.
3. **Time-unrolled causal structure:** recorded history precedes slate assignment; exposure is represented explicitly; post-exposure feedback is not adjusted for as a pre-treatment confounder.
4. **Interference treatment:** ecosystem outcomes are evaluated at cohort/platform level in CURE-Sim. Logged individual-level analyses do not claim identification of arbitrary cross-user spillovers.
5. **Partial identification:** unmeasured policy-assignment confounding is represented by a declared sensitivity set, not wished away.

Let \(\mathcal K\) be causal/domain constraints and \(\mathcal C_r\) a confidence region based on evidence up to update index \(r\). The logged action is a slate or factorized slate action \(A_t\), the recorded pre-treatment history is \(H_t\), and \(U_t\) denotes unrecorded policy-assignment factors. The primary sensitivity model is a sequential generalized odds-ratio restriction:

\[
\frac{1}{\Gamma}
\leq
\frac{
\Pr_M(A_t=a\mid H_t,U_t)/\Pr_M(A_t=a'\mid H_t,U_t)
}{
\Pr_M(A_t=a\mid H_t)/\Pr_M(A_t=a'\mid H_t)
}
\leq\Gamma
\]

for supported action pairs \(a,a'\) at every decision time. For a factorized slate policy, this restriction is applied to the declared action factors and the factorization is part of \(\mathcal K\); it is never assumed silently for an opaque deterministic ranker. The resulting ambiguity set is:

\[
\mathfrak M_{\Gamma,r}=\left\{M: M\models\mathcal K,\;P_M\in\mathcal C_r,\;M\text{ satisfies the sequential }\Gamma\text{-restriction}\right\}.
\]

For a coalition \(S\), the operational lower improvement is computed through an explicitly stated adversarial weighted sequential-DR program:

\[
\underline{\Delta V}_{\Gamma,r}(S)=
\min_{w\in\mathcal W_\Gamma(S)}
\widehat{\Delta V}_{\mathrm{SDR}}(S;w),
\]

where \(\mathcal W_\Gamma(S)\) contains normalized trajectory or declared time-factor weights obeying the above \(\Gamma\)-bounds. The manuscript must state whether this program is trajectory-level or time-factorized, its solver class, and the exact variables allowed to be unobserved. The implementation approximates the full set with adversarially weighted and sampled response models, while theory remains stated over \(\mathfrak M_{\Gamma,r}\). A bootstrap ensemble alone is **not** called a valid partial-identification set.

## 3.5 Long-term policy-improvement estimand, robust constraints, and abstention

For a coalition \(S\), define:

\[
V_M(S)=
\mathbb E_{P_M^{\pi_S}}
\left[
\sum_{h=0}^{H-1}\gamma^h r(\mathcal S_{t+h},L_{t+h},Y_{t+h})
\right]-\lambda_c C(S).
\]

The main reward prioritizes long-term user satisfaction/retention and penalizes fatigue. Relevance and provider exposure are constraints rather than silently chosen scalar weights:

\[
\inf_{M\in\mathfrak M_{\Gamma,r}}\Delta_{\mathrm{rel},M}(S)\geq-\epsilon_{\mathrm{rel}},
\]

\[
\sup_{M\in\mathfrak M_{\Gamma,r}}D_{\mathrm{provider},M}(S)\leq\epsilon_{\mathrm{prov}}.
\]

Define policy improvement relative to the deployed base policy:

\[
\Delta V_M(S)=V_M(S)-V_M(\emptyset),\qquad \pi_{\emptyset}=\pi_0.
\]

The robust selected portfolio is:

\[
S^\star=\arg\max_{S\in\mathcal F_{\mathrm{safe}}}\inf_{M\in\mathfrak M_{\Gamma,r}}\Delta V_M(S).
\]

The abstention rule is explicit:

\[
\max_{S\in\mathcal F_{\mathrm{safe}}}\underline{\Delta V}(S)\leq0
\quad\Longrightarrow\quad S^\star=\emptyset.
\]

This removes model-specific baseline offsets and answers the actual operational question—what policy change can robustly improve on the deployed recommender? The Shapley values are unchanged by this baseline shift because all marginal coalition differences are unchanged. This direct robust objective, not an additive sum of Shapley lower bounds, is the deployment decision rule.

---

# 4. CURE-REC FRAMEWORK

## 4.1 Overview

**Figure 1:** baseline recommender → fixed intervention library → causal response/ambiguity model → exact coalition sweep → model-consistent Shapley and interaction regions → direct robust portfolio selection → transformed slate plus explanation/certificate/defer decision.

**Algorithm 1:** CURE-Rec exact intervention game and robust portfolio selection.

## 4.2 Intervention library and collision semantics

Reproduce the six-player table from §3.3 and specify every threshold, slot count, eligibility condition, and cost in a configuration file. There are only \(2^6=64\) coalitions. Exact enumeration is a deliberate design choice that avoids Monte-Carlo Shapley approximation and makes the portfolio optimizer auditable.

### Collision and no-candidate semantics

The three injection interventions—exploration, long-tail, and novelty—cannot be left as informal requests for a “slot.” Fix injection capacity \(q=2\) in the primary experiments. Each active intervention \(i\) proposes eligible item-score pairs \(C_i(\mathcal S_t)=\{(j,b_i(j))\}\). Resolve the joint allocation deterministically:

\[
A_S^\star=\arg\max_A\sum_{(i,j)\in A} b_i(j)
\]

subject to \(|A|\leq q\), at most one proposal per intervention, distinct selected items, slate availability, and every intervention’s eligibility rule. An item may satisfy several eligibility rules but is assigned to at most one intervention in \(A_S^\star\). If an active intervention has no eligible candidate after repeat filtering and availability checks, it produces an explicit no-op with a logged reason. Ties are broken by a fixed item identifier. This exact three-player assignment is enumerated, not greedily approximated.

**Important separation:** policy compatibility is resolved by canonical composition and collision assignment. Budget, relevance, provider-exposure, and capacity restrictions are applied in \(\mathcal F_{\mathrm{safe}}\) during final portfolio selection, not by silently skipping players in Shapley permutations.

## 4.3 Coalition policy semantics and composition-order sensitivity

Every coalition mask maps to one fully specified transformed policy: active operators, canonical order version, collision-allocation result, thresholds, candidate availability, and deterministic tie-breaking are recorded in an immutable coalition manifest. This makes \(\pi_S\) reproducible and lets reviewers distinguish an intervention effect from an unlogged implementation difference.

The primary result uses the pre-registered canonical order in §3.3. A mandatory robustness analysis evaluates a small set of semantically defensible alternatives, for example: injection → diversity → provider balance; injection → provider balance → diversity; and diversity-constrained injection → provider balance. Report \(V_M^{(o)}(S)\), \(\phi_i^{(o)}(M)\), interactions, and selected portfolios for every \(o\in\mathcal O\). If an intervention’s sign or the robust portfolio changes materially, report order dependence as a limitation rather than treating the canonical order as a neutral fact.

## 4.4 Sequential causal response model

Use two estimators depending on evidence source:

- **CURE-Sim:** exact SCM rollouts, with oracle response dynamics available for evaluation only.
- **Logged data:** sequential doubly robust policy evaluation where logging support permits it, combined with an explicit \(\Gamma\)-sensitivity analysis for unmeasured policy assignment confounding.

A learned response/world model may accelerate trajectory evaluation, but it must be calibrated against held-out logged-policy data and never be confused with causal identification by itself.

## 4.5 Partially identified coalition values

For every coalition:

\[
\mathcal V_{\Gamma,r}(S)=
\left[
\underline V_{\Gamma,r}(S),\overline V_{\Gamma,r}(S)
\right]
=
\left[
\inf_{M\in\mathfrak M_{\Gamma,r}}V_M(S),
\sup_{M\in\mathfrak M_{\Gamma,r}}V_M(S)
\right].
\]

The key implementation distinction:

- **model-consistent extrema** optimize the complete coalition value under one common \(M\);
- **coalition-wise interval arithmetic** is cheaper but can combine incompatible extrema and is reported only as a conservative outer bound.

## 4.6 Statistical uncertainty and estimated regions

Separate population ambiguity from finite-sample uncertainty. Let \(\Phi_i^{\mathrm{ID}}(\Gamma)\) denote the population sensitivity/identified region induced by \(\mathfrak M_\Gamma\). Let \(\widehat\Phi_{i,1-\alpha}^{\mathrm{CI}}(\Gamma)\) denote an estimated confidence region that additionally accounts for trajectory, nuisance-model, and finite-sample uncertainty. Report separately:

- **sensitivity validity:** whether the true \(\phi_i(M^\star)\) lies in \(\Phi_i^{\mathrm{ID}}(\Gamma)\) when \(M^\star\in\mathfrak M_\Gamma\);
- **statistical coverage:** repeated-sample coverage of \(\widehat\Phi_{i,1-\alpha}^{\mathrm{CI}}(\Gamma)\);
- **ambiguity width** versus **sampling-confidence width**.

Bootstrap/model-ensemble dispersion is labelled sampling/model uncertainty unless it is part of a declared sensitivity-set construction.

## 4.7 Exact intervention Shapley values and model-consistent regions

For a fixed model \(M\):

\[
\phi_i(M)=
\sum_{S\subseteq\mathcal I\setminus\{i\}}
\frac{|S|!(n-|S|-1)!}{n!}
\left[\Delta V_M(S\cup\{i\})-\Delta V_M(S)\right].
\]

This is algebraically identical to applying Shapley to \(V_M\), because \(V_M(\emptyset)\) is a coalition-independent constant. Efficiency now has the operational interpretation \(\sum_i\phi_i(M)=\Delta V_M(\mathcal I)\): the full intervention library’s improvement over the deployed policy is allocated exactly. With six interventions, compute the exact value from all 64 coalition evaluations. The partially identified Shapley region is:

\[
\Phi_{i,\Gamma,r}=
\left[
\underline\phi_{i,\Gamma,r},
\overline\phi_{i,\Gamma,r}
\right]
=
\left[
\inf_{M\in\mathfrak M_{\Gamma,r}}\phi_i(M),
\sup_{M\in\mathfrak M_{\Gamma,r}}\phi_i(M)
\right].
\]

The full scientifically correct object is the joint set:

\[
\boldsymbol\Phi_{\Gamma,r}=
\{\boldsymbol\phi(V_M):M\in\mathfrak M_{\Gamma,r}\}\subseteq\mathbb R^6.
\]

Coordinate intervals are explanatory projections. They must not be summed as if their endpoints necessarily arise from the same causal model.

## 4.8 Interaction regions

For selected pairs, use the **Grabisch–Roubens Shapley interaction index** under each \(M\), with its normalization fixed before coding:

\[
\mathcal I_{ij}(M)=
\sum_{S\subseteq\mathcal I\setminus\{i,j\}}
\frac{|S|!(n-|S|-2)!}{(n-1)!}
\big[\Delta V_M(S\cup\{i,j\})-\Delta V_M(S\cup\{i\})-\Delta V_M(S\cup\{j\})+\Delta V_M(S)\big].
\]

Unit tests cover additive games (zero interaction), pure pair synergy, symmetry, and this exact normalization convention.

Report:

\[
\left[\inf_M\mathcal I_{ij}(M),\sup_M\mathcal I_{ij}(M)\right].
\]

Interpretation:

- lower bound \(>0\): robust complementarity;
- upper bound \(<0\): robust antagonism;
- interval containing zero: interaction sign is not determined.

Interactions are diagnostic and candidate-screening tools. The final decision still evaluates \(\underline{\Delta V}(S)\) directly.

## 4.9 Robust portfolio selection

Since \(n=6\), enumerate every feasible portfolio exactly:

```text
for every S in powerset(interventions):
    if S satisfies robust budget, capacity, relevance, provider, and safety constraints:
        estimate lower improvement inf_M Delta V_M(S)
return empty coalition if every feasible lower improvement is non-positive;
otherwise return the feasible S with maximum lower improvement
```

This avoids unjustified greedy/submodularity claims in the first paper. Include greedy, independent-treatment, lower-Shapley, and leave-one-intervention-out heuristics as baselines—not as the proposed decision rule. The non-Shapley robust selector uses the identical direct optimization rule and is therefore an explanatory-layer ablation, not a recommendation-utility competitor.

## 4.10 Explanation, certification, and abstention

For the selected \(S^\star\), output:

- \(\underline{\Delta V}(S^\star),\overline{\Delta V}(S^\star)\) relative to \(\pi_0\);
- each selected intervention’s Shapley region;
- key interaction regions;
- relevance, provider-exposure, capacity, and cost diagnostics;
- the ambiguity assumptions \(\Gamma\), graph, and support audit.

Decision semantics:

| Condition | Interpretation | System action |
|---|---|---|
| \(\underline\phi_i>0\) | positive order-averaged marginal contribution under every model in the stated ambiguity set | eligible for robust screening/selection |
| \(\overline\phi_i<0\) | consistently harmful average marginal contribution | reject unless external constraint requires it |
| \(0\in\Phi_i\) | sign unresolved | defer, gather evidence, or use only if direct robust portfolio value supports it |

Do not call a positive Shapley lower bound a universal safety certificate. It is an attribution-sign certificate conditioned on the ambiguity set and averaging context.

## 4.11 Computational complexity

Let \(n=6\), \(L\) be the number of computational representatives of \(\mathfrak M_\Gamma\), \(B=2^n=64\), and \(R\) the number of trajectory/OPE rollouts per coalition.

| Stage | Cost | Comment |
|---|---:|---|
| Base policy training | policy dependent | paid once |
| Coalition-policy construction | \(O(Bn)\) | exact and negligible |
| Causal utility sweep | \(O(LBR C_{\mathrm{eval}})\) | dominant term |
| Exact Shapley aggregation | \(O(nB)\) per model | arithmetic only |
| Pair interactions | \(O(n^2B)\) per model | exact at six players |
| Robust portfolio selection | \(O(BL)\) | exact enumeration |

The computational contribution is not large-scale combinatorial optimization. It is making the intervention space small, auditable, and exact while preserving a nontrivial causal uncertainty problem.

---

# 5. THEORETICAL ANALYSIS

The theory should be substantial but disciplined. Do not claim a generic greedy approximation theorem: the proposed method enumerates all six-player portfolios exactly.

## 5.1 Identified sets of cooperative games

Let:

\[
\mathcal G_{\Gamma,r}=\{V_M:M\in\mathfrak M_{\Gamma,r}\}
\]

be the set of scalar cooperative games induced by plausible causal models. State regularity assumptions explicitly: nonempty compact ambiguity set, bounded discounted rewards, and continuous map from model parameters to each coalition value.

## 5.2 Existence of model-consistent Shapley regions

**Theorem 1 (Existence).** If \(\mathcal G_{\Gamma,r}\) is nonempty and compact and the Shapley operator is applied to bounded coalition values, then:

\[
\{\phi_i(V):V\in\mathcal G_{\Gamma,r}\}
\]

is compact. If \(\mathcal G_{\Gamma,r}\) is connected—convexity is sufficient—its image in \(\mathbb R\) is an interval and equals \(\Phi_{i,\Gamma,r}\).

**Reviewer-proofing note:** compactness alone does not imply an interval. Connectedness/convexity must be stated. The joint set \(\boldsymbol\Phi\) need not be a hyperrectangle.

## 5.3 Valid coalition-wise outer bounds

Define valid utility bounds \([\underline V(S),\overline V(S)]\). For Shapley weights \(w(S)\geq0\), prove:

\[
\sum_Sw(S)\big[\underline V(S\cup\{i\})-\overline V(S)\big]
\leq\phi_i(M)\leq
\sum_Sw(S)\big[\overline V(S\cup\{i\})-\underline V(S)\big].
\]

These are valid outer bounds, not necessarily tight model-consistent bounds. Report their conservativeness empirically.

## 5.4 Robust attribution-sign certificate

**Proposition 1.** If:

\[
\underline\phi_{i,\Gamma,r}>0,
\]

then \(\phi_i(M)>0\) for every \(M\in\mathfrak M_{\Gamma,r}\). If \(\overline\phi_{i,\Gamma,r}<0\), then \(\phi_i(M)<0\) for every model in the set.

Use the exact phrase **robust order-averaged marginal-contribution certificate**. It does not imply benefit in every individual state or every coalition.

## 5.5 Simulation and exact-game estimation error

There is no permutation error because \(n=6\) and the Shapley game is exact. The remaining error arises from estimating coalition values.

If, with probability at least \(1-\delta\), every coalition value for one model is estimated within \(\varepsilon\):

\[
\max_{S\subseteq\mathcal I}|\widehat V_M(S)-V_M(S)|\leq\varepsilon,
\]

then:

\[
|\widehat\phi_i(M)-\phi_i(M)|\leq2\varepsilon.
\]

Derive a finite-rollout bound for \(\varepsilon\) under bounded rewards using a union bound over the 64 coalitions. Add a separate caveat that fitted OPE/world-model error is not eliminated by this Monte-Carlo result.

## 5.6 Robust selection regret and feasibility under value-estimation error

**Theorem 2 (Robust selection stability).** Suppose:

\[
\sup_{M\in\mathfrak M_\Gamma}\sup_{S\in\mathcal F_{\mathrm{safe}}}
|\widehat{\Delta V}_M(S)-\Delta V_M(S)|\leq\varepsilon.
\]

Let \(S^\star\) maximize true robust improvement and \(\widehat S\) maximize estimated robust improvement over the same feasible set. Then:

\[
\inf_M\Delta V_M(S^\star)-\inf_M\Delta V_M(\widehat S)\leq2\varepsilon.
\]

This theorem directly supports the decision layer: uniform coalition-value error, rather than Shapley approximation error, controls robust portfolio regret.

**Robust feasibility margin.** For any estimated constraint \(\widehat g_M(S)\) with uniform error at most \(\varepsilon_g\), enforce \(\sup_M\widehat g_M(S)\leq-\varepsilon_g\). Then the true robust constraint satisfies \(\sup_M g_M(S)\leq0\). Apply this margin to provider-exposure, relevance-loss, and safety constraints in estimated-policy experiments.

## 5.7 Nested-evidence interval contraction

**Proposition 2.** Under a stationary environment and an update rule that yields nested valid ambiguity sets:

\[
\mathfrak M_{\Gamma,r+1}\subseteq\mathfrak M_{\Gamma,r},
\]

then:

\[
\underline\phi_{i,\Gamma,r}
\leq
\underline\phi_{i,\Gamma,r+1}
\leq
\overline\phi_{i,\Gamma,r+1}
\leq
\overline\phi_{i,\Gamma,r}.
\]

Under regime change, do not claim contraction; the ambiguity set may expand and that is the correct response.

---

# 6. CURE-SIM: GROUND-TRUTH SEQUENTIAL RECOMMENDATION BENCHMARK

## 6.1 Motivation and design goals

Ordinary recommendation logs cannot reveal oracle long-horizon counterfactuals, hidden confounding, or true intervention Shapley values. CURE-Sim is the primary causal evidence source. It must support exhaustive intervention rollouts and disclose every structural equation.

## 6.2 State dynamics

For each user \(u\), simulate latent interest \(z_{u,t}\), novelty appetite \(n_{u,t}\), fatigue \(f_{u,t}\), and exposure history. For each item/provider, simulate popularity and exposure state. Platform state includes aggregate concentration and provider exposure disparity.

Core mechanisms:

- exposure increases probability of interaction but does not equal preference;
- repeated exposure increases fatigue after a user-specific threshold;
- novelty can increase delayed satisfaction for some users and harm immediate click probability for others;
- popularity has a self-reinforcing feedback effect;
- provider exposure is a shared platform outcome, creating controlled interference;
- logged policy assignment includes an adjustable unobserved confounder to generate a known \(\Gamma\)-sensitivity regime.

## 6.3 Intervention interaction regimes

Generate at least six regimes:

1. additive interventions;
2. exploration–repeat-suppression complementarity;
3. novelty–long-tail redundancy;
4. provider balancing–immediate relevance antagonism;
5. delayed benefit from fatigue mitigation;
6. hidden-confounding and policy-shift stress tests.

## 6.4 Oracle values and release

For each environment seed, enumerate all 64 intervention coalitions under the true SCM. Release:

- oracle \(\Delta V_{M^\star}(S)\) and \(V_{M^\star}(S)\);
- oracle Shapley values and interactions;
- **oracle-optimal portfolio** \(S_{\mathrm{oracle}}^\star=\arg\max_S\Delta V_{M^\star}(S)\);
- **oracle-robust portfolio** only for a pre-declared stress set \(\mathfrak M_{\mathrm{stress}}^\star\), defined by disclosed parameter perturbations before outcomes are inspected;
- logged trajectories with propensities;
- environment configuration and deterministic seeds.

Evaluate both **in-set** cases, where \(M^\star\in\mathfrak M_\Gamma\), and deliberately **misspecified** cases, where \(M^\star\notin\mathfrak M_\Gamma\). The latter demonstrates how attribution certificates fail when their assumptions are wrong rather than treating every miss as an implementation defect.

## 6.5 Benchmark metrics

Attribution MAE, rank correlation, sign accuracy, interaction recovery, sensitivity validity, finite-sample confidence coverage, ambiguity and sampling widths, robust regret, false feasibility/false rejection, relevance-constraint violation, provider-exposure violation, fatigue, and runtime.

---

# 7. EXPERIMENTAL SETUP

## 7.1 Evidence hierarchy and datasets

| Evidence source | Role | Permitted claim |
|---|---|---|
| CURE-Sim | Main causal benchmark | Oracle causal credit, interval validity, long-horizon feedback, robust regret |
| Coat / Yahoo! R3 | Randomized-vs-biased evaluation | Short-horizon exposure/selection-bias and OPE behavior |
| A propensity-logged sequential dataset, only after audit | Sequential OPE stress test | Claims limited to recorded support/horizon |
| MovieLens / Amazon | Optional semi-synthetic reproducibility | Not real causal exposure evidence |

Do not present MIND, MovieLens, Amazon review data, or static ratings as evidence of real long-term causal ecosystem effects unless exposure policies, propensities, and outcome semantics are available. Every real dataset requires a causal/logging audit in Appendix C.

## 7.2 Base recommender policies

Use one reproducible primary base policy and one robustness policy. Example:

- primary: BPR-MF or a standard sequential ranker with fully documented code;
- robustness: LightGCN or SASRec through a maintained public implementation.

The base policy is fixed during a rollout horizon. Report CURE-Rec over both bases to demonstrate that it is a policy layer rather than an architecture claim. Do not make DyHuCoG a code dependency.

## 7.3 Causal estimators and ambiguity-set construction

- **Oracle SCM evaluation:** CURE-Sim only.
- **Sequential DR evaluation:** real logs with support and propensities.
- **Sensitivity analysis:** \(\Gamma\in\{1,1.25,1.5,2\}\), with \(\Gamma=1\) representing the sequential-ignorability reference case.
- **Ambiguity approximation:** candidate response models / adversarial weights approximate extrema; quantify approximation gaps against CURE-Sim oracle sets.

## 7.4 Baselines

### Recommendation/policy baselines

- no intervention (\(\pi_0\));
- immediate-reward reranking;
- independently best nominal intervention;
- independently best lower-bound intervention;
- greedy nominal coalition utility;
- direct leave-one-intervention-out policy contribution \(\operatorname{LOO}_i=\Delta V(N)-\Delta V(N\setminus\{i\})\);
- robust causal-policy selector without Shapley explanation (explanatory-layer ablation);
- deployment-prior weighted semivalue sensitivity analysis, if the coalition prior is defensible;
- oracle coalition(s) (CURE-Sim only).

### Attribution baselines

- predictive SHAP over policy-control features, where meaningful;
- observational/conditional feature attribution;
- causal Shapley or do-Shapley adapted to a point response model;
- leave-one-intervention-out;
- point-estimate interventional Shapley;
- bootstrap/ensemble Shapley uncertainty without a formal sensitivity set.

## 7.5 Metrics

**Causal attribution:** MAE to oracle Shapley, Spearman/Kendall correlation, top-\(k\) intervention precision, sign accuracy, and interaction-recovery accuracy.

**Uncertainty:** sensitivity validity of the population identified region, finite-sample confidence coverage of the estimated region, ambiguity width, sampling-confidence width, false robust-positive rate, false robust-negative rate, and sensitivity to \(\Gamma\).

**Decision quality:** robust regret, mean and worst-case policy improvement, harmful-policy rate, abstention frequency, false feasibility/false rejection, cost, and constraint violations.

**Recommendation and ecosystem:** NDCG/Recall as descriptive ranking diagnostics; causal relevance/satisfaction, cumulative reward, fatigue, catalogue coverage, long-tail exposure, provider exposure Gini/disparity, and concentration/popularity feedback.

## 7.6 Reproducibility and statistics

- five independent environment/base-policy seeds;
- fixed intervention library and canonical composition before results;
- paired per-environment or per-cohort tests, with Holm–Bonferroni correction;
- bootstrap confidence intervals for real-log OPE where appropriate;
- full coalition-value tables for all 64 coalitions in the appendix;
- code, generator, config files, logging audits, and hardware details released.

---

# 8. RESULTS AND DISCUSSION

## 8.1 RQ1 — causal credit recovery on CURE-Sim

**Table 3:** attribution MAE/rank/sign accuracy by intervention regime.  
**Figure 2:** predicted versus oracle Shapley values.  
Show where predictive/observational methods assign high importance to exposure-correlated but causally non-beneficial controls.

## 8.2 RQ2 — interval validity and ambiguity sensitivity

**Table 4:** coverage and width by \(\Gamma\), graph/response misspecification, and policy shift.  
**Figure 3:** coverage–width frontier.  
Critical claim: point estimates become overconfident as unobserved confounding grows; CURE-Rec intervals should widen rather than report false precision.

## 8.3 RQ3 — robust portfolio quality

**Table 5:** robust regret, worst-case utility, harmful-policy rate, and cost.  
**Figure 4:** intervention portfolio selection under additive, complementary, redundant, and antagonistic regimes.  
Compare direct robust enumeration with independent intervention ranking and Shapley-score heuristics. The proposed decision method must be direct robust coalition evaluation.

## 8.4 RQ4 — long-term trade-offs

**Figure 5:** trajectories of reward, fatigue, popularity concentration, and provider exposure.  
**Table 6:** relevance/provider/fatigue constraint satisfaction.  
Do not claim a universal fairness solution. Show the explicit trade-off frontier and failure modes.

## 8.5 Real logged-data results

Report only what each log supports. Include:

- propensity/support diagnostics;
- effective sample size for OPE;
- overlap failures;
- short-horizon utility bounds;
- sensitivity to \(\Gamma\).

A null result or wide interval is a valid outcome: it means the log cannot justify a policy claim at the required robustness level.

## 8.6 Ablations

1. replace \(do(\pi_S)\) with observational prediction;
2. collapse \(\mathfrak M_\Gamma\) to one point response model;
3. use outer bounds instead of model-consistent extrema;
4. select by lower Shapley score rather than direct robust coalition utility;
5. remove interaction diagnostics;
6. remove exposure/fatigue state from the response model;
7. vary base recommender;
8. vary horizon, slate capacity, and \(\Gamma\).

## 8.7 Efficiency and scalability

Report exact coalition sweep runtime at six interventions; show scaling to \(n=8,10\) in CURE-Sim as a stress test. Do not imply exact enumeration remains free for arbitrary intervention libraries.

## 8.8 Case study

Present one selected portfolio. Include baseline slate, transformed slate, long-term value interval, selected interventions’ Shapley regions, pairwise interaction region, relevance/provider constraints, and the reason uncertain alternatives were deferred.

## 8.9 Limitations

State explicitly:

- partial-identification validity is conditional on the declared ambiguity set and assumptions;
- individual logged-data outcomes cannot identify unrestricted platform interference;
- long-horizon real evidence is difficult and synthetic ground truth is necessary but insufficient alone;
- six interventions are deliberately controlled and do not exhaust platform policy space;
- utility and provider-exposure targets are normative choices;
- fixed-horizon evaluation does not model unrestricted recommender retraining;
- model-consistent intervals may be wide, which is an honest abstention signal rather than a method failure.

---

# 9. CONCLUSION AND FUTURE WORK

Restate the transition:

\[
\text{entity credit in recommendation learning}
\rightarrow
\text{causal credit for changes to recommendation policy}.
\]

Future work:

- active evidence acquisition to shrink action-attribution regions near the decision boundary;
- continual/online CURE-Rec with regime-change detection;
- policy-component games for feedback requests, human review, and model updates;
- federated/privacy-preserving causal intervention games;
- richer multi-sided fairness and welfare constraints;
- structured/Owen/Myerson values for hierarchical intervention families.

---

# DECLARATIONS

Funding · competing interests · ethics approval · consent · data availability · code availability · author contributions (CRediT) · AI-tool disclosure according to venue policy.

---

# APPENDICES

- **A. Proofs:** existence, outer bounds, sign certificate, finite-rollout propagation, nested-evidence contraction.
- **B. CURE-Sim:** all structural equations, intervention definitions, seeds, oracle computation, and stress-test regimes.
- **C. Causal/logging audit:** propensity availability, slate factorization, support, treatment timing, exposure measurement, outcome timing, missingness, and permitted claims per dataset.
- **D. Additional results:** all 64 coalition values, full interaction tables, base-policy robustness, hyperparameters, and ablation tables.

---

# NOTATION LIST

| Symbol | Meaning |
|---|---|
| \(\mathcal U,\mathcal I\) | user and item sets |
| \(\mathcal S_t\) | platform-level recommendation state |
| \(H_{u,t}\) | user \(u\)’s observed history/state |
| \(E_t\) | exposure/repetition state |
| \(Q_t\) | aggregate item/provider ecosystem state |
| \(\theta_t\) | deployed base-policy state |
| \(\pi_0\) | baseline recommendation policy |
| \(I_i=(g_i,\mathcal E_i,c_i)\) | intervention player |
| \(S\) | intervention coalition/portfolio |
| \(\pi_S\) | policy transformed by coalition \(S\) |
| \(M\) | one causal recommendation-response model |
| \(\mathfrak M_{\Gamma,r}\) | causal ambiguity set at evidence update \(r\) |
| \(V_M(S)\) | long-term coalition value under \(M\) |
| \(\underline V(S),\overline V(S)\) | lower/upper coalition values |
| \(\phi_i(M)\) | exact Shapley contribution of intervention \(i\) under \(M\) |
| \(\Phi_i=[\underline\phi_i,\overline\phi_i]\) | model-consistent Shapley region |
| \(\mathcal I_{ij}(M)\) | pairwise Shapley interaction |
| \(H\) | evaluation horizon |
| \(\Gamma\) | unmeasured-confounding sensitivity parameter |
| \(\mathcal F_{\mathrm{safe}}\) | feasible portfolios satisfying constraints |

---

# PLANNED FIGURES & TABLES

| ID | Type | Content |
|---|---|---|
| Fig. 1 | Diagram | CURE-Rec: base policy → intervention game → ambiguity set → robust selection → explanation |
| Fig. 2 | Scatter | Oracle versus estimated intervention Shapley values on CURE-Sim |
| Fig. 3 | Curve | Interval coverage versus width across \(\Gamma\) and misspecification regimes |
| Fig. 4 | Heatmap | Interaction regions across the six policy interventions |
| Fig. 5 | Trajectory | Long-run reward, fatigue, concentration, and provider exposure under selected portfolios |
| Fig. 6 | Case study | Baseline and transformed slate plus intervention explanation card |
| Fig. 7 | Runtime | Coalition evaluation/scaling with number of interventions |
| Fig. 8 | Sensitivity | Robust utility and action selection versus \(\Gamma\), horizon, and slate capacity |
| Tab. 1 | Related work | Positioning and differentiation |
| Tab. 2 | Benchmark | CURE-Sim/environment and logged-data statistics |
| Tab. 3 | Results | Causal attribution recovery |
| Tab. 4 | Results | Interval validity and calibration |
| Tab. 5 | Results | Robust portfolio utility and regret |
| Tab. 6 | Results | Long-term relevance/provider/fatigue trade-offs |
| Tab. 7 | Results | Ablations and base-policy robustness |
| Tab. 8 | Results | Runtime/memory and statistical tests |

---

# PLANNING NOTES (NOT PART OF THE MANUSCRIPT)

## Why this is a substantive fourth chapter

The prior work develops a single importance-allocation argument across static features, hierarchical explanation, and dynamic recommendation entities. CURE-Rec adds the missing decision layer: the players are now changes that can be made to the recommendation process, and the utility is a causal future outcome rather than a predictive/model-defined score alone.

## Non-negotiable claims discipline

1. Never call an ensemble spread a partially identified causal interval without an explicit ambiguity-set argument.
2. Never sum coordinate-wise Shapley lower bounds as a certified portfolio utility.
3. Never claim real long-term feedback-loop identification from static ratings data.
4. Never silently skip infeasible actions in a Shapley permutation; separate game semantics from budgeted selection.
5. Never call \(\underline\phi_i>0\) a universal safety guarantee; it is an ambiguity-set-conditional, order-averaged attribution certificate.
6. Never claim a new recommender architecture; the policy layer must work over documented base recommenders.
7. Never call direct robust selection a Shapley performance gain over an identical non-Shapley selector; Shapley’s role is credit recovery, diagnosis, certification, and auditability.

## Minimum acceptance criteria before writing results prose

- CURE-Sim oracle values and all exact-game unit tests pass.
- The causal/logging audit establishes what each real dataset can actually support.
- Model-consistent interval computation is compared with conservative outer bounds.
- At least one intervention regime produces non-additive effects; otherwise the coalition-game contribution is not tested.
- Direct robust portfolio selection beats at least the nominal and independent-intervention baselines on worst-case utility in the stress-test regimes.
- If real logs yield wide intervals, report them as abstention evidence rather than tuning until they disappear.

## Recommended build order

1. Write CURE-Sim and exact oracle coalition evaluator before using any real dataset.
2. Implement six canonical intervention operators and prove composition/unit tests.
3. Implement exact 64-coalition game and fixed-model Shapley/interactions.
4. Add response-model/OPE interface and \(\Gamma\)-sensitivity ambiguity interface.
5. Implement direct robust enumeration and explanation cards.
6. Run synthetic identification, misspecification, and interaction experiments.
7. Audit Coat/Yahoo! R3 and any sequential propensity log before allowing real-data claims.
8. Add second base recommender and final statistical/reporting layer.

## Follow-up papers deliberately excluded

- active intervention/experiment acquisition;
- continual CURE-Rec under regime change;
- causal games over retraining/model-update operations;
- federated CURE-Rec;
- multi-sided fairness beyond the declared provider-exposure constraint.
