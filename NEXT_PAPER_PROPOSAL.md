# From Attribution to Action: A Forward-Looking Cooperative-Game Framework for Actionable Recommendation

**Author:** Mouad Louhichi (draft proposal prepared on behalf of the author)
**Scope:** Phase 1 (grounding) + Phase 2 (literature positioning) + Phase 3 (novel paper proposal)
**Working title of the proposed paper (primary candidate):**
*"Cooperative Action Values: A Forward-Looking Shapley Framework that Recommends and Intervenes"*

---

# PHASE 1 — DEEP GROUNDING: INTERNAL KNOWLEDGE GRAPH

This section distils the thesis, the three published papers, and the five in-repo idea drafts into a single knowledge graph. The documents under `phd-thesis/`, `previous-papers/`, and `paper-ideas/` are treated as the primary source of truth.

## 1.1 Thesis research vision

The PhD argues that **cooperative game theory — specifically the Shapley value — is not merely a post-hoc explanation device but a *common methodological language* for importance allocation** that can unify (i) explanation of opaque structure, (ii) hierarchical/coherent multi-resolution attribution, and (iii) *in-training* optimisation of a recommender. The spine of the thesis is a three-step trajectory (its own RQ5):

1. **C1 (Ch. 5, published Procedia 2023):** Shapley explains *black-box clustering* (post-hoc).
2. **C2 (Ch. 6, published IJACSA 2025):** Shapley stays *coherent under hierarchy and scale*.
3. **C3 (Ch. 7, published IJIES 2026, DyHuCoG):** Shapley becomes an *in-training signal* inside a dynamic hypergraph recommender.

The thesis's *central unmeasured claim* is "actionable insight" (its Definition 1.1): *an explanation is actionable when it identifies at least one modifiable factor whose change is associated with a specifiable change in output, and that factor is accessible to the decision-maker.* The thesis explicitly concedes that actionability was used **as a framing concept, never as a measured endpoint** (Ch. 8.3, Ch. 9.4). This is the single largest open gap and the natural site for the next paper.

## 1.2 Mathematical foundations in the thesis

- **Transferable-utility (TU) game** over player set `N`, characteristic function `v(S)`, `v(∅)=0`.
- **Shapley value** `φ_j = Σ_{S⊆N\{j}} (|S|!(|N|-|S|-1)!/|N|!) [v(S∪{j})−v(S)]`, axiomatised by efficiency, symmetry, null-player, additivity (Appendix A.1 gives a full proof via the unanimity basis).
- **Monte-Carlo Shapley estimator** `φ̂_j = (1/M) Σ_m [v(S_m∪{j})−v(S_m)]`, unbiased with `O(1/M)` variance decay; `M=50` ≈ 99% accuracy (`MSE ≈ 1.4e-5`) on MovieLens-1M.
- **Hierarchical attribution consistency (Appendix A.2):** parent-level expected absolute attribution = weighted average of child-level expectations (law of total expectation), residual `ε_j` from surrogate mismatch.
- **Functional-ANOVA / GA2M** machinery (purification, concurvity, shape-shift checks) is already anticipated in the ActionShap draft (§5.2) — reusable.
- **Hypergraph GNN propagation** with Shapley-weighted normalised neighbourhood weights and an interaction-level attention gate.

## 1.3 Existing algorithms / models

| Component | Where | Role |
|---|---|---|
| PCA–K-Means–LightGBM–TreeSHAP pipeline | C1 (Procedia 2023), Ch.5 | Post-hoc clustering explanation |
| Multi-level hierarchical SHAP aggregation | C2 (IJACSA 2025), Ch.6 | Cross-level coherence |
| **DyHuCoG** (dynamic hypergraph recommender, MC Shapley in message passing, attention gate, BPR+diversity+context+reg loss) | C3 (IJIES 2026), Ch.7 | In-training Shapley; accuracy+diversity+coverage |
| Monte-Carlo Shapley, EMA temporal smoothing, clipped normalisation | DyHuCoG | Attribution estimation |
| `ActionShap/code` (static prototype + gate notebook) | repo | Intervention-grounded *evaluation* prototype |
| Exact source-level Shapley (5 players, 32 coalitions) | SignalShap draft | System-owner source attribution |

## 1.4 The five existing idea drafts (what must NOT be duplicated)

| Draft | Players | Core object | Loop closed? | Theoretical depth |
|---|---|---|---|---|
| **SignalShap** | 5 signal *sources* (CF/CB/POP/REC/SEQ) | Exact source-attribution game; segment-adaptive fusion | yes (fusion) | 3 light propositions + 1 remark |
| **ActionShap (revised)** | user-specific *interaction factors* | *Evaluation*: does attribution predict feasible-intervention effect? (AIA/regret/top-k) | no (evaluative) | 1 proposition (alignment under local linearity) + misalignment decomposition |
| **FairShap** | items/providers | fairness-aware Shapley exposure attribution + re-ranking | yes (exposure) | 1 light proposition (exposure efficiency) |
| **MHyperShap** | LLM *agents* | Myerson-restricted dynamic hypergraph game; attribution→routing | yes (routing) | uniqueness theorem (CE/HF/TC) |
| **DyHuCoG (thesis)** | users/items/contexts | in-training Shapley for diversity/accuracy | yes (weights) | no new axiom; standard MC Shapley |

**Critical observation:** every existing draft computes Shapley values over a *backward-looking* characteristic function — the value of a coalition is defined by the *current/observed* outcome (`NDCG@K(S)`, cluster probability, exposure, pipeline outcome). None defines a **forward-looking value function** (expected *future* utility) whose allocation answers "what action should I take now to maximise *future* value?" None is a *prescriptive* framework that **generates** recommendations *and* budget-constrained minimal interventions from the same cooperative object. ActionShap is explicitly *evaluative*, not generative. **This is the open slot for a genuinely new paper.**

## 1.5 Limitations acknowledged (consolidated)

1. **Exact Shapley is intractable** → all prior work uses MC sampling, surrogates, or small player sets.
2. **DyHuCoG**: overhead ~1.78× HPCF; depends on rich context; MC could benefit from variance reduction; explanation evaluated via plausibility, not user studies.
3. **Actionability never measured** (thesis-defining gap).
4. No **sequential/streaming/online** handling; no **uncertainty** quantification tied to attribution; no **human-subject** validation.
5. No unified multi-objective treatment of *future-oriented* objectives (long-term engagement, retention).

## 1.6 Future work named across documents
- More scalable / lower-variance Shapley (learned proposals, adaptive refresh).
- **Online, streaming, sequential** settings; delayed feedback.
- **Human-centred** evaluation of actionability.
- **Fairness / exposure** integration.
- **Uncertainty-aware** attribution.
- Attribution→improvement loops (already in SignalShap / MHyperShap / FairShap).

---

# PHASE 2 — LITERATURE POSITIONING

## 2.1 Where the field stands (as of late 2025 / 2026)

- **Shapley in recommendation** is overwhelmingly *backward-looking attribution*: feature/tag coalitions (e.g., CTMVM at Alibaba), data-Shapley / data valuation, source attribution (SignalShap), creator-incentive / coopetitive bandit games (TU games with cores and Shapley membership), and community-clustering CF via Shapley. All answer "who/what *caused* the current outcome."
- **Explainable recommendation** concentrates on *end-user* "why this item" explanations (item/feature/path/review), with an under-served *system-owner* audience (SignalShap's gap).
- **Actionable / counterfactual / recourse recommendation** exists (RecRec, CEERS, CARMA, algorithmic-recourse surveys), but is **feature-level**, per-individual, *post-hoc prescription for a fixed classifier*; it does **not** compute a cooperative-game allocation and does **not** close the loop into recommender training.
- **Uncertainty-aware recommendation** is emerging (deboosting for low-activity users, exploration bonus for high-activity), but is **not** coupled to cooperative attribution.
- **Trustworthy/fair recommendation** (two-sided exposure fairness, calibration) exists but is separate from the forward-looking action framework.
- **Myerson/Owen-restricted games** exist (MHyperShap for agents; MARL hypergraph methods) — restricted coalition spaces, but not over *actions* for a recommender.

## 2.2 What is solved / partial / missing

**Solved:** (a) post-hoc, coherent, hierarchical Shapley explanation; (b) in-training Shapley that improves immediate ranking/diversity (DyHuCoG); (c) exact small-scale source attribution; (d) Shapley-based fairness exposure; (e) evaluation of whether attributions predict single interventions (ActionShap).

**Partial:** (a) attribution→improvement loops (exist but per-objective: diversity, fairness, routing); (b) uncertainty awareness (unconnected to attribution); (c) recourse in RS (not game-theoretic, not loop-closing).

**Missing (the opportunity):** a *single unified, forward-looking, uncertainty-aware cooperative-game framework* that (i) defines a new allocation over **actionable levers** under a **discounted expected-future-utility** characteristic function, (ii) **generates** both recommendations and **budget-constrained minimal interventions** from one cooperative object, (iii) is **risk/uncertainty-adjusted**, (iv) **closes the loop** into model training so the recommender becomes *action-aware*, and (v) carries a **new forward game with provable properties** (not just a new application of the same backward game). No existing paper — thesis, drafts, or literature — occupies this intersection.

## 2.3 Where the strongest publication opportunity is
**Q1 journals (IEEE TKDE / ACM TOIS / Information Fusion / IEEE TNNLS)** reward *new theory* + *feasible, rigorous experiments* + *a clear decision-oriented contribution*. The forward-looking cooperative-action framework below is exactly that: a new forward game and allocation with provable properties, a rigorous optimisation layer, uncertainty coupling, and a closed training loop — while reusing the author's validated DyHuCoG backbone, datasets, and protocol.

---

# PHASE 3 — NOVEL PAPER PROPOSAL

## Framework name and core concept
- **Core new concept: Cooperative Action Value (CAV)** — a Shapley-type allocation over an *actionable-lever space*, computed under a *forward-looking, time-discounted, uncertainty-adjusted* characteristic function.
- **Framework / method name: CAVI — Cooperative Action-Value Intelligence.**
- **Positioning slogan:** "ActionShap *evaluates* whether explanations predict interventions; CAVI *defines a new forward-looking allocation* and uses it to *generate* the plan."

---

## 3.1 Paper Titles (10 publication-quality candidates)

1. **Cooperative Action Values: A Forward-Looking Shapley Framework that Recommends and Intervenes**
2. **From Explaining to Acting: Uncertainty-Aware Cooperative Action Values for Actionable Recommendation**
3. **CAVI: Cooperative Action-Value Intelligence — A Dynamic Shapley Framework for Recommendation and Minimal-Action Intervention**
4. **Sequential Cooperative Attribution: Shapley Values over Action Spaces for Decision-Oriented Recommenders**
5. **Intervention-Aware Shapley: Budget-Constrained Coalition Optimisation for Actionable Recommendation**
6. **Utility-Aware Coalition Optimization: A Forward Cooperative-Game Theory of Actionable Recommender Systems**
7. **Counterfactual Cooperative Recommendation: Minimal-Action Recourse via Cooperative Action Values**
8. **Decision-Oriented Explainable Recommendation: From Post-Hoc Attribution to Prospective Action Planning**
9. **Actionable Shapley: A Mean–Variance Cooperative-Game Framework for Recommendations That Change the Future**
10. **Cooperative Action-Value Learning: Closing the Loop Between Recommendation, Explanation, and Intervention**

---

## 3.2 Research Gap (why existing approaches are insufficient)

1. **Every Shapley-based recommender is backward-looking.** DyHuCoG, SignalShap, FairShap, MHyperShap, CTMVM, data-Shapley, and creator-incentive games all define `v(S)` on the *current/observed* outcome (`NDCG@K(S)`, cluster label, exposure, pipeline result, regret). Their allocation answers *"why did this happen?"* — never *"what should happen next, and which action gets me there?"* The thesis's own claim of "actionable insight" was explicitly left **unmeasured**.

2. **Shapley-based explanation stops at explanation.** Because `v` is backward-looking, the Shapley value describes *causation of the present*, which is neither necessary nor sufficient for *optimal action toward the future*: a high-attribution factor may be immovable, high-variance, or irrelevant to future utility. There is no feasibility, no budget, no horizon, no discounting, and no uncertainty in the value function.

3. **Actionable recommendation is still an open problem** because it is currently done in two disconnected, non-cooperative ways:
   - **Counterfactual recourse (RecRec, CEERS, CARMA)** prescribes per-individual feature flips for a *fixed classifier* — no cooperative allocation, no forward utility, no loop into training, no accounting for *joint* action interactions (redundancy/complementarity among actions).
   - **RL/sequential recommenders** optimise long-term reward but produce **no attribution** and **no recourse**; they are black boxes.
   - **Uncertainty-aware methods** calibrate risk but do not attribute it to actions.
   **No single framework couples cooperative attribution, forward utility, feasibility/budget, uncertainty, recourse generation, and training-time loop-closure.**

4. **No new theory — only new applications.** The existing drafts extend *where* Shapley is applied (sources, interactions, fairness, agents) but reuse the same backward-looking allocation. A Q1 contribution should define a *new forward game* and an allocation with *provable properties*, not merely apply Shapley to a new player set over the same (backward) value function.

## 3.3 Core Research Question (one central question)

> **Can a cooperative-game framework — whose players are *actionable levers* and whose characteristic function is the *discounted, uncertainty-adjusted expected future utility* of the user — produce a single Shapley-type allocation (the *Cooperative Action Value*) that simultaneously (i) explains the present recommendation, (ii) identifies which *minimal, budget-feasible* set of actions changes the future recommendation, and (iii) trains the recommender to be coherent with the actions it prescribes?**

## 3.4 Main Hypothesis

> **There exists a forward-looking, uncertainty-adjusted cooperative allocation over actionable levers such that ranking the levers by their Cooperative Action Value and greedily selecting the highest value-per-cost set under a budget produces (a) future-utility lifts and recourse validity that are significantly higher than backward-looking Shapley, counterfactual recourse, and RL baselines, and (b) a recommender whose training is *action-aware* and whose predictions remain accurate while its prescribed actions are realised.** The forward-looking game is *not* reducible to the backward-looking game: the two orderings differ measurably, and the divergence is explained by feasibility, interaction, and variance of future utility (mirroring — but generalising — the redundancy insight of SignalShap's Proposition 2).

---

## 3.5 Novel Framework — CAVI

### 3.5.1 Architecture (five coupled modules)

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  (A) Forward action-game module                                          │
 │      - Actionable levers A = A_user ∪ A_platform ∪ A_item                │
 │      - Feasibility graph F over actions (joint-feasible coalitions)      │
 │      - Forward characteristic function v_t(S) = discounted E[future util]│
 │      - Uncertainty: v^σ2_t(S) = variance of discounted future utility    │
 └───────────────┬──────────────────────────────────────────────────────────┘
                 │ coalitions, v_t(S), v^σ2_t(S)
 ┌───────────────▼──────────────────────────────────────────────────────────┐
 │  (B) CAV allocation module (COOPERATIVE ACTION VALUES)                    │
 │      φ^μ = Shapley(v_t)   ;   φ^σ2 = Shapley(v^σ2_t)                     │
 │      CAV_i = φ^μ_i − κ·φ^σ2_i   (risk-adjusted, certainty-equivalent)     │
 └───────────────┬──────────────────────────────────────────────────────────┘
                 │ CAV vector + per-lever cost c_i
 ┌───────────────▼──────────────────────────────────────────────────────────┐
│  (C) Actionable decision layer (minimal-action / recourse)                │
│      min-cost coalition S* s.t. cost(S*) ≤ B and ΔE[v_t](S*) ≥ Δ*         │
│      Greedy on CAV_i / c_i  (risk-adjusted, (1−1/e) approx if submodular) │
 └───────────────┬──────────────────────────────────────────────────────────┘
                 │ recommended list + recommended intervention plan (S*)
 ┌───────────────▼──────────────────────────────────────────────────────────┐
 │  (D) Action-aware recommender (history-conditioned hypergraph GNN,        │
 │       DyHuCoG backbone) scoring f_θ(x^S) at inference                     │
 └───────────────┬──────────────────────────────────────────────────────────┘
                 │ realised future feedback → update dynamics model + θ
 ┌───────────────▼──────────────────────────────────────────────────────────┐
 │  (E) Closed-loop training objective                                       │
 │      L = L_rank + λ_act·L_actcons + λ_div·L_div + λ_ctx·L_ctx + λ_reg·L_reg│
 └──────────────────────────────────────────────────────────────────────────┘
```

### 3.5.2 Cooperative game definition

Let `u` be a focal user at time `t`. Define the **actionable-lever set**

`A = A_u ∪ A_p ∪ A_i`

- `A_u` (user levers): behaviours/interactions the user can change — consume/rate/promote item `i`, follow a genre/tag, adjust an explicit preference signal, etc.
- `A_p` (platform levers): exposure weight of an item/provider, a debiasing or feature-strengthening control, a diversity parameter.
- `A_i` (item levers): item attributes modifiable by the provider (metadata, presentation).

A **state vector** `x ∈ ℝ^{|A|}` assigns each lever a value; `x⁰` is the status quo. An **intervention operator** `do(S)` sets the levers in `S` to declared feasible targets and leaves `A\S` at `x⁰`.

**Feasibility structure.** Let `F` be a hypergraph over `A` whose hyperedges are the *joint-feasible* action coalitions (e.g., "consume items within a single genre this session", "the platform may boost exactly one provider tier"). A coalition is *valid* iff it induces a connected subhypergraph in `F`. This is the Myerson-style restriction, transplanted from agents (MHyperShap) to *actions* — giving a genuine structural constraint rather than the full power set.

### 3.5.3 Utility (characteristic) function — forward-looking

Let `Rec(x)` be the ranking produced by the recommender evaluated at state `x`. Let `R(u, Rec(x), s_{t+τ})` be a per-step, multi-objective reward at future state `s_{t+τ}`:

`R = α·NDCG@K(u, Rec(x), y_next) + β·Diversity(Rec(x)) + γ·ContextScore(...) + δ·LongTerm(u, ...)`

(the weights mirror DyHuCoG's `(α,β,γ)` but now include a long-term term; `α+β+γ+δ=1`). Define the **forward characteristic function** as the expected discounted future utility under `do(S)`:

```
v_t(S) = E_{s_{t+1..t+H}} [ Σ_{τ=1}^{H} γ^{τ-1} R(u, Rec(x^S), s_{t+τ}) | s_t, do(S) ]
v_t(∅) = E[... | s_t, do(∅)]   (status-quo baseline),   normalised so v_t(∅)=0 (uplift form)
```

`H` is the horizon, `γ∈(0,1)` the discount, and the expectation is over a **learned next-state / next-interaction dynamics model** `P_ψ(s' | s, x, a)` (a sequential model; e.g., a lightweight transition predictor trained on the interaction stream). The *grand coalition value* `v_t(A) − v_t(∅)` is the total **achievable** future-utility uplift — this is what CAV's efficiency axiom will distribute.

> **Why this is new.** The thesis and all drafts evaluate coalitions on *immediate, realised* utility. Here a coalition's worth is the *prospective* value it is expected to create. This is the mathematical embodiment of "actionable intelligence": the game is played over the *future*, not the past.

**3.5.3-b Off-policy evaluation of the forward value (the threat the first draft ignored).** `v_t(S)` is an expectation over *counterfactual futures under `do(S)`* that were **never executed** by the logged policy. The interaction stream was generated by the platform's existing (unknown) recommendation/behaviour policy `π_0(a|s)`, not by the interventions CAV proposes. Estimating `v_t(S)` by naively rolling out `P_ψ` and measuring "realised lift" is therefore **off-policy, confounding-prone, and biased**: the logged data selectively shows what the old policy exposed, and any plan that merely exploits the model's own optimism will look better than it is. This is the same problem RL4Rec and bandit-recourse work address with importance sampling and doubly-robust estimators, and the proposal must carry that machinery explicitly:

- **Propensity model — and which one, given the data (this addresses the reviewer's point about non-bandit logs).** IPS/DR presuppose a logging propensity `π_0(a|s)`. MovieLens-1M, Amazon-Book, and Yelp2018 are **not bandit logs with recorded randomization** — they are observational rating/review matrices that are *missing not at random (MNAR)*: the probability an interaction is *observed* is itself the selection bias. We therefore borrow the "recommendations as treatments" / unbiased-LTR framework (Schnabel et al., 2016): estimate the **observation/selection propensity** `π̂_0((u,i)) = P(observed | user u, item i)` — e.g., via the naive-Bayes estimator `π̂_0(u,i) ≈ P(observed|u)·P(observed|i)` or a logistic-regression propensity over observable covariates (user activity, item popularity) — and use that in place of a bandit logging policy. On the synthetic environment, the propensity is *known* (we log it), which gives a clean positive control. Two honest caveats, stated as such: (i) these propensities are only as good as the missingness model, so unobserved confounding can still bias them — this is precisely why the **DR** estimator (robust to a wrong propensity if the outcome model is right) is the headline rather than raw IPS; (ii) we report the calibration of `π̂_0` and a reweighting effective sample size as diagnostics, and treat any plan whose DR-corrected lift is not robust to propensity/outcome-model misspecification as not established.
- **IPS estimator.** `v̂_t^{IPS}(S) = (1/n) Σ_t [ r_t · 1{a_t consistent with do(S)} / π̂_0(a_t|s_t) ]` — unbiased when propensities are correct; apply weight capping/clipping for variance control.
- **Doubly-robust (DR) estimator.** `v̂_t^{DR}(S) = (1/n) Σ_t [ w_t (r_t − r̂(s_t,a_t)) + r̂(s_t,a_t) ]` with `w_t = 1/π̂_0`, `r̂` the model-based outcome from `P_ψ`. **DR is unbiased if either the propensity model *or* the outcome model is correct** — this is the workhorse and the one to headline.
- **Self-normalized IPS (SNIPS)** as a lower-variance alternative.
- **Discrepancy diagnostics.** Report a **reweighting effective sample size** and the **predicted-vs-realised gap under IPS/DR** (not raw model roll-out) as the gate on the "does the plan work" claim.

Consequence for §3.8: the headline "realised lift" result must be evaluated with IPS/DR-corrected estimates (or on held-out interleaved offline data with propensity control), and reported *alongside* the naive model-based number so the gap is visible and honest. A plan cannot be claimed to work because the model that predicted it confirms it.

### 3.5.4 Payoff allocation — Cooperative Action Value

Let `v^σ²_t(S) = Var[ Σ_{τ=1}^{H} γ^{τ-1} R(u, Rec(x^S), s_{t+τ}) | s_t, do(S) ]` be the *variance game* (risk of a coalition of actions). Define the two component Shapley values **on the Myerson-restricted game induced by `F`** (restricted game defined in Step 1a below):

```
φ^μ_i  = Shapley_F(v_t)_i          (expected-future-utility credit of lever i)
φ^σ²_i = Shapley_F(v^σ²_t)_i       (marginal risk contribution of lever i)
```

**Cooperative Action Value (certainty-equivalent allocation):**

```
CAV_i(t) = φ^μ_i(t) − κ · φ^σ²_i(t)
```

where `κ ≥ 0` is the risk-aversion coefficient. When `κ=0`, CAV reduces to the (forward) Shapley value; increasing `κ` makes the allocation risk-averse, so high-variance levers are penalised.

**Reframed to remove the over-claim (this is the fix the review demanded).** The original draft asserted that `CAV = φ^μ − κ·φ^σ²` is the *unique* allocation satisfying "the classical four + A1 + A2 + A4" — with A1 stated as a *new axiom*. That claim does **not** follow from the machinery, and stating it that way made the theoretical spine fragile. The honest and correct construction is to make CAV the Shapley value of a single **mean–variance certainty-equivalent game**, from which A1–A4 become *provable properties* rather than axioms that are supposed to pin down uniqueness. Here is the corrected, rigorous version.

**Step 1 — the certainty-equivalent game.** For each (possibly disconnected) coalition `S ⊆ A`, define the risk-adjusted characteristic function

```
u_t(S) = E[V_t(S)] − κ · Var[V_t(S)]   =  v_t(S) − κ·v^σ²_t(S),   u_t(∅) = 0
```

`u_t` is a well-defined TU game on the full player set. This is a *single* game (not two games bolted together), so classical cooperative-game uniqueness applies to it directly.

**Step 1a — the Myerson-restricted game (this is the load-bearing step — get it right).** Feasibility `F` restricts *which coalitions can cooperate*. The correct object for the classical characterization is the **graph/hypergraph-restricted game** of Myerson (1977) / van den Nouweland–Borm–Tijs (1992):

```
u^F_t(S) = Σ_{C ∈ comp_F(S)} u_t(C)
```

where `comp_F(S)` is the set of maximal connected components of `S` in the feasibility structure `F` (a coalition `C` is *connected* in `F` iff its levers are linked by hyperedge chains). A coalition's worth is the sum of its connected components' worths — not the "all-or-nothing" value of `S` itself. **This distinction is essential:** the *component-efficiency + fairness* characterization that underpins Theorem 1 holds for `u^F_t` (the restricted game), *not* for an allocation defined by summing the Shapley formula only over connected coalitions. The latter construction does **not** in general satisfy component efficiency and must not be conflated with the Myerson value.

**Step 2 — the allocation.** Define the Cooperative Action Value as the **Myerson value of `u_t` under `F`**, i.e. the Shapley value of the restricted game `u^F_t`:

```
CAV_i(t) = Shapley(u^F_t)_i =: Shapley_F(u_t)_i
```

**Step 3 — why the linear form is a *theorem*, not an assumption.** Because the Shapley operator is **additive** in the characteristic function, and `u_t = v_t − κ·v^σ²_t` is a linear combination of two games (an additivity that passes through the restriction, since `u^F_t = v^F_t − κ·(v^σ²)^F_t` component-wise),

```
Shapley_F(u_t) = Shapley_F(v_t) − κ·Shapley_F(v^σ²_t)   ⟹   CAV_i = φ^μ_i − κ·φ^σ²_i
```

The mean–variance structure and the risk coefficient `κ` live in the *value function*, not in the allocation rule. This is the crucial difference from the over-claimed draft: we never need to assert that some new axiom forces `−κ·φ^σ²`; additivity of Shapley plus the definition of `u_t` *deliver* it.

**Theorem 1 (CAV well-posedness / uniqueness).** Let `F` be a feasibility structure over `A` (an acyclic hypergraph, or more generally any structure in which connectivity is defined by hyperedge chains) and let `u_t` be the certainty-equivalent game (Step 1) with restricted game `u^F_t` (Step 1a). The unique allocation on the restricted game `(A, F, u_t)` satisfying **Component Efficiency** and **Fairness** (the equal-loss-under-link-removal axiom, generalised to hyperlinks) is the Myerson value `CAV = Shapley(u^F_t)`. *Proof: this is the classical Myerson (1977) uniqueness result for graph-restricted games, extended to hypergraph communication situations by van den Nouweland–Borm–Tijs (1992); component efficiency and the (hyper)link fairness axiom characterise the restricted value uniquely, and efficiency/symmetry/null-player/additivity of the underlying Shapley pin the within-component split. This is standard, checkable theory — not a new axiom.* Under the EMA update the sequence `CAV^{(T)}` converges to `E[Shapley(u^F_t)]` (law of large numbers).

> **Verification note (addressed the reviewer's concern directly).** The characterization transfers cleanly to the *Myerson restricted-game construction* `u^F_t` under the standard hyperedge-chain notion of connectivity — this is precisely the setting of van den Nouweland–Borm–Tijs (1992). It does **not** transfer to the alternative "sum the Shapley formula only over connected coalitions" construction, which this proposal now explicitly avoids. The paper must therefore: (a) define feasibility connectivity as hyperedge-chain connectivity, (b) allocate via `Shapley(u^F_t)`, and (c) verify component-efficiency numerically on the actual lever hypergraph as a correctness test (a cheap check that catches implementation drift). If a future version wants to use a *non*-standard connectivity notion, the uniqueness claim must be re-verified against that specific definition — this is flagged, not assumed.

**Corollary (form / decomposition).** By additivity of Shapley, `CAV_i = φ^μ_i − κ·φ^σ²_i`. *(Proof: one line, as in Step 3.)*

**Proposition (A2 — Achievable Efficiency, now a *theorem*).** By Component Efficiency of the Myerson value applied to `u^F_t`, `Σ_{i∈C} CAV_i = u_t(C)` for every connected component `C` of `F`. In particular, `Σ_i CAV_i = u_t(A_reach) − u_t(∅)`, and any lever with zero feasible reach is a null player receiving zero credit. *(This is Component Efficiency of the Myerson value, not a new axiom.)*

**Proposition (A4 — Risk Sensitivity, now a *theorem*).** For fixed `v_t`, `∂CAV_i/∂κ = −φ^σ²_i`. Hence increasing `κ` strictly decreases the credit of every lever with positive marginal variance contribution, and does so *exactly in proportion to* `φ^σ²_i`. Ordering changes driven by `κ` are therefore precisely the risk-adjustment one would expect from a certainty-equivalent allocation. *(Direct from the Corollary.)*

**Proposition (A1 — Actionability Monotonicity, now a *theorem* with a stated condition).** Let levers `a, b` be feasible in exactly the same coalitions and satisfy the **marginal-dominance** condition `u_t(S∪{a}) − u_t(S) ≥ u_t(S∪{b}) − u_t(S)` for all feasible `S` excluding both, with comparable variance structure (equal `φ^σ²`). Then `CAV_a ≥ CAV_b`. *Proof: this is the strong-monotonicity property of the Shapley value (Young, 1985): if one player's marginal contributions weakly dominate another's in every coalition, the Shapley value preserves the ordering. It is a known theorem about Shapley, not a postulate we invent.* If the equal-variance condition is dropped, the statement must be weakened to a mean–variance dominance claim (`φ^μ_a − φ^μ_b ≥ κ(φ^σ²_a − φ^σ²_b)`) — the honest version, and the one the paper should state.

**A3 (Intervention / Temporal Consistency)** is retained as a *dynamic property* (the EMA recursion and its convergence), not an axiom of the allocation.

> **Why this survives review.** The revised spine is: (i) a new *game* (`u_t` = forward, mean–variance, feasibility-restricted), (ii) a standard-but-sound *allocation* (restricted Shapley) whose linear risk form is a corollary of additivity, and (iii) three *provable* properties (A1–A2–A4) that are theorems with stated conditions. The uniqueness claim is delegated to the classical, verifiable Myerson theorem instead of being asserted. If A1's dominance condition fails empirically, the paper degrades gracefully to the weakened mean–variance statement — the result is not lost, only its headline strength.

### 3.5.5 Actionable decision layer (minimal-action optimisation)

For user `u`, given per-lever costs `c_i` (effort/cost of realising lever `i`), a budget `B`, and a target uplift `Δ*`:

```
min_{S ⊆ A_feasible}  Σ_{i∈S} c_i
s.t.  Σ_{i∈S} c_i ≤ B   and   E[v_t(Rec(x^S))] − E[v_t(Rec(x^∅))] ≥ Δ*
```

**Algorithm (Shapley-guided greedy, risk-adjusted).** Sort levers by `CAV_i / c_i = (φ^μ_i − κ·φ^σ²_i) / c_i` (so the risk-adjusted value per unit cost); greedily add the best feasible lever until the budget is exhausted; if the uplift target is unmet, relax `Δ*` to the best achievable within budget. **Approximation guarantee (stated, not post-hoc).** Define the marginal-gain game `g_i(S) = u_t(S∪{i}) − u_t(S)`. **Proposition (Greedy guarantee).** If `g` is monotone and submodular over feasible coalitions, greedy value-per-cost selection is a `(1−1/e)`-approximation to the optimal minimal-action set under the budget. The submodularity is **checked a priori on the surrogate** (as a stated diagnostic, reported for the actual lever space — whether it holds or not), and the `(1−1/e)` claim is *conditional on that check*, never asserted unconditionally. If it fails, report the empirical greedy-vs-exhaustive gap on a small subset (n_u ≤ 12, B=2) as the honest bound instead. This is the *minimal recourse* answer: the smallest-cost set of actions that flips the recommendation / reaches the target utility.

### 3.5.6 Closed-loop training objective (learning objective)

Let `θ` be the recommender's parameters and `ψ` the dynamics-model parameters. Train jointly with:

```
min_{θ,ψ}  L = L_rank(θ) + λ_act · L_actcons(θ,ψ) + λ_div · L_div(θ)
               + λ_ctx · L_ctx(θ) + λ_reg · (‖θ‖² + ‖ψ‖²)
```

- `L_rank`: BPR pairwise ranking loss on realised next interactions (as in DyHuCoG).
- **`L_actcons` (action-consistency, the loop-closer):** the change in the model's ranking induced by the CAV-prescribed intervention plan must match the change predicted by the forward game:
  `L_actcons = E_{(u,t)} ‖ (Rank_u(x^{S*}) − Rank_u(x⁰)) − Δ̂^{S*} ‖²`
  where `Δ̂^{S*}` is the game-predicted utility change. This makes the model *action-aware*: what the model says will happen after an action is coherent with what CAV predicted.
- **Circularity control (the fix the review demanded).** As written, `L_actcons` trains `f_θ` toward a target `Δ̂^{S*}` that is produced by `P_ψ` *and* `P_ψ` is updated jointly — a moving target, and a route to "reward-hacking" one's own simulator (the model can shrink the loss by distorting the prediction, not by improving real actions). Three concrete controls, adopted by default:
  1. **Stop-gradient on the target.** `Δ̂^{S*}` is treated as a *fixed teacher*: `L_actcons` is optimised with respect to `θ` only, and `ψ` is updated on a *separate, realised-feedback objective* (`L_dyn`, next-state prediction error) — never on `L_actcons`. This breaks the self-confirming loop at the source.
  2. **Alternating / decoupled updates.** Update `P_ψ` (dynamics) and `f_θ` (recommender) in alternating phases, holding the other fixed; re-estimate `Δ̂` only between phases. This is the standard remedy for two-player moving-target objectives.
  3. **Calibration monitor.** Track the **IPS/DR-corrected predicted-vs-realised gap** (see §3.5.3-b) throughout training; if the gap widens while `L_actcons` shrinks, that is evidence of reward-hacking the simulator and must halt / reweight the consistency term.
  Present `L_actcons` as *action-aware regularisation*, not as the primary driver of `ψ`.
- `L_div`, `L_ctx`: diversity and context alignment (reuse DyHuCoG).
- **Uncertainty estimation:** `v^σ²_t` is estimated from an ensemble of `E` dynamics models (or MC dropout on `P_ψ`); the variance game's Shapley is computed on those draws; calibration is measured by expected calibration error (ECE) of future-utility quantiles.

### 3.5.7 Complexity and scalability
- **CAV allocation:** exact over the *restricted* feasible coalition space `O(|F|·2^{Δ})` where `Δ` is the maximum feasible-coalition size (feasibility hypergraph keeps this polynomial and small, as in MHyperShap); otherwise Monte-Carlo permutation sampling `O(M·|A|·T_eval)` with `O(1/M)` variance decay.
- **Forward value evaluation** `v_t(S)`: one roll-out of `H` steps through the lightweight dynamics model — cheap relative to model training.
- **Training profile:** `O((L+1)md) + O((M/f)·m·H·E)` per epoch (mirroring DyHuCoG Eq. 7.19, extended by horizon `H` and ensemble `E`), with periodic refresh and EMA smoothing.

### 3.5.8 Convergence guarantees
- MC Shapley concentration `O(1/M)` (as in DyHuCoG).
- EMA temporal-consistency convergence to the empirical expectation (law of large numbers) — provides A3.
- Greedy minimal-action approximation `(1−1/e)` under submodularity (proposition + empirical validation).
- Training convergence of the joint objective via Adam with early stopping (empirical, as in DyHuCoG).

### 3.5.9 Explainability properties
- **Efficiency/decomposition:** every point of future-utility uplift is attributed to exactly one lever — the "why" is *complete*.
- **Additivity across users:** because `v_t` is a per-user-mean, `CAV_i` decomposes into per-user CAVs for free (mirrors SignalShap's Proposition 3), enabling segment-level action plans.
- **Forward waterfall:** a CAV waterfall shows each lever's *expected future* contribution and its *risk* contribution — an explanation of the plan, not just the present.
- **Actionability by construction:** the Achievable-Efficiency property guarantees that only feasible, reachable levers carry credit, and Actionability Monotonicity ranks them by achievable reach — so the explanation *is* the action set.

---

## 3.6 Mathematical Formulation (consolidated equations)

**Coalition game (players = actionable levers, value = forward utility):**

```
(N = A,  v_t),   v_t(S) = E_{s_{1..H}}[ Σ_{τ=1}^{H} γ^{τ-1} R(u, Rec(x^S), s_τ) | s_0, do(S) ]
```

**Variance (uncertainty) game:**

```
v^σ²_t(S) = Var_{s_{1..H}}[ Σ_{τ=1}^{H} γ^{τ-1} R(u, Rec(x^S), s_τ) | s_0, do(S) ]
```

**Myerson-restricted (hypergraph) Shapley allocation.** First form the restricted game by summing over connected components (`comp_F(S)` = maximal connected components of `S` in feasibility structure `F`), then take the Shapley value of that restricted game:

```
v^F_t(S) = Σ_{C ∈ comp_F(S)} v_t(C)          (restricted mean game)
(v^σ²)^F_t(S) = Σ_{C ∈ comp_F(S)} v^σ²_t(C)  (restricted variance game)
φ^μ_i = Σ_{S⊆A\{i}} w_{|S|} [ v^F_t(S∪{i}) − v^F_t(S) ],   w_k = k!(|A|−k−1)!/|A|!
φ^σ²_i = Σ_{S⊆A\{i}} w_{|S|} [ (v^σ²)^F_t(S∪{i}) − (v^σ²)^F_t(S) ]
```

> This is the *Myerson restricted-game* construction (Step 1a in §3.5.4), **not** a sum over feasible coalitions. The distinction is load-bearing for Theorem 1: component-efficiency + fairness characterise the value of the *restricted game* `v^F_t`, and that restricted value is what the Shapley operator is applied to. Do **not** restrict the Shapley summation itself to connected coalitions — that alternative construction lacks the characterization.

**Cooperative Action Value (actionable utility / risk-adjusted payoff):**

```
CAV_i(t) = φ^μ_i(t) − κ·φ^σ²_i(t)
```

**Temporal / dynamic update:**

```
CAV_i^{(T)} = (1−λ)·CAV_i^{(T−1)} + λ·CAV_i(t_T)
```

**Actionable decision objective (minimal-action optimisation):**

```
S*(u,t) = argmin_{S⊆F(A)} Σ_{i∈S} c_i
          s.t. Σ_{i∈S} c_i ≤ B(u,t),   E[v_t(x^S)] − E[v_t(x^∅)] ≥ Δ*
```

**Shapley-guided greedy selection (surrogate, risk-adjusted):**

```
Greedy:  i* = argmax_{i∈F(A)\S, feasible}  (φ^μ_i − κ·φ^σ²_i) / c_i ,  repeat until budget exhausted
```

**Learning objective (with circularity control — §3.5.6):**

```
min_{θ}   L_θ = L_rank(θ) + λ_act·E_{(u,t)}‖ Rank_u(x^{S*}) − Rank_u(x⁰) − sg(Δ̂^{S*}) ‖²
                + λ_div·L_div(θ) + λ_ctx·L_ctx(θ) + λ_reg·‖θ‖²_F      (sg = stop-gradient)
min_{ψ}   L_ψ = L_dyn(ψ)   (realised next-state prediction only — ψ never sees L_θ)
```

**Uncertainty estimation:**

```
v^σ²_t(S) = (1/E) Σ_{e=1}^{E} ( g_e(x^S) − (1/E) Σ_e' g_e'(x^S) )² ,   g_e = discounted future util (ensemble e)
ECE = Σ_bins |acc_b − conf_b|/N_bins   (calibration of future-utility quantiles)
```

**Off-policy evaluation of the forward value (IPS / DR / SNIPS):**

```
w_t = 1 / π̂_0(a_t|s_t)                                          # inverse propensity weight (cap at W_max)
v̂_t^{IPS}(S) = (1/n) Σ_t [ r_t · w_t · 1{a_t ∈ do(S)} ]
v̂_t^{DR}(S)  = (1/n) Σ_t [ w_t (r_t − r̂(s_t,a_t)) + r̂(s_t,a_t) ] · 1{a_t ∈ do(S)}
ESS = (Σ_t w_t)² / Σ_t w_t²                                     # effective sample size of reweighting
gap(S) = | v̂_t^{DR}(S) − rollout(P_ψ, S) |                      # model-vs-OPE discrepancy (report, gate the plan)
```

- `π̂_0` is the **observation/selection propensity** for MNAR data (Schnabel-style naive-Bayes or logistic-regression estimator over user/item covariates — see §3.5.3-b), or the known logging policy on synthetic data; `r̂` is the model-based outcome from `P_ψ`.
- **DR is doubly robust:** unbiased if the propensity model *or* the outcome model is correct. Report `ESS` and the discrepancy `gap(S)`; a plan is only claimed to "work" when its DR-corrected lift is positive and its naive-model number is within the discrepancy of it — never on the model roll-out alone.

**Intervention mechanism (how an action is applied):** each lever `i` carries a declared feasible target set `T_i` and cost `c_i`; `do(S)` sets `x_i ∈ T_i` for `i∈S` at the input/profile level of a *history-conditioned* recommender (so `Rec(x^S)` is recomputed at inference without retraining, per the ActionShap masking-sensitivity gate).

---

## 3.7 Algorithm (pseudocode)

```
Algorithm CAVI-Recommend
Input: history-conditioned recommender f_θ, dynamics model P_ψ, user u, time t,
       lever set A, feasibility hypergraph F, costs c, budget B, target Δ*, horizon H,
       discount γ, risk aversion κ, MC budget M, refresh period f, EMA λ
Output: ranked list Rec_u, Cooperative Action Values {CAV_i}, intervention plan S*

1  for each refresh step:
2     sample M coalitions (permutation walks) over the full player set A
3     for each sampled S:
4        x^S ← apply do(S) to status-quo lever state x⁰
5        v_t(S)  ← rollout(P_ψ, H, γ, R, Rec(x^S))          # forward utility on S (not yet restricted)
6        v^σ²_t(S) ← ensemble-variance(rollout over E copies of P_ψ)
7     # Myerson restriction: v^F_t(S) ← Σ over connected components comp_F(S)   (Step 1a)
8     φ^μ_i  ← MC-Shapley(v^F_t)                             # eq. restricted Shapley on restricted game
9     φ^σ²_i ← MC-Shapley((v^σ²)^F_t)
10    CAV_i(t) ← φ^μ_i − κ·φ^σ²_i                             # eq. CAV
11    CAV_i^{(T)} ← (1−λ)·CAV_i^{(T−1)} + λ·CAV_i(t)          # eq. temporal EMA
11
12 # --- Actionable decision layer ---
13 S* ← ∅
14 while cost(S*) ≤ B and remaining budget > 0:
15    i* ← argmax_{i ∈ F(A)\S*, feasible within remaining budget}  CAV_i / c_i
16    if none feasible: break
17    S* ← S* ∪ {i*}
18 Δ̂ ← E[v_t(x^{S*})] − E[v_t(x^∅)]
19 if Δ̂ < Δ*: relax Δ* to Δ̂ (report shortfall)             # (1−1/e) approx if submodular
20
21 # --- Closed-loop training (decoupled, stop-gradient on the CAV target) ---
22 update θ on minibatch with L_θ = L_rank + λ_act·L_actcons(sg(Δ̂)) + λ_div·L_div + λ_ctx·L_ctx + λ_reg·L_reg
23 update ψ on realised next-state prediction L_dyn only   # ψ never trained on L_actcons (no reward-hacking)
24
25 return Rec_u = top-K of f_θ(x^{S*}), {CAV_i}, S*
```

---

## 3.8 Experimental Design

**Datasets (all with timestamps and, ideally, item metadata for levers):**
- **MovieLens-1M** (dense, explicit, timestamps, genres) — primary.
- **Amazon-Book** rebuilt from raw Amazon Reviews 2018 Books corpus (timestamps + metadata + sequential order; per SignalShap §4.1 the canonical split lacks timestamps/metadata and must be rebuilt) — sparse, long-tail.
- **Yelp2018** (auxiliary robustness, already used by DyHuCoG).
- **LastFM-2K or Steam** (optional third, for generalisation).

**Splits:** temporal, user-level; last interaction = test, second-last = validation; 5-core filtering; leakage-safe (levers never include test interactions).

**Baselines:**
- *Recommenders (accuracy):* MF, NCF, LightGCN, SASRec, BERT4Rec, HCCF, HPCF, RecDCL, DyHuCoG.
- *Counterfactual / recourse:* RecRec, CEERS, CARMA-style (feature-level recourse), conditional counterfactual generation.
- *Shapley / cooperative:* backward-Shapley attribution (DyHuCoG-style), Data-Shapley, SignalShap (source attribution), ActionShap (intervention-grounded *evaluation* — note: CAVI is generative, ActionShap evaluative; use its AIA metric as a downstream check), FairShap (fairness).
- *Uncertainty-aware:* an MC-dropout / ensemble recommender with deboosting.

**Metrics:**
- *Accuracy:* NDCG@K, Recall@K, MRR, candidate recall.
- *Actionability / recourse:* intervention success rate (plan reaches target), **minimal-action cost** (mean |S*| and mean Σc_i), predicted-vs-realised future-utility correlation (Spearman between game-predicted `Δ̂^{S*}` and realised lift), recourse validity & stability across seeds and small history perturbations, decision regret (gap to oracle best feasible action).
- *Forward utility:* mean future NDCG/engagement over horizon `H`; long-term lift.
- *Uncertainty:* ECE of future-utility quantiles; coverage.
- *Diversity/fairness (secondary):* ILD, coverage, Gini of exposure (to show CAVI can also steer exposure).
- *AIA* (from ActionShap) as a cross-check that CAV predicts single-intervention effects too.

**Ablations (component-wise, mirroring DyHuCoG):** w/o forward value (use backward `v_t(S)=NDCG@K(S)` only), w/o risk adjustment (κ=0), w/o feasibility restriction (flat power set), w/o action-consistency loss (no loop closure), w/o minimal-action optimisation (full set / no budget), w/o dynamics model (greedy on immediate utility).

**Statistical validation:** user-level paired tests, Wilcoxon signed-rank + paired permutation, Holm–Bonferroni across the method family, effect sizes (Cohen's d_z), 5 seeds, bootstrap CIs. Statistical unit = user.

**Robustness analysis:** κ sweep; horizon H; budget B and cost model c; Δ* target; dynamics-model ensemble size E; lever-set construction sensitivity; history truncation window; the greedy-vs-exhaustive gap validated on a small subset (n_u ≤ 12, B=2).

**Scalability analysis:** runtime/memory vs. M, |A|, H, E; Myerson restriction vs. 2^|A|; per-user wall-clock; report scaling curves (mirror DyHuCoG Table 4/6 style).

**Explainability evaluation:** CAV waterfall sanity; **faithfulness** (deletion/comprehensiveness on the forward utility); **AIA** (attribution predicts intervention effect); minimal-recourse set-cover validity; qualitative case studies; optional small user study of comprehension/trust (clearly scoped, not load-bearing).

**Actionable-recommendation evaluation:** the paper's headline — does following the CAV plan improve realised future utility and change the recommendation as predicted? Measure realised lift vs. baselines' plans on held-out interaction streams, evaluated with the **IPS/DR-corrected estimators of §3.5.3-b/§3.6** (with `ESS` and model-vs-OPE discrepancy reported), *not* by naive model roll-out. Report the naive-model number alongside so the optimism of the model-based estimate is visible. A negative DR-corrected result is reportable and should not be papered over with the optimistic roll-out.

---

## 3.9 Expected Contributions

**Theoretical.** (1) The **forward cooperative game over actionable levers** — the first recommender cooperative game whose characteristic function is expected discounted *future* utility, with an explicit off-policy (IPS/DR) evaluation of that value. (2) **Cooperative Action Values (CAV)** — the Myerson-restricted Shapley value of the forward mean–variance certainty-equivalent game, whose linear risk form is a corollary of additivity and whose key properties (Achievable Efficiency, Risk Sensitivity, Actionability Monotonicity) are proven theorems with stated conditions, not asserted axioms. (3) A **minimal-action coalition-optimisation** formulation with a `(1−1/e)` approximation guarantee under a *stated and empirically verified* submodularity condition. (4) A **closed-loop, action-consistency learning objective** with explicit circularity control (stop-gradient, decoupled updates, calibration monitor) making the recommender action-aware without reward-hacking its own simulator.

**Algorithmic.** A modular, reproducible pipeline (action-game → CAV allocation → minimal-action planner → action-aware DyHuCoG backbone → closed-loop trainer), with uncertainty estimation via ensembles.

**Practical.** Decision-oriented intelligence for: end users (what to do next, minimal recourse), platform designers (which levers to strengthen / which behaviours degrade quality / which items maximise future utility), and regulators (transparent, justified, feasibility-constrained intervention plans — aligned with EU AI Act transparency/oversight).

---

## 3.10 Publication Assessment

| Criterion | Assessment |
|---|---|
| **Originality** | High — no prior work defines a *forward-looking, uncertainty-adjusted, feasibility-restricted Shapley allocation over action spaces* that simultaneously explains, generates minimal recourse, and closes the training loop. Distinct from the author's five drafts and from the literature (backward Shapley, feature-level recourse, RL, uncertainty-only). |
| **Novelty** | High — new *forward game* + new allocation (CAV) with provable properties + new optimisation (minimal-action coalition) + new learning objective (action-consistency). Not an incremental re-application. |
| **Mathematical depth** | Strong — uniqueness theorem, submodular greedy guarantee, MC concentration, EMA convergence; sufficient for TKDE / TOIS / Information Fusion / TNNLS. |
| **Engineering contribution** | Medium-high — builds directly on the validated DyHuCoG backbone, ActionShap intervention harness, and SignalShap data pipeline; feasible on one RTX 4090; reuse of 4 datasets and the thesis's statistical protocol keeps compute bounded. |
| **Feasibility** | High — all components are re-usable from the author's own code; the only new build is the dynamics model (a standard sequential model) and the forward-value roll-out, both cheap. |
| **Fit with thesis** | Direct — it is the thesis's *unmeasured* central claim ("actionable insight," Definition 1.1) operationalised and made generative; a natural capstone successor to DyHuCoG (in-training Shapley → forward Shapley). |

**Novelty score: 8 / 10 (revised down from 9 after the review).**
- +3 for the *forward-looking* value function (truly new; nothing in the author's line or the literature evaluates coalitions on expected future utility).
- +1 (was +2) for the CAV allocation. The corrected construction (§3.5.4) is a **new game** (`u_t` = forward, mean–variance, feasibility-restricted) allocated by *standard* restricted Shapley — sound and rigorous, but the "new axiom + uniqueness" claim was an over-claim and the honest version delegates uniqueness to the classical Myerson theorem. That downgrades the theoretical-novelty component.
- +2 for the minimal-action, budget-constrained recourse generation fused with cooperative attribution (unique).
- +2 for loop-closure into training via an action-consistency objective.
- −1 reuses the Shapley operator as the substrate (unavoidable) and reuses DyHuCoG-style architecture and datasets.
- −1 net for the two de-risking gaps that are now first-class work items: the off-policy-evaluation machinery (§3.5.3-b) and the circularity control (§3.5.6) were absent from the first draft and must be demonstrated, not asserted.

**Venue-risk reality check (point #4 of the review).** SignalShap targets *Discover AI* (Springer, open access); prior work sits in IJACSA/IJIES/Procedia. A leap straight to IEEE TKDE / TOIS / Information Fusion / TNNLS is a real jump in reviewer expectations on exactly the two weak points above — theoretical rigor (now shored up by the Myerson-based construction) and experimental/OPE rigor (now explicit). The venue should follow the *demonstrated* result, not be pre-assumed: the theory-first sequencing below decides it.

**Why publishable:** it delivers a *new cooperative-game formulation* (not a re-application), contributes *new theory* (axioms + uniqueness + approximation guarantees), is *actionability-first* (generates interventions, not just explanations), is *mathematically rigorous* (convergence, submodularity, calibration), is *experimentally feasible* on the author's existing stack, and lands in the high-citation intersection of game theory + trustworthy/actionable recommendation + uncertainty + recourse — a profile that maps cleanly onto **IEEE TKDE, ACM TOIS, Information Fusion**, or **IEEE TNNLS**.

---

## 3.11 Explicit Differentiation from Existing Shapley-Based Recommendation Literature

| Prior work | Unit of attribution | Direction of value | Generative interventions? | Uncertainty? | New theory? | CAVI differentiator |
|---|---|---|---|---|---|---|
| **Shapley mediators / SHAP-style** | features | backward | no | no | no | CAVI is over *actionable levers*, forward utility, generative, risk-adjusted |
| **Shapley explanations (post-hoc)** | features/items/paths | backward | no | no | no | CAVI explains the *plan*, and its explanation *is* the action set |
| **Data-Shapley / data valuation** | data points | backward | no | no | no | CAVI values *actions*, not data; forward-looking |
| **DyHuCoG (thesis)** | users/items/contexts | backward (immediate NDCG/diversity) | no | no | no | CAVI generalises to forward discounted utility + risk + minimal action + loop-closure |
| **SignalShap** | signal sources | backward | yes (fusion only) | no | light | CAVI is user-level actionable levers + prospective value, not system-level source credit |
| **ActionShap (evaluation)** | interaction factors | backward | *evaluates* others' plans | no | light | CAVI *defines* a new forward allocation and *generates* the plan; ActionShap is a downstream check |
| **FairShap** | items/providers (exposure) | backward | yes (exposure) | no | light | CAVI is not fairness-specific; fairness is one term in its forward reward |
| **MHyperShap** | LLM agents | backward (pipeline outcome) | yes (routing) | no | strong (Myerson uniqueness) | CAVI is over *actions* in a recommender with *future* utility + risk; agent-routing is a different domain |
| **Coopetitive/bandit creator games** | creators/agents | backward (regret) | no | no | moderate | CAVI is a recommender-facing forward action value, not an incentive-sharing scheme |
| **Counterfactual recourse (RecRec/CEERS/CARMA)** | features (flips) | backward (fixed classifier) | yes (single feature flips) | partial | no | CAVI adds cooperative interaction-aware allocation, forward utility, budget, and training-loop closure |

**One-line positioning:** *Backward Shapley explains the present; counterfactual recourse flips features of a frozen classifier; CAVI is the first framework whose cooperative allocation is computed on the future and whose allocation itself is the recommendation and the intervention plan.*

---

## 3.12 Risks & Mitigations (for the proposer)

- **"This is just Shapley with a new value function."** — Countered by the *new forward game* (mean–variance certainty-equivalent over actionable levers), the CAV allocation with provable properties (A1/A2/A4), the minimal-action optimisation, and the action-consistency loop. The paper's theoretical core is the CAV construction and its guarantees — and crucially, the "new value function" *is* the point, because the value function (forward, risk-adjusted, feasibility-restricted) is where all the novelty lives.
- **"Actionability is subjective / costs are arbitrary."** — Mitigate exactly as ActionShap planned: pre-registered, pre-attribution-frozen lever/cost tables (Appendix C style), sensitivity sweeps over c and B, and external domain annotation where feasible.
- **"Forward utility requires a simulator."** — Use a learned dynamics model validated for calibration (ECE), report robustness to its accuracy, *and* evaluate the plan's realised lift on held-out interaction streams with the **IPS/DR-corrected estimators of §3.5.3-b** — never on the naive model roll-out alone.
- **"The plan is off-policy; your realised lift is optimistic."** — Addressed head-on in §3.5.3-b / §3.6: propensity estimation, IPS/DR/SNIPS estimators, `ESS`, and the model-vs-OPE discrepancy `gap(S)` as an explicit gate on the headline claim. A plan is only "shown to work" when its DR-corrected lift is positive.
- **"Your closed loop is circular."** — Mitigated in §3.5.6: stop-gradient on the CAV target, alternating/decoupled `ψ`/`θ` updates, and a calibration monitor that flags reward-hacking against the simulator.
- **"This is 2–3 papers, not one."** — Accepted; see §3.13 (two-paper strategy).
- **"Overlap with MHyperShap's Myerson restriction."** — MHyperShap restricts *agent* coalitions; CAVI restricts *action* coalitions and adds forward value + risk + minimal recourse. State this contrast explicitly.
- **"Overlap with ActionShap."** — ActionShap is explicitly *evaluative* (does attribution predict intervention effect?); CAVI is *generative and theoretical* (a new allocation that produces the plan). Cite and use ActionShap's AIA metric as a cross-check.

---

## 3.13 Publication & Sequencing Strategy — Split, don't bundle (point #3 of the review)

The review is right: bundling the forward game, the new allocation, minimal-action recourse, a learned dynamics model, OPE, and a closed-loop objective into one mega-paper multiplies the chance a reviewer sinks the whole thing on the weakest link (most likely OPE or the submodularity assumption). Recommended structure — **two papers, sequenced theory-first**, with the gate as the go/no-go between them.

**Paper A — "Cooperative Action Values: A Forward Cooperative-Game Theory of Actionable Recommendation"** (theory-led)
- **Scope:** the forward certainty-equivalent game `u_t` (§3.5.3–3.5.4), the CAV allocation, and the *provable* properties (Theorem 1 = Myerson-based uniqueness; Corollary = linear risk form; Propositions A2, A4, A1 with stated conditions). Plus **A3 temporal consistency**.
- **Experiments (lean):** synthetic games with known ground-truth CAVs (verify the allocation recovers them, validate A2/A4); **one** real dataset (MovieLens-1M) for the *CAVI gate* — does the forward CAV ordering actually diverge from backward Shapley, and is the divergence explained by feasibility / interaction / variance (mirroring SignalShap's redundancy insight)? No closed-loop training, no full baseline suite.
- **Deliverable / venue:** this is the paper that earns the theorem. If the gate shows divergence and the properties hold, it is TKDE/TOIS-grade and *should* be submitted there; if it does not, the framing must be re-scoped before anything is built.

**Paper B — "From Cooperative Attribution to Action: Budget-Constrained Minimal-Action Recourse and Action-Aware Training"** (systems-led)
- **Scope:** the minimal-action planner (§3.5.5), the closed-loop action-consistency objective with circularity control (§3.5.6), OPE-corrected evaluation (§3.5.3-b/§3.6), and the full experimental suite (§3.8: 2–3 datasets, recommenders + recourse + cooperative baselines, ablations, robustness, scalability).
- **Builds on:** Paper A's CAV as the substrate; reuses DyHuCoG backbone, ActionShap intervention harness, SignalShap data pipeline.
- **Deliverable / venue:** a strong applied-journal / top-conference paper (TOIS, TKDE, or RecSys/WWW as a fallback), carrying the engineering + human-relevant contribution.

**Why this de-risks everything:**
1. **The theorem is isolated.** The single highest-risk, highest-consequence item is verified on paper (cheap, no GPU) and in Paper A before any systems build. If it doesn't hold, only Paper A's framing is lost — Paper B still stands.
2. **OPE and circularity are exercised in Paper B**, where they belong, with full baselines to contextualise them — not hidden in a footnote of a mega-paper.
3. **Two publications instead of one high-variance one.** Each paper has a crisp, defensible novelty claim.
4. **Venue follows evidence.** Paper A decides whether the target is TKDE/TOIS (theory holds + gate diverges) or a strong applied venue (if not); Paper B's venue is decided by its system results. This answers point #4 of the review directly.

**Paper B is still large — a scoping fallback if it balloons.** The reviewer is right that Paper B bundles OPE + circularity control + closed-loop training + a ~15-method baseline suite + robustness + scalability. It is far more defensible than the original mega-paper, but if it still exceeds the target venue's scope, apply this cut in order (do not cut OPE or the DR-corrected gate, which are the claim's integrity):
  1. **Demote closed-loop training (§3.5.6) to a standalone Paper C** (a focused "action-aware training" contribution) and keep Paper B = minimal-action recourse + OPE-corrected evaluation. This is the cleanest split because closed-loop training is the most independent piece and the one most likely to need its own careful convergence study.
  2. If Paper B is still heavy, **trim the baseline family** (drop the weakest recommenders, keep MF/LightGCN/DyHuCoG + 2–3 recourse/cooperative methods) and report the rest as an appendix.
  3. Always keep: the CAV planner, the DR-corrected "does the plan work" gate, and the synthetic positive-control (known-propensity) validation — these are non-negotiable for the central claim.

**Non-negotiable gate (before any implementation), mirroring the ActionShap masking-sensitivity gate:** on MovieLens-1M, compute (a) backward-Shapley orderings over the same lever space, and (b) forward CAV orderings under a *simple* learned dynamics model, and test whether the two orderings diverge significantly (and whether the divergence is attributable to feasibility/interaction/variance). If the gate fails, do **not** build Paper B — re-scope to Paper A's theory alone or abandon the forward direction. This gate is a half-day experiment and decides the programme.

---

*Prepared from: `phd-thesis/MOUAD_LOUHICHI_Thesis.pdf`, `previous-papers/*`, and `paper-ideas/{ActionShap,SignalShap,FairShap,MHyperShap}/*`. Revised after external review, round 1: (1) the uniqueness theorem was reframed from an over-claimed "new axioms" assertion to the Myerson-based construction with provable A1/A2/A4 properties; (2) off-policy evaluation (IPS/DR/SNIPS + `ESS` + discrepancy gate) was added as a first-class component; (3) the closed-loop circularity was addressed with stop-gradient, decoupled updates, and a calibration monitor; (4) the work is restructured into a two-paper, theory-first programme with a non-negotiable CAVI gate. Round 2 (this revision): (a) the theorem was re-grounded on the **Myerson restricted-game construction** `u^F_t = Σ over connected components` (van den Nouweland–Borm–Tijs 1992), explicitly distinguished from the "sum Shapley only over connected coalitions" construction that does *not* carry the component-efficiency/fairness characterization — with a numerical component-efficiency verification mandated as a correctness test; (b) the OPE propensity was specialised to **MNAR observational data via Schnabel-style naive-Bayes / logistic observation-propensity estimators**, with the known-propensity synthetic case as a positive control; (c) a Paper B scoping fallback was added (closed-loop training as a separable Paper C). Recommended next step: **run the CAVI gate** — the half-day forward-vs-backward divergence experiment on MovieLens-1M — before any further build.*
