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

**Missing (the opportunity):** a *single unified, forward-looking, uncertainty-aware cooperative-game framework* that (i) defines a new allocation over **actionable levers** under a **discounted expected-future-utility** characteristic function, (ii) **generates** both recommendations and **budget-constrained minimal interventions** from one cooperative object, (iii) is **risk/uncertainty-adjusted**, (iv) **closes the loop** into model training so the recommender becomes *action-aware*, and (v) carries **new axioms** (not just new applications). No existing paper — thesis, drafts, or literature — occupies this intersection.

## 2.3 Where the strongest publication opportunity is
**Q1 journals (IEEE TKDE / ACM TOIS / Information Fusion / IEEE TNNLS)** reward *new theory* + *feasible, rigorous experiments* + *a clear decision-oriented contribution*. The forward-looking cooperative-action framework below is exactly that: new axioms, a new allocation, a rigorous optimisation layer, uncertainty coupling, and a closed training loop — while reusing the author's validated DyHuCoG backbone, datasets, and protocol.

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

4. **No new theory — only new applications.** The existing drafts extend *where* Shapley is applied (sources, interactions, fairness, agents) but reuse the same allocation. A Q1 contribution should define a *new allocation* with *new axioms*, not merely apply Shapley to a new player set.

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
 │      Greedy on φ^μ_i / c_i  (Shapley-guided, (1−1/e) approx if submodular)│
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

### 3.5.4 Payoff allocation — Cooperative Action Value

Let `v^σ²_t(S) = Var[ Σ_{τ=1}^{H} γ^{τ-1} R(u, Rec(x^S), s_{t+τ}) | s_t, do(S) ]` be the *variance game* (risk of a coalition of actions). Define the two component Shapley values **restricted to feasible coalitions** (Myerson restriction over `F`):

```
φ^μ_i  = Shapley_{F}(v_t)_i          (expected-future-utility credit of lever i)
φ^σ²_i = Shapley_{F}(v^σ²_t)_i       (marginal risk contribution of lever i)
```

**Cooperative Action Value (certainty-equivalent allocation):**

```
CAV_i(t) = φ^μ_i(t) − κ · φ^σ²_i(t)
```

where `κ ≥ 0` is the risk-aversion coefficient. When `κ=0`, CAV reduces to the (forward) Shapley value; increasing `κ` makes the allocation risk-averse, so high-variance levers are penalised.

**New axioms (the theory).** In addition to the classical four (efficiency, symmetry, null-player, additivity — each applied component-wise to `v_t` and `v^σ²_t`), CAV is defined to satisfy:

- **A1 Actionability Monotonicity.** If lever `a` has weakly greater *feasible marginal reach* than `b` over every feasible coalition (its maximal achievable effect within its feasibility interval dominates), then `CAV_a ≥ CAV_b`. Credit tracks *achievable* change, not raw counterfactual effect.
- **A2 Achievable Efficiency.** The allocation sums to the *reachable* future-utility uplift: `Σ_i CAV_i = v_t(A_reach) − v_t(∅)`, where `A_reach` is the maximal feasible coalition; levers with zero feasible reach are null players.
- **A3 Intervention / Temporal Consistency.** Across the interaction stream, the CAV sequence satisfies an EMA recursion `CAV^{(T)} = (1−λ)CAV^{(T−1)} + λ·CAV(t_T)`, and its limit satisfies A1–A2 with respect to the empirical state distribution.
- **A4 Risk Sensitivity.** A lever with non-negligible marginal variance contribution is penalised: `CAV_i` is decreasing in `φ^σ²_i` at rate `κ`. (This is a *certainty-equivalent* allocation in the mean–variance sense.)

**Theorem (CAV well-posedness / uniqueness, statement).** Under acyclic `F` and the EMA update, the allocation satisfying efficiency, symmetry, null-player, additivity, A1, A2 (with respect to the reachable game), and A4 is unique and equals `CAV_i = Shapley_F(v_t)_i − κ·Shapley_F(v^σ²_t)_i`. *(Existence: component Shapley values satisfy the classical axioms; risk-adjustment subtracts the variance Shapley. Uniqueness: classical Shapley uniqueness on the restricted game (Myerson 1977, extended to hypergraphs as in MHyperShap's Thm. 1) applied component-wise, with `κ` fixing the risk axis. Full proof in the paper's appendix.)*

### 3.5.5 Actionable decision layer (minimal-action optimisation)

For user `u`, given per-lever costs `c_i` (effort/cost of realising lever `i`), a budget `B`, and a target uplift `Δ*`:

```
min_{S ⊆ A_feasible}  Σ_{i∈S} c_i
s.t.  Σ_{i∈S} c_i ≤ B   and   E[v_t(Rec(x^S))] − E[v_t(Rec(x^∅))] ≥ Δ*
```

**Algorithm (Shapley-guided greedy).** Sort levers by `φ^μ_i / c_i`; greedily add the best value-per-cost feasible lever until the budget is exhausted; if the uplift target is unmet, relax `Δ*` to the best achievable within budget. **Approximation guarantee.** If the forward value function is submodular and monotone over feasible coalitions (a condition we verify empirically on the surrogate dynamics model and state as a proposition), the greedy policy is a `(1−1/e)`-approximation to the optimal minimal-action set. This is the *minimal recourse* answer: the smallest-cost set of actions that flips the recommendation / reaches the target utility.

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
- **Actionability by construction:** A2 (achievable efficiency) guarantees that only feasible, reachable levers carry credit, and A1 ranks them by achievable reach — so the explanation *is* the action set.

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

**Enhanced (Myerson-restricted) Shapley allocation:**

```
φ^μ_i = Σ_{S⊆F(A)\{i}} w_{|S|} [ v_t(S∪{i}) − v_t(S) ],   w_k = k!(|A|−k−1)!/|A|!
φ^σ²_i = Σ_{S⊆F(A)\{i}} w_{|S|} [ v^σ²_t(S∪{i}) − v^σ²_t(S) ]
```

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

**Shapley-guided greedy selection (surrogate):**

```
Greedy:  i* = argmax_{i∈F(A)\S, feasible}  φ^μ_i / c_i ,  repeat until budget exhausted
```

**Learning objective:**

```
min_{θ,ψ} L = L_rank(θ) + λ_act·E_{(u,t)}‖ Rank_u(x^{S*}) − Rank_u(x⁰) − Δ̂^{S*} ‖²
              + λ_div·L_div(θ) + λ_ctx·L_ctx(θ) + λ_reg·(‖θ‖²_F + ‖ψ‖²_F)
```

**Uncertainty estimation:**

```
v^σ²_t(S) = (1/E) Σ_{e=1}^{E} ( g_e(x^S) − (1/E) Σ_e' g_e'(x^S) )² ,   g_e = discounted future util (ensemble e)
ECE = Σ_bins |acc_b − conf_b|/N_bins   (calibration of future-utility quantiles)
```

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
2     sample M coalitions (permutation walks) over feasible F
3     for each sampled S:
4        x^S ← apply do(S) to status-quo lever state x⁰
5        v_t(S)  ← rollout(P_ψ, H, γ, R, Rec(x^S))          # forward utility
6        v^σ²_t(S) ← ensemble-variance(rollout over E copies of P_ψ)
7     φ^μ_i  ← MC-Shapley(v_t)                                # eq. restricted Shapley
8     φ^σ²_i ← MC-Shapley(v^σ²_t)
9     CAV_i(t) ← φ^μ_i − κ·φ^σ²_i                             # eq. CAV
10    CAV_i^{(T)} ← (1−λ)·CAV_i^{(T−1)} + λ·CAV_i(t)          # eq. temporal EMA
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
21 # --- Closed-loop training ---
22 update θ,ψ on minibatch with L = L_rank + λ_act·L_actcons + λ_div·L_div + λ_ctx·L_ctx + λ_reg·L_reg
23
24 return Rec_u = top-K of f_θ(x^{S*}), {CAV_i}, S*
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

**Actionable-recommendation evaluation:** the paper's headline — does following the CAV plan improve realised future utility and change the recommendation as predicted? Measure realised lift vs. baselines' plans under the *same* simulator/dynamics model *and* on held-out interaction streams.

---

## 3.9 Expected Contributions

**Theoretical.** (1) The **forward cooperative game over actionable levers** — the first recommender cooperative game whose characteristic function is expected discounted *future* utility. (2) **Cooperative Action Values (CAV)**, a new risk-adjusted, feasibility-restricted, forward-looking allocation with new axioms (Actionability Monotonicity, Achievable Efficiency, Intervention/Temporal Consistency, Risk Sensitivity) and a uniqueness theorem. (3) A **minimal-action coalition-optimisation** formulation with a `(1−1/e)` approximation guarantee under submodularity. (4) A **closed-loop, action-consistency learning objective** making the recommender action-aware.

**Algorithmic.** A modular, reproducible pipeline (action-game → CAV allocation → minimal-action planner → action-aware DyHuCoG backbone → closed-loop trainer), with uncertainty estimation via ensembles.

**Practical.** Decision-oriented intelligence for: end users (what to do next, minimal recourse), platform designers (which levers to strengthen / which behaviours degrade quality / which items maximise future utility), and regulators (transparent, justified, feasibility-constrained intervention plans — aligned with EU AI Act transparency/oversight).

---

## 3.10 Publication Assessment

| Criterion | Assessment |
|---|---|
| **Originality** | High — no prior work defines a *forward-looking, uncertainty-adjusted, feasibility-restricted Shapley allocation over action spaces* that simultaneously explains, generates minimal recourse, and closes the training loop. Distinct from the author's five drafts and from the literature (backward Shapley, feature-level recourse, RL, uncertainty-only). |
| **Novelty** | High — new axioms + new allocation (CAV) + new optimisation (minimal-action coalition) + new learning objective (action-consistency). Not an incremental re-application. |
| **Mathematical depth** | Strong — uniqueness theorem, submodular greedy guarantee, MC concentration, EMA convergence; sufficient for TKDE / TOIS / Information Fusion / TNNLS. |
| **Engineering contribution** | Medium-high — builds directly on the validated DyHuCoG backbone, ActionShap intervention harness, and SignalShap data pipeline; feasible on one RTX 4090; reuse of 4 datasets and the thesis's statistical protocol keeps compute bounded. |
| **Feasibility** | High — all components are re-usable from the author's own code; the only new build is the dynamics model (a standard sequential model) and the forward-value roll-out, both cheap. |
| **Fit with thesis** | Direct — it is the thesis's *unmeasured* central claim ("actionable insight," Definition 1.1) operationalised and made generative; a natural capstone successor to DyHuCoG (in-training Shapley → forward Shapley). |

**Novelty score: 9 / 10.**
- +3 for the *forward-looking* value function (truly new; nothing in the author's line or the literature evaluates coalitions on expected future utility).
- +2 for the new CAV allocation + axioms + uniqueness theorem (new theory, not new application).
- +2 for the minimal-action, budget-constrained recourse generation fused with cooperative attribution (unique).
- +2 for loop-closure into training via an action-consistency objective.
- −1 because it reuses the Shapley operator as the substrate (unavoidable — the request is explicitly a *unified Shapley-based* formulation) and reuses DyHuCoG-style architecture and datasets.

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

- **"This is just Shapley with a new value function."** — Countered by the new axioms + uniqueness theorem + minimal-action optimisation + action-consistency loop; the paper's theoretical core is the CAV allocation and its guarantees, not the operator itself.
- **"Actionability is subjective / costs are arbitrary."** — Mitigate exactly as ActionShap planned: pre-registered, pre-attribution-frozen lever/cost tables (Appendix C style), sensitivity sweeps over c and B, and external domain annotation where feasible.
- **"Forward utility requires a simulator."** — Use a learned dynamics model validated for calibration (ECE) and report robustness to its accuracy; also evaluate the plan's *realised* lift on held-out interaction streams.
- **"Overlap with MHyperShap's Myerson restriction."** — MHyperShap restricts *agent* coalitions; CAVI restricts *action* coalitions and adds forward value + risk + minimal recourse. State this contrast explicitly.
- **"Overlap with ActionShap."** — ActionShap is explicitly *evaluative* (does attribution predict intervention effect?); CAVI is *generative and theoretical* (a new allocation that produces the plan). Cite and use ActionShap's AIA metric as a cross-check.

---

*Prepared from: `phd-thesis/MOUAD_LOUHICHI_Thesis.pdf`, `previous-papers/*`, and `paper-ideas/{ActionShap,SignalShap,FairShap,MHyperShap}/*`. Recommended next step: run the CAVI gate (a mini experiment computing forward vs. backward CAV ordering divergence on MovieLens-1M) before full implementation, mirroring the masking-sensitivity gate the ActionShap spec already mandates.*
