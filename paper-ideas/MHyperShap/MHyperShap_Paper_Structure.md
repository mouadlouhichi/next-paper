# MHyperShap — Full Paper Structure, TOC & Embedded Content

**Target journal:** *Discover Artificial Intelligence* (Springer Nature, open access, Q1 — Information Systems)
**Article type:** Research article
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco
**Corresponding author:** mouad_louhichi@um5.ac.ma

> Working blueprint **with the MHyperShap technical content embedded and refactored** for an academic venue. The original proposal's vendor-specific framing (Salesforce/Agentforce/Atlas/MuleSoft/Apex) has been generalized to domain-neutral enterprise-agent terminology, and the meta sections (publication strategy, risk assessment, scorecard) have been moved out of the manuscript and kept only as planning notes at the end. Structure mirrors DyHuCoG's proven format. Target ≈ 9,000–11,000 words, 8–10 figures, 8–10 tables.

---

## Working Title (primary + alternates)
- **Primary:** *MHyperShap: Myerson-Restricted Dynamic Hypergraph Cooperative Games for Credit Assignment and Routing in Multi-Agent LLM Systems*
- Alt 1: *Beyond Flat Shapley: Axiomatic Credit Assignment for Structured Multi-Agent LLM Pipelines*
- Alt 2: *Cooperative-Game-Theoretic Attribution and Routing for Agentic LLM Systems via Dynamic Hypergraph Games*

## One-paragraph thesis (the spine)
Agentic LLM pipelines orchestrate heterogeneous agents constrained by a task-dependency DAG, yet existing Shapley-based credit assignment treats agents as a flat, exchangeable coalition — axiomatically incorrect under structural constraints, producing misleading attributions. We model the pipeline as a **Myerson-restricted dynamic hypergraph cooperative game (MHyperShap)**, prove its allocation is the *unique* one satisfying Component Efficiency, Hyperedge Fairness, and Temporal Consistency, evaluate it on a new analytic-ground-truth benchmark (**SynAgentBench**) plus real agent benchmarks, and **close the attribution→routing loop** so the signal that explains a pipeline also improves it.

---

# TABLE OF CONTENTS
```
Abstract / Keywords
1. Introduction
   1.1 Background and motivation
   1.2 The credit-assignment gap in multi-agent LLM systems
   1.3 Why flat Shapley fails for structured pipelines
   1.4 From DyHuCoG to MHyperShap
   1.5 Contributions
   1.6 Organization
2. Related Work
   2.1 Shapley value in ML and explainable AI
   2.2 Graph/hypergraph-restricted cooperative games (Myerson, Owen)
   2.3 Credit assignment in multi-agent and agentic LLM systems
   2.4 Attribution-guided orchestration and routing
   2.5 Benchmarks for agent evaluation
   2.6 Positioning and differentiation (prior-art table)
3. Preliminaries and Problem Formulation
   3.1 Notation
   3.2 Multi-agent LLM pipeline as a cooperative game
   3.3 Background: Shapley and Myerson values
   3.4 Agent dependency DAG and the induced hypergraph (Def. 1)
4. The MHyperShap Framework
   4.1 Framework overview
   4.2 Agent hypergraph construction (Def. 1)
   4.3 Causal characteristic function and agent masking (Def. 2, Def. 6, Prop. 1)
   4.4 Multi-objective coalition value (Def. 3)
   4.5 Myerson-restricted Shapley value (Def. 4)
   4.6 Dynamic update (Def. 5)
   4.7 Monte Carlo estimation
   4.8 Attribution-driven routing
5. Theoretical Analysis
   5.1 Axioms (CE, HF, TC, DA)
   5.2 Main theorem: uniqueness of MHyperShap (Thm. 1)
   5.3 Corollary: axiomatic incorrectness of flat Shapley (Cor. 1)
   5.4 Routing-convergence proposition (Prop. 2)
   5.5 Computational complexity
6. SynAgentBench: A Ground-Truth Attribution Benchmark
   6.1 Motivation
   6.2 Task families A–D with analytic ground truth
   6.3 Evaluation metrics
   6.4 Generation protocol and release
7. Experimental Setup
   7.1 Benchmarks
   7.2 Baselines
   7.3 Agent backbone and implementation
   7.4 Metrics
   7.5 Hardware and reproducibility
8. Results and Discussion
   8.1 Attribution accuracy on SynAgentBench
   8.2 Axiom adherence
   8.3 Attribution and routing on real benchmarks
   8.4 Ablation study
   8.5 Sensitivity analysis (λ, M, n)
   8.6 Computational efficiency and scalability
   8.7 Statistical significance
   8.8 Interpretability case study
   8.9 Limitations
9. Conclusion and Future Work
Declarations
References
Appendix A — Proof of the uniqueness theorem
Appendix B — Statistical methodology
Appendix C — SynAgentBench full specification
Appendix D — Hyperparameters and prompts
Notation list
```

---

# ABSTRACT (refactored, ~220 words)
Multi-agent large language model (LLM) systems orchestrate heterogeneous agents — retrievers, tool-callers, reasoners, executors — across task pipelines whose cooperation is constrained by a task-dependency directed acyclic graph (DAG). When such a pipeline succeeds or fails, there is no principled mechanism to attribute the outcome to individual agents or coalitions. Existing Shapley-based approaches treat the agent set as a flat, exchangeable coalition, an assumption that is axiomatically incorrect for structured workflows where coalition validity is DAG-constrained and the agent pool evolves over time. We introduce **MHyperShap**, a framework that models agentic pipelines as **Myerson-restricted dynamic hypergraph cooperative games**. We prove that MHyperShap yields the *unique* allocation satisfying Component Efficiency, Hyperedge Fairness, and Temporal Consistency for DAG-structured agent workflows, and that flat Shapley violates Component Efficiency on disconnected coalitions. A causal agent-masking protocol preserves pipeline coherence during coalition evaluation. We further introduce **SynAgentBench**, the first benchmark with analytically known ground-truth agent attributions across additive, complementary, redundant, and DAG-constrained configurations, and validate on GAIA, AgentBench, and τ-bench. MHyperShap outperforms flat-Shapley, leave-one-out, and gating baselines on attribution accuracy and axiom adherence, and — by closing the attribution→routing feedback loop — improves downstream task-resolution rates with a provable convergence guarantee. *(Replace closing sentence with the headline quantitative result once experiments are complete.)*

**Keywords:** Cooperative game theory · Shapley value · Myerson value · Hypergraph · Multi-agent systems · Large language models · Credit assignment · Agent orchestration · Explainable AI · Routing

---

# 1. INTRODUCTION

**1.1 Background and motivation.** Agentic LLM systems (AutoGen, CrewAI, LangGraph, and enterprise orchestration stacks) increasingly solve tasks by chaining specialized agents. Consider a generic enterprise customer-resolution pipeline: a *retrieval* agent fetches account history, a *connector* agent queries external APIs, an *execution* agent runs business logic, and a *knowledge* agent grounds the final response. These agents form a coalition whose contributions are non-additive and order-dependent.

**1.2 The credit-assignment gap.** When the pipeline produces a wrong resolution, an SLA breach, or a compliance violation, there is no principled mechanism to identify *which agent caused the failure* or *which coalition drove the success*. This is not a logging problem but a **game-theoretic** one: the space of valid coalitions is not the full power set of agents but is constrained by the task-dependency DAG (the retrieval agent must run before the executor; the knowledge agent is only useful paired with the reasoner). Naive Shapley treats all subsets as valid and ignores these structural constraints.

**1.3 Why flat Shapley fails.** Three structural facts break flat cooperative-game treatments:
1. **Coalition validity is DAG-constrained.** Not all agent subsets form coherent pipelines; evaluating $v(\{\text{executor}\})$ without its upstream retriever is meaningless because the executor has no data to act on.
2. **Agent pools evolve.** Deployments add/remove agents continuously, so a static Shapley computation becomes stale quickly.
3. **Naive removal is out-of-distribution.** Removing an agent and running the pipeline feeds downstream agents null/undefined inputs — an OOD failure, not a principled counterfactual.

We forward-reference Corollary 1 (§5.3), which formalizes the resulting Component-Efficiency violation.

**1.4 From DyHuCoG to MHyperShap.** Our prior framework DyHuCoG [Louhichi et al., 2026] introduced dynamic hypergraph cooperative games with Monte-Carlo preference-aware Shapley for recommendation, modeling user–item–context interactions as dynamic coalitions and injecting Shapley weights into hypergraph message passing. The mathematical structure transfers directly to agent orchestration: the same dynamic-hypergraph-game backbone, with (i) a new domain (agents as players), (ii) new axioms (DAG/Myerson restriction), (iii) new theory (axiomatic uniqueness), and (iv) a new closed loop (attribution-driven routing). This continuity also connects to our real-time Shapley adjustment [Nesmaoui et al., 2025] and Shapley-based XAI work [Louhichi et al., 2023; 2025].

**1.5 Contributions.**
1. **Framework:** MHyperShap — the first Myerson-restricted *dynamic hypergraph* cooperative game for multi-agent LLM credit assignment, with a causal agent-masking protocol preserving pipeline coherence.
2. **Theory:** A uniqueness theorem (Component Efficiency + Hyperedge Fairness + Temporal Consistency under DAG-acyclicity) and a corollary showing flat Shapley is axiomatically incorrect for structured pipelines.
3. **Benchmark:** SynAgentBench — the first agent-attribution benchmark with *analytically computable* ground-truth Shapley values.
4. **Closed loop:** An attribution-driven routing policy with a convergence guarantee.
5. **Empirical:** Comprehensive evaluation vs. flat-Shapley, LOO, and gating baselines on SynAgentBench + GAIA/AgentBench/τ-bench, with ablations, sensitivity, significance testing, and scalability analysis.

**1.6 Organization.** Standard roadmap paragraph.

---

# 2. RELATED WORK
~1.5 pages; close with the differentiation table.

- **2.1 Shapley in ML/XAI** — SHAP, Data-Shapley, KNN-Shapley; our clustering-SHAP and "Game Theory Meets XAI"; note all are *flat/unstructured*.
- **2.2 Graph/hypergraph-restricted games** — Myerson (1977); van den Nouweland, Borm & Tijs (1992); Owen values; DyHuCoG. Establish the formal lineage we extend.
- **2.3 Credit assignment in multi-agent/agentic LLMs** — Shapley-Coop, SHARP, "Agent That Matters", AgentFlow-Shapley (flat, or training-time only); MARL methods COMA, QMIX, HIVE, HYGMA, MOHITO (RL-based, not Shapley-axiomatic, not LLM).
- **2.4 Attribution-guided routing** — current routing is heuristic; no prior work feeds principled attribution back into routing.
- **2.5 Agent benchmarks** — GAIA, AgentBench, τ-bench, SWE-Bench measure task success, not per-agent ground-truth attribution → motivates SynAgentBench.

**2.6 Prior-art differentiation table.**

| Method | Shapley? | Hypergraph? | Dynamic? | DAG-constrained? | Routing loop? | LLM agents? |
|---|---|---|---|---|---|---|
| Shapley-Coop | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| SHARP | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Agent That Matters | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| AgentFlow-Shapley | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| HIVE | ✗ (VDN) | ✓ | ✗ | ✗ | ✗ | ✗ |
| HYGMA | ✗ (MARL) | ✓ | ✓ | ✗ | ✗ | ✗ |
| MOHITO | ✗ (MARL) | ✓ | ✓ | ✗ | ✗ | ✗ |
| DyHuCoG (ours, 2026) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ (recommender) |
| **MHyperShap (this work)** | ✓ | ✓ | ✓ | **✓** | **✓** | **✓** |

**Precise novelty claim (for the text):** MHyperShap is the first framework to apply Myerson-restricted dynamic hypergraph cooperative games to LLM agent orchestration, proving axiomatic uniqueness under DAG constraints, using causal masking to preserve pipeline coherence, and closing the attribution→routing loop. The claim is over this *specific combination* of five conditions, not over Shapley-for-multi-agent-LLMs in general.

---

# 3. PRELIMINARIES AND PROBLEM FORMULATION

**3.1 Notation.** Agents $\mathcal{A} = \{a_1, \ldots, a_n\}$; task $t$; dependency DAG $\mathcal{D}_t = (\mathcal{A}, \mathcal{E}_t^{\mathrm{DAG}})$; agent hypergraph $\mathcal{H}_t = (\mathcal{A}, \mathcal{E}_t)$; coalition $S \subseteq \mathcal{A}$; coalition value $v_t(S)$; restricted value $v_t^{\mathcal{H}}(S)$; Myerson-restricted Shapley $\phi_i^{\mathcal{H}}(t)$; EMA factor $\lambda$; Monte-Carlo samples $M$. Full table at end of paper.

**3.2 Pipeline as cooperative game.** Players are agents; the coalition value is the pipeline outcome obtained with only the agents in $S$ active. A coalition is *valid* iff it induces a connected subgraph in $\mathcal{H}_t$.

**3.3 Background: Shapley and Myerson.** Exact Shapley value (mirroring DyHuCoG Eq. 3):

$$
\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\,\bigl[v(S \cup \{j\}) - v(S)\bigr]
$$

The **Myerson value** is the Shapley value of the graph-restricted game where the characteristic function of a disconnected coalition equals the sum over its connected components.

**3.4 Agent DAG → hypergraph.**
> **Definition 1 (Agent Hypergraph).** Given agents $\mathcal{A}$ and a task $t$ with dependency DAG $\mathcal{D}_t = (\mathcal{A}, \mathcal{E}_t^{\mathrm{DAG}})$, the agent hypergraph is $\mathcal{H}_t = (\mathcal{A}, \mathcal{E}_t)$, where each hyperedge $e \in \mathcal{E}_t$ is a maximal subset of agents that can operate as a coherent sub-pipeline in $\mathcal{D}_t$. A coalition $S \subseteq \mathcal{A}$ is *valid* iff it induces a connected subgraph in $\mathcal{H}_t$.

---

# 4. THE MHYPERSHAP FRAMEWORK
Methodological core — match DyHuCoG's depth. Fig. 1 = workflow diagram.

**4.1 Overview.** Six stages: (1) hypergraph construction → (2) causal masking → (3) characteristic-function evaluation → (4) Myerson-restricted Shapley → (5) dynamic update → (6) attribution-driven routing. Present as Algorithm 1:
```
Input: task t, agent pool A, dependency DAG D_t
1. Hypergraph construction:  derive H_t from D_t; enumerate connected coalitions C(H_t)
2. Causal masking:           for each S in C(H_t), mask absent agents via do(output = null embedding),
                             propagate through downstream DAG
3. Characteristic function:  v_t(S) = a*r + b/l + g/c + d*k   (Monte Carlo, M samples)
4. MHyperShap value:         phi_i^H(t) = Myerson-restricted Shapley (Def. 4)
5. Dynamic update:           phi_i^(T) = (1 - lambda)*phi_i^(T-1) + lambda*phi_i^H(t)
6. Routing:                  S* = argmax over S in C(H_t') of  sum_i phi_i^(T) * sim(t', T_i)
Output: per-agent attributions phi^(T); routed coalition S* for future tasks
```

**4.2 Hypergraph construction.** Definition 1 (above); enumerate $\mathcal{C}(\mathcal{H}_t)$ (connected coalitions only).

**4.3 Causal characteristic function and agent masking.**
> **Definition 2 (Causal Characteristic Function).**
> $$
> v_t(S) = \mathbb{E}\!\left[\,\text{outcome}(t) \;\middle|\; do\!\left(S\ \text{active},\ \forall a_i \notin S:\ \text{output}_{a_i} = \mathbf{null}\right)\right]
> $$
> For a disconnected coalition, $v_t^{\mathcal{H}}(S) = \sum_{C \in S_{/\mathcal{H}}} v_t(C)$, where $S_{/\mathcal{H}}$ partitions $S$ into the connected components of $\mathcal{H}_t[S]$.

*The problem with naive masking.* Simply removing an absent agent and running the pipeline feeds downstream agents undefined inputs — an OOD failure rather than a principled counterfactual (the same issue causal Shapley [Janzing et al., 2020] identified for tabular ML). We solve it with the do-operator:
> **Definition 6 (Causal Agent Masking).** When $a_i \notin S$, replace its output via $do(\text{output}_{a_i} = \mathbf{null})$ propagated through the downstream DAG: (1) for each downstream $a_j$, replace $a_i$'s contribution to $a_j$'s context with a learned *null embedding* $\mathbf{z}_\emptyset$, trained to minimize distribution shift on held-out traces; (2) $\mathbf{z}_\emptyset$ is task-type conditioned, $\mathbf{z}_\emptyset^{(t)} = f_\theta(\text{task\_type}(t))$.

> **Proposition 1.** Under causal masking, $v_t(S)$ is measurable and bounded for all connected $S \subseteq \mathcal{A}$, and equals zero for all disconnected $S$ (downstream agents receive null embeddings producing no-op outputs).

*Causal vs. naive masking (paper table):*

| Property | Naive masking | Causal masking |
|---|---|---|
| Pipeline coherence | ✗ (undefined states) | ✓ (null propagation) |
| Distribution shift | High (OOD inputs) | Low (learned null embedding) |
| Axiom satisfaction | Partial (CE violated on disconnected coalitions) | Full |
| Implementation cost | Trivial | One-time null-embedding training |

**4.4 Multi-objective coalition value.** Directly parallels DyHuCoG Eq. 1.
> **Definition 3 (Multi-Objective Outcome).**
> $$
> \text{outcome}(t, S) = \alpha \cdot r(t,S) + \beta \cdot \ell(t,S)^{-1} + \gamma \cdot c(t,S)^{-1} + \delta \cdot \kappa(t,S)
> $$
> where $r$ = task-resolution rate, $\ell$ = latency, $c$ = API cost, $\kappa$ = compliance score; weights $(\alpha, \beta, \gamma, \delta)$ tuned via Pareto-frontier analysis on a validation set (reuse DyHuCoG's grid-search protocol).

**4.5 Myerson-restricted Shapley value.**
> **Definition 4 (MHyperShap Value).**
> $$
> \phi_i^{\mathcal{H}}(t) = \sum_{\substack{S \subseteq \mathcal{A} \setminus \{i\} \\ S \cup \{i\}\ \text{connected in}\ \mathcal{H}_t}} \frac{|S|!\,(|\mathcal{A}|-|S|-1)!}{|\mathcal{A}|!}\,\bigl[v_t^{\mathcal{H}}(S \cup \{i\}) - v_t^{\mathcal{H}}(S)\bigr]
> $$
> The summation is restricted to coalitions where $S \cup \{i\}$ is connected in $\mathcal{H}_t$ (the Myerson restriction), replacing the $2^n$ flat coalitions with the polynomial-size set of connected subgraphs.

**4.6 Dynamic update.**
> **Definition 5 (Dynamic Update).** Across a task stream $\{t_1, \ldots, t_T\}$:
> $$
> \phi_i^{(T)} = (1-\lambda)\,\phi_i^{(T-1)} + \lambda \cdot \phi_i^{\mathcal{H}}(t_T), \qquad \lambda \in (0,1)
> $$
> enabling incremental maintenance without full recomputation as the agent pool evolves.

**4.7 Monte Carlo estimation.** Permutation sampling with $M$ samples; variance decreases as $\mathcal{O}(1/M)$. Reuse DyHuCoG's empirical precedent ($M=50 \approx 99\%$ accuracy, $\mathrm{MSE} \approx 1.4\times10^{-5}$) as justification and report a SynAgentBench convergence table.

**4.8 Attribution-driven routing.** Accumulated $\phi_i^{(T)}$ define a Shapley routing policy:

$$
S^* = \arg\max_{S \in \mathcal{C}(\mathcal{H}_{t'})} \sum_{i \in S} \phi_i^{(T)} \cdot \mathrm{sim}(t', \mathcal{T}_i)
$$

where $\mathcal{C}(\mathcal{H}_{t'})$ is the set of valid coalitions for new task $t'$, and $\mathrm{sim}(t', \mathcal{T}_i)$ is the cosine similarity between $t'$ and the past tasks where $a_i$ had high attribution. Structurally identical to DyHuCoG's Shapley-guided item selection.

---

# 5. THEORETICAL ANALYSIS

**5.1 Axioms.**
- **A1 — Component Efficiency (CE):** for every connected component $C$ of $\mathcal{H}_t$, $\sum_{i \in C} \phi_i^{\mathcal{H}} = v_t(C)$.
- **A2 — Hyperedge Fairness (HF):** for any hyperedge $e$ and $i, j \in e$, removing $e$ changes both members' allocations by the same amount:
  $$\phi_i^{\mathcal{H}}(\mathcal{H}) - \phi_i^{\mathcal{H}}(\mathcal{H} \setminus e) = \phi_j^{\mathcal{H}}(\mathcal{H}) - \phi_j^{\mathcal{H}}(\mathcal{H} \setminus e).$$
- **A3 — Temporal Consistency (TC):** the dynamic-update sequence $\{\phi_i^{(T)}\}$ converges, and its limit satisfies CE and HF w.r.t. the empirical task distribution.
- **A4 — DAG Acyclicity (DA):** $\mathcal{D}_t$ is acyclic, so the connected components of $\mathcal{H}_t$ are well-defined.

**5.2 Main theorem.**
> **Theorem 1 (MHyperShap Uniqueness).** Let $(N, v, \mathcal{H})$ be a hypergraph cooperative game where $\mathcal{H}$ is the agent dependency hypergraph derived from an acyclic task DAG satisfying DA. The unique allocation $\phi : 2^N \to \mathbb{R}^N$ satisfying Component Efficiency (A1), Hyperedge Fairness (A2), and Temporal Consistency (A3) is the Myerson-restricted dynamic Shapley value $\phi^{\mathcal{H}}$ of Definitions 4–5.

*Proof sketch (full proof in Appendix A).* **Existence:** $\phi^{\mathcal{H}}$ satisfies CE (the restricted $v^{\mathcal{H}}$ distributes value within components; Shapley efficiency reduces to CE component-wise), HF (the difference upon removing $e$ depends only on coalitions disconnected by removal; by the symmetric roles of $i, j \in e$ these are equal), and TC (the EMA converges to $\mathbb{E}_t[\phi_i^{\mathcal{H}}(t)]$ by the law of large numbers). **Uniqueness:** following Myerson (1977) extended to hypergraphs (van den Nouweland, Borm & Tijs, 1992): for a single-hyperedge game, CE + HF uniquely determine $\psi_i$; induction on $|\mathcal{E}|$ with Shapley linearity extends to all hypergraph games; DA grounds the base case; TC uniquely extends to the dynamic setting via the EMA. $\square$

**5.3 Corollary.**
> **Corollary 1 (Flat Shapley is axiomatically incorrect).** A flat game $(N, v)$ ignoring $\mathcal{H}$ violates Component Efficiency whenever the agent set is disconnected in $\mathcal{H}$, assigning non-zero credit to agents in isolated components.

This is the formal justification for why flat-Shapley credit-assignment methods produce incorrect attributions on structured workflows (validated empirically in §8.2).

**5.4 Routing convergence.**
> **Proposition 2 (Routing Convergence).** Under the dynamic update (Def. 5) with $\lambda \in (0,1)$ and a stationary task distribution $P(t)$, the Shapley routing policy converges to the optimal coalition-selection policy $\pi^*(t') = \arg\max_S \mathbb{E}_P[v(S) \mid t']$ as $T \to \infty$.

*Proof sketch.* By TC, $\phi_i^{(T)} \to \mathbb{E}_P[\phi_i^{\mathcal{H}}(t)]$; under the regularity condition $\mathrm{sim}(t', \mathcal{T}_i) \propto P(t' \mid a_i\ \text{useful})$, the routing objective converges to $\mathbb{E}[v(S) \mid t']$. $\square$

**5.5 Complexity.** Per-task cost in terms of $n$, $M$, and $|\mathcal{C}(\mathcal{H}_t)|$; contrast with flat $2^n$. The Myerson restriction collapses the coalition space to connected subgraphs; in realistic pipelines $n \leq 10$. Mirror DyHuCoG's Big-O treatment and report wall-clock vs. flat Shapley and HYGMA.

---

# 6. SYNAGENTBENCH

**6.1 Motivation.** No existing benchmark has known ground-truth agent attributions; GAIA/AgentBench/τ-bench measure task success only. SynAgentBench constructs pipelines whose ground-truth Shapley values are analytically computable, so attribution accuracy can be measured directly (not just via downstream routing as a proxy).

**6.2 Task families (analytic ground truth).**
- **Family A — Additive** (ground truth: $\phi_i = q_i$). Agents contribute independently: $v(S) = \sum_{i \in S} q_i$. A correct method recovers each $q_i$ exactly.
- **Family B — Complementary** (superadditive pairs). Agents operate in synergistic pairs: $v(\{a_i, a_j\}) = 2\,v(\{a_i\}) = 2\,v(\{a_j\})$. Ground truth: $\phi_i = \phi_j = v(\{a_i, a_j\})/2$. Tests capture of superadditivity.
- **Family C — Redundant** (null player). Duplicate agents: $v(S \cup \{a_k\}) = v(S)$ whenever $S$ contains an equivalent agent. Ground truth: $\phi_k = 0$. Tests null-player enforcement.
- **Family D — DAG-constrained** (Myerson value). Agents linked by a fixed DAG making some coalitions invalid; ground truth = the analytically computed Myerson value. Tests correct application of the Myerson restriction.

**6.3 Metrics.** Attribution MAE; Axiom-Violation Rate (fraction of coalitions violating CE/HF); Null-Player Precision; Complementarity Recall.

**6.4 Generation and release.** 500 tasks/family $\times$ 4 = **2,000 tasks**; agent-pool sizes $n \in \{4, 6, 8, 10\}$; ground truth computed analytically with **no LLM calls** (fully reproducible); open-source generator + dataset (GitHub/Zenodo). Designed to be reused by future agent-attribution papers.

---

# 7. EXPERIMENTAL SETUP
Mirror DyHuCoG §4.1 rigor (shared protocol, seeds, CIs, paired tests).

**7.1 Benchmarks.** SynAgentBench (ground truth) + GAIA, AgentBench, τ-bench (and optionally a SWE-Bench subset for code pipelines).

**7.2 Baselines.**

| Baseline | Represents |
|---|---|
| Flat Shapley (no Myerson) | best flat cooperative attribution; ablation of the DAG constraint |
| Causal LOO | leave-one-out ablation (standard) |
| Gating / heuristic credit | proxy credit (e.g., routing/gate weights) |
| Shapley-Coop-style | flat Shapley for LLM agents |
| SHARP-style | Shapley reward attribution (training-time) |
| Static MHyperShap | ablation: no dynamic update |
| Naive-masking MHyperShap | ablation: removal instead of causal masking |

**7.3 Agent backbone and implementation.** Open 7–8B model (e.g., Qwen2.5-7B / Llama-3.1-8B) as the agent backbone; Python + PyTorch; reuse the DyHuCoG Monte-Carlo Shapley estimator as the computational core; one-time null-embedding training on agent traces. Optional frontier-API comparison row (GPT/Claude/Gemini) for one table.

**7.4 Metrics.** Attribution metrics (§6.3) + routing success rate, coalition-coherence score (fraction of $v(S)$ evaluations avoiding OOD states), computation time, Pareto routing curve (resolution vs. cost).

**7.5 Hardware and reproducibility.** Single RTX 4090 (consistent with the DyHuCoG hardware table); 5 seeds $\{42, 43, 44, 45, 46\}$; mean $\pm$ std; 95% CIs; public code + benchmark release.

---

# 8. RESULTS AND DISCUSSION
Mirror DyHuCoG's results depth.

- **8.1 Attribution accuracy** — MAE vs. analytic ground truth on Families A–D; MHyperShap recovers true values; baselines fail on D.
- **8.2 Axiom adherence** — axiom-violation-rate table; flat Shapley violates CE on disconnected coalitions (empirical validation of Corollary 1).
- **8.3 Real benchmarks** — attribution quality + routing-success improvement on GAIA/AgentBench/τ-bench.
- **8.4 Ablation** — (1) Myerson vs. flat (Family D); (2) causal vs. naive masking (OOD rate + stability); (3) dynamic vs. static (routing convergence speed); (4) hypergraph vs. graph vs. unstructured (axiom-violation rates). Table mirrors DyHuCoG Table 5.
- **8.5 Sensitivity** — $\lambda \in \{0.01, 0.05, 0.1, 0.3\}$; $M \in \{10, 25, 50, 100\}$ (reuse DyHuCoG convergence framing); $n \in \{4, 6, 8, 10, 15\}$ scaling.
- **8.6 Efficiency & scalability** — runtime/memory table; Myerson restriction vs. $2^n$; near-linear scaling (mirror DyHuCoG Table 6).
- **8.7 Statistical significance** — paired t-tests + Holm–Bonferroni + Wilcoxon on per-task attribution error (reuse your Appendix-A methodology style).
- **8.8 Interpretability case study** — one pipeline: per-agent Shapley waterfall + how routing changes (analogous to DyHuCoG SHAP waterfall Fig. 4).
- **8.9 Limitations** — null-embedding quality; reliance on DAG availability; MC variance; frontier-model generalization.

---

# 9. CONCLUSION AND FUTURE WORK
Recap the five contributions; restate headline numbers; future work: weighted/heterogeneous-player Shapley (different agent cost/types), online streaming agents, federated/differentially-private variant, multimodal agent pipelines. Tie to EU AI Act auditability (consistent with the thesis framing).

---

# DECLARATIONS (required by Discover AI)
- **Funding** — state grant/none.
- **Competing interests** — "The authors declare no competing interests."
- **Data availability** — SynAgentBench + generator on GitHub/Zenodo; real benchmarks public.
- **Code availability** — repository link.
- **Author contributions** — reuse DyHuCoG CRediT split (Louhichi: conceptualization/methodology/software/writing; Nesmaoui: software/data; Lazaar: supervision/analysis).
- **Ethics approval** — not applicable.

# APPENDICES
- **A** — full uniqueness proof (existence + induction-on-$|\mathcal{E}|$ uniqueness, EMA limit).
- **B** — statistical methodology (paired tests, Holm–Bonferroni, Wilcoxon) — adapt DyHuCoG Appendix A.
- **C** — SynAgentBench full specification (per-family generators + ground-truth derivations).
- **D** — hyperparameters, agent prompts, weight-tuning grid.

# NOTATION LIST
Reuse DyHuCoG's notation-table format, updated for agents / hypergraph / Myerson / routing symbols.

---

# PLANNED FIGURES & TABLES
**Figures:** (1) MHyperShap workflow; (2) DAG→hypergraph example; (3) causal vs. naive masking schematic; (4) attribution-MAE bars (A–D); (5) axiom-violation comparison; (6) routing-success curves; (7) $\lambda$/$M$/$n$ sensitivity; (8) per-agent Shapley waterfall; (9) runtime/scalability; (10) per-task difference distribution + Q–Q.

**Tables:** (1) prior-art differentiation; (2) notation; (3) SynAgentBench statistics; (4) main attribution results; (5) axiom-adherence; (6) real-benchmark attribution+routing; (7) ablation; (8) sensitivity; (9) runtime/memory; (10) paired significance.

---

# PLANNING NOTES (NOT part of the manuscript)

**Execution timeline (4–6 months).**

| Phase | Weeks | Output |
|---|---|---|
| Related work + differentiation | 1–2 | §2 + prior-art table |
| Theorem + proof | 2–4 | §5 + Appendix A |
| SynAgentBench generator | 3–5 | §6 + open benchmark |
| Adapt DyHuCoG MC-Shapley to agents + causal masking | 5–8 | working codebase |
| Experiments (SynAgentBench + real) | 8–12 | §8 tables |
| Ablations + sensitivity + significance | 12–15 | §8.4–8.7 |
| Full draft + figures | 15–20 | manuscript |
| Internal review + polish + submit | 20–24 | Discover AI submission |

**Key prior art to cite.** Shapley (1953); Myerson (1977); van den Nouweland, Borm & Tijs (1992); Janzing et al. (2020); Lundberg & Lee (2017, SHAP); Ghorbani & Zou (2019, Data-Shapley); Shapley-Coop; SHARP; HIVE; HYGMA; MOHITO; your own lineage — DyHuCoG (2026), Real-time Shapley adjustment (2025), GNN+Shapley hierarchical recommendation (2025), Game-Theory-Meets-XAI (2025), clustering-SHAP (2023).

**Submission checklist (Discover AI).**
- [ ] Abstract ≤ ~250 words, no citations.
- [ ] All declarations present.
- [ ] Data + code availability with working links.
- [ ] Figures ≥ 300 dpi; vector where possible.
- [ ] Reproducibility: seeds, hyperparameters, hardware reported.
- [ ] Self-citation lineage included.
- [ ] Statistical tests with multiple-comparison correction.
- [ ] Check Morocco APC discount eligibility (Research4Life) before acceptance.
