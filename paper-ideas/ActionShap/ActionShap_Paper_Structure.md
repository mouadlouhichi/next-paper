> **LEGACY CROSS-DOMAIN DRAFT — NOT THE CANONICAL RECOMMENDATION-ONLY PAPER.** Use `ActionShap_Recommendation_Paper_Structure.md` and `ActionShap_Recommendation_Spec.md` instead. This file is retained for historical reference.

# ActionShap — Full Paper Structure, TOC & Embedded Content

**Target journal:** *Discover Artificial Intelligence* (Springer Nature, open access; Scopus Q1 in Information Systems, Q2 in Artificial Intelligence; CiteScore 6; median first decision ≈ 23 days; APC $1,690 / €1,390 / £1,190)
**Article type:** Research article
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco
**Corresponding author:** mouad_louhichi@um5.ac.ma

> **Why this paper, and why it is easy to accept.** It is the *capstone of the thesis's own argument*: RQ5 (the unifying cooperative-attribution perspective) was never published standalone, and the thesis subtitle promises "Actionable Insight" while explicitly conceding that actionability was never measured. This paper closes that gap. It reuses all four thesis datasets and both existing codebases, needs **no new theorem** (one light proposition plus a one-line corollary of Shapley efficiency), **no GPU beyond the existing RTX 4090**, and **no human subjects** — so no ethics approval, no recruitment. Structure mirrors DyHuCoG and the MHyperShap/FairShap blueprints. Target ≈ 8,000–10,000 words, 8–9 figures, 10–12 tables (cut order specified in the planning notes).

---

## Working Title (primary + alternates)
- **Primary:** *ActionShap: Intervention-Grounded Evaluation of Cooperative Attribution — A Unified Actionability Framework for Explainable Clustering and Recommendation*
- Alt 1: *Do Shapley Values Predict What Happens When You Intervene? An Actionability Benchmark for Explainable AI*
- Alt 2: *From Attribution to Action: Measuring the Actionability of Cooperative-Game Explanations Across Clustering and Recommendation*

## One-paragraph thesis (the spine)
Explainable AI evaluates attributions with *faithfulness* proxies (sufficiency, comprehensiveness) that measure what happens when a factor is **deleted** — but practitioners need to know what happens when a factor is **feasibly modified**. These are different claims, and the second is what "actionable" actually means. We introduce **ActionShap**, a unified cooperative-game framework that (i) operationalizes actionability into two measurable quantities — an **Actionability Score (AS)** and **Attribution–Intervention Alignment (AIA)** — (ii) formalizes static feature-player games (clustering) and dynamic interaction-player games (recommendation) as one importance-allocation problem, and (iii) empirically tests, across four datasets and five attribution methods, whether high attribution predicts realized intervention effect. We further propose **actionability-guided attribution reranking**, which surfaces modifiable drivers first and measurably changes which intervention a practitioner would select.

## Research questions of *this paper* (and their link to the thesis)
- **PRQ1.** Can the qualitative notion of actionable insight be operationalized into metrics computable without human subjects? *(operationalizes thesis Definition 1.1)*
- **PRQ2.** Do Shapley attributions predict realized intervention effects better than LIME, attention, gradient, and permutation baselines? *(tests the thesis's core unmeasured claim)*
- **PRQ3.** Does the actionability ranking of attribution methods hold consistently across static (clustering) and dynamic (recommendation) tasks? *(answers thesis RQ5 empirically)*
- **PRQ4.** Does actionability-guided reranking change the intervention a practitioner would choose, and does it improve realized outcomes?

---

# TABLE OF CONTENTS
```
Abstract / Keywords
1. Introduction
   1.1 Background and motivation
   1.2 The faithfulness-vs-actionability gap
   1.3 Why cooperative attribution is the right substrate
   1.4 From post-hoc explanation to intervention-grounded evaluation
   1.5 Contributions
   1.6 Organization
2. Related Work
   2.1 Feature attribution and SHAP
   2.2 Evaluating explanations: faithfulness metrics and their limits
   2.3 Counterfactual explanation and algorithmic recourse
   2.4 Interpretable clustering
   2.5 Explainable recommendation and attention-based explanation
   2.6 Positioning and differentiation (comparison table)
3. Preliminaries and Problem Formulation
   3.1 Notation
   3.2 A unified cooperative game over heterogeneous players
   3.3 Static feature-player games (clustering)
   3.4 Dynamic interaction-player games (recommendation)
4. The ActionShap Framework
   4.1 Framework overview
   4.2 Modifiability
   4.3 Feasible interventions and the intervention effect
   4.4 Attribution stability
   4.5 The Actionability Score
   4.6 Attribution-Intervention Alignment
   4.7 Intervention-decision metrics: top-k precision and intervention regret
   4.8 Actionability-guided attribution reranking (A-Shapley)
5. Analysis
   5.1 When does the attribution ordering coincide with the intervention ordering?
   5.2 Sources of misalignment: curvature, interaction, infeasibility
       5.2.1 Identifiability of the curvature/interaction split
       5.2.2 A Shapley decomposition of the misalignment gap
   5.3 Computational complexity
6. Experimental Setup
   6.1 Datasets and task instantiation
   6.2 Modifiability elicitation protocol
   6.3 Intervention designs per dataset
   6.4 Attribution methods compared
   6.5 Metrics
   6.6 Implementation, hardware, reproducibility
7. Results and Discussion
   7.1 Attribution-Intervention Alignment across methods and tasks
   7.2 Actionability Score profiles (with the modifiability-held-out control)
   7.3 Faithfulness vs. actionability: do they agree?
   7.4 Shapley vs. attention inside the same model
   7.5 Intervention decisions: top-k precision and regret
   7.6 Decomposing misalignment: curvature, interaction, infeasibility
   7.7 Actionability-guided reranking
   7.8 Sensitivity, stability, and robustness
   7.9 Statistical significance
   7.10 Case studies: air-quality regime and a recommendation decision
   7.11 Limitations
8. Conclusion and Future Work
Declarations
References
Appendix A - Proposition proof and metric derivations
Appendix B - Statistical methodology
Appendix C - Modifiability tables per dataset
Appendix D - Intervention protocols and hyperparameters
Notation list
```

---

# ABSTRACT (draft, ~225 words)
Explanation methods are routinely evaluated with faithfulness proxies such as sufficiency and comprehensiveness, which quantify how a model responds when a factor is removed. Practitioners, however, act by *modifying* factors, not deleting them, and many highly attributed factors are not modifiable at all. Consequently, an explanation may be faithful yet operationally useless. We introduce **ActionShap**, a unified cooperative-game framework for measuring the *actionability* of attributions. ActionShap formalizes clustering and recommendation as instances of one importance-allocation problem — static feature-player games and dynamic interaction-player games — and defines a suite of computable quantities: an **Actionability Score**, combining domain-elicited modifiability, realized effect under a feasible intervention, and attribution stability; **Attribution–Intervention Alignment (AIA)**, the rank agreement between an attribution ordering and the measured intervention-effect ordering; and two decision-level metrics — **top-$k$ intervention precision** and **intervention regret** — that score an attribution by the quality of the intervention a practitioner would select from it. We prove that the two orderings coincide under local linearity and unit modifiability, and characterize curvature, interaction, and infeasibility as the sources of divergence. Across four datasets (Wine Quality, Beijing Multi-Site Air Quality, MovieLens-1M, Amazon-Book) and five attribution methods (TreeSHAP, KernelSHAP, LIME, attention, gradient-based), we find that faithfulness and actionability rank methods differently, and that cooperative attribution aligns most closely with realized interventions. We further propose actionability-guided reranking, which changes the recommended intervention in a substantial fraction of cases while improving realized outcomes. *(Insert headline numbers once experiments complete.)*

**Keywords:** Explainable AI · Shapley value · Cooperative game theory · Actionability · Intervention · Attribution evaluation · Interpretable clustering · Recommender systems · Trustworthy AI

---

# 1. INTRODUCTION

**1.1 Background and motivation.** Attribution methods now mediate how analysts, regulators, and system designers interrogate opaque models. The implicit promise is that a high attribution tells the user *where to act*. Yet the standard evaluation criteria never test that promise.

**1.2 The faithfulness–actionability gap.** Faithfulness metrics — sufficiency, comprehensiveness, deletion/insertion curves — ask: *if this factor were absent, how would the output change?* Actionability asks a different question: *if this factor were changed by an amount the decision-maker can actually realize, how would the output change?* The two diverge for two reasons:
1. **Infeasibility.** Many high-attribution factors are not modifiable. In air-quality monitoring, temperature and dew point may dominate a cluster explanation while being entirely outside operator control; only pollutant emissions are actionable.
2. **Non-locality.** Deletion moves a factor to a baseline that may be far outside any feasible operating range, so the measured effect need not reflect the effect of a realistic adjustment.

**1.3 Why cooperative attribution is the right substrate.** The Shapley value allocates credit under explicit axioms (efficiency, symmetry, null-player, additivity), which makes attributions *complete* and *comparable* across factors — a precondition for ranking candidate interventions against one another. Heuristic or attention-based scores lack this guarantee, so their magnitudes are not commensurable.

**1.4 From post-hoc explanation to intervention-grounded evaluation.** Prior work of ours established cooperative attribution for black-box clustering [Louhichi et al., 2023], extended it to hierarchical large-scale settings [Louhichi et al., 2025], and embedded it as an in-training signal in a dynamic hypergraph recommender [Louhichi et al., 2026]. Those studies demonstrated that Shapley attribution is *transferable* across tasks. This paper takes the next step and asks whether it is *actionable*, using intervention experiments rather than illustrative cases.

**1.5 Contributions.**
1. **Actionability formalism:** a computable operationalization of actionable insight via modifiability, feasible-intervention effect, and stability.
2. **A four-metric evaluation suite,** none of which requires human subjects: the Actionability Score (per factor), Attribution–Intervention Alignment (per method/task), and two decision-level metrics — top-$k$ intervention precision and intervention regret — that measure an attribution by the quality of the action it induces rather than by its rank correlation alone.
3. **Unified game formulation** covering static feature-players and dynamic interaction-players, with a proposition identifying exactly when attribution and intervention orderings coincide.
4. **Algorithmic component:** actionability-guided attribution reranking (A-Shapley), which prioritizes modifiable drivers.
5. **Cross-task empirical study:** four datasets, five attribution methods, showing that faithfulness and actionability induce *different* method rankings — a result with direct consequences for how XAI is evaluated.

**1.6 Organization.** Standard roadmap paragraph.

---

# 2. RELATED WORK
~1.5 pages; close with the comparison table.

- **2.1 Feature attribution and SHAP** — Shapley (1953); Lundberg & Lee (2017); TreeSHAP/KernelSHAP; our clustering-SHAP and Game-Theory-Meets-XAI works.
- **2.2 Evaluating explanations** — faithfulness (sufficiency/comprehensiveness), deletion/insertion curves, stability/robustness metrics, sanity checks. Argue: all are *deletion-based* or *perturbation-based*, none are *feasibility-constrained*.
- **2.3 Counterfactual explanation and algorithmic recourse** — Wachter et al.; Ustun et al. (actionable recourse); Karimi et al. (recourse under causal constraints). **Key differentiation:** recourse produces a *per-individual prescription* for supervised classifiers; ActionShap produces a *per-factor evaluation of an attribution method*, and generalizes to unsupervised clustering and to recommendation, where no notion of individual recourse exists.
- **2.4 Interpretable clustering** — pre-/in-/post-clustering interpretability; SHAP-based cluster explanation; LIME baselines.
- **2.5 Explainable recommendation** — attention-based, path-based, counterfactual explanation in recommendation; note that attention weights are widely *interpreted as* importance without validation.

**2.6 Comparison table.**

| Approach | Evaluates attribution quality? | Feasibility-constrained? | Unsupervised tasks? | Recommendation? | Model-agnostic metric? |
|---|---|---|---|---|---|
| Faithfulness (sufficiency/comprehensiveness) | ✓ | ✗ | partial | partial | ✓ |
| Deletion/insertion curves | ✓ | ✗ | ✗ | ✗ | ✓ |
| Stability / sanity checks | ✓ | ✗ | partial | ✗ | ✓ |
| Algorithmic recourse | ✗ (prescribes, not evaluates) | ✓ | ✗ | ✗ | ✗ |
| Counterfactual explanation | ✗ | partial | ✗ | partial | ✗ |
| **ActionShap (this work)** | **✓** | **✓** | **✓** | **✓** | **✓** |

**Precise novelty claim:** ActionShap is the first framework to evaluate attribution methods against *feasibility-constrained interventions* rather than deletions, and the first to do so under a single cooperative-game formulation spanning unsupervised clustering and recommendation.

---

# 3. PRELIMINARIES AND PROBLEM FORMULATION

**3.1 Notation.** Player set $N$; coalition $S \subseteq N$; characteristic function $v(S)$; Shapley value $\phi_j$; Monte-Carlo estimate $\hat{\phi}_j$; modifiability $m_j \in [0,1]$; feasible perturbation budget $\tau_j$; intervention effect $\Delta_j$; stability $s_j$; Actionability Score $\mathrm{AS}_j$; model output functional $f$.

**3.2 A unified cooperative game.** Both thesis regimes are instances of one template: a player set $N$, a task-specific characteristic function $v$, and a Shapley allocation $\phi$. The regimes differ only in *what a player is* and *what $v$ measures*.

| | Static regime (clustering) | Dynamic regime (recommendation) |
|---|---|---|
| Players $N$ | features $\mathcal{F}$ | $\mathcal{U} \cup \mathcal{I} \cup \mathcal{C}$ (users, items, contexts) |
| $v(S)$ | surrogate class-probability of cluster assignment using $S$ | $\alpha\,\mathrm{NDCG@}K(S) + \beta\,\mathrm{Div}(S) + \gamma\,\mathrm{Ctx}(S)$ |
| Output $f$ | cluster membership / regime label | top-$N$ ranked list quality |
| Intervention | adjust a physicochemical or pollutant variable | reweight an interaction, alter a context signal |

**3.3 Static feature-player game.** Reuse the PCA–K-Means–LightGBM–TreeSHAP formulation: features are players, and $v(S)$ is the surrogate's predicted probability of the observed cluster using only $S$.

**3.4 Dynamic interaction-player game.** Reuse the DyHuCoG coalition utility:
$$
v(S) = \alpha\,\mathrm{NDCG@}20(S) + \beta\,\mathrm{Diversity}(S) + \gamma\,\mathrm{ContextScore}(S), \qquad \alpha+\beta+\gamma = 1
$$
with the preference-weighted variant $v_{\mathrm{pref}}(S) = v(S) + \lambda_{\mathrm{pref}}\sum_{(u,i)\in S}\mathrm{sim}(u,i)$ and the Monte-Carlo estimator
$$
\hat{\phi}_j = \frac{1}{M}\sum_{m=1}^{M}\bigl[v(S_m \cup \{j\}) - v(S_m)\bigr].
$$

---

# 4. THE ACTIONSHAP FRAMEWORK
Fig. 1 = workflow: model → attribution → modifiability elicitation → feasible intervention → measured effect → AS / AIA → reranking.

**4.1 Overview.** Five stages, presented as Algorithm 1:
```
Input: trained model f, player set N, attribution method g, modifiability map m, budget map tau
1. Attribution:        phi_j = g(f, j)  for all j in N        (repeat over R seeds)
2. Stability:          s_j from dispersion of phi_j across R runs
3. Intervention:       for each j: apply feasible perturbation of size tau_j, measure Delta_j
4. Scoring:            AS_j = m_j * |Delta_j| * s_j
5. Alignment:          AIA = rank_corr( ordering by |phi_j| , ordering by |Delta_j| )
6. Reranking:          phi_tilde_j = actionability-guided reweighting of phi_j
Output: per-factor AS; per-method AIA; reranked attribution ordering
```

**4.2 Modifiability.**
> **Definition 1 (Modifiability).** $m_j \in [0,1]$ quantifies the degree to which factor $j$ is under the decision-maker's control. $m_j = 1$ denotes a directly controllable factor, $m_j = 0$ an immutable one; intermediate values encode partial or indirect control.

Elicitation is protocolized (§6.2) and reported transparently per dataset (Appendix C), so results are reproducible and the subjective step is auditable. The thesis already draws this distinction qualitatively for air quality — pollutant concentrations are intervention targets, whereas temperature, pressure, and dew point are interpretive only. ActionShap makes it explicit and numeric.

**4.3 Feasible interventions and the intervention effect.**
> **Definition 2 (Feasible Intervention Effect).** For factor $j$ with feasible perturbation budget $\tau_j$,
> $$
> \Delta_j = \mathbb{E}_{x}\Bigl[f\bigl(x \mid do(x_j \leftarrow x_j + \tau_j)\bigr) - f(x)\Bigr]
> $$
> where $\tau_j$ is calibrated to a realistic operating range (e.g., one within-domain standard deviation, or a domain-specified achievable adjustment), **not** to a deletion baseline.

Contrast with faithfulness, which uses $x_j \leftarrow \text{baseline}$ — an unconstrained and often out-of-distribution move.

**4.4 Attribution stability.**
> **Definition 3 (Stability).** Over $R$ repeated attribution runs (different seeds, Monte-Carlo draws, or perturbation neighbourhoods),
> $$
> s_j = \max\left(0,\; 1 - \frac{\mathrm{sd}(\hat{\phi}_j)}{|\overline{\hat{\phi}_j}| + \varepsilon}\right) \in [0,1].
> $$

This is where LIME is expected to suffer, consistent with the thesis's finding that LIME attributions are unstable under changing perturbation neighbourhoods.

**4.5 The Actionability Score.**

*State the modeling assumption before the formula.* Actionability requires three conditions **jointly**: the factor must be controllable, the controllable change must move the output, and the attribution must be reliable enough to act on. Each condition is *necessary*, so a factor failing any one of them is not actionable at all, regardless of how well it scores on the other two — an immovable factor with a huge effect is as useless as a movable factor with no effect. A functional form that is zero whenever any argument is zero, and monotone increasing in each, is therefore required; the product is the simplest such form. A weighted sum is explicitly rejected because it permits compensation across conditions, which contradicts the definition. **This is an assumption, not a derivation**, and §7.8 tests it by re-running the headline rankings under two alternatives (a normalized weighted sum, and a min-based conjunctive score) to show which conclusions are form-dependent.

> **Definition 4 (Actionability Score).**
> $$
> \mathrm{AS}_j = m_j \cdot |\Delta_j| \cdot s_j
> $$
> matching Definition 1.1 of the thesis, which requires both a modifiable factor and a specifiable change in output.

**Circularity guard (important — read before writing §7).** Because $m_j = 0$ forces $\mathrm{AS}_j = 0$, any statement of the form "AS ranks unmodifiable factors last" is true *by construction* and must never be presented as an empirical finding. Every AS-based result must therefore be reported alongside the **modifiability-held-out score** $\mathrm{AS}^{-m}_j = |\Delta_j| \cdot s_j$, which carries no such tautology. The empirical claims live on the attribution side, not the AS side: the finding is that *attribution methods place high $|\phi_j|$ on factors with $m_j = 0$*, and that AIA drops once feasibility is enforced. Phrase every headline accordingly.

**4.6 Attribution–Intervention Alignment.**
> **Definition 5 (AIA).** For attribution method $g$ on task $T$,
> $$
> \mathrm{AIA}(g, T) = \rho\Bigl(\bigl[\,|\phi_j|\,\bigr]_{j \in N},\ \bigl[\,|\Delta_j|\,\bigr]_{j \in N}\Bigr)
> $$
> with $\rho$ Spearman (primary) and Kendall $\tau$ (reported for robustness). We additionally report **modifiability-restricted AIA**, computed only over $\{j : m_j > 0\}$, which isolates alignment on the factors a practitioner could actually act upon.

**4.7 Intervention-decision metrics.** AIA scores an *ordering*; a practitioner acts on a *choice*. Two metrics close that gap and carry the decision-level evidence in §7.5 and §7.7.

> **Definition 6 (Top-$k$ Intervention Precision).** Let $\mathcal{K}_g$ be the top-$k$ factors under attribution method $g$. Then
> $$
> \mathrm{P}@k(g) = \frac{1}{k}\bigl|\{\, j \in \mathcal{K}_g : m_j > 0 \ \wedge\ \mathrm{sign}(\Delta_j) = \mathrm{sign}(\phi_j) \ \wedge\ |\Delta_j| \geq \delta \,\}\bigr|
> $$
> the fraction of top-attributed factors that are modifiable and whose feasible intervention produces an effect of the predicted sign and at least a minimum practical magnitude $\delta$.

> **Definition 7 (Intervention Regret).** With $j^\star_g = \arg\max_{j} |\phi_j|$ restricted to $\{j : m_j > 0\}$ and $j^{\mathrm{opt}} = \arg\max_{j : m_j > 0} |\Delta_j|$,
> $$
> \mathrm{Reg}(g) = |\Delta_{j^{\mathrm{opt}}}| - |\Delta_{j^\star_g}| \ \geq 0
> $$
> the outcome forgone by acting on the attribution's top recommendation instead of the best feasible one. Reported normalized by $|\Delta_{j^{\mathrm{opt}}}|$ for cross-dataset comparability.

*Scope note.* Regret requires the per-factor intervention sweep to be exhaustive over the feasible set. This is trivial for the 11-feature Wine task and cheap for air quality, but the recommendation tasks have far too many players for an exhaustive sweep — there, define regret over a **pre-declared candidate intervention set** (top-$C$ players by attribution under *any* method, plus a random control sample), and state $C$ explicitly so the metric is reproducible and not silently method-dependent.

**4.8 Actionability-guided reranking (A-Shapley).**
> **Definition 8 (A-Shapley).**
> $$
> \tilde{\phi}_j = (1-\eta)\,\frac{|\phi_j|}{\max_k |\phi_k|} + \eta \cdot \frac{\mathrm{AS}_j}{\max_k \mathrm{AS}_k}, \qquad \eta \in [0,1]
> $$
> interpolating between pure attribution ($\eta = 0$) and pure actionability ($\eta = 1$). Report **Intervention Switch Rate** — the fraction of cases where the top-ranked recommended intervention changes — and the realized outcome improvement.

---

# 5. ANALYSIS (deliberately light — one proposition plus a one-line corollary, no heavy theorem)

**5.1 When do the orderings coincide?**
> **Proposition 1 (Alignment under local linearity).** Let $f$ be locally linear on the feasible region, $f(x + \tau) \approx f(x) + \sum_j w_j \tau_j$, let all factors share a common budget $\tau_j = \tau$, and let $m_j = 1$ for all $j$. Then $|\Delta_j| = |w_j|\tau$ and the Shapley ordering coincides with the intervention-effect ordering, so $\mathrm{AIA} = 1$.

*Proof in Appendix A* (uses linearity + efficiency of the Shapley value on an additive game).

**5.2 Sources of misalignment.** Proposition 1 fails in exactly three ways, and the switches that restore it must be defined so that each targets *one* hypothesis and the three jointly restore *all* of them. The functional-ANOVA (Hoeffding) decomposition of $f$ over the feasible region,

$$
f(x) = f_0 + \sum_j f_j(x_j) + \sum_{j<k} f_{jk}(x_j, x_k) + \cdots,
$$

separates the two ways local linearity can fail — nonlinear main effects (curvature) and cross terms (interaction) — and gives each its own switch.

| | Mechanism | Hypothesis of Proposition 1 that fails | Switch that restores it |
|---|---|---|---|
| **H1** | **Curvature** | local linearity, through nonlinear *main effects* $f_j$ | replace each main effect by its linear part, $f_j(x_j) \to w_j x_j$, retaining all interaction terms |
| **H2** | **Interaction** | local linearity, through *cross terms* $f_{jk}$ and higher | drop all higher-order terms, retaining nonlinear main effects |
| **H3** | **Infeasibility** | common budget $\tau_j = \tau$ and $m_j = 1$ for all $j$ | impose a common budget and restrict to $\{j : m_j = 1\}$ |

**5.2.1 Identifiability of the H1/H2 split — do not skip this.** The two switches are only well defined if the decomposition of $f$ into main effects and interactions is *unique*. It is not, in general. Under the standard Hoeffding decomposition, orthogonality of the components requires the inputs to be independent; under dependent inputs the split is unidentified, and a main-effect function can silently absorb interaction signal. This is not a hypothetical for these datasets — the Beijing air-quality features are strongly dependent (temperature with dew point, PM2.5 with PM10), which is precisely the regime where naive fitting misallocates. Two consequences follow, and both must be built in from the start:

- **Fit one purified surrogate, not two independent ones.** Fit a single GA2M (EBM with pairwise terms) over the feasible region and *purify* it — the canonical procedure that moves interaction mass into main effects until the interaction terms have zero conditional means, yielding a unique decomposition. Derive **both** switches from that one fitted object: H2 drops the purified $f_{jk}$; H1 replaces the purified $f_j$ with their linear parts. Fitting an interaction-free GAM and an interaction-bearing GAM separately would let the two fits disagree about what the main effects *are*, which reintroduces exactly the absorption problem the purification removes. One fit, two projections.
- **Report a concurvity diagnostic and a shape-shift check.** Refit main effects with pairwise terms present and absent, and report how far the main-effect shape functions move. Large movement means absorption is occurring and the H1/H2 attribution is unstable; small movement licenses the split. This costs one extra fit and converts an unfalsifiable assumption into a reported number.

Cite Hooker (2007) on generalized functional ANOVA under dependence and Lengerich et al. (2020) on purifying interaction effects; both are standard and make this a known, solved problem rather than an improvisation. **Disclose the ceiling:** a GA2M captures interactions only up to order two, so any third-order and higher structure falls into the residual rather than into $\psi_{\mathrm{H2}}$.

**Why this definition and not the obvious one.** The intuitive switch for interaction is to replace single-factor interventions with joint perturbation of the top-$k$ set. That is *not* rigorous: Proposition 1's additivity comes from Shapley efficiency over the full player set $N$, so a top-$k$ joint perturbation narrows the gap without provably closing it unless $k = |N|$, leaving H2 weaker than H1 and H3 and quietly converting part of the residual into a fourth, unnamed mechanism. The functional-ANOVA switches avoid this: $\{$H1, H2$\}$ together yield $f_0 + \sum_j w_j x_j$, which is *exactly* local linearity, so the grand coalition restores all three hypotheses by construction rather than by appeal. The top-$k$ coalitional perturbation is retained, but as a robustness check in §7.8 where it carries no load-bearing claim.

**5.2.2 Making the decomposition additive — a Shapley decomposition of the misalignment gap.**

*The problem with the obvious approach.* The natural estimator for each mechanism is a one-at-a-time difference: turn on one switch, see how much AIA improves. This does **not** license "share of the total gap" language, for three reasons, and the paper must not use that language without earning it:
1. AIA is a rank correlation, not a variance. Differences between Spearman $\rho$ values have no additive structure, so three one-at-a-time differences have no reason to sum to the total gap.
2. The switches interact *in their effect on AIA* even though they target orthogonal components of $f$: removing interaction changes which main effects dominate the ranking, so the benefit of linearizing main effects depends on whether cross terms are already gone. One-at-a-time estimates therefore double-count.
3. Sequential ablation is order-dependent: attributing the gap by turning switches on in some order gives different answers for different orders, with no principled choice among them.

*The fix, which the paper is uniquely positioned to make.* Treat the three switches as players in a cooperative game and allocate the gap by the Shapley value — using the same solution concept the paper studies to explain why that concept's attributions misalign. Let $\Sigma = \{\mathrm{H1}, \mathrm{H2}, \mathrm{H3}\}$ and let $\mathrm{AIA}(Q)$ denote alignment measured with the switches in $Q \subseteq \Sigma$ enabled. Define the gap-closing game — written $u$, **not** $v$, since $v(S)$ is already the attribution game's characteristic function over factor-players in §3 and the two must stay visibly distinct:

$$
u(Q) = \mathrm{AIA}(Q) - \mathrm{AIA}(\varnothing), \qquad u(\varnothing) = 0
$$

where $\mathrm{AIA}(\varnothing)$ is the observed alignment. Enabling all three switches restores every hypothesis of Proposition 1 on the surrogate, so $\mathrm{AIA}(\Sigma) = 1$ would hold were the surrogate exact; the measured value is **reported rather than assumed**, and the shortfall is left unattributed pending the diagnostics (see Corollary 1 and the residual below). Allocate by

$$
\psi_c = \sum_{Q \subseteq \Sigma \setminus \{c\}} \frac{|Q|!\,(|\Sigma|-|Q|-1)!}{|\Sigma|!}\,\bigl[u(Q \cup \{c\}) - u(Q)\bigr], \qquad c \in \Sigma .
$$

> **Corollary 1 (Exact additivity of the misalignment decomposition).** By efficiency of the Shapley value, $\psi_{\mathrm{H1}} + \psi_{\mathrm{H2}} + \psi_{\mathrm{H3}} = u(\Sigma) = \mathrm{AIA}(\Sigma) - \mathrm{AIA}(\varnothing)$. The three mechanisms therefore partition the *measured* closed portion of the misalignment gap exactly, and the allocation is order-independent and symmetric. Under the switch definitions of §5.2 the grand coalition satisfies the hypotheses of Proposition 1 on the surrogate, so $\mathrm{AIA}(\Sigma) = 1$ would hold exactly were the surrogate exact; $\mathrm{AIA}(\Sigma)$ is therefore reported empirically and the shortfall $1 - \mathrm{AIA}(\Sigma)$ carried as a separate, **unattributed** residual whose interpretation is settled by the diagnostics below, not by the corollary. *(One line from efficiency; Appendix A.)*

Note what the corollary does and does not need. Efficiency holds for **any** characteristic function, so the additivity, order-independence, and no-double-counting guarantees are unconditional — they do not depend on the switches being well designed. The switch definitions matter only for the *interpretation* of $u(\Sigma)$ as the full gap, which is why $\mathrm{AIA}(\Sigma)$ is measured rather than asserted. The load-bearing part of the argument is therefore insulated from any residual doubt about switch fidelity.

*Cost.* One purified GA2M fit per dataset (§5.2.1), from which all eight configurations are obtained as projections — plus one extra fit for the shape-shift diagnostic. In the static regime this is trivial. In the dynamic regime the surrogate is fit over the **pre-declared candidate set $C$ from §4.7**, not the full item space — the same scoping already used for intervention regret, so no new cost and no new arbitrary parameter. State $C$ once and use it for both.

*Three implementation details that must not be skipped.*
- **Recompute both $\phi$ and $\Delta$ on the switched model.** A configuration $Q$ defines a model variant $f_Q$; attribution *and* intervention effect must both be evaluated on $f_Q$. Attributing $f$ while intervening on the surrogate would break $\mathrm{AIA}(\Sigma) = 1$ for a purely mechanical reason and is the single easiest way to get this experiment wrong.
- **Cardinality matching.** $\mathrm{AIA}$ computed over all factors and over $\{j : m_j = 1\}$ are correlations on sets of different size, whose sampling distributions differ; a raw difference conflates the feasibility effect with a set-size artifact. Hold cardinality fixed by comparing against the mean AIA over random subsets of the unrestricted factor set of matching size, and report that this matching was done.
- **Report the residual, and do not pre-name it.** The shortfall $1 - \mathrm{AIA}(\Sigma)$ is **ambiguous by construction** between three causes: surrogate approximation error, third-order-and-higher interaction beyond the GA2M ceiling, and a genuine fourth mechanism the three switches do not capture. Calling it "approximation error" in the write-up would assume the answer. Report it as an unattributed residual, and adjudicate it with the diagnostics rather than by assertion: surrogate fidelity $R^2$ against $f$ on the feasible region, and the §5.2.1 shape-shift check. **A large residual should prompt inspection of surrogate fidelity before being read as unexplained misalignment** — and if fidelity is high while the residual stays large, say plainly that a mechanism outside H1–H3 is implicated rather than absorbing it into the surrogate.

*Fallback if the decomposition proves noisy.* Report $\mathrm{AIA}(Q)$ for all eight configurations descriptively with confidence intervals and drop the share language entirely. This costs the paper nothing structural, since §7.6 is on the cut list for Appendix A anyway.

**We hypothesize that H3 dominates**, because it is the only mechanism that operates even when the model is perfectly linear and perfectly estimated. That is a prediction, not a result: §7.6 reports $\psi_{\mathrm{H1}}, \psi_{\mathrm{H2}}, \psi_{\mathrm{H3}}$ with CIs and the paper's conclusion follows the data. **If H3 does not dominate, the paper is unharmed** — the contribution is the decomposition and the measurement, not the specific ordering of terms. Do not write "infeasibility is the dominant term" anywhere before §7.6 reports it.

**5.3 Complexity.** Attribution cost is unchanged from the underlying method. ActionShap adds $\mathcal{O}(|N| \cdot R)$ attribution repetitions for stability and $\mathcal{O}(|N|)$ intervention evaluations, both embarrassingly parallel and cheap relative to model training.

---

# 6. EXPERIMENTAL SETUP

**6.1 Datasets and task instantiation.**

| Dataset | Regime | Players | Output $f$ | Source |
|---|---|---|---|---|
| Portuguese Wine Quality | static | 11 physicochemical features | cluster membership probability | thesis Ch. 5 |
| Beijing Multi-Site Air Quality (>380k obs.) | static, hierarchical | pollutant + meteorological features | pollution-regime assignment | thesis Ch. 6 |
| MovieLens-1M | dynamic | users / items / contexts | NDCG@20, coverage, ILD | thesis Ch. 7 |
| Amazon-Book | dynamic | users / items / contexts | NDCG@20, coverage, ILD | thesis Ch. 7 |

**6.2 Modifiability elicitation protocol.**

**Say who the annotators are, in the main text, on first mention.** The elicitation is performed by the three authors, independently and blind to the attribution results, against a rubric fixed *before* any attribution was computed. This is **author elicitation, not blind third-party annotation**, and the paper states so here — not only in the Declarations — because a reader who discovers the discrepancy later will discount everything else. Write it as: *"Modifiability was elicited by the three authors, working independently and blind to model outputs, against a pre-specified rubric."*

Protocol:
- **Rubric (fixed in advance, published verbatim in Appendix C):** directly controllable by the decision-maker = 1.0; indirectly controllable, or controllable at substantial cost or delay = 0.5; observable but immutable = 0.0.
- **Blinding:** annotation is completed and frozen (timestamped in the repository) before any attribution or intervention result is inspected, so $m_j$ cannot be tuned to produce a favourable finding. This is the substantive defence — stronger than annotator independence, and worth stating as such.
- **Agreement:** report Krippendorff's $\alpha$ across the three authors, described accurately as *intra-team* agreement. Do not present it as external inter-rater reliability; it measures rubric clarity, not domain consensus.
- **External check (do this if at all possible):** have two domain-adjacent colleagues outside the author team annotate one dataset — air quality is the natural choice, since its modifiability structure is the least contestable — and report agreement with the author labels. Two external annotators on one of four datasets costs almost nothing and converts the single most predictable objection into a reported number. If it cannot be arranged, say plainly that no external annotation was obtained.
- Disagreements resolved by discussion; final table published (Appendix C).

**Sensitivity to $m_j$ is reported in §7.8**, and the self-elicitation is restated in Limitations (§7.11) and Declarations. Three consistent statements, no fine print.

**6.3 Intervention designs.**
- *Wine:* adjust a physicochemical variable within an achievable production range; measure change in cluster-assignment probability.
- *Air quality:* reduce a pollutant concentration by a policy-realistic percentage; measure regime-reassignment rate. Meteorological variables receive $m_j = 0$ and serve as the negative control — a method that ranks them highly is *faithful but not actionable*, which is exactly the phenomenon under study.
- *Recommendation:* reweight or remove a specific interaction, inject/suppress a context signal, or adjust an item's exposure; measure realized change in NDCG@20, coverage, and ILD.

**6.4 Attribution methods compared.** TreeSHAP (thesis pipeline), KernelSHAP, LIME, permutation importance, gradient×input / integrated gradients (dynamic regime), and **the DyHuCoG attention gate**. The last is a highlight: DyHuCoG contains *both* a Shapley weighting and an attention mechanism, so the paper can ask which of two importance signals *inside the same trained model* is more actionable — a clean, self-contained comparison no prior work has run.

**6.5 Metrics.** All four are defined in §4 and are only *instantiated* here — no metric appears for the first time in the setup section. AIA (Spearman primary, Kendall $\tau$ for robustness) and modifiability-restricted AIA (§4.6); Actionability Score distributions together with the modifiability-held-out control $\mathrm{AS}^{-m}$ (§4.5); top-$k$ intervention precision at $k \in \{1,3,5\}$ and normalized intervention regret (§4.7), with the candidate set size $C$ declared per dataset; stability $s_j$ (§4.4); and, for contrast, standard sufficiency/comprehensiveness so the faithfulness-vs-actionability comparison in §7.3 is like-for-like.

**6.6 Implementation, hardware, reproducibility.** Python 3.8 / PyTorch 2.0.1; single RTX 4090 (thesis hardware table); $R = 5$ attribution repetitions with seeds $\{42,43,44,45,46\}$; mean $\pm$ std and 95% CIs; public code release.

---

# 7. RESULTS AND DISCUSSION

- **7.1 AIA across methods and tasks** — main table (Table 5): AIA and modifiability-restricted AIA per (method, dataset). Expect cooperative attribution highest, LIME lowest, attention weak. *Report both AIA variants in the same table*: the gap between them is the visible signature of the infeasibility term and previews §7.6.
- **7.2 Actionability Score profiles, with the modifiability-held-out control** — per-dataset AS rankings reported next to $\mathrm{AS}^{-m}_j = |\Delta_j| \cdot s_j$. **Do not claim "AS ranks unmodifiable factors last"** — that is forced by $m_j = 0$ (see the circularity guard in §4.5). The defensible claim is the mirror image: *attribution methods rank unmodifiable factors highly*, quantified as the mass of total $|\phi|$ that TreeSHAP/KernelSHAP/LIME/attention place on $m_j = 0$ factors. That single number is the headline for the air-quality case and it is a property of the attribution methods, not of AS.
- **7.3 Faithfulness vs. actionability** — scatter and rank correlation between the two criteria over (method, dataset) pairs. **Pre-commit to the write-up under both outcomes** (see planning notes): divergence establishes the central claim; convergence is reported as a bounded-scope finding — faithfulness proxies for actionability on modifiable factors but not on the full factor set — which keeps §7.2 and §7.5 as the paper's evidentiary core.
- **7.4 Shapley vs. attention inside DyHuCoG** — **its own subsection and its own table (Table 6), never folded into Table 5.** Same trained model, two importance signals, one intervention set: report AIA, top-$k$ precision, regret, and stability for the Shapley weighting and the attention gate side by side, per dataset. This is the comparison no prior work has run and the most quotable result in the paper; give it a dedicated table and Figure 6, and reference it from the abstract.
- **7.5 Intervention decisions: top-$k$ precision and regret** — Table 7. The decision-level counterpart to 7.1: AIA can be moderate while the *chosen* intervention is still wrong, and this subsection is where that is shown. Report P@$k$ for $k \in \{1,3,5\}$ and normalized regret per method and dataset.
- **7.6 Decomposing misalignment** — Table 9: the Shapley allocations $\psi_{\mathrm{H1}}, \psi_{\mathrm{H2}}, \psi_{\mathrm{H3}}$ from §5.2.2 with bootstrap CIs, plus the measured $\mathrm{AIA}(\Sigma)$ and the residual $1 - \mathrm{AIA}(\Sigma)$. These are shares of the *measured* closed gap **by efficiency (Corollary 1)**, not by normalization — say so in the caption, since the additivity of a rank-correlation decomposition is exactly what a methods reviewer will interrogate. Note explicitly that cardinality matching was applied to the H3 switch and that $\mathrm{AIA}(\Sigma)$ is measured rather than assumed. This subsection exists so the dominance claim about infeasibility is a measured result rather than an assertion; state the outcome even if H3 does not dominate.
- **7.7 Actionability-guided reranking** — Intervention Switch Rate and realized-outcome improvement as $\eta$ varies; tie the improvement back to the regret reduction from 7.5 so reranking is evaluated on the same currency as the diagnosis.
- **7.8 Sensitivity, stability, and robustness** *(merged to protect the word budget)* — perturbation budget $\tau$; jittered $m_j$; the AS functional-form alternatives promised in §4.5 (weighted sum, min-based); top-$k$ coalitional perturbation as an alternative interaction probe (reported here, deliberately *not* used as the H2 switch, since it does not provably restore additivity for $k < |N|$); surrogate fidelity $R^2$, concurvity, and the main-effect shape-shift check from §5.2.1; Monte-Carlo samples $M \in \{10,25,50,100\}$ (reuse DyHuCoG convergence framing); repetitions $R$; $s_j$ by method; cross-dataset consistency.
- **7.9 Statistical significance** — paired t-tests + Holm–Bonferroni + Wilcoxon over per-factor / per-user differences, following the thesis protocol exactly.
- **7.10 Case studies** — (a) an air-quality regime where the dominant SHAP driver is meteorological (unactionable) and the top-AS driver is a pollutant (actionable); (b) a single recommendation where reranking changes the recommended intervention.
- **7.11 Limitations** — modifiability is **elicited by the authors**, not by blind external annotators (mitigated by the pre-registered rubric, pre-attribution freezing, the external check on air quality if obtained, and the §7.8 sensitivity analysis) — state this in the same words used in §6.2 and Declarations; the multiplicative AS form is an assumption, tested but not derived; interventions are simulated rather than deployed; tabular and top-$N$ settings only; regret on the recommendation tasks is defined over a candidate set rather than the full feasible space; no human-subject validation of perceived usefulness.

---

# 8. CONCLUSION AND FUTURE WORK
Recap the five contributions; restate headline numbers; argue that XAI evaluation should add a feasibility-constrained axis alongside faithfulness. Future work: deployed/online interventions, causal-graph-aware feasibility constraints, human-subject validation of the AS ranking, extension to sequential and multimodal settings. Tie to EU AI Act oversight requirements (consistent with the thesis framing).

---

# DECLARATIONS (required by Discover AI)
- **Funding** — state grant/none.
- **Competing interests** — "The authors declare no competing interests."
- **Data availability** — all four datasets are public (UCI Wine Quality, UCI Beijing Multi-Site Air Quality, MovieLens-1M, Amazon-Book); modifiability tables and intervention protocols released with the code.
- **Code availability** — repository link (GitHub/Zenodo DOI).
- **Author contributions** — reuse DyHuCoG CRediT split (Louhichi: conceptualization/methodology/software/writing; Nesmaoui: software/data; Lazaar: supervision/analysis).
- **Ethics approval** — not applicable; no human participants. The modifiability elicitation was performed by the authors against a pre-specified rubric (§6.2) and is reported as a methodological protocol, not as a study of human annotators. *Use wording consistent with §6.2 and §7.11 — all three must describe the same procedure.*

# APPENDICES
- **A** — proof of Proposition 1; one-line proof of Corollary 1 from Shapley efficiency, with the switch game $u(Q)$ stated explicitly and all eight configurations tabulated; the functional-ANOVA basis for the H1/H2 split, the identifiability argument under dependent inputs, the purification procedure, and the argument that $\{$H1, H2$\}$ jointly yield local linearity; surrogate specifications, fidelity, concurvity, and shape-shift diagnostics; the cardinality-matching procedure for the H3 switch; derivations of AS and AIA.
- **B** — statistical methodology (paired t-tests, Holm–Bonferroni, Wilcoxon, effect sizes) — adapt thesis Appendix F.
- **C** — full modifiability tables per dataset with annotator rubric and agreement statistics.
- **D** — intervention protocols, perturbation budgets, hyperparameters.

# NOTATION LIST
Reuse the thesis notation-table format, extended with $m_j$, $\tau_j$, $\Delta_j$, $s_j$, $\mathrm{AS}_j$, $\mathrm{AS}^{-m}_j$, $\mathrm{AIA}$, $\eta$, $R$, $C$, and the switch-game symbols $\Sigma$, $Q$, $u(Q)$, $\psi_c$. Flag in the notation table that the document runs **two cooperative games at different levels** and that their symbols must never be mixed: $v(S)$ over factor-players with allocation $\phi_j$, and $u(Q)$ over mechanism-switches with allocation $\psi_c$. This is the most likely source of reviewer confusion in the paper — give it its own row.

---

# PLANNED FIGURES & TABLES
**Figures:** (1) ActionShap workflow; (2) faithfulness vs. actionability conceptual contrast; (3) AIA by method and dataset; (4) AS vs. $|\phi|$ scatter with the unactionable-but-high-attribution quadrant highlighted; (5) air-quality case study (SHAP ranking vs. AS ranking); (6) Shapley vs. attention in DyHuCoG; (7) reranking switch rate vs. $\eta$; (8) sensitivity to $\tau$ and $m$; (9) per-factor difference distribution + Q–Q.

**Tables:** (1) comparison/differentiation; (2) notation; (3) dataset and task instantiation; (4) modifiability summary with agreement statistics; (5) main AIA results, unrestricted and modifiability-restricted; **(6) Shapley vs. attention inside DyHuCoG — dedicated, not merged into Table 5**; (7) Actionability Score rankings with the $\mathrm{AS}^{-m}$ control column; (8) top-$k$ intervention precision and regret; (9) misalignment decomposition — Shapley allocations $\psi_{\mathrm{H1}}/\psi_{\mathrm{H2}}/\psi_{\mathrm{H3}}$ plus residual; (10) reranking outcomes; (11) stability by method; (12) paired significance.

*If the table count needs cutting, merge (11) into (5) as extra columns and move (9) to Appendix A — never cut (6) or (8).*

---

# PLANNING NOTES (NOT part of the manuscript)

**The gap, in the thesis's own words.** The thesis is subtitled *"A Shapley Framework for Actionable Insight"* and gives a precise Definition 1.1 of actionability, yet states: *"actionable insight is used primarily as a framing concept rather than as a separately measured endpoint: the empirical chapters illustrate it through domain-grounded explanatory cases rather than through an independent user study or intervention experiment."* Chapter 8.3 adds that the thesis *"does not provide a dedicated human-subject study of actionability or contestability."* Separately, Chapters 5, 6, and 7 each map to a published paper, but the thesis-level synthesis answering **RQ5** was never published standalone. ActionShap closes both gaps at once — and does so *without* human subjects, by substituting intervention experiments for user studies.

**Why acceptance risk is low.**
- No new theorem (one proposition with a short proof).
- No new datasets, no new models — everything already implemented and validated.
- No GPU cost beyond the existing RTX 4090; interventions are cheap perturbation loops.
- No ethics approval or participant recruitment.
- Reviewer-friendly framing: "we propose a new evaluation axis and show it changes conclusions" is a well-established, well-received journal contribution type.

**Main reviewer objection to pre-empt.** *"Modifiability is subjective."* Answer it in four places: the elicitation protocol with a pre-registered rubric, pre-attribution freezing, and intra-team agreement (§6.2); the optional external annotation check on the air-quality dataset (§6.2); the published per-dataset tables (Appendix C); and the sensitivity analysis showing conclusions are stable under jittered $m_j$ (§7.8). Restate it in Limitations (§7.11). Pre-empting beats defending — but only if the four statements agree with each other word for word.

**Pre-commit to the write-up before the data arrives (§7.3).** The framing must survive a null result. Two outcomes, both publishable, decided *now* rather than during results-writing:
- *Faithfulness and actionability diverge* (expected): the central claim lands as written; lead with the method-ranking disagreement.
- *They converge*: retitle §7.3 to "When faithfulness suffices" and report the bounded-scope finding — faithfulness is an adequate proxy **on modifiable factors**, and fails only on the full factor set, where high-attribution unmodifiable factors dominate. In that case the paper's weight shifts to §7.2 (attribution mass on $m_j=0$ factors), §7.5 (regret), and §7.7 (reranking), all of which stand independently of whether the two criteria correlate. Write the abstract so that swapping one sentence covers both cases.

**Self-citation.** Five self-cites in an explicitly declared capstone paper is defensible, and the differentiation table (2.6) already contrasts against external lines of work — recourse, faithfulness metrics, counterfactual explanation — not against the author's own papers, so the "differing from ourselves" failure mode does not apply. The one thing to avoid is treating self-citation as a box to tick: cite each prior work where it does argumentative work (§1.4, §2.1, §3, §6.1) and nowhere else. Target self-cites below roughly 15% of the reference list, which means the bibliography needs ~35+ external references.

**Word budget.** Eleven result subsections in 8,000–10,000 words leaves ~150–250 words each, with no slack for a messy result. Cut order if the draft runs long, in this sequence: merge §7.9 into the table captions; move §7.6's full decomposition to Appendix A and keep a three-sentence summary; drop case study (b) from §7.10. **Never cut §7.4 or §7.5** — they carry the paper's most distinctive evidence.

**Execution timeline (≈3 months).**

| Phase | Weeks | Output |
|---|---|---|
| Related work + differentiation table | 1 | §2 |
| Formalize AS/AIA + prove Proposition 1 | 1–2 | §4–5 + Appendix A |
| Modifiability elicitation + intervention protocols | 2–3 | §6.2–6.3 + Appendix C |
| Implement intervention harness over existing codebases | 3–5 | working code |
| Run static-regime experiments (Wine, Beijing) | 5–7 | §7.1–7.3 partial |
| Run dynamic-regime experiments (ML-1M, Amazon-Book) | 6–8 | §7.1–7.5 |
| Misalignment decomposition + reranking + sensitivity + significance | 8–10 | §7.6–7.9 |
| Case studies, figures, full draft | 10–12 | manuscript |
| Internal review + submit | 12–13 | Discover AI submission |

**Key prior art to cite.** Shapley (1953); Lundberg & Lee (2017); Ribeiro et al. (2016, LIME); Wachter et al. (2017, counterfactuals); Ustun et al. (2019, actionable recourse); Karimi et al. (recourse under causal constraints); Janzing et al. (2020, causal feature relevance); Hooker (2007, generalized functional ANOVA under dependent inputs); Lengerich et al. (2020, purifying interaction effects with the functional ANOVA); Lou et al. (2013, GA2M / intelligible models with pairwise interactions); DeYoung et al. (ERASER, faithfulness); Adebayo et al. (sanity checks); Jain & Wallace (attention is not explanation); your lineage — clustering-SHAP (2023), Game-Theory-Meets-XAI (2025), GNN+Shapley hierarchical recommendation (2025), Real-time Shapley adjustment (2025), DyHuCoG (2026).

**Submission checklist (Discover AI).**
- [ ] Abstract ≤ ~250 words, no citations.
- [ ] All declarations present.
- [ ] Data + code availability with working links (Zenodo DOI for the modifiability tables).
- [ ] Figures ≥ 300 dpi; vector where possible.
- [ ] Reproducibility: seeds, hyperparameters, hardware reported.
- [ ] Self-citation lineage cited where it does argumentative work; self-cites under ~15% of the reference list.
- [ ] Annotator identity stated identically in §6.2, §7.11, and Declarations.
- [ ] No AS-based claim that is true by construction (circularity guard, §4.5).
- [ ] No claim that infeasibility dominates stated anywhere before §7.6 reports it.
- [ ] Every "share of the gap" statement traceable to Corollary 1 (efficiency), never to post-hoc normalization; residual reported separately.
- [ ] Cardinality matching applied and disclosed wherever AIA is compared across factor sets of different size.
- [ ] $\mathrm{AIA}(\Sigma) = 1$ reported empirically, never asserted; residual left **unattributed** and adjudicated by surrogate fidelity $R^2$ and the shape-shift check, never pre-labelled as approximation error.
- [ ] H1 and H2 derived as two projections of **one purified GA2M**, not from two independently fitted surrogates; purification and concurvity diagnostics reported.
- [ ] GA2M pairwise ceiling disclosed: third-order and higher interaction falls into the residual, not into $\psi_{\mathrm{H2}}$.
- [ ] Each switch targets exactly one hypothesis of Proposition 1, and H1+H2 jointly yield local linearity — verify against the Appendix A configuration table before submission.
- [ ] Both $\phi$ and $\Delta$ recomputed on the switched model $f_S$ in every one of the eight configurations.
- [ ] Statistical tests with multiple-comparison correction.
- [ ] Check Morocco APC discount eligibility (Research4Life / Springer waiver policy) before acceptance.
