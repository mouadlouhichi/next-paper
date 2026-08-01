# SignalShap — Full Paper Structure, TOC & Embedded Content

**Target journal:** *Discover Artificial Intelligence* (Springer Nature, open access, Q1 — Information Systems)
**Article type:** Research article
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco
**Corresponding author:** mouad_louhichi@um5.ac.ma

**Structure:** six-section layout of *Game Theory Meets Explainable AI* (IJACSA 2025), not the nine-section DyHuCoG layout.
**Target:** ≈ 8,500 words, 7 figures, 8 tables.

> **Why this is the easy-to-accept paper.** It needs no new architecture, no new dataset, and no heavy theorem. Every base component is an off-the-shelf, well-documented recommender that runs on CPU in minutes. The cooperative game has five players, so the Shapley values are **computed exactly** — there is no sampling error for a reviewer to attack, and no approximation-quality section to defend. The single most expensive experiment is 32 refits of a linear fusion layer over precomputed score matrices, which is seconds of compute. The novelty is in *what the players are*, not in how hard the computation is.

> **Critical dependency note (read before starting).** This paper is deliberately built to be **independent of DyHuCoG**. An extraction audit of the DyHuCoG manuscript found 63 gaps that make faithful reimplementation impossible: the hypergraph construction is never defined, two load-bearing similarity functions are unspecified, equations 11–12 are dimensionally inconsistent and mutually contradictory, equations 6–7 give conflicting propagation rules, and the data protocol is self-contradictory. **The `FairShap` blueprint currently assumes it can reuse the DyHuCoG codebase — that assumption is no longer safe and FairShap needs re-planning.** SignalShap avoids the problem entirely by using only standard, independently documented components.

---

## Working Title (primary + alternates)

- **Primary (selected):** *Game Theory Meets Recommendation: Exact Shapley Credit Assignment over Collaborative, Content, and Contextual Signals*
- Alt 1: *SignalShap: Exact Cooperative Attribution of Signal Sources in Hybrid Recommender Systems*
- Alt 2: *Which Signal Recommends? Shapley Decomposition of Ranking Quality Across Hybrid Recommendation Sources*
- Alt 3: *From Feature Attribution to Source Attribution: A Cooperative-Game View of Hybrid Recommenders*

*(The primary deliberately echoes the IJACSA title so the venue reads this as a continuation of that line. **SignalShap** is retained throughout as the name of the method and the codebase, so the two are not in conflict — the title names the theme, the method keeps its own identity.)*

## One-paragraph thesis (the spine)

Modern recommenders are **hybrids**: they fuse a collaborative signal, a content signal, a popularity prior, a recency signal, and a sequential signal into a single ranking score. Practitioners know the hybrid works, but not **which source is actually earning the accuracy** — and the field's standard answer, ablation or leave-one-out, is provably wrong whenever two sources are redundant, which in recommendation they almost always are. We recast hybrid recommendation as a **cooperative game whose players are the signal sources** and whose value is ranking quality, and compute the **exact** Shapley value over the resulting small player set. Because the value function is a per-user mean and the Shapley operator is linear, the global attribution decomposes *for free* into exact per-user attributions, which we aggregate into **segment-level source profiles**. These profiles reveal that a single global answer is a fiction: the sources that serve heavy users are not the sources that serve cold users. We close the loop by using the segment profiles to set **segment-adaptive fusion weights**, which improves NDCG at zero additional inference cost.

## Research questions of *this paper* (and their link to the thesis)

| RQ | Question | Thesis link |
|---|---|---|
| **RQ1** | Can hybrid recommendation be posed as a cooperative game whose players are signal sources, such that the Shapley allocation exactly decomposes ranking-quality uplift? | Extends the thesis' Shapley-for-black-box spine from *features* (clustering papers) to *architectural sources* |
| **RQ2** | Does source attribution differ from what leave-one-out ablation reports, and is the difference explained by redundancy? | Motivates the axiomatic machinery the thesis argues for throughout |
| **RQ3** | Is source attribution homogeneous across the user population, or does a global number conceal opposing segment-level stories? | Reconnects to the author's clustering line (IJACSA 2025) — segments are the unit of explanation |
| **RQ4** | Can attribution be turned back into an improvement, i.e. does segment-adaptive fusion beat global fusion? | The attribution→intervention loop that recurs across the thesis |

---

# TABLE OF CONTENTS

```
Abstract / Keywords
1. Introduction
   1.1 Background and motivation
   1.2 The source-attribution gap in hybrid recommendation
   1.3 Why leave-one-out fails under redundancy
   1.4 Contributions
   1.5 Organization
2. Literature Review
   2.1 Hybrid recommendation and score-level fusion
   2.2 Explainability in recommender systems
   2.3 Shapley values in machine learning and XAI
   2.4 Cooperative games over model components, data, and features
   2.5 User segmentation and heterogeneous explanation
   2.6 Positioning and differentiation (comparison table)
3. Methodology
   3.1 Notation and problem formulation
   3.2 The five signal sources
   3.3 The score-level fusion recommender
   3.4 The source-attribution game
   3.5 Exact Shapley computation and its cost
   3.6 Per-user and per-segment decomposition
   3.7 Segment-adaptive fusion (SignalShap-Fuse)
   3.8 Comparative analysis against alternative attributions
   3.9 Theoretical justification
   3.10 Complexity analysis
   3.11 Practical implementation
4. Experimental Results
   4.1 Datasets and preprocessing
   4.2 Protocol, metrics, and baselines
   4.3 RQ1 — exact source attribution
   4.4 RQ2 — Shapley versus leave-one-out under redundancy
   4.5 RQ3 — segment heterogeneity
   4.6 RQ4 — segment-adaptive fusion
   4.7 Sensitivity, stability, and ablations
   4.8 Statistical significance
5. Discussion and Broader Implications
   5.1 What the attributions mean for system design
   5.2 Cold start as an attribution phenomenon
   5.3 Relation to feature-level explanation
   5.4 Limitations and threats to validity
6. Conclusion and Future Work
Declarations
Appendices
```

---

# ABSTRACT (draft, ~215 words)

Hybrid recommender systems combine collaborative, content, popularity, temporal, and sequential signals into a single ranking, yet practitioners have no principled account of which source earns the resulting accuracy. The prevailing answer — remove one source and measure the drop — is provably misleading whenever sources are redundant, a condition that is the norm rather than the exception in recommendation, because two mutually substitutable sources each receive zero credit. We recast hybrid recommendation as a cooperative game in which the players are the **signal sources** and the characteristic function is ranking quality, and we compute the **exact** Shapley allocation. The player set is small by construction, so no sampling approximation is required, and because the base scorers are trained once and only a lightweight fusion layer is refitted per coalition, the full game costs seconds. We prove that efficiency yields an exact additive decomposition of ranking-quality uplift, that leave-one-out collapses to zero under perfect redundancy where Shapley splits credit evenly, and that linearity makes per-user attribution available at no additional cost. Aggregating per-user attributions into behavioural segments shows that global source credit conceals opposing segment-level stories. Exploiting this, segment-adaptive fusion weights improve NDCG@10 over a globally tuned hybrid at no additional inference cost. Experiments span a dense and a sparse benchmark, MovieLens-1M and Amazon-Book.

**Keywords:** Shapley value; cooperative game theory; explainable AI; recommender systems; hybrid recommendation; credit assignment; user segmentation

---

# 1. INTRODUCTION

## 1.1 Background and motivation

Open flat, in the IJACSA voice. Deployed recommenders are almost never a single model — they are fusions. State the practical consequence: a team maintaining five signal pipelines has a fixed engineering budget and no defensible way to decide which pipeline deserves it. Ground it with the maintenance framing: each source carries a cost (a content pipeline needs metadata ingestion and cleaning; a sequential model needs session logging and low-latency state; a collaborative model needs periodic retraining), and cost is only justifiable against credit.

## 1.2 The source-attribution gap in hybrid recommendation

Explainable recommendation has concentrated overwhelmingly on **why this item was shown to this user** — item-level, feature-level, or path-level explanation aimed at the end user. Almost nothing addresses **which part of my system is producing the value**, aimed at the system owner. Name this second audience explicitly and claim it; it is under-served and the claim is easy to defend.

## 1.3 Why leave-one-out fails under redundancy

This is the motivating engine of the paper and should be concrete before any formalism. Give the two-line intuition: if the content signal and the collaborative signal happen to rank the same items highly for a given user, deleting either one alone changes nothing, so ablation reports that *neither matters*, while deleting both is catastrophic. Ablation therefore fails to sum to the total and can report zero for a source that is genuinely load-bearing. Forward-reference Proposition 2, which turns this intuition into a statement.

## 1.4 Contributions

Numbered, bold lead-ins, IJACSA style. Four substantive plus one artifact.

1. **A cooperative game over signal sources.** We formalize hybrid recommendation as a transferable-utility game whose players are signal sources rather than features or items, and whose characteristic function is ranking quality relative to a null ranker. Because the player set is small by construction, the Shapley allocation is computed exactly, with no Monte-Carlo estimation and no approximation error to bound.

2. **Three short results that make the allocation usable.** Efficiency yields an exact additive decomposition of ranking-quality uplift into per-source shares (Proposition 1). Under perfect redundancy, leave-one-out assigns zero to both members of a redundant pair while Shapley splits the credit evenly, which formally identifies the failure mode motivating the work (Proposition 2). Linearity of the Shapley operator over a per-user-mean value function makes per-user attribution available at no computational cost beyond the global game (Proposition 3).

3. **Segment-level source profiles.** Aggregating per-user attributions over behavioural segments shows that global source credit is an average over heterogeneous and sometimes opposing populations. We quantify the heterogeneity with a permutation test and show which segments invert the global ordering.

4. **Segment-adaptive fusion.** We use the segment profiles to set per-segment fusion weights, converting an explanation into a measurable improvement in NDCG@10 at no additional inference cost, since segment assignment is a table lookup.

5. **A reproducible artifact.** All five sources are standard published methods with public implementations; base scorers are trained once and cached; the entire game is reproducible on a laptop CPU. Code and configurations are released.

## 1.5 Organization

One short roadmap paragraph naming sections 2–6.

---

# 2. LITERATURE REVIEW

Thematic subsections, each closing with a one-sentence gap statement. Target ~1,400 words.

## 2.1 Hybrid recommendation and score-level fusion
Burke's hybridization taxonomy; weighted, switching, cascade, feature-combination designs. Establish that score-level (weighted) fusion is both the most common production pattern and the one that makes source-level ablation well-defined — this is why we adopt it. *Gap: the literature optimizes fusion weights but never asks what each fused source contributes.*

## 2.2 Explainability in recommender systems
Survey the end-user-facing line: post-hoc item explanation, knowledge-graph paths, counterfactual explanation, review-based justification. *Gap: the system-owner audience is unaddressed.*

## 2.3 Shapley values in machine learning and XAI
SHAP, KernelSHAP, TreeSHAP; the axiomatic case for Shapley; the computational objection and the sampling estimators built to answer it. Make the point that the computational objection is a consequence of treating *features* as players, and evaporates when the players are *sources*. Cite the author's prior clustering work here as the feature-level antecedent.

## 2.4 Cooperative games over model components, data, and features
Data Shapley, feature-group Shapley, ensemble-member attribution, Owen and Myerson values for structured player sets. Position source attribution as a member of this family that has not yet been instantiated for recommendation. Cite MHyperShap and DyHuCoG lineage here as the author's cooperative-game antecedents.

## 2.5 User segmentation and heterogeneous explanation
Clustering for user modelling; the observation that global explanations average over populations. Cite the author's own clustering-plus-Shapley work as direct precedent for treating segments as the explanatory unit.

**Cite the group's *Explanation Drift* paper here and in §2.3.** Nesmaoui, Louhichi & Lazaar define explanation drift and the TAD/ESS/EHL metrics for Shapley attributions in dynamic recommenders. The relationship to SignalShap is clean and worth one explicit sentence: that paper asks whether attributions are stable **across time** for a fixed unit of attribution, whereas this one asks whether they are stable **across the population** for a fixed point in time. The two are orthogonal axes on the same object, which positions SignalShap as a companion rather than a competitor and gives the group a coherent two-paper story. It also supplies a natural future-work sentence — source attribution measured over a temporal stream.

## 2.6 Positioning and differentiation
**Table 1.** Rows: representative prior methods. Columns: *unit of attribution* (item / feature / data point / source), *axiomatic guarantee*, *exact or approximate*, *heterogeneity-aware*, *closes the loop to improvement*. SignalShap is the only row with source-level, exact, heterogeneity-aware, and loop-closing all marked.

---

# 3. METHODOLOGY

## 3.1 Notation and problem formulation

| Symbol | Meaning |
|---|---|
| $\mathcal{U}, \mathcal{I}$ | user and item sets |
| $\mathcal{G} = \{g_1,\dots,g_K\}$ | set of signal sources, the players; $K=5$ |
| $s_g(u,i)$ | score assigned to $(u,i)$ by source $g$ |
| $C \subseteq \mathcal{G}$ | a coalition of sources |
| $f_{\theta}^{C}$ | fusion model refitted using only the sources in $C$ |
| $v(C)$ | characteristic function: ranking quality of $f_{\theta}^{C}$ |
| $\varphi_g$ | Shapley value of source $g$ |
| $\varphi_g(u)$ | per-user Shapley value of source $g$ for user $u$ |
| $\mathcal{S}_1,\dots,\mathcal{S}_M$ | user segments |
| $\pi$ | null ranker (uniform random, fixed seed) |

Keep the notation table in the manuscript — the IJACSA paper has one and reviewers in this venue expect it.

## 3.2 The five signal sources

Each source is a standard, independently published method. Nothing here is novel and that is deliberate: the contribution is the game, not the players.

| $g$ | Source | Concrete instantiation | Cost it imposes in production |
|---|---|---|---|
| CF | Collaborative | BPR-MF (implicit feedback, 64 factors) | interaction store, periodic retraining |
| CB | Content | TF-IDF or sentence-embedding similarity over item metadata | metadata ingestion and cleaning |
| POP | Popularity | log interaction count, time-windowed | negligible |
| REC | Recency | exponentially time-decayed interaction score | event timestamps |
| SEQ | Sequential | first-order Markov item-transition score | session logging, low-latency state |

The final column is not decoration — it is what makes the attribution *decision-relevant*, and it should be referenced again in the discussion.

**On the deliberate exclusion of a demographic source.** An earlier draft included a sixth, demographic player. It is dropped, for two reasons worth stating in the paper in one sentence. First, demographic attributes are present in MovieLens but absent from Amazon, so including it would force $K$ to differ across datasets and make the two attribution vectors non-comparable — the cleanliness of a uniform player set across every experiment is worth more than the extra player. Second, a demographic source invites an ethics discussion that this paper is not otherwise equipped to have. The privacy-versus-credit argument is a good paper, but it is a *different* paper; note it in future work instead.

## 3.3 The score-level fusion recommender

Candidate generation retrieves the top-$N$ items per user by a union over sources ($N = 200$). Each source scores the candidates. Scores are z-normalized **per user and per source**, i.e. across that user's candidate list within a single source, to make sources commensurable. Fusion is a **regularized logistic ranker** over the $K$ normalized scores, trained with BPR-style pairwise sampling.

> **Pin the degenerate case before the build starts.** If a source returns the same value for every item in a user's candidate list, then $\sigma_{u,g} = 0$ and the normalization divides by zero. This is not hypothetical: content similarity returns all-zeros for a user whose candidates share no metadata with anything in their history, which is a routine occurrence for cold users on the sparse dataset, and a cold-start collaborative model can return a near-constant column for the same users. Left unguarded it produces `NaN` or `inf` that propagates into fusion and surfaces as a crash or, worse, as silently corrupted attributions mid-run. **Define the fallback explicitly: set $z_{u,g,i} = 0$ for all $i$ when $\sigma_{u,g} = 0$**, which is the semantically right answer since a constant column carries no ranking information for that user, and prefer it to the usual $\sigma + \epsilon$ trick, which instead amplifies floating-point noise into a spurious ranking signal. Log how often the fallback fires per source and per dataset, and report the rate — it is a one-line diagnostic that doubles as evidence about which sources are actually informative for cold users, which is precisely the §4.5 story.

The critical design property: **base scorers are trained exactly once**. A coalition $C$ is realized by masking the score columns outside $C$ and refitting only the fusion layer. This is what makes the exact game cheap, and it should be stated in the methodology and again in the complexity paragraph.

State the modelling assumption honestly: masking a source removes its *contribution to fusion*, not its influence on candidate generation. Two options, and the paper should pre-commit to one — either fix the candidate set from the grand coalition for all coalitions (clean, slightly favourable to the grand coalition) or regenerate candidates per coalition (more faithful, more expensive, and makes $v$ non-monotone). **Recommendation: fix the candidate set, and report the regenerate-per-coalition variant as a robustness check in the appendix.**

## 3.4 The source-attribution game

**Definition 1 (Source-attribution game).** The pair $(\mathcal{G}, v)$ with
$$v(C) = \mathrm{NDCG@10}\!\left(f_{\theta}^{C}\right) - \mathrm{NDCG@10}(\pi), \qquad v(\emptyset) = 0 .$$

Subtracting the null ranker is what makes $v(\emptyset)=0$ and turns the grand-coalition value into an *uplift*, which is the quantity a system owner actually cares about. Fix $\pi$'s seed and report its NDCG so the subtraction is auditable.

Close one small formal gap while you are here: $f_\theta^{\emptyset}$ — a fusion over no sources — is otherwise undefined, so **define the empty coalition to be the null ranker**, $f_\theta^{\emptyset} \equiv \pi$. Then $v(\emptyset)=0$ follows rather than being imposed by fiat, which is a one-line fix to a thing a careful reviewer will notice.

**Definition 2 (Source Shapley value).**
$$\varphi_g = \sum_{C \subseteq \mathcal{G}\setminus\{g\}} \frac{|C|!\,(K-|C|-1)!}{K!}\Big[v(C \cup \{g\}) - v(C)\Big].$$

**Definition 3 (Normalized source share).** $\bar\varphi_g = \varphi_g / \sum_{h}\varphi_h$, reported as a percentage. Flag that shares are only interpretable as percentages when all $\varphi_g \ge 0$; report raw values alongside, and treat a negative $\varphi_g$ as a finding (a source that actively harms ranking in expectation) rather than hiding it.

## 3.5 Exact Shapley computation and its cost

$2^K = 32$ coalitions, each requiring one fusion refit over precomputed scores. Give the wall-clock in the results and make the contrast explicit — feature-level SHAP on the same system would need sampling, source-level does not. Note for the reader that exactness is a property of the *design choice to make sources the players*, not a lucky accident of these datasets: any system with a handful of fused sources inherits it.

## 3.6 Per-user and per-segment decomposition

Define $v_u(C)$ as the same quantity restricted to user $u$, so that $v(C) = \frac{1}{|\mathcal{U}|}\sum_u v_u(C)$. Proposition 3 then gives $\varphi_g = \frac{1}{|\mathcal{U}|}\sum_u \varphi_g(u)$ with **no extra coalition evaluations** — the per-user values fall out of the same 32 refits. This is the cheapest contribution in the paper and should be sold as such.

> **Granularity warning — per-user values are exact but coarse.** Under the leave-one-out protocol of §4.1 each user has exactly **one** held-out test item, so $v_u(C)$ takes one of only about eleven values: $1/\log_2(r+1)$ when the item lands at rank $r \le 10$, and zero otherwise. Whether a user's item falls at rank 10 or 11 flips $v_u$ between $0.289$ and $0$. The per-user values are therefore *exact* with respect to the game as defined, but a **high-variance estimate** of that user's underlying source dependence. Consequence: **report segment aggregates as the primary result** and treat individual-user attributions as an intermediate quantity, since averaging over a segment is what restores stability. Do not present per-user attribution as if it were a deployable per-user explanation.

Segments: define both (a) **behavioural segments** from activity level and profile statistics, and (b) **attribution segments** from clustering the per-user $\varphi(u)$ vectors directly. If the two disagree, attribution is not reducible to behaviour, which is a publishable observation on its own.

> **Specify the null concretely — this is the paper's most fragile claim.** "Attribution segments differ from behavioural segments" is exactly the kind of finding that can be a clustering of noise, given the coarseness above, so name the test rather than gesturing at one. Use the **same permutation test already planned for §4.5** so the paper has one null procedure rather than two: cluster the observed $\varphi(u)$ vectors, record the between-cluster variance of the source shares, then recompute that statistic many times with the segment labels shuffled, and report the observed value against the shuffled distribution with a $p$-value. Reusing the §4.5 machinery costs nothing and keeps the methods section short. If you want a second check, **cluster stability across the five seeds** measured by adjusted Rand index between segmentations fitted on different seeds is the natural companion: high agreement means the clusters are a property of the data rather than of one training run. Report the Rand index; if it is low, the attribution-segment claim should be dropped rather than defended.

## 3.7 Segment-adaptive fusion (SignalShap-Fuse)

Fit fusion weights per segment rather than globally, with the segment profile as initialization and shrinkage toward the global weights controlled by $\lambda$:
$$\theta_m = (1-\lambda)\,\theta_{\text{global}} + \lambda\,\theta_{\text{segment}(m)} .$$
Sweep $\lambda \in [0,1]$ and report the curve; $\lambda = 0$ recovers the global hybrid, so the baseline is nested and the comparison is honest. Inference cost is unchanged because segment assignment is a lookup.

Guard against the obvious reviewer objection: **the segments must be fitted on training users only**, and the improvement must be measured on held-out users assigned to segments by the frozen segmenter. Say this explicitly in the methodology, not just the appendix.

## 3.8 Comparative analysis against alternative attributions

Closing subsection in the IJACSA pattern. Compare on: leave-one-out, forward selection, permutation importance over source score columns, Monte-Carlo Shapley, and feature-level SHAP on the fusion model. For each, state what it measures, what axiom it violates, and its cost. The table should make the redundancy failure visible as an empirical column, not just a theoretical claim.

## 3.9 Theoretical justification

**Calibration note.** This subsection was deliberately cut back to match the complexity level of the group's *Explanation Drift* paper (Nesmaoui, Louhichi & Lazaar), which carries **no propositions at all** — it defines its metrics, states that Shapley values are justified by efficiency, symmetry, additivity and consistency, and moves on. SignalShap needs slightly more than that because the leave-one-out contrast *is* the argument, but only slightly. Two short propositions and one remark, proofs in Appendix A, each a few lines. An earlier draft carried four propositions including an $\varepsilon$-near-redundancy bound with a convex-combination argument; that machinery is **cut** as disproportionate to the venue and to the rest of the group's published work. What it was defending against is handled in §4.4 by wording the empirical claim honestly instead of by proving a theorem.

**Proposition 1 (Exact additive decomposition).** $\sum_{g}\varphi_g = v(\mathcal{G}) = \mathrm{NDCG@10}(f^{\mathcal{G}}) - \mathrm{NDCG@10}(\pi)$. Immediate from efficiency; the value is interpretive, not mathematical — every point of uplift is assigned to exactly one source.

**Proposition 2 (Redundancy collapse of leave-one-out).** Let $g_1, g_2$ be perfectly redundant, i.e. $v(C \cup \{g_1\}) = v(C \cup \{g_2\}) = v(C \cup \{g_1, g_2\})$ for all $C \subseteq \mathcal{G}\setminus\{g_1,g_2\}$. Then $\mathrm{LOO}(g_1) = \mathrm{LOO}(g_2) = 0$, while $\varphi_{g_1} = \varphi_{g_2} = \tfrac{1}{2}\big[v(\mathcal{G}) - v(\mathcal{G}\setminus\{g_1,g_2\})\big]$. *This is the theoretical centre of the paper* — it converts the motivation in §1.3 into a proved statement, and the proof is a few lines by symmetry.

**Proposition 3 (Free per-user decomposition).** Since $v = \frac{1}{|\mathcal{U}|}\sum_u v_u$ and the Shapley operator is linear in the characteristic function, $\varphi_g = \frac{1}{|\mathcal{U}|}\sum_u \varphi_g(u)$. One line to prove, and it is what makes the segment analysis in §4.5 free rather than a second experiment.

**Remark 1 (Scale invariance).** Because each source's scores are z-normalized per user before fusion, replacing a source's raw scores by $ax+b$ with $a>0$ leaves the normalized scores pointwise unchanged, hence leaves every $\varphi_g$ unchanged. This answers the natural question of whether the shares are an artifact of the units each source happens to report in — raw probabilities against log-odds against arbitrary similarity scales.

> **Do not extend Remark 1 to nonlinear monotone maps.** z-normalization is a location-scale operation, so it cancels affine maps exactly but does not undo a nonlinear reshaping: log-transforming raw scores and then z-normalizing changes the spacing and tail behaviour even though the within-source item order is identical, and because fusion is a logistic ranker over normalized **values** rather than ranks, the fitted $\theta^C$ and therefore $v(C)$ generically shift. Keep the remark affine, and let §4.7 *measure* the nonlinear case instead of claiming it. A one-sentence remark that is true beats a corollary a reviewer can falsify in one experiment.

## 3.10 Complexity analysis

The *Explanation Drift* paper devotes a full subsection to complexity and derives its costs step by step; match that convention, because here the analysis is a **selling point rather than an apology**. That paper concludes that exact Shapley at $O(2^{|F|})$ is "computationally infeasible for large feature sets" and therefore adopts Monte-Carlo sampling with $M$ permutations, at $O(|\mathcal{B}|M|F|)$ per batch. SignalShap is the direct counterpoint: by making sources rather than features the players, $2^K = 32$ and the exact computation is simply affordable.

Derive it in the same style. Let $B$ be the cost of training one base scorer, $\Phi$ the cost of one fusion refit over cached score columns, and $P = |\mathcal{U}| \cdot N$ the number of cached user-candidate scores per source.

| Stage | Cost | Note |
|---|---|---|
| Base scorers | $O(KB)$ | paid **once**, not per coalition |
| Score caching | $O(KP)$ memory | the enabling design choice |
| Coalition sweep | $O(2^K \Phi)$ | $32\Phi$, and $\Phi$ is a small convex fit |
| Shapley aggregation | $O(K 2^K)$ | arithmetic only, negligible |
| Per-user decomposition | $O(1)$ extra | free by Proposition 3 |

The headline sentence to write: total cost is $O(KB + 2^K\Phi)$, dominated by the one-time $KB$ term, so **the attribution is cheaper than training the recommender it explains**. Contrast explicitly with the Monte-Carlo route the group's prior work required, and note that the sampling variance those methods must bound is simply absent here. Report measured wall-clock beside the asymptotics.

## 3.11 Practical implementation

Library versions, seeds, hardware, wall-clock, candidate-set size, negative sampling ratio, hyperparameter grids, and the caching scheme. Also pin, because each has already been identified as a place the pipeline can break or drift: the $\sigma_{u,g}=0$ fallback from §3.3, the choice of monotone transform in §4.7, and the fact that $f_\theta^{\emptyset} \equiv \pi$. Include the statement that the full game is reproducible on a laptop CPU.

Follow the house convention of the *Explanation Drift* paper and give a **hyperparameter table** plus **two numbered algorithm blocks** — one for cached score generation and the coalition sweep, one for the Shapley aggregation and per-user decomposition. Both are short, and the group's reviewers evidently expect pseudocode.

---

# 4. EXPERIMENTAL RESULTS

Setup and results combined, per the IJACSA structure.

## 4.1 Datasets and preprocessing

| Dataset | Users | Items | Interactions | Density | Metadata for CB |
|---|---|---|---|---|---|
| MovieLens-1M | 6,040 | 3,706 | 1,000,209 | ~4.5% (dense) | genres, year, title |
| Amazon-Book | ~50,000 | ~90,000 | ~1–3M | ~0.05% (sparse) | category, brand, title |

**These are the two benchmarks of the group's published recommendation work** (DyHuCoG, *IJIES* 2026), which is the point: reusing them makes SignalShap read as a continuation of the same experimental line rather than as an unrelated study, and it lets the discussion compare source attribution against the architecture-level results already published on the same data. Do not substitute a smaller Amazon category for convenience.

Two datasets, chosen for **contrast rather than coverage**: they differ in density by roughly two orders of magnitude, which is precisely the axis along which source attribution is predicted to move. Say this explicitly in the paper — two datasets picked for a reason reads far better than two datasets picked for convenience, and it pre-empts the "why not more benchmarks" question by making the choice part of the argument. If a reviewer presses for a third, LastFM-2K is the cheap addition and the pipeline will already support it unchanged.

> **Amazon-Book cannot be taken off the shelf here — read this before writing §4.1.** The canonical Amazon-Book split used in the DyHuCoG paper and throughout the LightGCN/HCCF line (52,643 users / 91,599 items / 2,984,108 interactions) ships as a fixed 80/20 random split carrying **no ratings, no timestamps, and no item metadata**. SignalShap needs all three: **REC** needs timestamps, **SEQ** needs interaction order, **CB** needs metadata, and the leave-one-out temporal split needs timestamps. Three of the five players and the entire evaluation protocol would break. **Rebuild Amazon-Book from the raw Amazon Reviews 2018 Books corpus** (Ni et al., EMNLP 2019), which carries ratings, timestamps, and the `meta_Books` metadata file, then apply the same 5-core and temporal-split protocol used for ML-1M. State in the paper that the split is rebuilt rather than reused, and why — a reviewer who knows the canonical split will otherwise assume the sequential and recency sources are impossible.
>
> **Second constraint: scale.** Raw Books 5-core is roughly 27M reviews over 8M users, and the caching design of §3.3 stores $|\mathcal{U}| \times N$ floats per source, so 8M users is ~32 GB of cache and the "runs on a laptop CPU" claim dies. **Subsample users to approximately the scale of the published Amazon-Book split** (~50k users), then re-apply 5-core to a fixed point. This is a disclosed preprocessing decision, not a hidden one: report the sampling rate and the seed, and note that the resulting density stays in the same regime as the published split (~0.05–0.06%), so the density-contrast argument against ML-1M is unaffected.

Both are public and standard, and both carry all five sources once Amazon-Book is rebuilt as above, so $K=5$ uniformly and the attribution vectors are directly comparable across datasets. Fill exact post-filtering counts from your own run — do not copy numbers from other papers, **including the group's own**, since the rebuilt split will not reproduce the canonical counts. Use 5-core filtering and state it. Leave-one-out temporal split (last interaction as test, second-last as validation), which is the RecSys convention and avoids a reviewer fight.

## 4.2 Protocol, metrics, and baselines

Metrics: NDCG@10 (primary, defines $v$), Recall@20, MRR, plus coverage and Gini for the discussion section. Report the null ranker's NDCG explicitly.

Baselines for the *recommender*: each single source alone, uniform-weight fusion, globally tuned fusion, and a strong single-model reference (LightGCN or SASRec) to establish that the hybrid is competitive — otherwise a reviewer will object that attributing credit in a weak system is uninteresting. **This baseline is not optional.**

Baselines for the *attribution*: leave-one-out, forward selection, permutation importance, Monte-Carlo Shapley at matched budget.

## 4.3 RQ1 — exact source attribution

**Table 4** and **Figure 2**: $\varphi_g$ and $\bar\varphi_g$ per dataset, with the efficiency check $\sum_g \varphi_g = v(\mathcal{G})$ shown numerically to machine precision. That row is a correctness audit and costs nothing to include.

Expected narrative, to be confirmed not assumed: CF dominates on MovieLens; CB and POP take a much larger share on sparse Amazon-Book. If that holds it is a clean, intuitive, defensible headline.

## 4.4 RQ2 — Shapley versus leave-one-out under redundancy

**Table 5**: $\varphi_g$ beside LOO, forward selection, and permutation importance, with a redundancy diagnostic (rank correlation between source score columns) as a final column. **Figure 3**: scatter of LOO against $\varphi$, annotated at the points where they most disagree.

Report this table as a **mean over the same five seeds used in §4.7, with standard deviations**, not as a single run. Both $\varphi_g$ and the LOO values are computed from trained fusion layers and inherit that training noise, so a gap that appears or vanishes by a small margin on one seed says nothing. This costs nothing extra — the five-seed sweep is already being run for §4.7, so Table 5 just reads off it.

The result to look for: **POP and REC are strongly rank-correlated**, LOO reports near-zero for both, Shapley assigns each a real share. Static popularity and time-decayed popularity are the same signal under two weightings, so the redundancy is close to structural rather than a hope about the data — see §A.5 of the implementation spec, where including both is a deliberate choice that keeps this experiment from depending on luck. POP↔CF is the secondary candidate, via the popularity bias in matrix factorization, and should be reported too but not relied on.

Report the **efficiency gap** as the headline number here: $\sum_g \mathrm{LOO}(g)$ against $v(\mathcal{G})$. Leave-one-out has no reason to sum to the total and under redundancy it falls badly short, so a single line showing the shortfall makes the argument more vividly than the per-source comparison does.

**Word the claim as an illustration, not an instantiation.** Real data gives *approximate*, not perfect, redundancy, so this observation is **not** a direct instance of Proposition 2 and must not be written as one — a perfect-case theorem cannot be left to carry an approximate-case empirical claim by implication, and that substitution is exactly what a careful reviewer catches. The honest and entirely sufficient framing is: Proposition 2 shows the failure mode exists in the idealized case; Table 5 shows the two methods disagreeing on real data in the direction the proposition predicts, with the measured redundancy between the sources reported alongside so the reader can judge how close to the idealized case we are. One sentence of care replaces a proposition. Say "consistent with" and "in the direction predicted by", never "an instance of".

**If the effect does not appear, say so and report the redundancy levels anyway** — a negative result here is survivable, but a fabricated one is not.

## 4.5 RQ3 — segment heterogeneity

**Figure 4**: stacked source shares per behavioural segment. **Figure 5**: attribution segments versus behavioural segments (confusion or Sankey). **Table 6**: per-segment shares with a permutation test for heterogeneity — shuffle segment labels, recompute the between-segment variance of $\varphi$, report the null distribution and $p$.

The headline to test: cold and light users derive most of their uplift from POP/CB, heavy users from CF/SEQ, so the global ordering inverts within at least one segment.

## 4.6 RQ4 — segment-adaptive fusion

**Table 7**: NDCG@10, Recall@20, MRR for uniform fusion, global fusion, and SignalShap-Fuse, with the $\lambda$ sweep as **Figure 6**. Report per-segment gains, since the expected pattern is a large gain on cold segments and roughly parity on heavy ones — that shape is more convincing than a single aggregate number, and it is the honest way to present a modest overall improvement.

## 4.7 Sensitivity, stability, and ablations

Seed stability of $\varphi$ across five seeds (report standard deviations — with exact Shapley the only variance source is base-model training, which is a nice thing to be able to say); sensitivity to candidate-set size $N$; the regenerate-candidates-per-coalition variant; sensitivity to the number of segments $M$; NDCG@$k$ for $k \in \{5,10,20\}$ to show the attribution is not a cutoff artifact.

**Nonlinear monotone rescaling (required, per the warning under Remark 1).** Apply a rank-preserving but shape-changing transform to each source's scores in turn, re-run the game, and report how far the shares move. Remark 1 guarantees nothing here and the paper must not imply otherwise. Both outcomes are publishable: if the shares barely move, that is an empirical robustness result offered *instead of* an overclaimed theorem; if they move, it is an honest statement of what the method is sensitive to, and reporting it yourself beats having a reviewer find it.

> **Implementation trap — do not apply a bare $\log$ to raw scores.** $\log$ requires a strictly positive argument, and the CF source is BPR-MF, whose raw scores are embedding dot products and therefore unbounded reals that are frequently negative or zero. A naive `np.log(scores)` produces `NaN` on **exactly the source the paper expects to matter most**, and it will fail silently into the fusion layer rather than raising. Use one of these instead, and state which in §3.10:
> - **Percentile rank** within each user's candidate list. Domain-safe for every source without exception, exactly rank-preserving, and it is the cleanest choice — it also happens to be the transform that would earn the general monotone claim, so it doubles as evidence for the discussion.
> - **Signed log**, $\mathrm{sgn}(x)\log(1+|x|)$. Defined on all of $\mathbb{R}$ and strictly increasing.
> - **Shift-then-log**, $\log(x - \min_u x + \eta)$ applied within each user's candidate list. Valid, and note the shift is itself affine so Remark 1 already covers that part, leaving the $\log$ as the only thing being tested.
>
> Percentile rank is the recommended default. Whichever you pick, apply it to **all five sources uniformly** so the comparison is not confounded by giving different sources different treatment.

## 4.8 Statistical significance

Paired tests over users, Holm–Bonferroni across the method family, Wilcoxon signed-rank as the non-parametric companion, and effect sizes. Pair over **users**, and state the unit of analysis explicitly.

---

# 5. DISCUSSION AND BROADER IMPLICATIONS

## 5.1 What the attributions mean for system design
Return to the cost column of §3.2. A source with a small share and a high maintenance cost is a decommissioning candidate; this is the concrete decision the paper enables. The most quotable version of this finding would be a source that is expensive to run — SEQ, which needs session logging and low-latency state — earning a share small enough that the engineering is not justified on the dense dataset while being clearly justified on the sparse one. Watch for that pattern in the results; if it appears, lead the discussion with it.

## 5.2 Cold start as an attribution phenomenon
Reframe cold start not as a data problem but as a *shift in which source carries the load*, which the segment profiles measure directly.

## 5.3 Relation to feature-level explanation
Source attribution and feature attribution answer different questions and compose: once a source is identified as load-bearing, feature-level SHAP inside that source is the natural next zoom level. This is where the author's prior clustering-plus-Shapley work slots in as the complementary layer, and where the thesis narrative closes.

## 5.4 Limitations and threats to validity
Be forthright: score-level fusion is one hybridization pattern among several and the method does not transfer unchanged to feature-combination or cascade hybrids; masking a source at fusion time is not the same as never having built it; offline ranking quality is a proxy for deployed value; the candidate set is shared across coalitions in the main results; five sources is a design choice and the exactness argument degrades if a system has thirty; attribution shares are invariant to how each source scales its scores but not to nonlinear reshaping of them, as quantified in §4.7; per-user attributions are exact with respect to the defined game but coarse as estimates, so segments rather than individuals are the reliable unit of explanation; and two datasets, however deliberately contrasted, are two datasets.

---

# 6. CONCLUSION AND FUTURE WORK

Restate the four contributions against the four RQs. Future work: **percentile-rank normalization in place of z-normalization**, which would upgrade Remark 1 from affine to full monotone invariance, at the cost of changing what the fusion layer conditions on — worth one sentence here precisely because §4.7 measures the nonlinear sensitivity but does not remove it; a demographic or otherwise privacy-bearing source, where the credit-versus-exposure trade-off becomes the object of study rather than a complication; Owen values when sources are grouped into a hierarchy (collaborative family versus content family); online or interleaved validation of the segment-adaptive gains; and extension to cascade hybrids where the coalition structure is genuinely constrained rather than free — the last is the natural bridge to the Myerson machinery in MHyperShap.

---

# DECLARATIONS (required by Discover AI)

Funding · Competing interests · Ethics approval (not applicable, public secondary data) · Consent · Data availability (both datasets public, with links) · Code availability (repository link) · Author contributions (CRediT) · **Use of AI tools** — the group's *Explanation Drift* paper carries an explicit declaration that ChatGPT was used for language clarity and code debugging with the authors taking full responsibility; include the equivalent statement here for consistency with the group's practice and with the venue's policy.

---

# APPENDICES

- **A.** Proofs of Propositions 1–3 and Remark 1 — three short arguments, roughly one page total
- **B.** Full coalition value tables $v(C)$ for all $2^K$ coalitions, per dataset — this is the paper's transparency showpiece and is only feasible *because* the game is exact and small. Include it.
- **C.** Hyperparameter grids and selected values
- **D.** Regenerate-candidates-per-coalition robustness results
- **E.** Per-segment attribution tables in full

---

# NOTATION LIST

Reproduce the §3.1 table as a standalone appendix-adjacent list, matching the IJACSA paper's convention.

---

# PLANNED FIGURES & TABLES

| # | Type | Content |
|---|---|---|
| Fig 1 | Diagram | Architecture: five sources → candidate generation → fusion → attribution game |
| Fig 2 | Bar | Source shares $\bar\varphi_g$ per dataset |
| Fig 3 | Scatter | LOO versus Shapley, annotated at maximal disagreement |
| Fig 4 | Stacked bar | Source shares per behavioural segment |
| Fig 5 | Sankey | Attribution segments versus behavioural segments |
| Fig 6 | Line | $\lambda$ sweep for SignalShap-Fuse |
| Fig 7 | Heatmap | Coalition value surface $v(C)$ |
| Tab 1 | Comparison | Positioning against prior work |
| Tab 2 | Descriptive | Dataset statistics |
| Tab 3 | Descriptive | The five sources and their production costs |
| Tab 4 | Results | $\varphi_g$, $\bar\varphi_g$, efficiency check |
| Tab 5 | Results | Shapley versus LOO, forward selection, permutation, redundancy |
| Tab 6 | Results | Per-segment shares with heterogeneity test |
| Tab 7 | Results | Recommendation quality, fusion variants |
| Tab 8 | Results | Significance tests and effect sizes |

---

# PLANNING NOTES (NOT part of the manuscript)

## Why this is likely to be accepted

The claim is narrow and fully supported; there is no oversold theorem. The computation is exact, which removes the single most common reviewer attack on Shapley papers. Every component is standard and public, so reproducibility objections are weak. The propositions are easy enough to verify in review and non-trivial enough to be worth stating, and Proposition 2 is the one that lifts this above an empirical note.

**Complexity is calibrated to the group's own published level.** The *Explanation Drift* paper (Nesmaoui, Louhichi & Lazaar) carries zero propositions, defines three metrics, and spends its analytical effort on a step-by-step complexity derivation, an ablation, a sensitivity sweep, and pseudocode. This blueprint was pulled back to that level: four propositions plus a corollary became **three short propositions plus a remark**, the $\varepsilon$-near-redundancy bound was **cut entirely** as disproportionate, and a complexity subsection was added in the house style — where, unusually, the analysis favours us, since the group's prior work had to adopt Monte-Carlo Shapley precisely because $2^{|F|}$ was infeasible over features, and moving the players to sources removes that constraint. If anything still reads as heavy after drafting, cut Proposition 3 next and state the per-user decomposition as a remark; Propositions 1 and 2 are the irreducible core.

## Build order (each step is independently checkable)

1. Data loaders and 5-core filtering for both datasets; freeze splits.
2. Five source scorers, cached to disk as score matrices over fixed candidate sets.
3. Fusion layer plus the coalition-masking harness; verify $v(\emptyset)=0$ and monotonicity spot-checks.
4. Exact Shapley over coalitions; **assert the efficiency identity in a unit test** — that single test catches most implementation errors. As a second test, assert **symmetry** on a synthetic game with two deliberately identical score columns: the two must receive equal $\varphi$ to machine precision, which independently validates the coalition-weighting code. *(An earlier draft named a pairwise-difference identity here; that test belonged to the ε-near-redundancy proposition, which has since been cut, so symmetry replaces it.)*
5. Per-user decomposition; assert it averages to the global values.
6. Segmentation, reusing the existing clustering pipeline from the ActionShap codebase.
7. Segment-adaptive fusion with train/test-clean segment assignment.
8. Statistics and the LaTeX table emitter — the ActionShap `stats.py` module ports over directly.

## What can be reused from existing work

The `stats.py` module (paired tests, Holm–Bonferroni, Cohen's $d_z$) and the k-means/quality-diagnostics pipeline from the ActionShap codebase transfer with essentially no change. Nothing from DyHuCoG is required, by design.

## Estimated effort

Roughly a week of implementation for someone with the ActionShap codebase already in hand, dominated by the five source scorers rather than by anything game-theoretic. Dropping to two datasets and five sources takes a meaningful bite out of that: the game itself is now 32 coalitions on two benchmarks, which is trivial, so effort is almost entirely in the scorers and the segmentation.

## Decisions taken

| Decision | Choice | Consequence |
|---|---|---|
| Number of datasets | Two: MovieLens-1M and Amazon-Book | Same pair as the group's published DyHuCoG work, so the paper reads as a continuation; justified by density contrast rather than convenience; LastFM-2K held in reserve if a reviewer asks |
| Amazon-Book provenance | Rebuilt from the raw Amazon Reviews 2018 Books corpus, not the canonical LightGCN split | The canonical split has no timestamps or metadata and would break REC, SEQ, CB and the temporal protocol; rebuild must be disclosed in §4.1 |
| Amazon-Book scale | User-subsampled to ~50k, then re-5-cored | Keeps the score cache and the laptop-CPU claim viable; sampling rate and seed reported |
| Demographic source | Dropped | $K=5$ uniform across datasets, attribution vectors directly comparable, no ethics section required; privacy angle moved to future work |
| Title | Explicit IJACSA continuation | *Game Theory Meets Recommendation…*; **SignalShap** kept as the method and codebase name |

## Remaining open questions

- Whether the strong single-model reference in §4.2 should be LightGCN or SASRec. SASRec is the better foil given SEQ is a player; LightGCN is cheaper and more standard. Pick one and justify in a sentence.
- Whether behavioural segments should come from k-means on profile statistics (reuses the existing pipeline, more defensible) or from a simple activity-quantile split (cruder, but far easier to explain and impossible to accuse of tuning). Consider reporting the quantile split in the main text and k-means in the appendix.
