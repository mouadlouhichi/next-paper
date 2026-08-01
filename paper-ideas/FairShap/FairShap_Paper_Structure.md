# FairShap — Full Paper Structure, TOC & Embedded Content

**Target journal:** *Discover Artificial Intelligence* (Springer Nature, open access, Q1 — Information Systems)
**Article type:** Research article
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco
**Corresponding author:** mouad_louhichi@um5.ac.ma

> **Why this is the "easy-to-accept" paper.** FairShap stays inside the author's established domain (Shapley + hypergraph + recommendation, 5 prior publications), reuses the DyHuCoG codebase, datasets, baselines, hardware, and statistical methodology, and requires **no new theorem** — it is an empirical methods paper with a clean, well-scoped contribution. Low compute, low reviewer risk, hot topic. Structure mirrors DyHuCoG and the MHyperShap blueprint. Target ≈ 8,000–10,000 words, 7–9 figures, 8–10 tables.

---

## Working Title (primary + alternates)
- **Primary:** *FairShap: Shapley-Value-Guided Two-Sided Fairness and Popularity Debiasing in Hypergraph Recommendation*
- Alt 1: *From Diversity to Fairness: Cooperative-Game Exposure Correction for Graph-Based Recommenders*
- Alt 2: *Two-Sided Fair Recommendation via Preference-Aware Shapley Exposure Attribution*

## One-paragraph thesis (the spine)
Graph- and hypergraph-based recommenders maximize accuracy but amplify **popularity bias**, starving long-tail items/providers of exposure and producing unequal recommendation quality across user groups. Existing fairness fixes either degrade accuracy or rely on heuristic reweighting without principled credit attribution. We introduce **FairShap**, which augments a hypergraph recommender with a **preference-aware Shapley estimator of each item/provider's marginal contribution to a fairness-aware utility**, then uses these attributions to correct exposure during propagation and re-ranking. FairShap jointly improves **provider-side** exposure equality and **consumer-side** quality parity while preserving NDCG/Recall, achieving a better accuracy–fairness trade-off than heuristic and re-ranking baselines on MovieLens-1M, Amazon-Book, and Yelp2018.

---

# TABLE OF CONTENTS
```
Abstract / Keywords
1. Introduction
   1.1 Background and motivation
   1.2 Popularity bias and the two-sided fairness gap
   1.3 Why principled attribution (Shapley) for fairness
   1.4 From DyHuCoG to FairShap
   1.5 Contributions
   1.6 Organization
2. Related Work
   2.1 Graph/hypergraph recommendation
   2.2 Popularity bias and debiasing
   2.3 Fairness in recommendation (provider/consumer, two-sided)
   2.4 Shapley value and cooperative games in recommendation/XAI
   2.5 Positioning and differentiation (comparison table)
3. Preliminaries and Problem Formulation
   3.1 Notation
   3.2 Two-sided fairness problem statement
   3.3 Background: Shapley value and exposure
4. The FairShap Framework
   4.1 Framework overview
   4.2 Fairness-aware coalition value
   4.3 Preference-aware Shapley exposure attribution
   4.4 Shapley-guided propagation reweighting
   4.5 Shapley-guided fair re-ranking
   4.6 Training objective
5. Analysis
   5.1 Exposure-efficiency property (light proposition)
   5.2 Accuracy–fairness trade-off characterization
   5.3 Computational complexity
6. Experimental Setup
   6.1 Datasets
   6.2 Baselines
   6.3 Evaluation metrics (accuracy + fairness)
   6.4 Implementation and hardware
7. Results and Discussion
   7.1 Main results: accuracy and two-sided fairness
   7.2 Accuracy–fairness Pareto trade-off
   7.3 Long-tail / cold-provider exposure analysis
   7.4 Ablation study
   7.5 Sensitivity analysis (fairness weight, M)
   7.6 Cross-validation and robustness
   7.7 Computational efficiency
   7.8 Statistical significance
   7.9 Interpretability case study
   7.10 Limitations
8. Conclusion and Future Work
Declarations
References
Appendix A — Exposure-efficiency proof and metric definitions
Appendix B — Statistical methodology
Appendix C — Hyperparameters
Notation list
```

---

# ABSTRACT (draft, ~210 words)
Graph- and hypergraph-based collaborative filtering achieves strong ranking accuracy but reinforces popularity bias: a small set of head items dominates exposure, long-tail providers are systematically under-served, and recommendation quality varies across user groups. Existing debiasing and fairness methods often trade away accuracy or rely on heuristic reweighting that lacks a principled notion of each item's contribution to fairness. We propose **FairShap**, a fairness-aware recommendation framework that estimates, via a preference-aware Monte-Carlo Shapley estimator, the marginal contribution of each item and provider to a multi-objective utility combining ranking quality, diversity, and exposure equality. These contributions are injected as fairness-corrected weights in hypergraph message passing and drive a Shapley-guided re-ranking step that redistributes exposure toward under-credited long-tail items without harming relevance. We evaluate FairShap on MovieLens-1M, Amazon-Book, and Yelp2018 against strong accuracy baselines (MF, NCF, LightGCN, HCCF, HPCF, DyHuCoG) and dedicated fairness baselines (reweighting, calibrated re-ranking, determinantal re-ranking). FairShap attains the best accuracy–fairness trade-off, reducing exposure inequality (Gini, average recommendation popularity) and consumer-side quality disparity while maintaining NDCG@20 and Recall@20, with gains confirmed by paired significance tests under Holm–Bonferroni correction. *(Replace closing sentence with headline numbers once experiments complete.)*

**Keywords:** Recommender systems · Fairness · Popularity bias · Shapley value · Cooperative game theory · Hypergraph neural networks · Exposure · Explainable AI

---

# 1. INTRODUCTION

**1.1 Background and motivation.** Recommender systems shape exposure on digital platforms; accuracy-only optimization concentrates exposure on popular items, creating filter bubbles and disadvantaging long-tail items and the providers behind them. (Reuse DyHuCoG's filter-bubble/popularity-bias framing.)

**1.2 Popularity bias and the two-sided fairness gap.** Distinguish **provider/item-side fairness** (equitable exposure across items/popularity tiers/providers) from **consumer/user-side fairness** (comparable recommendation quality across user groups). Graph/hypergraph propagation amplifies head-item signals, worsening both. State the gap: most recommenders optimize accuracy (and sometimes diversity, as in DyHuCoG) but do not explicitly equalize exposure with a principled attribution.

**1.3 Why principled attribution (Shapley) for fairness.** Heuristic reweighting (inverse-popularity, temperature scaling) cannot capture interaction effects: an item's fair-exposure value depends on which other items are present in the list/coalition. The Shapley value uniquely satisfies efficiency, symmetry, null-player, and linearity, providing a fair, interaction-aware credit for each item's marginal contribution to an exposure-fairness objective. Forward-reference the exposure-efficiency property (§5.1).

**1.4 From DyHuCoG to FairShap.** DyHuCoG [Louhichi et al., 2026] injected preference-aware Shapley values into hypergraph message passing to jointly optimize accuracy and **diversity**. FairShap retargets the same machinery to **fairness/exposure equality** — a distinct objective, distinct metrics, and a new Shapley-guided re-ranking stage — while reusing the dynamic-hypergraph backbone, the Monte-Carlo estimator, and the multi-objective coalition-value design. Connect to the real-time Shapley adjustment [Nesmaoui et al., 2025] and Shapley-XAI works [Louhichi et al., 2023; 2025].

**1.5 Contributions.**
1. **FairShap framework:** a fairness-aware hypergraph recommender that uses preference-aware Shapley attributions to correct two-sided exposure.
2. **Fairness-aware coalition value** combining ranking quality, diversity, and exposure equality, with a Shapley-guided fair re-ranking stage.
3. **Exposure-efficiency property:** a light proposition showing the Shapley-corrected exposure allocation distributes total exposure across items consistently with their fairness-aware marginal value.
4. **Comprehensive evaluation** on three public datasets against accuracy and fairness baselines, reporting the accuracy–fairness Pareto front, long-tail exposure, ablations, sensitivity, cross-validation, and paired significance tests.

**1.6 Organization.** Standard roadmap paragraph.

---

# 2. RELATED WORK
Close with the comparison table.

- **2.1 Graph/hypergraph recommendation** — LightGCN, HCCF, HPCF, RecDCL, DyHuCoG (reuse DyHuCoG's literature framing).
- **2.2 Popularity bias and debiasing** — inverse-propensity scoring, popularity-aware negative sampling, causal debiasing; note accuracy trade-offs.
- **2.3 Fairness in recommendation** — provider/consumer fairness, two-sided fairness, exposure fairness, calibrated recommendation, re-ranking (xQuAD, DPP/determinantal). Note most are post-hoc or heuristic.
- **2.4 Shapley/cooperative games in recommendation & XAI** — Data-Shapley, Shapley-driven data pruning for recommenders (Zhang et al., 2025), your DyHuCoG and XAI works; note none target two-sided exposure fairness via Shapley.

**2.5 Comparison table.**

| Method | Accuracy | Diversity | Provider fairness | Consumer fairness | Principled attribution | Hypergraph |
|---|---|---|---|---|---|---|
| LightGCN / HCCF / HPCF | ✓ | partial | ✗ | ✗ | ✗ | HCCF/HPCF ✓ |
| Popularity debiasing (IPS) | ✓ | partial | partial | ✗ | ✗ | ✗ |
| Calibrated / DPP re-ranking | ✓ | ✓ | partial | ✗ | ✗ | ✗ |
| DyHuCoG (ours, 2026) | ✓ | ✓ | ✗ | ✗ | ✓ (Shapley) | ✓ |
| **FairShap (this work)** | ✓ | ✓ | **✓** | **✓** | **✓ (Shapley)** | ✓ |

---

# 3. PRELIMINARIES AND PROBLEM FORMULATION

**3.1 Notation.** Users $\mathcal{U} = \{u_1, \ldots, u_{|\mathcal{U}|}\}$; items $\mathcal{I} = \{i_1, \ldots, i_{|\mathcal{I}|}\}$; providers/item-groups $\mathcal{G}$; contexts $\mathcal{C}$; interaction hypergraph $\mathcal{H} = (\mathcal{U} \cup \mathcal{I}, \mathcal{E})$; recommended list for user $u$ is $\mathcal{R}_u$ with cutoff $K$; coalition $S$; coalition value $v(S)$; Shapley value $\phi_j$; Monte-Carlo estimate $\hat{\phi}_j$; fairness weight $\gamma$; MC samples $M$.

**3.2 Two-sided fairness problem.** Learn a scoring function $f(u, i, c_{u,i})$ and produce lists $\{\mathcal{R}_u\}$ that (i) preserve ranking accuracy, (ii) equalize item/provider **exposure** $\text{Exp}(i) = \sum_{u} \mathbb{1}[i \in \mathcal{R}_u] \cdot \omega(\text{rank}_u(i))$ across popularity tiers/providers, and (iii) reduce the **consumer-side quality gap** $\Delta_{\text{NDCG}} = |\text{NDCG}_{g_1} - \text{NDCG}_{g_2}|$ across user groups.

**3.3 Background: Shapley value and exposure.** Exact Shapley value (mirroring DyHuCoG Eq. 3):

$$
\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\,\bigl[v(S \cup \{j\}) - v(S)\bigr]
$$

Monte-Carlo estimator (DyHuCoG Eq. 4):

$$
\hat{\phi}_j = \frac{1}{M}\sum_{m=1}^{M}\bigl[v(S_m \cup \{j\}) - v(S_m)\bigr]
$$

---

# 4. THE FAIRSHAP FRAMEWORK
Methodological core; Fig. 1 = workflow (input → cooperative game with fairness utility → Shapley attribution → fairness-corrected hypergraph propagation → fair re-ranking → output).

**4.1 Overview.** Five stages, paralleling DyHuCoG but with a fairness objective and an added re-ranking stage.

**4.2 Fairness-aware coalition value.** Extend DyHuCoG Eq. 1 with an exposure-equality term:
> **Definition 1 (Fairness-Aware Coalition Value).**
> $$
> v(S) = \alpha \cdot \text{NDCG@}K(S) + \beta \cdot \text{Diversity}(S) + \gamma \cdot \text{Fairness}(S)
> $$
> where $\text{Fairness}(S) = 1 - \text{Gini}\bigl(\{\text{Exp}(i)\}_{i \in S}\bigr)$ measures exposure equality among items in the coalition; weights $(\alpha, \beta, \gamma)$ tuned on validation with $\alpha + \beta + \gamma = 1$.

**4.3 Preference-aware Shapley exposure attribution.** Compute $\hat{\phi}_j$ under $v(S)$ to obtain each item's marginal contribution to the fairness-aware utility; items that improve exposure equality without harming relevance receive higher Shapley credit. (Reuse the preference-weighted variant and variance-reduction tricks from DyHuCoG §3.4.)

**4.4 Shapley-guided propagation reweighting.** Inject normalized Shapley coefficients as hyperedge weights in message passing (reuse DyHuCoG Eqs. 6–9):
> $$
> w_{jk} = \frac{\hat{\phi}_{jk}}{\sum_{k' \in \mathcal{N}(j)} \hat{\phi}_{jk'}}
> $$
> so that under-credited long-tail items receive relatively more propagation weight.

**4.5 Shapley-guided fair re-ranking.** Post-scoring, re-rank each user's candidate list to redistribute exposure:
> **Definition 2 (Fair Re-Ranking Score).**
> $$
> \tilde{y}_{u,i} = (1-\gamma)\,\hat{y}_{u,i} + \gamma \cdot \frac{\phi_i^{\text{fair}}}{\max_{i'}\phi_{i'}^{\text{fair}}}
> $$
> where $\hat{y}_{u,i}$ is the relevance score and $\phi_i^{\text{fair}}$ the item's fairness Shapley value; $\gamma$ controls the accuracy–fairness trade-off.

**4.6 Training objective.** Composite loss extending DyHuCoG Eq. 13 with a fairness regularizer:
> $$
> \mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}} + \lambda_{\text{fair}}\mathcal{L}_{\text{fair}} + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}
> $$
> with $\mathcal{L}_{\text{rec}}$ the BPR pairwise ranking loss and $\mathcal{L}_{\text{fair}}$ penalizing exposure inequality (e.g., variance of $\text{Exp}(i)$ across popularity tiers).

---

# 5. ANALYSIS (deliberately light — this is the easy paper)

**5.1 Exposure-efficiency property.**
> **Proposition 1 (Exposure Efficiency).** The Shapley-corrected exposure allocation satisfies $\sum_{i \in \mathcal{I}} \phi_i^{\text{fair}} = \text{Fairness}(\mathcal{I})$, i.e., the total fairness utility is fully distributed across items, with null (already-fairly-exposed) items receiving zero correction. *(Direct consequence of the Shapley efficiency and null-player axioms applied to $\text{Fairness}(S)$.)*

**5.2 Accuracy–fairness trade-off.** Characterize how $\gamma$ interpolates between pure relevance ($\gamma = 0$) and maximal exposure equality ($\gamma = 1$); show the trade-off is smooth and report the Pareto front empirically (§7.2).

**5.3 Complexity.** Same order as DyHuCoG plus a re-ranking pass: $\mathcal{O}((L+1)md) + \mathcal{O}((M/f)m)$ for training and $\mathcal{O}(|\mathcal{I}|\log|\mathcal{I}|)$ per user for re-ranking. Reuse DyHuCoG's complexity treatment (Eq. 24).

---

# 6. EXPERIMENTAL SETUP
Reuse DyHuCoG §4.1 protocol verbatim in style.

**6.1 Datasets.** MovieLens-1M, Amazon-Book, Yelp2018 (you already use all three). Report users/items/interactions/density and define popularity tiers (head/mid/tail by interaction-count deciles) and user groups (e.g., activity level) for fairness evaluation.

**6.2 Baselines.** Accuracy: MF, NCF, LightGCN, HCCF, HPCF, DyHuCoG. Fairness: inverse-popularity reweighting, calibrated re-ranking (xQuAD-style), determinantal (DPP) re-ranking, and a fairness-regularized GNN.

**6.3 Metrics.**
- *Accuracy:* NDCG@20, Recall@20.
- *Provider/item fairness:* Gini coefficient of exposure, Average Recommendation Popularity (ARP), long-tail coverage, provider-exposure disparity.
- *Consumer fairness:* NDCG gap across user groups $\Delta_{\text{NDCG}}$.
- *Diversity:* intra-list diversity (ILD) and catalog coverage (continuity with DyHuCoG).

**6.4 Implementation and hardware.** Python 3.8 / PyTorch 2.0.1; single RTX 4090 (reuse DyHuCoG hardware table); 5 seeds $\{42, 43, 44, 45, 46\}$; mean $\pm$ std; 95% CIs; early stopping on validation NDCG@20.

---

# 7. RESULTS AND DISCUSSION
Mirror DyHuCoG's results depth.

- **7.1 Main results** — accuracy + fairness leaderboard per dataset; FairShap best accuracy–fairness balance.
- **7.2 Accuracy–fairness Pareto** — sweep $\gamma$; plot NDCG@20 vs. Gini/ARP; show FairShap dominates baselines' frontier.
- **7.3 Long-tail / cold-provider exposure** — exposure by popularity decile; FairShap lifts tail exposure with minimal head loss.
- **7.4 Ablation** — w/o fairness term, w/o Shapley reweighting, w/o re-ranking, w/o hypergraph (table mirroring DyHuCoG Table 5).
- **7.5 Sensitivity** — $\gamma \in \{0, 0.1, 0.25, 0.5, 0.75, 1\}$; $M \in \{10, 25, 50, 100\}$ (reuse DyHuCoG convergence framing).
- **7.6 Cross-validation & robustness** — 5-fold + repeated sub-sampling (reuse DyHuCoG Table 4 design).
- **7.7 Computational efficiency** — runtime/memory vs. baselines (mirror DyHuCoG Table 6).
- **7.8 Statistical significance** — paired t-tests + Holm–Bonferroni + Wilcoxon (reuse Appendix-A methodology).
- **7.9 Interpretability case study** — SHAP-style waterfall showing relevance vs. fairness contributions to a re-ranked item (analogous to DyHuCoG Fig. 4).
- **7.10 Limitations** — reliance on group definitions; static popularity tiers; potential head-accuracy cost at high $\gamma$.

---

# 8. CONCLUSION AND FUTURE WORK
Recap contributions; restate headline numbers; future: dynamic/streaming exposure fairness, individual (not just group) fairness, multi-stakeholder objectives, and integration with the dynamic-update mechanism of DyHuCoG. Tie to EU AI Act fairness/transparency requirements (consistent with thesis framing).

---

# DECLARATIONS (required by Discover AI)
- **Funding** — state grant/none.
- **Competing interests** — "The authors declare no competing interests."
- **Data availability** — all three datasets are public (links); processed splits + code released.
- **Code availability** — repository link.
- **Author contributions** — reuse DyHuCoG CRediT split.
- **Ethics approval** — not applicable.

# APPENDICES
- **A** — exposure-efficiency proof + full fairness-metric definitions (Gini, ARP, disparity).
- **B** — statistical methodology (paired tests, Holm–Bonferroni, Wilcoxon) — adapt DyHuCoG Appendix A.
- **C** — hyperparameters and weight-tuning grid.

# NOTATION LIST
Reuse DyHuCoG's notation-table format, updated with fairness/exposure symbols.

---

# PLANNED FIGURES & TABLES
**Figures:** (1) FairShap workflow; (2) popularity-bias illustration; (3) accuracy–fairness Pareto front; (4) exposure by popularity decile; (5) consumer-fairness gap bars; (6) ablation; (7) $\gamma$/$M$ sensitivity; (8) SHAP waterfall (relevance vs. fairness); (9) per-user difference distribution + Q–Q.

**Tables:** (1) comparison/differentiation; (2) notation; (3) dataset statistics; (4) main accuracy+fairness results; (5) long-tail exposure; (6) ablation; (7) sensitivity; (8) cross-validation; (9) runtime/memory; (10) paired significance.

---

# PLANNING NOTES (NOT part of the manuscript)

**Why FairShap is faster to accept than MHyperShap.** No new theorem (only a light proposition); same domain/datasets/baselines/code as DyHuCoG; fairness is an established, reviewer-friendly topic; all experiments run on a single RTX 4090 with public data — minimal compute, minimal risk.

**Execution timeline (≈3 months).**

| Phase | Weeks | Output |
|---|---|---|
| Related work + comparison table | 1 | §2 |
| Add fairness utility + Shapley re-ranking to DyHuCoG code | 1–3 | working codebase |
| Experiments (3 datasets, accuracy + fairness) | 3–6 | §7 main tables |
| Ablations + sensitivity + Pareto + cross-validation | 6–8 | §7.2–7.6 |
| Significance + interpretability | 8–9 | §7.8–7.9 |
| Full draft + figures | 9–11 | manuscript |
| Internal review + submit | 11–12 | Discover AI submission |

**Key prior art to cite.** Shapley (1953); Lundberg & Lee (2017); Ghorbani & Zou (2019); popularity-bias/IPS debiasing; two-sided/provider fairness; calibrated & DPP re-ranking; Zhang et al. (2025, Shapley data pruning for recommenders); your lineage — DyHuCoG (2026), Real-time Shapley adjustment (2025), GNN+Shapley hierarchical recommendation (2025), Game-Theory-Meets-XAI (2025), clustering-SHAP (2023).

**Submission checklist (Discover AI).**
- [ ] Abstract ≤ ~250 words, no citations.
- [ ] All declarations present.
- [ ] Data + code availability with working links.
- [ ] Figures ≥ 300 dpi; vector where possible.
- [ ] Reproducibility: seeds, hyperparameters, hardware reported.
- [ ] Self-citation lineage included.
- [ ] Statistical tests with multiple-comparison correction.
- [ ] Check Morocco APC discount eligibility (Research4Life) before acceptance.
