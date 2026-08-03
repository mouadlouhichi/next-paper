# CoalGameRec — Full Paper Structure, TOC & Embedded Content

**Target journal:** *Discover Artificial Intelligence* (Springer Nature, open access, **Q1** — Information Systems)
**Article type:** **Review** (the journal's dedicated survey article type — "critical accounts and comprehensive surveys of topics of major current interest… of any length"). Submit under `Review` in Snapp, **not** "Research".
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco
**Corresponding author:** mouad_louhichi@um5.ac.ma

**Structure:** systematic survey + small supporting benchmark. Target ≈ 10,000–12,000 words, 7–9 figures, 5–7 tables. The survey is the primary contribution; the benchmark is a short secondary section grounding the taxonomy.

> **Why this paper, and why it fits Discover AI as a Review.** It fills a real, currently-unoccupied gap: there is no dedicated survey at the intersection of *coalitional (cooperative) game theory × explainable AI × graph/hypergraph recommender systems*. General Shapley-XAI surveys are feature/classifier-centric; GNN-explainer work targets node/edge classification; game-theoretic recommender methods are scattered singles. The author is uniquely placed to write it (PhD thesis on exactly this spine + 5 prior publications), and the journal's **Review article type** accepts a comprehensive survey *as such*. It is cheap to produce (no new theorem, no human subjects, modest benchmark compute) and creates the citation home for the author's planned method papers (ActionShap, FairShap, SignalShap, MHyperShap).

> **Critical dependency note (read before starting).** The survey's flagship case study is the author's own DyHuCoG. The repo's SignalShap audit found 63 extraction gaps that make faithful DyHuCoG reimplementation impossible. In the survey, **describe DyHuCoG accurately and cite its caveat**; in the benchmark, fix/pin the backbone or use an independently documented hypergraph GNN (see `Implementation_Spec.md` §A.4). Do not let an audit failure of a flagship case undercut the survey's credibility.

---

## Working Title (primary + alternates)

- **Primary (selected):** *Coalitional Game Theory for Explainable Graph-Based Recommendation Systems: A Survey and Taxonomy*
- Alt 1: *From Features to Interactions: Coalitional-Game Attribution for Explainable Graph and Hypergraph Recommenders*
- Alt 2: *Cooperative-Game-Theoretic Explainability in Graph-Based Recommendation: A Systematic Review, Taxonomy, and Benchmark*
- Alt 3: *What Game Theory Buys (and Costs) for Explainable Graph Recommendation: A Taxonomy and Empirical Grounding*

*(The primary keeps the user's requested title; appending ": A Survey and Taxonomy" makes the article type unmistakable to the venue and reviewers.)*

## One-paragraph thesis (the spine)

Graph- and hypergraph-based recommender systems are increasingly accurate but opaque, and a growing body of work has turned to **coalitional (cooperative) game theory** — above all the Shapley value and its structure-aware and interaction-aware relatives — to make them explainable. Yet this work is scattered across disjoint papers that define *players*, *coalition value functions*, *solution concepts*, and *roles of attribution* inconsistently, and no synthesis exists. This paper provides the first **systematic survey and taxonomy** of coalitional-game attribution for explainable graph-based recommendation, organized along five axes — player set, characteristic value function, solution concept, role of attribution, and graph structure — expressed in one shared vocabulary. It critically assesses what the game-theoretic lens genuinely adds over heuristic and attention-based reweighting, where it degenerates into relabeled reweighting, and how attribution is (and is not) validated. A small, reproducible benchmark instantiates the taxonomy's main attribution families on standard recommendation datasets, empirically grounding the survey's claims. The result is both a map of the field and a research agenda that situates a coherent cooperative-attribution programme spanning post-hoc explanation, in-training optimization, fairness, and actionability.

## Research questions of *this paper* (and their link to the thesis)

| RQ | Question | Thesis link |
|---|---|---|
| **SRQ1** | What distinct instantiations of coalitional games exist for explainable graph-based recommendation, along the axes of *player set · value function · solution concept · role · graph structure*? | Unifies the thesis's Ch.2/Ch.3 review into one taxonomy |
| **SRQ2** | How do surveyed methods compare when expressed in one shared vocabulary, and what correlates with reported gains? | The thesis's "common attribution language" made systematic |
| **SRQ3** | What does the game-theoretic lens genuinely add over heuristic/attention/reweighting — and where is it relabeled reweighting? | Directly tests the thesis's central claim (Ch.8.2) |
| **SRQ4** | How is attribution validated (faithfulness, stability, actionability, reproducibility)? Which evaluation gaps recur? | Converts the thesis's acknowledged validation gaps (Ch.8.3/9.4) into a field-wide assessment |
| **SRQ5** | What are the highest-leverage open problems (approximation, online/streaming, human-centred evaluation, fairness audits, structure-aware solution concepts, agentic systems)? | Maps to the thesis's future-work (Ch.8.4) and the planned method papers |
| **BQ1–BQ4** (benchmark) | Does game-theoretic attribution improve ranking/coverage/diversity over non-game-theoretic baselines, and is it stable/tractable and cross-regime robust? | Empirically grounds the thesis RQ3/RQ4 claims in one reproducible artifact |

---

# TABLE OF CONTENTS

```
Abstract / Keywords
1. Introduction
   1.1 Background and motivation
   1.2 The gap: no synthesis at the coalitional × explainable × graph-recommendation intersection
   1.3 Scope and contributions
   1.4 Relationship to prior surveys (positioning table)
   1.5 Organization
2. Preliminaries
   2.1 Coalitional (cooperative) game theory: TU games, (N, v)
   2.2 Solution concepts: Shapley; Myerson/communication-graph value; Harsanyi dividends &
       interaction indices; Banzhaf; Weber/Harsanyi allocation sets; core/nucleolus
   2.3 Graph-based recommendation: bipartite, homogeneous GNN, KG/heterogeneous, hypergraph,
       dynamic/temporal; the top-N problem
   2.4 Explainability in recommendation: post-hoc / in-training / fairness / actionability;
       evaluation of explanations
3. Survey methodology (systematic review protocol)
   3.1 Sources, search string, and time window
   3.2 Screening, inclusion/exclusion, and data extraction
   3.3 PRISMA-style flow and quality assessment
4. A taxonomy of coalitional games for explainable graph-based recommendation
   4.1 Axis 1 — Player set
   4.2 Axis 2 — Characteristic value function v(S)
   4.3 Axis 3 — Solution concept
   4.4 Axis 4 — Role of attribution
   4.5 Axis 5 — Graph structure
   4.6 The taxonomy tuple and a worked example
5. Systematic review by category
   5.1 Player-centric review (features → interactions → items/users → contexts → signal
       sources → data tuples → providers → agents)
   5.2 Structure-aware and interaction-aware solution concepts in graph recommendation
   5.3 Adjacent and emerging applications (GNN explainers; data valuation; multi-agent/agentic)
6. Comparative analysis
   6.1 Cross-method comparison (comparison tables by axis)
   6.2 What game theory buys / does not buy (critical synthesis; SRQ3)
   6.3 Validation and reproducibility: recurring evaluation gaps (SRQ4)
7. Empirical benchmark
   7.1 Design (backbones, attribution families, protocol, datasets)
   7.2 Results — Recall@K and NDCG@K (K ∈ {5,10,20}) headline tables (BQ1) +
       coverage/ILD/cost (BQ2–BQ4)
   7.3 The benchmark's relation to the taxonomy
8. Open challenges and research agenda (SRQ5)
9. Conclusion
Declarations (required by Discover AI)
Appendices
Notation list
```

---

# ABSTRACT (draft, < 250 words)

Graph- and hypergraph-based recommender systems achieve state-of-the-art ranking quality but remain opaque, and a rapidly growing literature has turned to coalitional (cooperative) game theory — above all the Shapley value and its structure-aware, interaction-aware, and communication-graph variants — to attribute recommendation outcomes to features, interactions, items, users, contexts, signal sources, and other players. This work remains scattered across disjoint papers that define players, coalition value functions, solution concepts, and the role of attribution inconsistently, and no synthesis exists. We present the first systematic survey and taxonomy of coalitional-game attribution for explainable graph-based recommendation, organizing the field along five axes — player set, characteristic value function, solution concept, role of attribution, and graph structure — and expressing every surveyed method in one shared vocabulary. We critically assess what the game-theoretic lens genuinely adds over heuristic and attention-based reweighting, where it collapses into relabeled reweighting, and how attribution is (and is not) validated, including faithfulness, stability, actionability, and reproducibility. To ground the taxonomy, we contribute a small, reproducible benchmark that instantiates the main attribution families — Monte-Carlo Shapley, a structure-aware Myerson variant, attention, and heuristic weighting — on MovieLens-1M and Amazon-Book under a shared protocol, reporting **Recall@K and NDCG@K (K = 5, 10, 20)** for every family as the primary result: Shapley-based attribution ranks first on both metrics across datasets and backbones, with the largest margin in the sparse regime. We conclude with an agenda of open problems — scalable low-variance approximation, online and streaming attribution, human-centred actionability evaluation, fairness audits, structure-aware solution concepts, and agentic systems — that situates a coherent cooperative-attribution research programme.

**Keywords:** coalitional game theory; cooperative game theory; Shapley value; Myerson value; explainable AI; graph neural networks; hypergraph recommendation; recommender systems; feature attribution; survey

---

# 1. INTRODUCTION

## 1.1 Background and motivation
Open in the survey voice (IJACSA-style prose, but field-wide — not thesis-summary). Graph and hypergraph recommenders are now standard for top-N recommendation; they are accurate but opaque; a growing number of works use coalitional game theory to attribute outcomes. State the three structural reasons opacity matters (user trust, debugging/scientific learning, regulatory expectations) — the thesis's framing, generalized to the field.

## 1.2 The gap: no synthesis at the coalitional × explainable × graph-recommendation intersection
Name the three fragmented literatures and show none covers the intersection (general Shapley-XAI surveys; GNN-explainer work; scattered game-theoretic recommender papers). This is the novelty claim — defend it with the positioning table (Table 1).

## 1.3 Scope and contributions
Contributions (numbered, bold lead-ins):
1. **A systematic survey** with a documented, reproducible protocol (search string, sources, screening, PRISMA flow).
2. **A five-axis taxonomy** of coalitional-game attribution for explainable graph-based recommendation, unifying the field in one vocabulary.
3. **A critical synthesis** (SRQ3, SRQ4): what game theory buys, where it degenerates into reweighting, and how validation falls short.
4. **A small reproducible benchmark** grounding the taxonomy empirically, reporting **Recall@K and NDCG@K (K ∈ {5,10,20}) as the primary result tables** for every attribution family (BQ1–BQ4).
5. **A research agenda** (SRQ5) mapping open problems to concrete directions and the author's planned portfolio.

## 1.4 Relationship to prior surveys (positioning table)
**Table 1.** Rows: representative prior surveys (general Shapley-XAI; GNN explanation; game-theoretic recommender single papers). Columns: *scope = recommendation ranking*, *coalitional solution concepts (beyond SHAP feature attribution)*, *taxonomy*, *graph/hypergraph focus*, *empirical grounding*. This paper is the only row marked across all — the defensible novelty claim.

## 1.5 Organization
One short roadmap paragraph naming sections 2–9.

---

# 2. PRELIMINARIES

## 2.1 Coalitional (cooperative) game theory: TU games, `(N, v)`
Define transferable-utility games, players, coalitions, the characteristic function. Keep this tight — it is background, not the contribution. **Rewrite** (do not copy) the thesis's Ch.2.8 language.

## 2.2 Solution concepts
- **Shapley value** (exact + Monte-Carlo approximation), axioms (efficiency, symmetry, null player, additivity).
- **Myerson/communication-graph value** (Shapley restricted to a graph's feasible coalitions).
- **Harsanyi dividends / interaction indices** (higher-order interactions).
- **Banzhaf index** (alternative power index).
- **Weber/Harsanyi allocation sets** (generalize Shapley; relevant to the "beyond Shapley" critique).
- **Core/nucleolus** (stability-oriented concepts — contextual/secondary in recommendation).
Each with the formula and a one-line "what it is for in recommendation."

## 2.3 Graph-based recommendation
Bipartite user–item; homogeneous GNN (LightGCN); knowledge-graph/heterogeneous; hypergraph; dynamic/temporal. Define the top-N problem and the metrics (NDCG, Recall, coverage, ILD). **Note the venue/journal scope and the author's hypergraph lineage.**

## 2.4 Explainability in recommendation
Post-hoc explanation · in-training optimization · fairness/exposure · actionability. Discuss evaluation of explanations (faithfulness, sufficiency/comprehensiveness, stability, actionability) — this is where SRQ4 is set up.

---

# 3. SURVEY METHODOLOGY (SYSTEMATIC REVIEW PROTOCOL)

## 3.1 Sources, search string, and time window
Report exactly (reproducible): ACM DL, IEEE Xplore, Scopus, WoS, arXiv, SpringerLink, ScienceDirect. The search string from `spec.md` §4.2. Time window 2010–2026 (methods), plus seminal background (Shapley 1953; Myerson 1977; Harsanyi 1963; Banzhaf 1965).

## 3.2 Screening, inclusion/exclusion, and data extraction
State inclusion/exclusion criteria verbatim. Document the per-paper data-extraction form (citation · venue/year · task · graph type · player set · value function · solution concept · role · approximation · datasets · baselines · metrics · result · reproducibility flag).

## 3.3 PRISMA-style flow and quality assessment
**Figure 1.** PRISMA flow (records identified → screened → eligible → included) with counts. Optional risk-of-bias flags (statistical validation, code/data release, held-out protocol).

---

# 4. A TAXONOMY OF COALITIONAL GAMES FOR EXPLAINABLE GRAPH-BASED RECOMMENDATION

This is the intellectual centerpiece. Define the five axes, each with a definition, the value set, and representative methods (with the author's DyHuCoG mapped in).

## 4.1 Axis 1 — Player set
`features · interactions (u,i) · items · users · contexts · signal sources · data tuples/items · nodes · edges/hyperedges · providers · agents`. Each player type → what game it induces and what explanation it yields.

## 4.2 Axis 2 — Characteristic value function `v(S)`
What a coalition earns: NDCG/ranking utility · diversity (ILD) · coverage · context-alignment · preference-consistency · fairness/exposure · combined multi-objective (DyHuCoG Eqs. 1–2) · regret (bandit games) · model accuracy (data valuation). Discuss the **value-function-arbitrariness critique** (the allocation cannot correct a poorly chosen `v`).

## 4.3 Axis 3 — Solution concept
Shapley (exact/MC) · Myerson · Harsanyi/interaction · Banzhaf · Weber/Harsanyi · core/nucleolus. Which concepts dominate in recommendation and why.

## 4.4 Axis 4 — Role of attribution
post-hoc explanation · in-training optimization signal · data/credit valuation · fairness/exposure correction · actionability/intervention. The thesis's "attribution as in-training signal" (RQ3) is a distinct role that few others play.

## 4.5 Axis 5 — Graph structure
bipartite · homogeneous GNN · KG/heterogeneous · hypergraph · dynamic/temporal.

## 4.6 The taxonomy tuple and a worked example
Each method maps to `(Axis1, Axis2, Axis3, Axis4, Axis5)`. **Worked example:** DyHuCoG = (interactions; multi-objective ranking+diversity+context; Shapley-MC; in-training signal; hypergraph). Walk one more external example to show the mapping is not self-serving.

---

# 5. SYSTEMATIC REVIEW BY CATEGORY

Organize the included works by player set (Axis 1), with sub-cases. For each category: what the game is, representative methods, what is established, and a one-sentence gap. Keep prose tight — the comparison tables (§6) carry the detail.

## 5.1 Player-centric review
- **Features as players** (SHAP-style feature attribution; the author's clustering-line precedent; why feature-attribution in recommendation is often off-model).
- **Interactions (u,i) as players** (DyHuCoG; in-training Shapley weighting).
- **Items / users as players** (node-level attribution; influential-user/item identification).
- **Contexts as players** (context-aware games).
- **Signal sources as players** (the SignalShap framing; source-level credit).
- **Data tuples as players** (data valuation for RS; Shapley pruning).
- **Providers as players** (fairness/exposure attribution — the FairShap framing).
- **Agents as players** (multi-agent LLM credit — the MHyperShap framing; TU-bandit creator-incentive games).

## 5.2 Structure-aware and interaction-aware solution concepts
Myerson/communication-graph values; Harsanyi interaction indices; how the graph structure constrains feasible coalitions — the natural bridge to MHyperShap.

## 5.3 Adjacent and emerging applications
GNN explainers for node/edge/classification (EdgeSHAPer, GraphSVX, GStarX, GISExplainer, GraphEXT) — clearly labelled *adjacent*, not core. TU-bandit creator-incentive games. Multi-agent/agentic systems.

---

# 6. COMPARATIVE ANALYSIS

## 6.1 Cross-method comparison
**Tables 2–4:** comparison matrices by axis — method × player set × `v(S)` × solution concept × role × graph type × approximation × evaluation × reproducibility. (At least 2 tables must place DyHuCoG alongside external game-theoretic recommender work, not just SHAP-on-tabular.)

## 6.2 What game theory buys / does not buy (SRQ3)
Critical synthesis. Genuine value: axiomatic fairness, principled credit under redundancy, interaction handling, in-training optimization, exposure/fairness, actionability. Degenerations: when "game-theoretic" is relabeled reweighting (attention/uniform masks with a Shapley name); the value-function-arbitrariness critique; computational intractability and the approximation problem. Cite the "Beyond Shapley Values" critique (Weber/Harsanyi sets) and answer it.

## 6.3 Validation and reproducibility (SRQ4)
Recurring gaps: faithfulness vs. actionability (the thesis's unmeasured-claim problem, generalized); stability across seeds/time; whether attribution is evaluated on held-out protocol; code/data release. **Include the DyHuCoG reproducibility caveat here** as an honest, field-level observation about evaluation standards.

---

# 7. EMPIRICAL BENCHMARK

Short, secondary. Design from `Implementation_Spec.md`. Do **not** turn this into a method bake-off.

## 7.1 Design
Backbones (hypergraph GNN + LightGCN), attribution families (`uniform`, `attention`, `heuristic-pop`, `shapley-mc`, `shapley-ai`, optional `myerson`), protocol (5 seeds, temporal LOO, BPR, Adam), datasets (MovieLens-1M, Amazon-Book), **primary metrics Recall@K and NDCG@K (K ∈ {5,10,20})**, secondary coverage/ILD, statistics (paired t-test, Holm–Bonferroni, Wilcoxon, Cohen's d_z). Reference `Implementation_Spec.md` §A.3–A.8 and §B.1a for the exact setup and the predicted result tables.

## 7.2 Results
**The paper's headline results are the Recall@K and NDCG@K tables** (Tables A/B/C in `Implementation_Spec.md` §B.1a): for every attribution family, on both backbones and both datasets, at K ∈ {5,10,20}, reported as mean ± std over 5 seeds. Present these as **Tables 5–7 / Figures 3–5** and lead the Results narrative with them:
- **Table 5 / Fig 3 — MovieLens-1M and Amazon-Book Recall@K / NDCG@K** across attribution families (primary). Highlight that `shapley-mc` ranks 1st on both metrics, with the largest margin on sparse Amazon-Book.
- **Table 6 — BQ2** (coverage / ILD), secondary to Table 5.
- **Table 7 — BQ3** (stability / cost) and **BQ4** (dense vs. sparse).
Report against the registered predictions (Part B of the implementation spec) and flag every miss explicitly.

## 7.3 The benchmark's relation to the taxonomy
Map each instantiated attribution family back to its taxonomy cell. This is the bridge that makes the benchmark *ground the survey* rather than sit apart from it. Concretely: the Recall@K/NDCG@K ordering across families is the empirical claim that answers the survey's central "what does game theory buy" question (SRQ3).

---

# 8. OPEN CHALLENGES AND RESEARCH AGENDA (SRQ5)

Organize by open problem:
- **Scalable, low-variance approximation** (learned proposal distributions, adaptive refresh) — thesis future-work R1.
- **Online / streaming attribution** — thesis future-work R2; incremental updates to `φ`.
- **Human-centred actionability evaluation** — thesis future-work R3; links to ActionShap.
- **Fairness / exposure audits** — thesis future-work R4; links to FairShap.
- **Structure-aware solution concepts** (Myerson, communication-graph) for graph recommendation — links to MHyperShap.
- **Agentic / multi-agent LLM systems** — links to MHyperShap.
- **Source-level attribution** — links to SignalShap.
Each challenge: one paragraph, the state of the art, why it is open, and the concrete direction. Explicitly frame this as the research agenda that the author's planned method papers address (without reporting their results).

---

# 9. CONCLUSION

Restate the five contributions against the SRQs and BQs. The survey gives the field a shared taxonomy, an honest critical assessment, a reproducible empirical grounding, and a roadmap. Close on the thesis-level claim, generalized: cooperative-game attribution provides a uniquely disciplined vocabulary for importance allocation in explainable graph-based recommendation — provided its choices are made explicit rather than assumed.

---

# DECLARATIONS (required by Discover AI — see `spec.md` §1.1)

- **Funding** — declare grant/none.
- **Competing interests** — "The authors declare no competing interests."
- **Ethics approval** — not applicable; no human/animal subjects. Survey + public-data benchmark; actionability discussed as an open challenge, not measured. No user study.
- **Consent** — not applicable.
- **Data availability** — MovieLens-1M and Amazon-Book are public; links and the rebuilt Amazon-Book split procedure disclosed.
- **Code availability** — repository link / Zenodo DOI for the benchmark.
- **Author contributions** — CRediT: Louhichi: conceptualization/methodology/software/writing; Nesmaoui: software/data/validation; Lazaar: supervision/analysis.
- **Use of AI tools** — include the group's standard statement (consistent with the group's *Explanation Drift* practice): ChatGPT/AI used for language clarity and code debugging, authors take full responsibility.
- **Dual publication / related work** — disclose prior related work (DyHuCoG, thesis) and that this survey is a new synthesis, not a repackaging; all reused content rewritten (see `spec.md` §1.1).

---

# APPENDICES

- **A.** Search string, full screening logs, PRISMA counts, and data-extraction form
- **B.** The complete list of reviewed works (index), with each mapped to a taxonomy tuple
- **C.** Solution-concept derivations (Shapley, Myerson, Harsanyi interaction indices) in one vocabulary
- **D.** Benchmark details: hyperparameter grids, seeds, runtime, full result tables
- **E.** Registered predictions and the realized-vs-predicted comparison table

---

# NOTATION LIST

Standalone table (matching the IJACSA/thesis convention): `N`, `S`, `v(S)`, `φ_i`, `φ̂_i`, `v_pref`, `α,β,γ,λ_pref`, `sim`, `M`, `f`, `NDCG`, `Recall`, `Coverage`, `ILD`, etc.

---

# PLANNED FIGURES & TABLES

| # | Type | Content |
|---|---|---|
| Fig 1 | Flow | PRISMA-style review flow (records → screened → eligible → included) |
| Fig 2 | Diagram | The five-axis taxonomy, with DyHuCoG mapped in as a worked example |
| Fig 3 | Bar | **Headline results**: NDCG@K (K=5,10,20) across attribution families per dataset |
| Fig 4 | Bar | **Headline results**: Recall@K (K=5,10,20) across attribution families per dataset |
| Fig 5 | Scatter/heat | BQ2/BQ3: coverage, ILD, stability, and cost (secondary to Fig 3–4) |
| Fig 6 | Matrix | Research agenda: open problems → planned directions (map to portfolio) |
| Tab 1 | Comparison | Positioning vs. prior surveys (novelty claim) |
| Tab 2–4 | Comparison | Cross-method comparison matrices by axis (DyHuCoG placed alongside external work) |
| Tab 5 | Results | **Recall@20 / NDCG@20 across all attribution families, both datasets (headline)** |
| Tab 6 | Results | Recall@5/10/20 and NDCG@5/10/20 full cutoffs for `shapley-mc` vs `uniform` |
| Tab 7 | Results | Coverage, ILD, stability, cost (BQ2/BQ3) and dense-vs-sparse (BQ4) |
| Tab 8 | Descriptive | Dataset statistics |

---

# PLANNING NOTES (NOT part of the manuscript)

## Why this is likely to be accepted at Discover AI (Review)
- The journal explicitly accepts comprehensive surveys of any length; this is a well-scoped one.
- Genuinely novel niche (no prior survey at this exact intersection), and the author is the natural author.
- Low risk: no new theorem, no human subjects, small benchmark, no ethics requirement; reuses the group's protocol.
- High leverage: it becomes the citation home for the author's planned method papers, and keeps the whole portfolio in one Q1 venue.

## Build order (each step independently checkable)
1. Lock scope + write §4 taxonomy (§1.1 and §5 of `spec.md`).
2. Run the systematic search + screening; produce PRISMA flow and extraction spreadsheet.
3. Write §2 preliminaries + §3 methodology + §5 taxonomy + §6 notation.
4. Write §5 systematic review + §6 comparison tables (the core content).
5. Implement + run the benchmark (`Implementation_Spec.md` Part A).
6. Write §7 benchmark results + §6.2/6.3 critical analysis + §8 agenda + §9 conclusion + abstract.
7. Compile bibliography, appendices, figures/tables; internal review (trajectory check: survey ≠ recap).
8. Assemble the Discover AI submission package (Review type, <250-word abstract, cover letter, all declarations — `spec.md` §13.1).

## What can be reused from existing work
- `stats.py` (paired tests, Holm–Bonferroni, Cohen's d_z) and clustering/quality diagnostics from `ActionShap/code/`.
- Thesis Ch.2/Ch.3 review material — as **source material only**, rewritten in survey voice, not copied.
- Backbone/metrics from the DyHuCoG line **only after** the reproducibility gaps are fixed or an independently documented backbone is used.

## Estimated effort
Roughly 6–10 weeks: ~3–4 weeks for the survey (on the thesis literature base) + ~2–4 weeks for the benchmark + ~1–2 weeks for integration, figures, tables, and the submission package.

## Decisions taken
| Decision | Choice | Consequence |
|---|---|---|
| Article type | **Review** at Discover AI (Q1) | Journal accepts surveys of any length; submits under `Review`, not "Research" |
| Benchmark role | Secondary, small | Grounds the taxonomy; kept out of scope for a method bake-off |
| Datasets | MovieLens-1M + Amazon-Book | Continuity with DyHuCoG; density contrast |
| Amazon-Book provenance | Rebuilt from raw corpus | Needed for temporal protocol; disclosed in §4.1 |
| DyHuCoG reuse | Fix/pin or fall back | Preserves the survey's reproducibility credibility |

## Remaining open questions
- Whether to include the optional `myerson` attribution family in the benchmark (adds the structure-aware cell to the empirical grounding, but adds compute). Default: include on MovieLens-1M only.
- Whether to add a third benchmark dataset (e.g., Yelp2018) — reserve for reviewer pressure; not in the default scope.
- Exact APC/budget confirmation at submission.
