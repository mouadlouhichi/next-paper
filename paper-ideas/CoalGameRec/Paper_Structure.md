# CoalGameRec — Full Paper Structure, TOC & Embedded Content

**Target journal:** *Discover Artificial Intelligence* (Springer Nature, open access). Quartile claims must cite a dated external source or be removed (review 2).
**Article type:** **Review** (the journal's dedicated survey article type — "critical accounts and comprehensive surveys of topics of major current interest… of any length"). Submit under `Review` in Snapp. **Because this package includes an empirical case study, the hybrid Review+benchmark format must be justified in the cover letter and Introduction; an editor may reclassify to `Research` or request the benchmark be split (review 1.2/2.1).**
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco
**Corresponding author:** mouad_louhichi@um5.ac.ma

**Structure:** systematic review + separately-scoped empirical case study. Target ≈ 10,000–12,000 words, **7 figures, 6 tables** in main text (detailed factorial tables + prediction register in supplementary Online Resources). The survey is the primary contribution; the case study is a short secondary section.

> **Why this paper, and why it fits Discover AI as a Review (private planning note — do NOT carry the portfolio language into the manuscript).** It targets a real gap: we hypothesize there is no dedicated survey at the intersection of *coalitional (cooperative) game theory × explainable AI × graph/hypergraph recommender systems*. General Shapley-XAI surveys are feature/classifier-centric; GNN-explainer work targets node/edge classification; game-theoretic recommender methods are scattered singles. The journal's **Review article type** accepts a comprehensive survey. The "uniquely placed author / citation home / planned method papers (ActionShap, FairShap, SignalShap, MHyperShap)" rationale is **private strategy only** — per review 1.4, keep it out of the manuscript and cover letter to avoid self-citation/self-promotion optics. The novelty claim ("first survey") is a **hypothesis pending the systematic search** (review 1.6/2.2), not an established fact.

> **Backbone dependency note (read before starting; review 1.3, authors' decision).** **The benchmark does NOT use DyHuCoG code.** DyHuCoG appears in this paper only as a **literature worked example in the taxonomy (§4.6)**, where it is coded as one case among several and described accurately (an internal audit found extraction gaps; treat this as an author-side implementation caveat, not a published field fact). The benchmark uses a **pinned independent backbone — HCCF (Hypergraph Contrastive Collaborative Filtering, Xia et al., SIGIR 2022)**, with a preregistered fallback rule; it does **not** rely on any DyHuCoG internals. The §4.6 worked example (DyHuCoG as a literature case) is decoupled from the benchmark backbone. See `Implementation_Spec.md` §A.4.

> **Prospective redesign after BPR-MF/MovieLens prototype (superseding older benchmark defaults).** The first five-seed BPR-MF/MovieLens pilot showed tiny Shapley-vs-uniform gains and a slightly stronger attention mean. The next preregistered empirical design therefore removes the additive preference term from the primary Shapley game (`λ_pref=0`), uses a smooth validation-only pairwise log-sigmoid coalition utility, treats diversity as a secondary/listwise reranking outcome rather than a primary coalition term, uses stratified bounded-player selection with sensitivity, adds `loo-marginal` as a serious control, and prefers backbone-native embedding aggregation over the external cosine kernel where available. The old `α=.70, β=.30, λ_pref=.20` NDCG/ILD coalition game is retained only as a diagnostic/sensitivity design, not as the prospective primary game.

---

## Working Title (primary + alternates)

- **Primary (selected):** *Coalitional Game Theory for Explainable Graph-Based Recommendation Systems: A Survey and Taxonomy*
- Alt 1: *From Features to Interactions: Coalitional-Game Attribution for Explainable Graph and Hypergraph Recommenders*
- Alt 2: *Cooperative-Game-Theoretic Explainability in Graph-Based Recommendation: A Systematic Review, Taxonomy, and Benchmark*
- Alt 3: *What Game Theory Buys (and Costs) for Explainable Graph Recommendation: A Taxonomy and Empirical Grounding*

*(The primary keeps the user's requested title; appending ": A Survey and Taxonomy" makes the article type unmistakable to the venue and reviewers.)*

## One-paragraph thesis (the spine)

Graph- and hypergraph-based recommender systems are increasingly accurate but opaque, and a growing body of work has turned to **coalitional (cooperative) game theory** — above all the Shapley value and its structure-aware and interaction-aware relatives — to make them explainable. Yet this work is scattered across disjoint papers that define *players*, *coalition value functions*, *solution concepts*, and *roles of attribution* inconsistently, and (to our knowledge, to be confirmed by the registered search) no synthesis currently organizes it. This paper provides a **systematic survey and taxonomy** of coalitional-game attribution for explainable graph-based recommendation, organized along five axes — player set, characteristic value function, solution concept, role of attribution, and graph structure — expressed in one shared vocabulary. (The "first survey" claim is a **hypothesis to be confirmed by the registered search**, review 1.6/2.2.) It critically assesses what the game-theoretic lens genuinely adds over heuristic and attention-based reweighting, where it degenerates into relabeled reweighting, and how attribution is (and is not) validated. A **separately-scoped empirical case study (planned for external preregistration)** instantiates the *interaction-player / ranking-utility* slice of the taxonomy on public source datasets processed under this paper's custom protocol; it is a supporting study, not the basis for the survey's field-wide claims. The result is both a map of the field and a research agenda argued from the literature's own gaps.

## Research questions of *this paper* (and their link to the thesis)

| RQ | Question | Thesis link |
|---|---|---|
| **SRQ1** | What distinct instantiations of coalitional games exist for explainable graph-based recommendation, along the axes of *player set · value function · solution concept · role · graph structure*? | Unifies the thesis's Ch.2/Ch.3 review into one taxonomy |
| **SRQ2** | What patterns and directions are reported across the coded corpus when methods are expressed in one shared vocabulary? (Synthesis: descriptive + structured vote/count — not correlation/meta-regression) | The thesis's "common attribution language" made systematic |
| **SRQ3** | What does the game-theoretic lens genuinely add over heuristic/attention/reweighting — and where is it relabeled reweighting? | Directly tests the thesis's central claim (Ch.8.2) |
| **SRQ4** | How is attribution validated (faithfulness, stability, actionability, reproducibility)? Which evaluation gaps recur? | Converts the thesis's acknowledged validation gaps (Ch.8.3/9.4) into a field-wide assessment |
| **SRQ5** | What are the highest-leverage open problems (approximation, online/streaming, human-centred evaluation, fairness audits, structure-aware solution concepts, agentic systems)? | Argued from the literature's own gaps (Ch.8.4); the author's planned work appears only as one direction among several (review 1.4) |
| **BQ1–BQ4** (case study) | In one interaction-player case study, does attribution-guided reweighting change held-out NDCG@K / HitRate@K (BQ1), coverage/ILD (BQ2), computational cost, MC approximation error, training-seed variability, and reranking-sensitivity (BQ3), and cross-dataset results (BQ4) vs. matched controls? | A separate study planned for external preregistration; it does not validate the whole taxonomy and does not decide SRQ3 (review 2.5) |

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
   3.1 Protocol and registration
   3.2 Information sources and database-specific searches
   3.3 Eligibility and evidence tiers
   3.4 Screening and data extraction
   3.5 Quality/risk-of-bias assessment and inter-coder agreement
   3.6 Synthesis approach
4. A taxonomy of coalitional games for explainable graph-based recommendation
   4.1 Taxonomy development and validation
   4.2 Axis 1 — Player set
   4.3 Axis 2 — Characteristic value function v(S)
   4.4 Axis 3 — Solution concept
   4.5 Axis 4 — Role of attribution
   4.6 Axis 5 — Graph structure
   4.7 The taxonomy tuple and a worked example
5. Systematic review by category
   5.1 Player-centric review (features → interactions → items/users → contexts → signal
       sources → data tuples → providers → agents)
   5.2 Structure-aware and interaction-aware solution concepts in graph recommendation
   5.3 Adjacent and emerging applications (GNN explainers; data valuation; multi-agent/agentic)
6. Comparative analysis
   6.1 Cross-method comparison (comparison tables by axis)
   6.2 What game theory buys / does not buy (critical synthesis; SRQ3)
   6.3 Validation and reproducibility: recurring evaluation gaps (SRQ4)
   6.4 Evidence-strength summary
7. Separately-scoped empirical case study (planned for external preregistration)
   7.1 Preregistered questions and estimands
   7.2 Data and leakage controls
   7.3 Backbones and attribution families
   7.4 Characteristic value and reranking operator
   7.5 Statistical analysis
   7.6 Results
   7.7 Sensitivity and feasibility results
   7.8 The case study's relation to one taxonomy slice (not the whole taxonomy)
8. Open challenges and research agenda (SRQ5, argued from the literature)
9. Limitations
10. Conclusion
Declarations (required by Discover AI)
Appendices
Notation list
```

---

# ABSTRACT (draft, must be < 250 words — re-verify final count at submission)

Graph- and hypergraph-based recommender systems are increasingly accurate yet opaque, and a growing literature uses coalitional (cooperative) game theory — above all the Shapley value and its structure-aware, interaction-aware, and communication-graph variants — to attribute recommendation outcomes to players such as features, interactions, items, users, contexts, signal sources, and providers. This work is scattered across papers that define players, coalition value functions, solution concepts, and the role of attribution inconsistently, and (to be confirmed by our systematic search) no synthesis currently organizes it. We present a systematic review and taxonomy of coalitional-game attribution for explainable graph-based recommendation, organizing the field along five axes — player set, value function, solution concept, role of attribution, and graph structure — and expressing surveyed methods in a shared vocabulary. We critically assess what the game-theoretic lens adds over heuristic and attention-based reweighting, where it degenerates into relabeled reweighting, and how attribution is (and is not) validated, including faithfulness, stability, actionability, and reproducibility. To illustrate the framework we report a separately-scoped empirical case study (planned for external preregistration) instantiating the interaction-player / ranking-utility slice of the taxonomy on MovieLens-1M and Amazon-Book, reporting HitRate@K and NDCG@K (K = 5, 10, 20). We close with an agenda of open problems — scalable low-variance approximation, online attribution, human-centred actionability evaluation, fairness audits, structure-aware solution concepts, and agentic systems.

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
4. **A separately-scoped empirical case study (planned for external preregistration)** instantiating the interaction-player / ranking-utility slice of the taxonomy, reporting **HitRate@K and NDCG@K (K ∈ {5,10,20}) as the primary realized result tables** (BQ1–BQ4) with matched controls. It is a supporting study, not the basis for the survey's field-wide claims (review 2.5).
5. **A research agenda** (SRQ5) argued from the literature's own gaps, with concrete directions (the author's planned work appears as one direction among several, review 1.4).

## 1.4 Relationship to prior surveys (positioning table)
**Table 1.** Separate blocks: (a) prior review/survey articles (general Shapley-XAI, Shapley-in-ML/data-analytics, GNN explanation, recommender explainability), and (b) representative primary near-neighbor methods clearly labelled as primary studies rather than surveys (ShaRP, data valuation for RS, TU-bandit creator-incentive games, graph/hypergraph recommender attribution). Columns: *scope = recommendation ranking*, *coalitional solution concepts (beyond SHAP feature attribution)*, *taxonomy*, *graph/hypergraph focus*, *empirical grounding*. The novelty claim is that this combination is not covered by any single prior work — **to be confirmed by the systematic search**, not asserted (review 1.6/2.2).

## 1.5 Organization
One short roadmap paragraph naming sections 2–10.

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
Bipartite user–item; homogeneous GNN (LightGCN); knowledge-graph/heterogeneous; hypergraph; dynamic/temporal. Define the top-N problem and the metrics (NDCG, Recall, coverage, ILD). **Note the venue/journal scope and define the recommendation task clearly.**

## 2.4 Explainability in recommendation
Post-hoc explanation · in-training optimization · fairness/exposure · actionability. Discuss evaluation of explanations (faithfulness, sufficiency/comprehensiveness, stability, actionability) — this is where SRQ4 is set up.

---

# 3. SURVEY METHODOLOGY (SYSTEMATIC REVIEW PROTOCOL)

## 3.1 Protocol and registration
Report the external protocol registry, timestamp, version, protocol-deviation policy, final update date, and whether the review protocol was registered before screening. Until those artifacts exist, write this section in future/protocol language only.

## 3.2 Information sources and database-specific searches
Report exactly (reproducible): ACM DL, IEEE Xplore, Scopus, WoS, arXiv, SpringerLink, ScienceDirect, plus the fixed Google Scholar citation-chasing procedure from `spec.md`. Give database-specific queries, field tags, search dates, export formats, deduplication keys, and time window 2010–2026 (methods), plus seminal background (Shapley 1953; Myerson 1977; Harsanyi 1963; Banzhaf 1965).

## 3.3 Eligibility and evidence tiers
State inclusion/exclusion criteria verbatim and define Core, Adjacent A, Adjacent B, Adjacent C, and Background-only evidence tiers. Core/adjacent status is based on task/graph/game eligibility, not citation count or popularity.

## 3.4 Screening and data extraction
Document independent screening, adjudication, and the per-paper extraction form (citation · venue/year · task · graph type · player set · value function · solution concept · role · approximation · datasets · baselines · metrics · result · reproducibility flag).

## 3.5 Quality/risk-of-bias assessment and inter-coder agreement
**Figure 1.** PRISMA 2020 flow (records identified → screened → eligible → included) with counts, plus PRISMA-S search reporting. **Quality/risk-of-bias assessment is required (not optional)** — score each included study on the §4.5 rubric of `spec.md`; use weighted Cohen's kappa for ordinal quality domains and a separate nominal agreement statistic for categorical taxonomy fields. Record screening counts, disagreements, and agreement measure; report included core vs. adjacent works separately.

## 3.6 Synthesis approach
State explicitly that the synthesis is descriptive plus structured vote/count. Heterogeneous tasks, metrics, and protocols prevent a defensible pooled effect-size meta-analysis; vote/count summaries are qualified by quality/risk-of-bias domains and publication-status labels.

---

# 4. A TAXONOMY OF COALITIONAL GAMES FOR EXPLAINABLE GRAPH-BASED RECOMMENDATION

This is the intellectual centerpiece. Define the five axes, each with a definition, the value set, and representative methods (with the author's DyHuCoG mapped in).

## 4.1 Taxonomy development and validation
Explain that the five axes are the initial deductive framework; categories are tested against the coded corpus; revisions are logged; multi-label coding is permitted; overlapping labels are operationally distinguished; and inter-coder agreement is measured. This prevents the taxonomy from reading as an author-imposed conceptual list rather than a validated coding framework.

## 4.2 Axis 1 — Player set
`features · interactions (u,i) · items · users · contexts · signal sources · data tuples · nodes · edges/hyperedges · providers · agents`. Each player type → what game it induces and what explanation it yields. **Define allowed values and multi-label rules; note that some labels overlap** ("edge"/"interaction"/"user–item tuple" may be the same player; "node" may be a user/item/context node; review 2.3/6.1).

## 4.3 Axis 2 — Characteristic value function `v(S)`
What a coalition earns: NDCG/ranking utility · diversity (ILD) · coverage · context-alignment · preference-consistency · fairness/exposure · combined multi-objective (DyHuCoG Eqs. 1–2) · regret (bandit games) · model accuracy (data valuation). Discuss the **value-function-arbitrariness critique** (the allocation cannot correct a poorly chosen `v`).

## 4.4 Axis 3 — Solution concept
Shapley (exact/MC) · Myerson · Harsanyi/interaction · Banzhaf · Weber/Harsanyi · core/nucleolus. Which concepts dominate in recommendation and why.

## 4.5 Axis 4 — Role of attribution
post-hoc explanation · in-training optimization signal · data/credit valuation · fairness/exposure correction · actionability/intervention. The thesis's "attribution as in-training signal" (RQ3) is a distinct role that few others play.

## 4.6 Axis 5 — Graph structure
bipartite · homogeneous GNN · KG/heterogeneous · hypergraph · dynamic/temporal.

## 4.7 The taxonomy tuple and worked examples
Each method maps to `(Axis1, Axis2, Axis3, Axis4, Axis5)`. **Work through several external examples first** (e.g., a data-valuation method, a GNN explainer, a TU-bandit game), then code the author's DyHuCoG as **one case among them** — explicitly marked as the authors' own method so the mapping is not self-serving (review 1.4/4.3).

---

# 5. SYSTEMATIC REVIEW BY CATEGORY

Organize the included works by player set (Axis 1), with sub-cases. For each category: what the game is, representative methods, what is established, and a one-sentence gap. Keep prose tight — the comparison tables (§6) carry the detail.

## 5.1 Player-centric review
- **Features as players** (SHAP-style feature attribution; the author's clustering-line precedent; why feature-attribution in recommendation is often off-model).
- **Interactions (u,i) as players** (DyHuCoG; in-training Shapley weighting).
- **Items / users as players** (node-level attribution; influential-user/item identification).
- **Contexts as players** (context-aware games).
- **Signal sources as players** (source-level credit for hybrid recommenders).
- **Data tuples as players** (data valuation for RS; Shapley pruning).
- **Providers as players** (fairness/exposure attribution).
- **Agents as players** (multi-agent LLM credit; TU-bandit creator-incentive games).
(Any references to the authors' planned methods — SignalShap/FairShap/MHyperShap — are **clearly labelled as the authors' own unpublished directions**, not field categories; review 2 §9.)

## 5.2 Structure-aware and interaction-aware solution concepts
Myerson/communication-graph values; Harsanyi interaction indices; how the graph structure constrains feasible coalitions.

## 5.3 Adjacent and emerging applications
GNN explainers for node/edge/classification (EdgeSHAPer, GraphSVX, GraphGI, GStarX, GISExplainer, GraphEXT) — **Adjacent B**, clearly labelled, with exact references (review 2 §9). TU-bandit creator-incentive games (Adjacent C). Multi-agent/agentic systems (Adjacent C). Report adjacent counts separately from the core corpus (review 2.3).

---

# 6. COMPARATIVE ANALYSIS

## 6.1 Cross-method comparison
**Tables 2–4:** comparison matrices by axis — method × player set × `v(S)` × solution concept × role × graph type × approximation × evaluation × reproducibility. (At least 2 tables must place DyHuCoG alongside external game-theoretic recommender work, not just SHAP-on-tabular.)

## 6.2 What game theory buys / does not buy (SRQ3)
Critical synthesis. Possible genuine value (argued, not assumed): principled credit under redundancy, interaction handling, in-training optimization, exposure allocation. **Do not present axiomatic fairness, faithful explanation, or actionability as automatic consequences of Shapley values** — state that axioms govern allocation conditional on a chosen game and do not select the player representation, missingness/baseline distribution, coalition semantics, or quantity of interest (review 2.6.2). Degenerations: when "game-theoretic" is relabeled reweighting; the value-function-arbitrariness critique; computational intractability and the approximation problem. Cite the "Beyond Shapley Values" critique (Weber/Harsanyi sets) and answer it.

## 6.3 Validation and reproducibility (SRQ4)
Recurring gaps: faithfulness vs. actionability (the thesis's unmeasured-claim problem, generalized); stability across seeds/time; whether attribution is evaluated on held-out protocol; code/data release. **On the DyHuCoG reproducibility caveat (6.6):** report it precisely as "An audit of one reviewed method identified reporting gaps; this illustrates a reproducibility risk but does not estimate its prevalence." It is an **author-side case**, not a field-level statistical observation, unless the audit is archived and independently verifiable.

## 6.4 Evidence-strength summary
For each major synthesis claim, identify whether the evidence is consistently reported, mixed, supported mainly by adjacent studies, methodologically weak, or not evaluated. This links the quality table to the conclusions and prevents structured vote/count synthesis from treating all studies as equally reliable.

---

# 7. EMPIRICAL CASE STUDY

A **separately-scoped empirical case study (planned for external preregistration)** (secondary to the review). Design from `Implementation_Spec.md`. Do **not** turn this into a method bake-off, and do **not** present it as empirical validation of the whole taxonomy (review 2.5).

## 7.1 Preregistered questions and estimands
State the BQs, four confirmatory HCCF contrasts, validation-only coalition-value target, test-only final evaluation target, and conditional user-population statistical estimand.

## 7.2 Data and leakage controls
Describe MovieLens-1M and custom Amazon-Book source processing, train-period-only 5-core filtering, temporal leave-one-out, full-catalogue candidates, train-only `x_i`, and the no canonical-split comparison rule.

## 7.3 Backbones and attribution families
Backbones (**HCCF selected as the primary backbone; a port will be pinned and validated before preregistration — no DyHuCoG code, DOI 10.1145/3477495.3532058, official repo github.com/akaxlh/HCCF — + secondary LightGCN**; HCCF's standard contrastive training is retained identically for all families). Attribution families with a **fixed hierarchy: Primary families** = `uniform` (control), `additive-pref` (matched non-game heuristic), `shapley-mc` (game-theoretic); **Secondary controls** = `attention` (fixed post-hoc attention-style similarity weighting; no learned parameters), `heuristic-pop`; **Exploratory** = `shapley-ai`, `myerson` (every promised cell run or dropped from headline claims). Protocol: 5 seeds, temporal leave-one-out, Adam, **full-catalogue evaluation** (no model-generated candidate pool). Datasets: MovieLens-1M, Amazon-Book (custom sample). **Primary metrics HitRate@K and NDCG@K (K ∈ {5,10,20})**, secondary coverage/ILD. **Statistics (selected): confirmatory tests = `shapley-mc` vs `uniform` on NDCG@20 & HitRate@20 × 2 datasets on the HCCF backbone = 4 tests in one Holm family**; all other cells secondary/exploratory; per-user paired differences within seed + seed-clustered bootstrap inference (`Implementation_Spec.md` §A.10). Reference `Implementation_Spec.md` §A.2–§A.10 and §B.1a. All families share a **matched objective and tuning budget**.

## 7.4 Characteristic value and reranking operator
Define `v_pref,u`, `Shapley(v_pref,u)`, family-specific weights, cached full-graph base scores, and the post-hoc kernel reranking operator. State that refreshed in-training attribution is outside the primary design and only an optional exploratory Study C.

## 7.5 Statistical analysis
Define the seed-clustered bootstrap, Holm family, descriptive seed variability, user-conditional `d_z`, and secondary/sensitivity status of all non-primary comparisons.

## 7.6 Results
The primary realized results are the **HitRate@K and NDCG@K (K ∈ {5,10,20}) tables** for the **primary families** (`uniform`, `additive-pref`, `shapley-mc`) on the **HCCF primary backbone, both datasets**, reported as mean ± std over seeds with significance flagged after the selected Holm correction (§A.10). **These tables are filled only with realized numbers; no predicted values appear as results** (review 1.1/2.1). Present per the figure/table map and report the measured ordering neutrally — **do not assume or instruct that `shapley-mc` ranks 1st**; report whichever ordering occurs (review 2.1):
- **Table 5 / Fig 3 — NDCG@K** across primary families per dataset (primary).
- **Table 6 / Fig 4 — HitRate@K** across primary families per dataset (primary).
- **Fig 5 — coverage/ILD (BQ2)**; **Fig 6 — cost/stability (BQ3)**; **Fig 7 — research agenda**.
- **Online Resource 1** — coverage/ILD/cost detail (BQ2–BQ4), dataset statistics, full factorial results + prediction register (distinct labels, not "Table A/B/C").
Report against the registered predictions (Part B) and flag every miss explicitly.

## 7.7 Sensitivity and feasibility results
Report value-function sensitivity, reranking-strength sensitivity, MC convergence, HCCF validation, runtime/memory, and feasibility-pilot outcomes descriptively, without adding them to the four-test Holm family.

## 7.8 The case study's relation to one taxonomy slice
Map the instantiated attribution families back to their taxonomy cells. This is an illustration of the interaction-player / ranking-utility slice only; it does **not** empirically validate the full taxonomy or answer SRQ3 (what game theory buys across the field). SRQ3 is answered by the survey's critical synthesis (§6.2); the case study provides one concrete, reproducible instance (review 2.5). If explanation quality is claimed, add the explanation metrics of §A.7b — otherwise frame this as an intervention study.

---

# 8. OPEN CHALLENGES AND RESEARCH AGENDA (SRQ5)

Argue each open problem from the **literature's own stated gaps**, with representative external evidence (review 1.4). The author's planned work, where it happens to align, appears as **one possible direction among several from independent groups** — never as the section's organizing structure, and never framed as "links to our upcoming papers."
- **Scalable, low-variance approximation** (learned proposal distributions, adaptive refresh).
- **Online / streaming attribution** (incremental updates to `φ`).
- **Human-centred actionability evaluation** (how explanations affect analyst/user decisions — flagged as generally unmeasured, review 4.3).
- **Fairness / exposure audits** (explicit group/provider metrics — coverage/ILD alone are not fairness, review 4.3).
- **Structure-aware solution concepts** (Myerson/communication-graph) for graph recommendation.
- **Agentic / multi-agent LLM systems** (credit assignment across agents).
- **Source-level attribution** for hybrid recommenders.
Each challenge: one paragraph — state of the art, why it is open, and concrete directions.

---

# 9. LIMITATIONS

A dedicated Limitations section is required (review 4.3/10.2). Cover: search bias and publication bias (preprints vs. peer-reviewed); heterogeneous tasks/protocols across the corpus; the custom Amazon preprocessing and its non-comparability to published splits; that the case study covers only one taxonomy slice; no human evaluation of actionability; no fairness audit; **training-randomness limitation** (only five seeds, with inference conditional on those trained models); **reviewer-author dependence** (the authors review their own methods and may have privileged implementation knowledge unavailable for external papers); author-conflict of interest in reviewing own methods; the value-function-arbitrariness sensitivity. State where "not reported" differs from "not applicable," and distinguish absent evidence, unreported evidence, inapplicable evaluation, and evidence actively showing no effect.

# 10. CONCLUSION

Restate the contributions against the SRQs and BQs. The survey gives the field a shared taxonomy, an honest critical assessment, and a roadmap; the case study provides one reproducible illustration. Close on the thesis-level claim, generalized and qualified: cooperative-game attribution offers a disciplined vocabulary for importance allocation in explainable graph-based recommendation — provided its choices (player set, value function, baseline, coalition semantics) are made explicit rather than assumed (review 2.6.2).

---

# DECLARATIONS (required by Discover AI — see `spec.md` §1.1)

- **Funding** — declare grant/none (actual, not placeholder; review 2).
- **Competing interests** — complete financial and non-financial disclosure; at minimum disclose the authors' prior authorship of reviewed methods (DyHuCoG) and the benchmark's relationship to those methods. Do not pre-fill "no competing interests" (review 2, P1).
- **Ethics / human data (review 2, P0)** — do **not** assert "not applicable" merely because there is no user study. MovieLens and Amazon data are human-generated; report the institutional ethics determination (datasets/fields used, public/pseudonymous status, exemption/approval reference, identifier protection). Actionability is discussed as an open challenge, not measured.
- **Consent** — per the institutional determination.
- **Data availability** — dataset citations, URLs, access dates, checksums/versions, final counts, split-generation code, license/terms; state whether raw data are redistributed (review 2).
- **Code availability** — actual repository/Zenodo DOI with code, configs, frozen splits, raw results, lockfile, tests, permanent archive (not a placeholder).
- **Author contributions** — actual, post-completion CRediT reflecting the writing-heavy survey and largely-reused benchmark code; every author's approval (review 2).
- **Use of AI tools** — transparent, risk-based disclosure of actual AI use (drafting, critique, code assistance) with human accountability; AI not an author (review 2).
- **Dual publication / text-recycling** — disclose prior related work (DyHuCoG, thesis) and that this survey is a new synthesis; reused formulations carry in-text attribution (e.g., "we adopt the value function of [DyHuCoG, Eq. X]"); an overlap table is prepared (review 1.5/2).

---

# APPENDICES

- **A.** Search string, full screening logs, PRISMA counts, and data-extraction form
- **B.** The complete list of reviewed works (index), with each mapped to a taxonomy tuple
- **C.** Solution-concept derivations (Shapley, Myerson, Harsanyi interaction indices) in one vocabulary
- **D.** Benchmark details: hyperparameter grids, seeds, runtime, full result tables
- **E.** The realized-vs-predicted comparison (predictions live in the external pre-registration; this appendix reports realized results and deviations). Prediction tables are kept **separate** from results, with distinct labels (review 2.2/C18).

---

# NOTATION LIST

Standalone table (synchronized with `spec.md` §6). Include and disambiguate: `p` (generic player; `i` reserved for item in `(u,i)`, `u` user), `N_u` (per-user player set), `S_u` (per-user coalition), `v_u(S_u)`, `v_pref,u`, `φ_{u,p}` (per-user attribution), `Φ_i` or other explicitly defined aggregate summaries (not a naive global average of user-specific interaction IDs), `φ̂` (MC estimate), `α, β, λ_pref` (with `Context` removed, `α+β=1`), `sim(u,i)` (user–item similarity), `sim(i,j)` (item–item similarity for ILD), `M`/`m` (MC sample count/index — not player count), `freq` (refresh frequency, not the model function `f`), `K`/`k` (cutoff/rank), `g` (communication graph), `R_u`, `U`, `I`, `rel`, the full-catalogue candidate set, baseline value `v(∅)`, and exact vs. MC attribution (review 6.3/7.1).

---

# PLANNED FIGURES & TABLES

| # | Type | Content |
|---|---|---|
| Fig 1 | Flow | PRISMA 2020 flow (records → screened → eligible → included) |
| Fig 2 | Diagram | The five-axis taxonomy, with several external worked examples then DyHuCoG coded as one case |
| Fig 3 | Bar | Primary results: NDCG@K (K=5,10,20) across attribution families per dataset (realized) |
| Fig 4 | Bar | Primary results: HitRate@K (K=5,10,20) across attribution families per dataset (realized) |
| Fig 5 | Bar | Secondary: coverage and ILD across primary families (BQ2) |
| Fig 6 | Scatter | Cost/stability: post-hoc attribution time, latency, memory, estimator variance (BQ3) |
| Fig 7 | Matrix | Research agenda: open problems → directions (argued from the literature, not a portfolio map) |
| Tab 1 | Comparison | Positioning vs. prior surveys + known near-neighbors (qualified novelty claim) |
| Tab 2–4 | Comparison | Cross-method comparison matrices by axis (DyHuCoG placed alongside external work) |
| Tab 5 | Results | NDCG@K (K=5,10,20) across primary families, both datasets (realized, primary) |
| Tab 6 | Results | HitRate@K (K=5,10,20) across primary families, both datasets (realized, primary) |
| Online Resource 1 | Supplement | Coverage/ILD/stability/cost detail (BQ2–BQ4), dataset statistics, full factorial results + prediction register (distinct labels, not "Table A/B/C"; review C18) |

---

# PLANNING NOTES (NOT part of the manuscript)

## Why this is likely to be accepted at Discover AI (Review) — private assessment
- The journal explicitly accepts comprehensive surveys of any length; this is a well-scoped one.
- Plausibly a novel niche, **to be confirmed by the systematic search** (not asserted beforehand).
- Low technical risk: no new theorem; modest benchmark compute.
- **But:** the hybrid Review+benchmark format and the human-data ethics determination are open risks to address in the cover letter and before submission (review 1.2/2.1/2.7). The "citation home / portfolio" rationale must not appear in the manuscript (review 1.4).

## Build order (each step independently checkable; aligns with `spec.md` §12)
1. Freeze scope + register the systematic-review protocol; run the search + screening (PRISMA 2020) **before** locking the taxonomy.
2. Code + validate the taxonomy against the corpus; write §2–§5.
3. Obtain the ethics determination for the human-generated data.
4. Run the Amazon feasibility spike + synthetic Shapley test.
5. Freeze the benchmark estimand/splits/baselines/stats; register predictions externally.
6. Implement + run the benchmark; write realized results only.
7. Write critical analysis + case-study results + agenda + limitations + conclusion + abstract (<250 words).
8. Compile bibliography, appendices, figures/tables; internal review (survey ≠ recap; survey independent of benchmark outcome).
9. Assemble the Discover AI submission package (Review type + justified hybrid format, <250-word abstract, cover letter, ethics determination, all declarations — `spec.md` §13.1).

## What can be reused from existing work
- `stats.py` (paired tests, Holm–Bonferroni, Cohen's d_z) and clustering/quality diagnostics from `ActionShap/code/`.
- Thesis Ch.2/Ch.3 review material — as **source material only**, rewritten in survey voice, not copied.
- **No DyHuCoG code.** The benchmark backbone is the pinned HCCF hypergraph GNN (official public implementation, pinned commit; preregistered HGNN fallback rule — `Implementation_Spec.md` §A.4).

## Estimated effort
Roughly 8–14 weeks: ~4–5 weeks for the systematic review (protocol, search, screening, extraction, taxonomy coding), ~3–5 weeks for the case-study benchmark (planned for external preregistration) (incl. Amazon preprocessing and the hypergraph-backbone decision), ~1–2 weeks for integration/figures/tables, ~1–2 weeks for the submission package (incl. ethics determination, formatting, cover letter). The benchmark is not "cheap" — budget it as a real empirical study (review 1/4.1).

## Decisions taken
| Decision | Choice | Consequence |
|---|---|---|
| Article type | **Review** at Discover AI | Journal accepts surveys; but justify the hybrid Review+benchmark format, and expect possible reclassification to `Research` (review 1.2/2.1) |
| Benchmark role | Secondary, small | Grounds the taxonomy; kept out of scope for a method bake-off |
| Datasets | MovieLens-1M + Amazon-Book | Standard public benchmarks for a dense/sparse contrast (independent of any prior codebase) |
| Amazon-Book provenance | Rebuilt from raw corpus | Needed for temporal protocol; disclosed in §4.1 |
| DyHuCoG code in benchmark | **Not used** | Benchmark uses the pinned HCCF backbone (independent); DyHuCoG appears only as a taxonomy worked example (§4.6) |

## Remaining open questions
- Whether to include the exploratory `myerson` / `shapley-ai` families. **If included, every promised cell must be run (no partial factorial — review C4);** otherwise drop them from headline claims. Do **not** default to "MovieLens-1M only," which creates a partial factorial.
- Whether to add a third benchmark dataset (e.g., Yelp2018) — reserve for reviewer pressure; not in the default scope.
- Whether explanation-quality metrics (§A.7b) are added or the case study is explicitly framed as intervention-only (review 1.7/2.5).
- Confirm the current Discover AI APC, median decision time, and article-type policy at submission (review 2).
