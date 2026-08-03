# CoalGameRec — Specification for "Coalitional Game Theory for Explainable Graph-Based Recommendation Systems"

**Status:** design specification, v1.1 (retargeted to **Q1 Discover Artificial Intelligence**)
**Paper type:** Review article (systematic survey + small empirical benchmark of coalitional-attribution methods)
**Target venue (fixed):** *Discover Artificial Intelligence* — Springer Nature, **open access**. Scopus-indexed; **CiteScore 2024 = 6.0**; Q1 quartile. Indexed in DOAJ, Ei Compendex and Scopus. **Accepts a dedicated `Review` article type** ("critical accounts and comprehensive surveys of topics of major current interest within the scope of the journal… of any length"), which is a precise fit for this paper's survey-first structure. Median first decision ≈ 23 days. **APC: US$1,690 / €1,390 / £1,190** (confirm current APC at submission).
**Corresponding author:** Mouad Louhichi (ENSIAS, UM5 Rabat) — `mouad_louhichi@um5.ac.ma`

> **Why this venue.** The author's planned method papers (ActionShap, FairShap, SignalShap, MHyperShap) are already aimed at *Discover Artificial Intelligence* (per the `paper-ideas/` blueprints). Targeting the **same Q1 journal** for this survey keeps the entire next-paper portfolio in one venue, gives every method paper a ready-made citation home, and uses the journal's dedicated **Review article type** so a field-wide survey is accepted *as such* (not forced into a "Research" slot). The journal's scope ("all aspects of artificial intelligence in theory and application") comfortably covers coalitional game theory + XAI + recommender systems.

> **Scope note.** This paper is a **field-wide survey/taxonomy** with a **small empirical benchmark**, not a new method paper. It introduces no acronym (unlike DyHuCoG/FairShap/ActionShap/SignalShap/MHyperShap). Its novelty is the *synthesis*: a taxonomy of coalitional (cooperative) games instantiated for explainable graph/hypergraph recommendation, a systematic review of the field, a comparative/critical analysis, and a reproducible empirical comparison of the taxonomy's main attribution families. The paper must read as a synthesis of the field in which the author's own work (DyHuCoG) is one case study among many — **not** as a recap of the thesis.

---

## 1. Purpose and central question

**Central question.** How are coalitional (cooperative) game-theoretic models instantiated to make graph- and hypergraph-based recommender systems explainable, and what — empirically and theoretically — does this attribution lens actually buy (and cost)?

The paper has **two deliverables**:

1. **Survey deliverable (primary).** A systematic, reproducible review of coalitional-game attribution for explainable graph-based recommendation, organized by a novel taxonomy, with comparison tables and a critical synthesis. This is the novel, citable contribution. **Framed for the `Review` article type of Discover Artificial Intelligence.**
2. **Benchmark deliverable (secondary, lightweight).** A small, reproducible empirical study that instantiates the taxonomy's main attribution families on standard recommendation datasets and compares them to non-game-theoretic baselines under a shared protocol. This grounds the taxonomy, makes the survey stronger than an annotated bibliography, and reuses existing codebases.

---

## 1.1 Target journal and submission readiness — Discover Artificial Intelligence (Q1)

**Journal profile (verified against Springer's current journal page, Aug 2026).**
- **Name:** *Discover Artificial Intelligence* (Springer Nature, Discover Series). ISSN 2731-0809.
- **Model:** Gold **open access**; **APC** (currently US$1,690 / €1,390 / £1,190).
- **Indexing / ranking:** **Scopus** (Q1 in Information Systems; Q2 in Artificial Intelligence), **Ei Compendex**, **DOAJ**. **CiteScore 2024 = 6.0**.
- **Peer review:** single-anonymous; at least two reviewers; technical + suitability checks before review; decisions (major/minor revisions, reject) with editor re-evaluation on minor revisions.
- **Scope fit:** "all aspects of artificial intelligence in theory and application" — directly covers coalitional game theory + explainable AI + graph-based recommender systems.
- **Article-type fit:** the journal explicitly accepts a **`Review`** article type — *"Review articles provide critical accounts and comprehensive surveys of topics of major current interest within the scope of the journal… we accept Review articles of any length."* This is the article type this paper must be submitted under. (Do **not** submit as a generic "Research" article.)

**Submission mechanics (Snapp — Springer Nature tracking system).**
- Upload: manuscript file + **abstract < 250 words** + **cover letter** (context/importance of the work, why it fits the journal). Optional: figures, tables, supplementary material.
- Choose article type **`Review`** in the Details tab.
- If a relevant topical collection is open, submit via that collection (same mechanics).

**Required statements (mandatory for this journal) — must be drafted before submission.**
- **Funding statement** (mandatory) — declare funding or "no funding received".
- **Author contributions** statement (CRediT split; reuse DyHuCoG pattern: Louhichi: conceptualization/methodology/software/writing; Nesmaoui: software/data/validation; Lazaar: supervision/analysis).
- **Competing interests** statement — "The authors declare no competing interests."
- **Data availability** statement — datasets used in the benchmark (MovieLens-1M, Amazon-Book) are public; code + results released.
- **Code availability** statement — GitHub/Zenodo DOI for the benchmark code and results.
- **Ethics statement** — *not applicable* (no human/animal subjects; survey + public-data benchmark; declare this explicitly). This survey involves **no user studies** — actionability is discussed as an open challenge, not measured.

**Formatting (Journal formatting guidelines — Springer Nature Discover Series).**
- Manuscript templates and style requirements per the journal's Submission Guidelines; confirm current template. Reference manager, figure/table formatting, and reference style follow Springer Nature guidelines.
- Language: standard academic English (Springer recommends professional language editing; check Step 5).

**Dual-publication note.** Because much of the survey's content overlaps the author's own prior publications and thesis (DyHuCoG, clustering papers, PhD thesis), the manuscript must be written in survey voice and **no text, equation, or figure may be reused verbatim** from those works (see §11 self-plagiarism/overlap mitigation). Disclose prior related work appropriately in the cover letter if asked.

**Relationship to prior work.** This paper fills a real, currently-unoccupied gap (verified by literature search, Aug 2026):
- General Shapley-XAI surveys (e.g., Zhao et al. 2025, *Dyn. Games & Appl.*; Li et al. 2024, *Auton. Intell. Syst.*) are feature/classifier-centric and do not target ranking/recommendation.
- GNN-explanation work (EdgeSHAPer, GraphSVX, GraphGI, GStarX, GISExplainer, GraphEXT) targets node/edge/graph classification, not top-N recommendation.
- Game-theoretic recommender methods are scattered singles (Shapley community-CF; Shapley data valuation/pruning for RS; TU-bandit creator-incentive games; DyHuCoG) with no unifying survey.

---

## 2. Research questions (SRQs)

- **SRQ1 (taxonomy).** What distinct instantiations of coalitional games exist for explainable graph-based recommendation, along the axes of *player set · characteristic value function · solution concept · role of attribution · graph structure*?
- **SRQ2 (comparison).** How do surveyed methods compare when expressed in one shared vocabulary, and what correlates with reported gains?
- **SRQ3 (value analysis).** What does the game-theoretic lens genuinely add over heuristic/attention/reweighting — and where is it relabeled reweighting?
- **SRQ4 (validity).** How is attribution validated (faithfulness, stability, actionability, reproducibility)? Which evaluation gaps recur?
- **SRQ5 (roadmap).** What are the highest-leverage open problems (approximation, online/streaming, human-centred evaluation, fairness audits, structure-aware solution concepts, agentic systems)?

**Benchmark questions (BQ, secondary).**
- **BQ1.** Do Shapley-weighted propagation and reranking improve ranking quality (NDCG@20, Recall@20) over heuristic/attention/uniform-weight baselines in graph recommenders?
- **BQ2.** Do Shapley-based weights improve catalogue coverage and intra-list diversity without sacrificing accuracy?
- **BQ3.** Does the game-theoretic module remain training-stable and tractable (runtime, memory) relative to baselines?
- **BQ4.** Do game-theoretic attributions generalize across a dense (MovieLens-1M) and a sparse (Amazon-Book) regime?

---

## 3. Scope — in scope / out of scope

### In scope
- **Tasks:** top-N item recommendation under implicit (or converted-implicit) feedback.
- **Graph structures:** bipartite user–item; homogeneous GNN; knowledge-graph/heterogeneous; hypergraph; dynamic/temporal. (Hypergraph is the priority, per the author's trajectory.)
- **Game-theoretic core:** transferable-utility (TU) cooperative games `(N, v)`; solution concepts: **Shapley value** (primary), **Myerson/communication-graph value**, **Harsanyi dividends / interaction indices**, **Banzhaf index**, **Weber/Harsanyi allocation sets**, **core/nucleolus** (secondary/contextual).
- **Roles of attribution:** post-hoc explanation · in-training optimization signal · data/credit valuation · fairness/exposure correction · actionability/intervention.
- **Player types:** features · interactions (u,i) · items · users · contexts · signal sources · data tuples · nodes · edges/hyperedges · providers · LLM agents.
- **Evaluation:** recommendation metrics (NDCG, Recall, coverage, ILD), plus explanation-quality metrics where reported.

### Out of scope (explicitly)
- General (non-recommendation) XAI surveys; we de-scope to recommendation ranking.
- Image/text/audio/multimodal recommenders and pure sequence/LLM-generative recommender papers (mention, don't review deeply).
- Non-cooperative game formulations (Stackelberg, adversarial, auction) — mention in related work only.
- Producing a new method or a new unified "meta-framework" — reserved for the author's planned method papers (ActionShap, FairShap, MHyperShap, SignalShap).
- No human-subject studies, no ethics approval (scope note: actionability is discussed as an open challenge, not measured here).

---

## 4. Survey methodology (systematic review protocol)

A credible survey must report a transparent protocol. Adopt a PRISMA-style flow.

### 4.1 Sources
- ACM Digital Library, IEEE Xplore, Scopus, Web of Science, arXiv, SpringerLink, Elsevier (ScienceDirect). Optionally Google Scholar for citation chasing.

### 4.2 Search string (documented, reproducible)
- Primary: `("coalitional" OR "cooperative game" OR "Shapley" OR "Myerson" OR "Banzhaf" OR "Harsanyi" OR "nucleolus") AND ("recommender" OR "recommendation" OR "collaborative filtering")`
- Graph constraint (for the core set): `AND ("graph" OR "hypergraph" OR "GNN" OR "graph neural" OR "knowledge graph")`
- Explainability constraint (applied at screening): presence of an explanation/attribution/evaluation purpose.

### 4.3 Time window
- 2010–2026 (methods) — prioritize 2018 onward; include all seminal prior work (Shapley 1953; Myerson 1977; Harsanyi 1963; Banzhaf 1965) as background.

### 4.4 Screening and eligibility (documented criteria)
- **Inclusion:** peer-reviewed papers (or stable arXiv) that (a) define a coalitional game over recommendation-relevant entities **or** use a coalitional solution concept for attribution in a recommendation task, **and** (b) operate on a graph/hypergraph structure or a graph-like interaction structure.
- **Exclusion:** papers whose game-theoretic component is only SHAP feature attribution on a tabular classifier with no ranking/recommendation framing; non-recommendation GNN explainers (kept as an *adjacent* category, clearly labelled); non-cooperative formulations.
- **Screening:** title → abstract → full-text, with counts recorded at each stage (PRISMA flow figure).
- **Data extraction form (per included paper):** citation · venue/year · task · graph type · player set · value function `v(S)` · solution concept · role · approximation method · datasets · baselines · metrics · reported result · reproducibility flag.

### 4.5 Quality/risk-of-bias assessment (optional but recommended)
- Flag whether results are statistically validated (paired tests, Holm–Bonferroni), whether code/data are released, whether evaluation uses a proper held-out protocol.

---

## 5. Taxonomy (the paper's core contribution)

Define the taxonomy explicitly; this is the intellectual centerpiece. Five axes:

### Axis 1 — Player set
`features · interactions (u,i) · items · users · contexts · signal sources · data tuples/items · nodes · edges/hyperedges · providers · agents`

### Axis 2 — Characteristic / coalition value function `v(S)`
What utility a coalition earns. Candidates from the corpus: NDCG/ranking utility · diversity (ILD) · coverage · context-alignment · preference-consistency · fairness/exposure · combined multi-objective (e.g., DyHuCoG Eq. 1–2) · regret (bandit games) · model accuracy (data valuation).

### Axis 3 — Solution concept
Shapley value (exact / MC-approx) · Myerson value (communication graph) · Harsanyi dividends & interaction indices · Banzhaf · Weber/Harsanyi allocation sets · core/nucleolus.

### Axis 4 — Role of attribution
post-hoc explanation · in-training optimization signal · data/credit valuation · fairness/exposure correction · actionability/intervention.

### Axis 5 — Graph structure
bipartite · homogeneous GNN · KG/heterogeneous · hypergraph · dynamic/temporal.

Each surveyed method is mapped into the tuple `(Axis1, Axis2, Axis3, Axis4, Axis5)`.

---

## 6. Unified notation (reuse thesis language)

Adopt the thesis's game notation (Ch. 2.8) so all methods are comparable:

- Players `N`, player `i`; TU game `v: 2^N → ℝ`, coalition `S ⊆ N`.
- **Shapley value:** `φ_i(v) = Σ_{S⊆N∖{i}} (|S|!(|N|−|S|−1)!/|N|!) [v(S∪{i}) − v(S)]`.
- **Monte-Carlo estimator:** `φ̂_i = (1/M) Σ_{m=1..M} [v(S_m ∪ {i}) − v(S_m)]`.
- **Myerson value:** Shapley restricted to a communication graph `g`; `φ^g_i(v)` = Shapley of `v` restricted to feasible coalitions (connected components of `g[S]`).
- **Harsanyi dividends / interaction index:** `I(S)` interaction among players in `S`; pairwise Shapley interaction `I_ij = Σ ... [v(S∪{i,j}) − v(S∪{i}) − v(S∪{j}) + v(S)]`.
- **Multi-objective value (for hypergraph recommendation, from DyHuCoG):**
  `v(S) = α·NDCG(S) + β·Diversity(S) + γ·Context(S)`; `v_pref(S) = v(S) + λ_pref·Σ_{(u,i)∈S} sim(u,i)`.

> Publication note: if text/equations are reused from the thesis or prior papers, **rewrite them** in survey voice to avoid self-plagiarism overlap; do not copy verbatim.

---

## 7. Benchmark design (secondary deliverable)

Purpose: empirically instantiate the taxonomy and compare the main attribution families under one shared protocol. **Reuse existing codebases** (DyHuCoG code, ActionShap code) where sound; **do not** build on unresolved DyHuCoG reproducibility gaps without fixing them (see §11 risk register).

### 7.1 Recommender backbone(s)
Pick a small, defensible set of graph/hypergraph backbones to host the attribution modules:
- **Hypergraph GNN backbone** (DyHuCoG-style message passing) — primary.
- **LightGCN** (homogeneous GNN) — secondary, to show transfer across graph types.

### 7.2 Attribution families to compare (players = interactions, value = ranking/diversity utility)
| Label | Attribution | Nature |
|---|---|---|
| `uniform` | uniform edge weight (no attribution) | degenerate baseline |
| `attention` | learned interaction-level attention gate | non-game-theoretic baseline |
| `heuristic-pop` | popularity/degree weighting | heuristic baseline |
| `shapley-mc` | preference-aware Monte-Carlo Shapley (DyHuCoG-style) | **game-theoretic (primary)** |
| `shapley-ai` | sampling/importance-based Shapley variant | game-theoretic (ablation of estimator) |
| `myerson` (optional) | communication-graph-restricted Shapley on the hypergraph projection | game-theoretic (structure-aware) |

> Keep the benchmark **small** (few methods × few backbones × few datasets) — it is illustrative, not the main contribution. Do not turn it into a large-scale method bake-off (that belongs to the separate method papers).

### 7.3 Protocol (shared, mirrors thesis Ch. 4)
- **Implicit feedback:** positive if rating > 3; min 5 interactions; temporal per-user split (70/10/20), leave-one-out for evaluation.
- **Optimizer:** Adam; early stopping patience 20 on validation NDCG@20; BPR-style pairwise loss; popularity-aware negative sampling with periodic hard-negative refresh.
- **Runs:** 5 seeds {42,43,44,45,46}; report mean ± std.
- **Reproducibility:** fixed seeds, recorded hyperparameters, released code.

### 7.4 Metrics
**Primary reported results — Recall@K and NDCG@K (K ∈ {5, 10, 20}).** The benchmark's headline deliverable is the result table reporting **NDCG@20 and Recall@20** for every attribution family on both backbones and both datasets (full cutoffs @5/@10/@20 included), per `Implementation_Spec.md` §B.1a.
- Ranking: **NDCG@K** = `(1/|U|) Σ_u DCG_u@K / IDCG_u@K`, `DCG_u@K = Σ_{k=1..K} rel_u,k / log2(k+1)`; **Recall@K** = `(1/|U|) Σ_u |rel_u ∩ R_u@K| / |rel_u|`. Report mean ± std over 5 seeds for every family.
- Exposure (secondary): **catalogue coverage** (`Coverage = |∪_u R_u| / |I|`), head/tail coverage by popularity decile.
- Diversity (secondary): **Intra-List Diversity (ILD)** (`ILD = 2/(K(K−1)) Σ_{1≤k<l≤K} [1 − sim(i_k,i_l)]`).
- Cost (secondary): training time, inference latency, peak GPU memory.
Coverage/ILD/cost never replace Recall@K/NDCG@K as the headline; they contextualize them.

### 7.5 Statistical analysis (reuse thesis protocol)
- Per-user paired **t-test**; **Holm–Bonferroni** correction across comparisons; **Wilcoxon signed-rank** as a distribution-free check; **Cohen's d_z** effect size; 95% CIs. Report a paired table analogous to the thesis Appendix A.

### 7.6 Hardware / software
- Same environment as the thesis: PyTorch 2.x, Python 3.10+, RTX 4090 (or equivalent), CUDA. Record exact versions.

---

## 8. Datasets

Primary benchmark (recommendation — dense + sparse pairing, mirroring the thesis):
| Dataset | Type | Why |
|---|---|---|
| **MovieLens-1M** | dense, explicit → implicit | primary, statistically clean |
| **Amazon-Book** | sparse, long-tail implicit | sparsity robustness, coverage/diversity test |

Optional extension (if needed for breadth): **Yelp2018** (cross-dataset robustness).

> Note: the thesis's clustering datasets (Wine Quality, Beijing Air Quality) are **out of scope** here — the paper is recommendation-scoped. Mention them only as historical context for the author's trajectory, not as benchmark data.

---

## 9. Deliverables and artifact requirements

1. **Manuscript** (survey + benchmark), ~9,000–12,000 words, 6–8 figures, 4–6 comparison tables. Structure in §10.
2. **PRISMA-style flow figure** and documented search/protocol (for reproducibility of the review).
3. **Comparison tables** mapping each reviewed method into the taxonomy tuple.
4. **Code + configs + scripts** for the benchmark, with recorded seeds and results (`results/{raw,tables,figures}`), following the repo's `paper-ideas/<name>/code/` layout convention.
5. **BibTeX bibliography** file with all reviewed works.

---

## 10. Manuscript structure (suggested)

```
Abstract / Keywords
1. Introduction (motivation, gap vs. prior surveys, scope, contributions)
2. Preliminaries
   2.1 Coalitional game theory: TU games, (N, v)
   2.2 Solution concepts: Shapley; Myerson/communication-graph; Harsanyi dividends &
       interaction indices; Banzhaf; Weber/Harsanyi sets; core/nucleolus
   2.3 Graph-based recommendation: bipartite, homogeneous GNN, KG/heterogeneous, hypergraph,
       dynamic/temporal; top-N problem
   2.4 Explainability in recommendation: post-hoc / in-training / fairness / actionability;
       evaluation of explanations
3. Survey methodology (sources, search string, screening, extraction, PRISMA flow)
4. A taxonomy of coalitional games for explainable graph-based recommendation (5 axes)
5. Systematic review by category (organized by player set, with sub-cases)
6. Comparative analysis (comparison tables)
7. Empirical benchmark (design, results, BQ answers)
8. Critical analysis (what game theory buys / does not buy; faithfulness vs. actionability;
   reproducibility; DyHuCoG audit caveat)
9. Open challenges and research agenda (maps to planned method papers)
10. Conclusion
Appendix A: notation/glossary · Appendix B: reviewed-works index
```

---

## 11. Risk register and mitigations

| Risk | Mitigation |
|---|---|
| Scope creep → shallow bibliography | Lock §3 scope; taxonomy is the backbone; de-scope non-recommendation GNN explainers |
| Reads as a thesis recap | Frame author's own work (DyHuCoG) as one case study among many; field-wide synthesis voice |
| Reviewers demand a real protocol | §4 PRISMA-style flow, documented search, data-extraction form, counts |
| "You missed X" reviews | Transparent inclusion/exclusion; time window; broad search string; record all screened titles |
| **DyHuCoG reproducibility flag** (repo's own SignalShap audit: 63 extraction gaps) | In the survey, describe DyHuCoG accurately and cite its caveat; in the benchmark, fix/pin the backbone before use or use an independently documented backbone |
| Self-plagiarism / overlap with thesis | Rewrite preliminaries in survey voice; do not copy equations/paragraphs verbatim |
| Scooping (hot intersection, 2025–26) | Ship first; a focused survey draft is feasible in ~3–4 weeks on the thesis literature base |
| Venue mismatch (survey vs. method) | **Resolved:** venue fixed to Discover AI, submitted under the **`Review`** article type (§1.1). Re-confirm article-type availability and length at submission in case policies change |
| Discover AI article-type mismatch | Submit as **`Review`**, not "Research"; title + structure must read as a survey, not a new method (§1.1) |
| Missing mandatory declarations / formatting | Use §1.1 checklist (funding, author contributions, competing interests, data/code availability, ethics "not applicable"); follow Springer Nature Discover templates |
| APC / open-access cost | Budget for APC (≈US$1,690) or a waiver/institutional OA agreement before submission (§1.1) |
| Abstract > 250 words | Keep abstract < 250 words and write the cover letter (both mandatory uploads in Snapp) |
| Benchmark scope creep | Keep benchmark small (§7.2, §7.3); it is illustrative |

---

## 12. Build order (recommended sequence)

1. **Lock scope + write §3 and the taxonomy (§5)** — 1-page scope + taxonomy sketch, before any drafting.
2. **Run the systematic search + screening** (§4); produce PRISMA flow and extraction spreadsheet.
3. **Write §2 preliminaries + §4 methodology + §5 taxonomy + §6 notation.**
4. **Write §7 systematic review + §6 comparison tables** (the core content).
5. **Implement the benchmark** (§7) using existing codebases; run and collect results.
6. **Write §8 critical analysis + §9 open challenges + §10 conclusion + abstract (< 250 words).**
7. **Compile bibliography, appendix, figures/tables; internal review (trajectory check: survey ≠ recap).**
8. **Prepare Discover AI submission package (§1.1):** manuscript in the current journal template, cover letter, and all mandatory statements (funding, author contributions, competing interests, data/code availability, ethics "not applicable").
9. **Draft the separate method papers** (ActionShap/FairShap/... ) citing this survey.

---

## 13. Minimum acceptance criteria

For the survey to be defensible, **all** must hold:
- [ ] PRISMA-style protocol documented with search string, sources, window, inclusion/exclusion, and counts.
- [ ] Taxonomy (§5) with all five axes defined and every reviewed method mapped into a tuple.
- [ ] ≥ 2 comparison tables placing the author's DyHuCoG alongside external game-theoretic recommender work (not just SHAP-on-tabular).
- [ ] Critical analysis section addressing *faithfulness vs. actionability*, the value-function-arbitrariness critique, and reproducibility — not just a listing.
- [ ] Benchmark (if included) runs the full shared protocol: ≥2 datasets, ≥4 attribution families including a game-theoretic and a non-game-theoretic baseline, 5 seeds, paired statistics with correction, and code released.
- [ ] **Primary results reported as Recall@K and NDCG@K (K ∈ {5,10,20}) result tables** for every attribution family, on both backbones and both datasets, with mean ± std over 5 seeds (per `Implementation_Spec.md` §B.1a).
- [ ] Scope respected: no clustering datasets in the benchmark; no new-method claim; no verbatim thesis text.
- [ ] Explicit roadmap linking to the planned method papers (without reporting their results).

### 13.1 Discover AI submission-readiness checklist (all must hold)

- [ ] Article type selected as **`Review`** in Snapp (not "Research").
- [ ] Abstract **< 250 words**; cover letter drafted (context/importance + fit for the journal).
- [ ] **Funding statement** present.
- [ ] **Author contributions** (CRediT) statement present.
- [ ] **Competing interests** statement present.
- [ ] **Data availability** + **Code availability** statements present (public datasets; code/Zenodo DOI).
- [ ] **Ethics statement** present — "not applicable" (no human/animal subjects; justify no user studies).
- [ ] Manuscript formatted per current **Springer Nature Discover Series** template; figures/tables/references conform to journal style.
- [ ] No verbatim reuse of thesis/prior-paper text/equations/figures (survey voice + disclosure in cover letter).
- [ ] APC funded or waiver/institutional OA agreement confirmed.

---

## 14. Definition of done

This paper is "done" when: the survey + benchmark manuscript is complete and internally consistent; the review protocol is reproducible; the benchmark code and results are committed under `paper-ideas/CoalGameRec/`; the bibliography is complete; the **Discover AI submission package is assembled** (Review article type, abstract < 250 words, cover letter, and all §1.1/§13.1 mandatory statements + formatting); and the trajectory check passes (the survey reads as a field synthesis that *situates* the author's method portfolio, and does not cannibalize ActionShap/FairShap/MHyperShap/SignalShap). The survey should be the **first** of the author's next-paper sequence so all subsequent method papers cite it.
