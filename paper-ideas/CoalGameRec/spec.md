# CoalGameRec — Specification for "Coalitional Game Theory for Explainable Graph-Based Recommendation Systems"

**Status:** design specification, v1.2 (revised to address two pre-submission reviews: `review1.md`, `review2/review2.md`)
**Paper type:** Review article — **systematic review + a separately-scoped empirical case study** (the benchmark is explicitly secondary and is not the basis for the survey's claims)
**Target venue:** *Discover Artificial Intelligence* — Springer Nature, **open access**. Scopus-indexed; **CiteScore 2024 = 6.0**; indexed in DOAJ, Ei Compendex and Scopus. **Accepts a dedicated `Review` article type** ("critical accounts and comprehensive surveys of topics of major current interest within the scope of the journal… of any length"). **Quartile note (review 2):** the official journal page confirms indexing and CiteScore but does not itself establish a "Q1" label; any quartile claim must cite a dated external source (Scopus/SCImago/JCR, edition, year) or be removed from the manuscript. Median first decision is currently reported as **16 days** on the journal page (not ≈23). **APC: US$1,690 / €1,390 / £1,190** (verify against the live page at submission; keep in private planning notes, not the manuscript).
**Corresponding author:** Mouad Louhichi (ENSIAS, UM5 Rabat) — `mouad_louhichi@um5.ac.ma`

> **Why this venue (private planning note — do NOT carry this framing into the manuscript).** The author's planned method papers are aimed at *Discover Artificial Intelligence*; the journal's dedicated **Review article type** fits a comprehensive survey, and its scope covers coalitional game theory + XAI + recommender systems. The choice is defensible on venue fit. However, the "citation home / keep the whole portfolio in one venue" rationale and the mapping of §8's agenda onto the author's planned papers must **not** appear in the manuscript or cover letter — per review 1.4, the research agenda must be argued from the literature's own gaps, with the author's planned work as one direction among several, to avoid self-citation/self-promotion optics (COPE).

> **Scope note (revised per review 1.2/2.1/C1/C15).** This is a **systematic review and taxonomy** (the primary, novel contribution) **plus an explicitly separate empirical case study** (the benchmark). It introduces no new method or unified "meta-framework." The paper must be honest that the benchmark is **original empirical work** (it is not "no new experiments"), that it **grounds only the interaction-player / ranking-utility slice** of the taxonomy (not the whole five axes), and that the survey's arguments do not depend on the benchmark's outcome. The paper must read as a field-wide synthesis in which the author's own work (DyHuCoG) is one case study among many — **not** as a recap of the thesis.

---

## 1. Purpose and central question

**Central question.** How are coalitional (cooperative) game-theoretic models instantiated to make graph- and hypergraph-based recommender systems explainable, and what — empirically and theoretically — does this attribution lens actually buy (and cost)?

The paper has **two deliverables**:

1. **Survey deliverable (primary).** A systematic, reproducible review of coalitional-game attribution for explainable graph-based recommendation, organized by a novel taxonomy, with comparison tables and a critical synthesis. This is the novel, citable contribution. **Framed for the `Review` article type of Discover Artificial Intelligence.**
2. **Case-study benchmark deliverable (secondary, separately-scoped).** A small, reproducible, **pre-registered** empirical study that instantiates the *interaction-player / ranking-utility* slice of the taxonomy on standard datasets and compares game-theoretic reweighting to matched non-game-theoretic controls. It is a **separate empirical study**, not the basis for the survey's field-wide claims; a null result does not invalidate the review.

---

## 1.1 Target journal and submission readiness — Discover Artificial Intelligence

**Journal profile (verified against Springer's current journal page, Aug 2026).**
- **Name:** *Discover Artificial Intelligence* (Springer Nature, Discover Series). ISSN 2731-0809.
- **Model:** Gold **open access**; **APC** (currently US$1,690 / €1,390 / £1,190).
- **Indexing / ranking:** **Scopus**, **Ei Compendex**, **DOAJ**; **CiteScore 2024 = 6.0**. Quartile claims must cite a dated external source (review 2); do not treat "Q1" as established by the journal page.
- **Peer review:** single-anonymous; at least two reviewers; technical + suitability checks before review; decisions (major/minor revisions, reject) with editor re-evaluation on minor revisions.
- **Scope fit:** "all aspects of artificial intelligence in theory and application" — covers coalitional game theory + XAI + recommender systems.
- **Article-type fit:** the journal explicitly accepts a **`Review`** article type — *"Review articles provide critical accounts and comprehensive surveys of topics of major current interest within the scope of the journal… we accept Review articles of any length."* **However (review 1.2/2.1):** because this package includes an extensively-powered benchmark, the hybrid Review+benchmark format must be justified explicitly in the cover letter and Introduction; a handling editor may reasonably reclassify to `Research` or ask that the benchmark be split into a separate paper. Anticipate this.

**Submission mechanics (Snapp — Springer Nature tracking system).**
- Upload: manuscript file + **abstract < 250 words** + **cover letter** (context/importance of the work, why it fits the journal). Optional: figures, tables, supplementary material.
- Choose article type **`Review`** in the Details tab.
- If a relevant topical collection is open, submit via that collection (same mechanics).
- **Recommended:** a **pre-submission inquiry** to the editor asking whether a Review article may contain a new benchmark of the proposed size, before investing in the manuscript (review 1.2/2.1).

**Required statements (mandatory for this journal) — must be drafted before submission.**
- **Funding statement** (mandatory) — declare funding or "no funding received".
- **Author contributions** statement — **actual, post-completion CRediT**, not a copy-paste from DyHuCoG; reflect the writing-heavy survey work and the largely-reused benchmark code. Obtain every author's approval (review 1/2).
- **Competing interests** statement — must be a complete financial and non-financial disclosure: at minimum, disclose the authors' prior authorship of reviewed methods (DyHuCoG) and the relationship between the benchmark and those methods. Do not pre-fill "no competing interests" (review 2, P1).
- **Data availability** statement — give dataset citations, URLs, access dates, file checksums/versions, final counts, split-generation code, and license/terms; state whether raw data are redistributed (review 2, P1).
- **Code availability** statement — actual repository/Zenodo DOI with code, configs, frozen splits, raw results, environment lockfile, tests, and a permanent archive (a future placeholder is not an availability statement; review 2, P1).
- **Ethics statement (review 2, P0)** — **do not** assert "not applicable" merely because there is no user study. MovieLens and Amazon review data are human-generated. Before data processing, obtain an **institutional determination** from the ethics committee; the final statement must identify the datasets/fields used, whether raw text/demographics are used, public/pseudonymous status, any exemption/approval reference, and how identifiers are protected. Report the institutional determination rather than promising "no ethics approval" as a design decision.

**Formatting (Journal formatting guidelines — Springer Nature Discover Series).**
- Manuscript in the current Springer template: consistent font ≥ 12 pt, ≤ 3 heading levels, square-bracket numeric citations, sequential Arabic table/figure numbering, captions in the body, accessibility (alt text, colour-blind-safe patterns, contrast) (review 2 §10).
- Reference manager and style follow Springer Nature; reference list limited to published/accepted works.
- **AI-use disclosure:** follow the journal's transparent, risk-based AI policy; disclose actual AI use (drafting, critique, code assistance) with human accountability. Do not list AI as an author (review 2).
- Language: standard academic English (Springer recommends professional language editing; check Step 5).

**Dual-publication / text-recycling note (review 1.5/2.1).** Because much of the survey's content overlaps the author's own prior publications and thesis (DyHuCoG, clustering papers, PhD thesis), the manuscript must be written in survey voice and **no text, equation, or figure reused verbatim** — and rewording alone is not enough. Springer Nature/COPE require **explicit in-text attribution** for any reused formulation (e.g., "we adopt the value function of [DyHuCoG, Eq. X]"). Prepare an **overlap table** covering the thesis, DyHuCoG, the two clustering papers, code, datasets, figures, equations, and results, and disclose relevant prior work **in the manuscript**, not only in the cover letter "if asked."

**Relationship to prior work (working hypothesis only — to be confirmed by the systematic search, review 1.6/2.2).** As of this revision, we have **not** yet run the registered search, so the "no dedicated survey exists" claim is a **hypothesis, not a finding**, and must be written as: "Our search, conducted on [date] using [databases and queries], identified no review meeting the following prespecified criteria."
- General Shapley-XAI surveys (e.g., Zhao, Liu & Parilina, *Dynamic Games & Applications* 2025; Li et al., *Autonomous Intelligent Systems* 2024) are feature/classifier-centric and do not target ranking/recommendation.
- GNN-explanation work (EdgeSHAPer, GraphSVX, GraphGI, GStarX, GISExplainer, GraphEXT) targets node/edge/graph classification, not top-N recommendation — kept as **Adjacent B**, not core.
- Game-theoretic recommender methods are scattered singles (Shapley community-CF; Shapley data valuation/pruning for RS; TU-bandit creator-incentive games, incl. the AISTATS 2026 oral; DyHuCoG) with no unifying survey.
- **Known near-neighbors the final search must explicitly check** (review 2 §9): *ShaRP* (ranking/preference Shapley), *Shapley Value: From Cooperative Game to XAI*, *The Shapley Value in Machine Learning*, *The Shapley Value Contribution to XAI*, *Shapley Value-driven Data Pruning for Recommender Systems*, *A Game-Theoretic Approach to Recommendation Systems with Strategic Content Providers*, *Creator Incentives in Recommender Systems* (TU-bandit), Jia et al., *A Comprehensive Study of Shapley Value in Data Analytics* (VLDB 2024/25), and *Beyond Shapley Values* (Weber/Harsanyi). Each is screened against the prespecified scope, not auto-included.

---

## 2. Research questions (SRQs)

- **SRQ1 (taxonomy).** What distinct instantiations of coalitional games exist for explainable graph-based recommendation, along the axes of *player set · characteristic value function · solution concept · role of attribution · graph structure*?
- **SRQ2 (comparison/synthesis).** How do surveyed methods compare when expressed in one shared vocabulary, and what patterns emerge across the coded corpus? (State the synthesis method — descriptive, vote/count, or meta-regression — rather than implying "what correlates with gains" without a quantitative synthesis; review 2.5.5/4.1.)
- **SRQ3 (value analysis).** What does the game-theoretic lens genuinely add over heuristic/attention/reweighting — and where is it relabeled reweighting?
- **SRQ4 (validity).** How is attribution validated (faithfulness, stability, actionability, reproducibility)? Which evaluation gaps recur?
- **SRQ5 (roadmap).** What are the highest-leverage open problems (approximation, online/streaming, human-centred evaluation, fairness audits, structure-aware solution concepts, agentic systems)?

**Benchmark questions (BQ, secondary — framed as hypotheses, review 1.1/2.1).**
- **BQ1.** In one interaction-player case study, does attribution-guided reweighting change held-out NDCG@K / Recall@K relative to matched no-attribution and heuristic controls?
- **BQ2.** What is the effect on coverage and intra-list diversity under the same protocol?
- **BQ3.** What are estimator error, runtime, memory, and seed/data-sample variability?
- **BQ4.** How do results compare across a dense (MovieLens-1M) and a sparse (Amazon-Book) setting (a descriptive cross-dataset comparison, not a causal density claim)?
The benchmark does **not** by itself answer the field-wide SRQ3; it tests one pre-registered slice (review 2.5).

---

## 3. Scope — in scope / out of scope

### In scope
- **Tasks:** top-N item recommendation under implicit (or converted-implicit) feedback.
- **Graph structures:** bipartite user–item; homogeneous GNN; knowledge-graph/heterogeneous; hypergraph; dynamic/temporal. (Hypergraph is the priority, per the author's trajectory.)
- **Game-theoretic core:** transferable-utility (TU) cooperative games `(N, v)`; solution concepts: **Shapley value** (primary), **Myerson/communication-graph value**, **Harsanyi dividends / interaction indices**, **Banzhaf index**, **Weber/Harsanyi allocation sets**, **core/nucleolus** (secondary/contextual).
- **Roles of attribution:** post-hoc explanation · in-training optimization signal · data/credit valuation · fairness/exposure correction · actionability/intervention.
- **Player types:** features · interactions (u,i) · items · users · contexts · signal sources · data tuples · nodes · edges/hyperedges · providers · LLM agents.
- **Evaluation:** recommendation metrics (NDCG, Recall, coverage, ILD), plus explanation-quality metrics where reported.

### Evidence tiers (review 2.3) — core vs. adjacent
Define and report separately (do not count adjacent work in the core corpus):
- **Core:** top-N recommendation; graph/hypergraph is an input, model, or explicit interaction structure; the game attributes a recommendation-relevant output or training signal.
- **Adjacent A:** ranking/recommendation attribution without a graph model.
- **Adjacent B:** graph/GNN attribution outside recommendation (e.g., node/edge classification explainers) — clearly labelled, not counted in the core corpus.
- **Adjacent C:** provider/data/agent incentive or valuation games without model explanation.
- **Background only:** general cooperative-game theory and general SHAP/XAI surveys.

### Out of scope (explicitly)
- General (non-recommendation) XAI surveys; de-scoped to recommendation ranking (mentioned as background only).
- Image/text/audio/multimodal recommenders and pure sequence/LLM-generative recommender papers (mention, don't review deeply).
- Non-cooperative game formulations (Stackelberg, adversarial, auction) — mention in related work only.
- Producing a new method or a new unified "meta-framework" — reserved for the author's planned method papers.
- No user studies; actionability is discussed as an open challenge, not measured (but see the ethics determination requirement, §1.1).

---

## 4. Survey methodology (systematic review protocol)

A credible survey must report a transparent protocol. Adopt and cite **PRISMA 2020** (the journal recommends PRISMA for systematic reviews), complete its checklist and flow diagram, and use PRISMA-S or equivalent for search reporting. The protocol must be **frozen and registered (timestamped, external)** before screening. "First systematic survey" language is only defensible after the search is run and reported.

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

### 4.5 Quality/risk-of-bias assessment (required, review 2.2)
Not optional: the paper promises a "critical synthesis" and evaluates validation gaps, so it must score/report each included study on a defined rubric (task and player-set clarity; coalition/value-function clarity; model/split/candidate transparency; leakage controls; baseline adequacy; seeds and uncertainty; explanation-faithfulness evaluation; robustness; code/data availability; reported limitations). Distinguish NR (not reported) from NA (not applicable). Use the assessment to qualify synthesis claims, not merely to rank papers.

### 4.6 Screening rigor (required, review 2.2)
Report: records per source; duplicates removed; title/abstract exclusions; full-text exclusions with one reason per paper; included core vs. adjacent counts; disagreements and adjudication; agreement measure. If only the three authors screen, assign independent roles and be blind to author identity where practical; the author's own papers must meet the same inclusion criteria. Record search dates, database-specific syntax, language/date/publication-type rules, and a final search update before submission.

---

## 5. Taxonomy (the paper's core contribution)

Define the taxonomy explicitly; this is the intellectual centerpiece. Five axes:

### Axis 1 — Player set
`features · interactions (u,i) · items · users · contexts · signal sources · data tuples · nodes · edges/hyperedges · providers · agents`

> **Taxonomy rigor (review 2.3/4.3/6.1):** the axes are a deductive starting framework and must be validated against the coded corpus (allow new categories; report how the final axes were revised). Player labels overlap ("edge", "interaction", "user–item tuple" may be the same player; "node" may be user/item/context; "feature"/"context"/"signal source" may nest) — define allowed values, multi-label rules, and distinguish **game target** from **player granularity**. The taxonomy must be coded by more than one independent coder with agreement evaluation.

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
- **Myerson value:** the Shapley value of the **graph-restricted game** `v^g(S) = Σ_{C∈𝒞(g[S])} v(C)`, where `g[S]` is the subgraph induced by `S` and `𝒞(g[S])` its connected components. Do not describe it loosely as "Shapley restricted to connected coalitions."
- **Harsanyi dividends vs. interaction indices:** these are distinct. Harsanyi dividends `λ_S` (the unique numbers with `v(S)=Σ_{T⊆S} λ_T`) are not the same as pairwise Shapley interaction indices. Give one precise definition per concept with a citation, or remove formulas not used. Weber set and Harsanyi set are also distinct allocation sets — do not conflate them.
- **Multi-objective value (for hypergraph recommendation, from DyHuCoG):**
  `v(S) = α·NDCG(S) + β·Diversity(S) + γ·Context(S)`; `v_pref(S) = v(S) + λ_pref·Σ_{(u,i)∈S} sim(u,i)`.

> Publication note: if text/equations are reused from the thesis or prior papers, **rewrite them** in survey voice to avoid self-plagiarism overlap; do not copy verbatim.

---

## 7. Benchmark design (secondary deliverable)

Purpose: empirically instantiate the taxonomy and compare the main attribution families under one shared protocol. **Reuse the ActionShap codebase (`stats.py`, clustering/quality diagnostics) where sound; the benchmark does NOT use DyHuCoG code** — it uses an independently documented hypergraph GNN (see §7.1, §11 risk register, and `Implementation_Spec.md` §A.4).

### 7.1 Recommender backbone(s)
Pick a small, defensible set of graph/hypergraph backbones to host the attribution modules. **The benchmark does NOT use DyHuCoG code** (authors' decision, review 1.3):
- **Hypergraph GNN backbone** — an **independently documented** hypergraph GNN with a public implementation (e.g., HNN/HGCN/HCCF-style), pinned before the benchmark — primary.
- **LightGCN** (homogeneous GNN) — secondary, to show transfer across graph types.

### 7.2 Attribution families to compare (players = interactions, value = ranking/diversity utility)
| Label | Attribution | Nature |
|---|---|---|
| `uniform` | uniform edge weight | **no-attribution control** (not "an attribution method") |
| `attention` | learned interaction-level attention gate (matched parameter count) | non-game-theoretic control |
| `heuristic-pop` | popularity/degree weighting | heuristic control |
| `additive-pref` | additive preference-similarity prior without game aggregation | **matched non-game heuristic (required, review 2.5)** |
| `shapley-mc` | preference-aware Monte-Carlo Shapley (game formulation defined independently in this paper, §A.6; **independent implementation** — no DyHuCoG code) | **game-theoretic (primary)** |
| `shapley-ai` | precisely-defined sampling/importance Shapley variant (§A.6d) | exploratory estimator ablation |
| `myerson` | Myerson value on a stated graph-restricted game (§A.6) | exploratory structure-aware |

**Matched-objective requirement (review 2.5):** all families share the same backbone, loss, scalar objective, and tuning budget; the game-theoretic method must not receive a matched multi-objective value/preference term that the controls lack. A factorial ablation (with/without preference term, with/without diversity/context, fixed vs. refreshed attribution) and a matched additive-similarity baseline (`additive-pref`) are required, or the experiment cannot attribute a gain to the Shapley rule.

> Keep the benchmark a **separately-scoped case study** — a small, fully-run method set. Do not call it both "illustrative" and "the headline" (review 1.7/2.2). `shapley-ai`/`myerson` are exploratory: run every promised cell or drop them from headline claims (C4).

### 7.3 Protocol (shared — one split definition, review C3)
- **Implicit feedback:** positive if rating ≥ 4 (state the exact threshold); min 5 interactions.
- **Split (single, fixed):** **temporal leave-one-out per user** — last interaction = test, second-last = validation, rest = train; break timestamp ties deterministically (stable key, not row order that changes on re-parse). This is the definition used everywhere; the earlier "70/10/20" wording is removed. State the minimum training history and handling of users with too few positives, and clarify whether the 5-core filter uses train-period data only (avoid future leakage).
- **Optimizer:** Adam; early stopping patience 20 on validation NDCG@20; BPR-style pairwise loss; popularity-aware negative sampling with periodic hard-negative refresh.
- **Runs:** 5 seeds {42,43,44,45,46}; report mean ± std.
- **Reproducibility:** fixed seeds, recorded hyperparameters, released code.

### 7.4 Metrics
**Primary reported results — Recall@K and NDCG@K (K ∈ {5, 10, 20}).** The benchmark's headline deliverable is the result table reporting **NDCG@20 and Recall@20** for every attribution family on both backbones and both datasets (full cutoffs @5/@10/@20 included), per `Implementation_Spec.md` §B.1a.
- Ranking: **NDCG@K** = `(1/|U|) Σ_u DCG_u@K / IDCG_u@K`, `DCG_u@K = Σ_{k=1..K} rel_u,k / log2(k+1)`; **Recall@K** = `(1/|U|) Σ_u |rel_u ∩ R_u@K| / |rel_u|`. Report realized mean ± std over seeds for every family actually run (fill results only after the experiment; per `Implementation_Spec.md` §B.1a).
- Exposure (secondary): **catalogue coverage** (`Coverage = |∪_u R_u| / |I|`), head/tail coverage by popularity decile.
- Diversity (secondary): **Intra-List Diversity (ILD)** (`ILD = 2/(K(K−1)) Σ_{1≤k<l≤K} [1 − sim(i_k,i_l)]`).
- Cost (secondary): training time, inference latency, peak GPU memory.
Coverage/ILD/cost never replace Recall@K/NDCG@K as the headline; they contextualize them.

### 7.5 Statistical analysis (see `Implementation_Spec.md` §A.10 for the full plan)
- **Unit of analysis: per-user** (paired per-user differences on the shared split); the 5 seeds quantify training variability, not the pairing unit.
- **Primary predeclared contrasts:** `shapley-mc` vs `uniform` on NDCG@20 and Recall@20, per dataset/backbone; these form the **Holm–Bonferroni family**; all else is secondary/exploratory.
- Paired t-test and Wilcoxon signed-rank used as **sensitivity analyses** (assumptions stated); report Cohen's d_z with 95% CI and permutation/bootstrap intervals over users; report per-seed values.
- State the number of comparisons in every significance caption; users are not strictly independent after shared training (state this caveat).

### 7.6 Hardware / software
- Same environment as the thesis: PyTorch 2.x, Python 3.10+, RTX 4090 (or equivalent), CUDA. Record exact versions.

---

## 8. Datasets

Primary benchmark (recommendation — dense + sparse pairing, mirroring the thesis):
| Dataset | Type | Why |
|---|---|---|
| **MovieLens-1M** | dense, explicit → implicit | primary, statistically clean |
| **Amazon-Book** | sparse, long-tail implicit | sparsity robustness, coverage/diversity test |

Optional extension (if needed for breadth): **Yelp2018** (cross-dataset comparison).

> **Amazon-Book provenance (review 2, C17/C11):** Amazon-Book is rebuilt from the raw Amazon Reviews 2018 `Books_5.json.gz` (5-core subset, 27,164,983 reviews) + `meta_Books.json.gz`, user-subsampled to ~50k and re-5-cored — a **custom dataset**, not the canonical Amazon-Book split. Cite the UCSD Amazon page (Ni, Li & McAuley 2019) and the exact custom preprocessing; release the user sample, final graph, and hash. Because it differs from the published split, published baseline numbers are **not** calibration targets. Consider reporting the canonical-split numbers as a secondary sanity check if the temporal protocol is preserved.
>
> Note: the thesis's clustering datasets (Wine Quality, Beijing Air Quality) are **out of scope** here — the paper is recommendation-scoped. Mention them only as historical context for the author's trajectory, not as benchmark data.

---

## 9. Deliverables and artifact requirements

1. **Manuscript** (systematic review + secondary case study). Target ~10,000–12,000 words, 7–9 figures, 5–7 tables in main text (reconciled across `spec.md` and `Paper_Structure.md`; detailed factorial result tables and the prediction register go to supplementary material with distinct labels). Structure in §10.
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
   reproducibility; DyHuCoG audit caveat) — placed **before** the benchmark, so the taxonomy and
   critique are not contingent on benchmark results (review 4.3)
9. Empirical case-study benchmark (design, results, BQ answers; framed as a separate study)
10. Open challenges and research agenda (argued from the literature; author's planned work as ONE
    direction among several — not the organizing structure; review 1.4)
11. Limitations
12. Conclusion
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
| **DyHuCoG reproducibility flag** (repo's own SignalShap audit: 63 extraction gaps) | **Resolved by decision: no DyHuCoG code in the benchmark.** In the survey, describe DyHuCoG accurately and cite its caveat; the benchmark uses an independently documented hypergraph GNN (`Implementation_Spec.md` §A.4) |
| Self-plagiarism / overlap with thesis | Rewrite preliminaries in survey voice; do not copy equations/paragraphs verbatim |
| Scooping (hot intersection, 2025–26) | Ship first; a focused survey draft is feasible in ~3–4 weeks on the thesis literature base |
| Review vs. Research article type | **Not merely asserted as resolved** — justify the hybrid Review+benchmark format explicitly in the cover letter and Introduction; a handling editor may reclassify to `Research` or ask to split the benchmark (review 1.2/2.1) |
| Predicted results mistaken for results | Remove pre-formatted predicted tables from manuscript-facing files; fill result tables only with realized numbers; external pre-registration (review 1.1/2.1) |
| Abstract > 250 words and contains predicted outcome | Keep abstract < 250 words and state hypotheses, not results, until data exist (review 2) |
| Missing mandatory declarations / formatting | Use §1.1 checklist (funding, actual CRediT, competing interests incl. prior authorship of reviewed methods, data/code availability with real identifiers, AI-use, ethics determination); follow Springer Nature Discover templates |
| Ethics misclassified as "not applicable" | Obtain and report an institutional determination for the human-generated data (review 2, P0) |
| Self-citation / self-promotion optics | Argue the agenda from the literature; author's planned work as one direction among several; keep portfolio strategy out of the manuscript (review 1.4) |
| Novelty claim unsupported | Run the search before finalizing the "first systematic survey" claim; qualified wording until the protocol is complete (review 1.6/2.2) |
| DyHuCoG audit treated as a field fact | Archive the audit or describe it as an author-side implementation risk, not a published fact (review 1.3/2) |
| APC / open-access cost | Budget for APC or a waiver; verify the live figure at submission (review 2) |
| Benchmark scope creep / "illustrative" vs. full-factorial inconsistency | Choose one framing: a separately-scoped case study (§7.2/§A.0) with a small, fully-run method set; do not call it both "illustrative" and "the headline" (review 1.7/2.2) |

---

## 12. Build order (recommended sequence — revised per review 1.6/2.2)

1. **Freeze core/adjacent scope and RQs (§3),** then **build and register the systematic-review protocol** (PRISMA 2020, search strings, screening rules) in a timestamped external repository — **before** the taxonomy is locked, so the novelty claim is supported by the search, not backfilled by it.
2. **Run the search + screening + extraction + quality audit (§4);** produce the PRISMA 2020 flow, full-text exclusion log, extraction sheet, and validated corpus.
3. **Code and validate the taxonomy against the corpus** (independent coders, agreement); write §2 preliminaries + §4 methodology + §5 taxonomy + §6 notation.
4. **Write the systematic review + comparison tables** (the core content).
5. **Obtain the institutional ethics determination** for the human-generated data (MovieLens/Amazon) before data processing (§1.1).
6. **Run the Amazon feasibility spike and a small synthetic Shapley test** (`Implementation_Spec.md` §B.7 milestone 0).
7. **Freeze the benchmark estimand, splits, baselines, and statistical plan (§A.10); register predictions externally (§B.0).**
8. **Implement + run the benchmark** with a documented backbone and complete tests; write realized results only.
9. **Write critical analysis + case-study results + agenda + limitations + conclusion + abstract (< 250 words).**
10. **Compile bibliography, appendices, figures/tables; internal review (trajectory check: survey ≠ recap; survey does not depend on benchmark outcome).**
11. **Prepare the Discover AI submission package:** manuscript in the current Springer template, cover letter (justifying the Review+benchmark format), and all mandatory statements (funding, actual CRediT, competing interests — including prior authorship of reviewed methods, data/code availability with real identifiers, AI-use disclosure, ethics determination).
12. **Draft the separate method papers** citing this survey (only after the survey's gap claim is independently grounded).

---

## 13. Minimum acceptance criteria

For the review to be defensible, **all** must hold:
- [ ] **PRISMA 2020** protocol registered externally before screening; checklist + flow figure + full exclusion log reported (review 2.2).
- [ ] Search strings, sources, window, dates, deduplication, screening counts, and agreement reported.
- [ ] Taxonomy validated against the corpus by >1 coder with agreement reported; core vs. adjacent evidence tiers defined and reported separately (review 2.3).
- [ ] ≥ 2 comparison tables placing the author's DyHuCoG alongside external game-theoretic recommender work (not just SHAP-on-tabular).
- [ ] Critical analysis section addressing *faithfulness vs. actionability*, the value-function-arbitrariness critique, and reproducibility — not just a listing.
- [ ] **The review's conclusions do not depend on the benchmark outcome** (a null benchmark is still a valid survey; review 2).
- [ ] Benchmark (if included) is a **separately-scoped case study** with a fixed estimand, external pre-registration, matched controls, per-user analysis unit, predeclared contrast family, and code released (§B.0/§A.10). It grounds only the interaction-player / ranking-utility slice (review 2.5/4.3).
- [ ] **Primary realized results reported as Recall@K and NDCG@K (K ∈ {5,10,20}) result tables** for every attribution family actually run, with mean ± std over seeds and significance flagged after correction. **No predicted values appear as results** (review 1.1/2.1); prediction register is a separate, clearly-labelled supplement linked to the external pre-registration.
- [ ] Consistent minimum vs. actual scope: the set of attribution families actually run matches what is claimed (no partial factorial; `myerson`/`shapley-ai` dropped or fully run; review C4).
- [ ] Scope respected: no clustering datasets in the benchmark; no new-method claim; no verbatim thesis text; reused equations carry in-text attribution (review 1.5).
- [ ] Ethics determination obtained for the human-generated data; accurate funding, COI (incl. prior authorship of reviewed methods), data/code, and AI-use statements (review 2).
- [ ] Roadmap argued from the literature's gaps; the author's planned work appears as one direction among several, not the organizing structure (review 1.4).

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

This paper is "done" when: (1) the **systematic review** (PRISMA 2020 protocol, validated corpus, coded taxonomy, critical analysis) is complete and reproducible on its own, independent of the benchmark; (2) the **case-study benchmark** is a separately-specified, pre-registered study with realized Recall@K/NDCG@K results and matched controls; (3) the review + case-study manuscript is internally consistent and free of the cross-document contradictions flagged in review 2 §3; (4) the bibliography, appendices, and figures/tables are complete and reconciled; (5) the **Discover AI submission package** is assembled (Review article type with justified hybrid format, abstract < 250 words, cover letter, ethics determination, and all §1.1/§13.1 mandatory statements + formatting); and (6) the framing check passes — the survey reads as a field-wide synthesis that does **not** read as a thesis recap or a promotional vehicle for the author's planned papers.
