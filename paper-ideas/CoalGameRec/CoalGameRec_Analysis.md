# Analysis — "Coalitional Game Theory for Explainable Graph-Based Recommendation Systems"

**Prepared by:** Arena agent, grounded in `phd-thesis/` and `previous-papers/`
**Date:** 2026-08-03
**Status:** Critical analysis / feasibility memo for a proposed *next paper* (a planning document, not a manuscript).
**Revision history:**
- **v1 (2026-08-03):** initial feasibility memo.
- **v1.1 (2026-08-03):** benchmark deliverable sharpened to report **HitRate@K and NDCG@K (K ∈ {5,10,20}) result tables**.
- **v1.2 (2026-08-03):** updated to reconcile with two pre-submission reviews (`review1.md`, `review2/review2.md`). **This memo's earlier "no GPU / no new experiments / no ethics approval / cheap" claims are SUPERSEDED:** the adopted design is a **systematic review + a separately-scoped empirical case study planned for external preregistration**, which is original empirical work requiring real compute and an institutional ethics determination for the human-generated data. The novelty claim is a **hypothesis pending the systematic search**, not an established fact. Publication-count claims were corrected (C12).

---

## 0. One-paragraph verdict

This proposed title reads as a **survey / taxonomy / perspective paper**, not a method paper
(there is no method acronym, unlike DyHuCoG, FairShap, ActionShap, etc.). That is a *strategically*
sound choice: it targets a plausible gap at the intersection — coalitional (cooperative) game theory ×
explainable AI × graph/hypergraph recommender systems — one the author is well-placed to write
(3 published papers + a PhD thesis on this spine; the thesis itself is a separate work, C12). The
paper is **feasible and low-technical-risk for the review**, but the adopted **empirical case study is
original work requiring real compute, external pre-registration, and an ethics determination** — it is
not "cheap" or "no new experiments." The novelty claim ("no dedicated survey exists") is a **hypothesis
to be confirmed by the registered systematic search**, not an established fact. Main risks: (1) scope
creep into shallowness; (2) reading as a recap of the author's own thesis rather than a field-wide
synthesis; (3) predicted-results-as-results integrity risk; (4) getting scooped. Recommendation: a
**tight, deep systematic review** with the case study kept separate and secondary.

---

## 1. What the proposed paper most plausibly is

The exact title — *Coalitional Game Theory for Explainable Graph-Based Recommendation Systems* —
uses "Coalitional" (the standard synonym for *cooperative*) and "Explainable Graph-Based
Recommendation Systems", and names *no specific method*. In the author's own corpus every
**method** paper carries an acronym in the title (DyHuCoG 2026; and the planned FairShap, ActionShap,
MHyperShap, SignalShap all follow suit). A title with no acronym, framing a *field*, is the signature
of a **survey / systematic review / taxonomy**, or at most a **position + research-agenda** paper.

Thematically it is exactly the intersection the thesis's own *spine* argues for. The thesis's RQ5
("What emerges when clustering explanation and recommendation learning are read as two stages of a
shared cooperative-game perspective?") is a *thesis-level, never-published-standalone* claim — and
the thesis itself (Ch. 8, §8.2) explicitly frames cooperative attribution as "a common formal
language for explanation across tasks that are usually studied separately." A survey is the natural
vehicle to make that thesis-level perspective citable on its own.

---

## 2. Where it fits in the author's trajectory

| Item | Type | Role the proposed survey can play |
|---|---|---|
| PhD thesis (2026): *Cooperative Game Theory for XAI in Recommendation Systems — A Shapley Framework for Actionable Insight* | Thesis | The survey is the **published "map" of the thesis spine** — converts Ch.2 Background + Ch.3 Related Work + RQ5 into a citable artifact |
| Paper 1 (2023, Procedia CS 220): Shapley for black-box **clustering** | Method | One case study of game-theoretic attribution; motivates "players = features" games |
| Paper 2 (2025, IJACSA 16): **Multi-level** hierarchical XAI for clustering | Method | Case study of attribution **coherence across granularities** |
| Paper 3 (2026, IJIESS 19): **DyHuCoG** dynamic hypergraph cooperative game for recommendation | Method | The flagship "players = interactions; in-training Shapley" case study |
| Planned: **ActionShap** (actionability of attributions) | Method | Survey's "attribution→intervention" open-challenge section; the survey pre-frames it |
| Planned: **FairShap** (two-sided fairness / popularity debiasing) | Method | Survey's "fairness/exposure" category; survey pre-frames it |
| Planned: **SignalShap** (exact Shapley over hybrid signal sources) | Method | Survey's "players = architectural/signal sources" category |
| Planned: **MHyperShap** (Myerson-restricted dynamic hypergraph games for multi-agent LLM credit) | Method | Survey's "structure-aware solution concepts (Myerson value / communication-graph games)" category |

**Publication count (C12 correction):** the author has **three published papers** (2023, 2025, 2026) plus a **PhD thesis** (2026) and **four planned method papers** (ActionShap, FairShap, SignalShap, MHyperShap). "Five prior publications" was inconsistent with the trajectory table and is corrected; cite each work precisely.

**Strategic takeaway (private planning only — keep out of the manuscript, review 1.4):** publishing the survey before the method papers gives them a ready-made related-work home. In the manuscript, the gap and the agenda must be argued from the literature's own gaps, with the author's planned work as one direction among several — not as the organizing structure or a "citation home" rationale.

---

## 3. The literature gap — why it may be novel (hypothesis pending the systematic search)

Preliminary grounding searches (Aug 2026) suggest a rich but *fragmented* landscape; the registered systematic search will test whether any prior review meets the prespecified scope at this exact intersection:

- **General Shapley-value XAI surveys exist** and are mature, e.g. Zhao, Liu & Parilina, *The Shapley
  Value Contribution to Explainable AI: A Comprehensive Survey* (Dynamic Games & Applications, 2025);
  Li et al., *Shapley value: from cooperative game to XAI* (Autonomous Intelligent Systems, 2024).
  These are **model/feature-centric** (mostly tabular classifiers, LIME/SHAP comparisons) — they do
  *not* target ranking/recommendation.
- **GNN-explanation literature is active** (EdgeSHAPer, GraphSVX, GraphGI, GStarX, GISExplainer,
  GraphEXT, and Myerson/communication-graph Shapley surveys such as Hu, Shan & Li). But these target
  **node/edge/graph classification and molecule/social graphs**, not **top-N recommendation ranking**.
- **Game-theoretic recommender methods are proliferating** (Shapley-based community CF; Shapley data
  valuation/pruning for RS; TU-bandit "creator-incentive" games for RS, e.g. the AISTATS 2026 oral;
  DyHuCoG; FairShap-class exposure-correction). These are **scattered single papers**, rarely
  cross-referenced, with inconsistent players/value functions and no unifying taxonomy.

**Gap claim (hypothesis, to be confirmed by the registered search — review 1.6/2.2):** we hypothesize
there is no survey that (a) is scoped to *recommendation ranking* rather than general XAI or GNN
classification, (b) treats the full family of *coalitional solution concepts* used in graph RS, and
(c) provides a comparative, critical synthesis. **Near-neighbors the search must explicitly check**
(review 2 §9): ShaRP (ranking Shapley), Shapley-value XAI surveys, Shapley data valuation for RS
(e.g., Ghorbani & Zou line, Jia et al., VLDB 2024/25), TU-bandit creator-incentive games (AISTATS 2026),
and "Beyond Shapley Values" (Weber/Harsanyi sets). Do not assume a paper is out of scope from its title.

---

## 4. Core contributions a good version must deliver

If this is a survey, its novelty must come from the **synthesis**, not from listing papers. The paper
should deliver:

1. **A taxonomy of coalitional games in graph-based recommendation**, organized along axes the field
   currently leaves implicit:
   - **Player set:** features · interactions (u,i) · items · users · contexts · signal sources ·
     data tuples/items · nodes · edges/hyperedges · providers · LLM agents.
   - **Characteristic / coalition value function** `v(S)`: what utility a coalition earns (NDCG,
     diversity, coverage, context-alignment, preference-consistency, fairness/exposure, regret...).
   - **Solution concept:** Shapley value and its structure-aware/interaction variants (Myerson value
     under a communication graph, Harsanyi dividends / interaction indices, Banzhaf, Weber/Harsanyi
     allocation sets, core/nucleolus).
   - **Role of attribution:** post-hoc explanation · in-training optimization signal · data/credit
     valuation · fairness/exposure correction · actionability/intervention.
   - **Graph structure:** bipartite user–item · homogeneous GNN · heterogeneous / knowledge-graph ·
     hypergraph · dynamic/temporal.
2. **A unified notation** reusing the thesis's game formulation `(N, v)`, the Shapley formula, and
   Monte-Carlo estimation — so every reviewed method can be expressed in one vocabulary.
3. **A systematic mapping** of methods into the taxonomy with **core vs. adjacent evidence tiers**
   (review 2.3), including the author's DyHuCoG *and* external work (EdgeSHAPer, GraphSVX, GraphGI,
   GStarX, Shapley community-CF, Shapley data-pruning for RS, TU-bandit creator-incentive games,
   Myerson-value recommender variants, etc.). Define coding rules for what counts as a substantive
   game-theoretic contribution vs. "relabeled reweighting" (review 4.1).
4. **A critical, comparative analysis** (not just annotation): what the game-theoretic lens genuinely
   buys — axiomatic fairness, principled credit under redundancy, interaction handling, in-training
   optimization, exposure/fairness, actionability — and what it costs — intractability and the
   approximation problem, the arbitrariness of the value function, ambiguity of player definitions,
   reproducibility.
5. **An open-challenges / research-agenda section** aligned with the thesis's future-work and the
   planned method papers (scalable low-variance approximation, streaming/online, human-centred
   actionability evaluation, fairness audits, communication-graph/Myerson structure for GNNs,
   multi-agent LLM credit assignment).

---

## 5. Feasibility and novelty assessment

### Strengths
- **Defensible gap** (see §3, subject to the search). A well-scoped survey has a citable contribution.
- **Well-positioned author:** 3 published papers + a thesis on this spine; can frame the field and
  honestly situate own work (avoid "uniquely placed"/"leading voice" advocacy in the manuscript, review 2 §4.1).
- **Review is low-technical-risk** (no new theorem); but the **case study is original empirical work**
  needing compute, pre-registration, and an ethics determination — not "cheap" (review 1/4.1).
- **High leverage** if the agenda is argued from the literature; keep the "programme/portfolio" framing out of the manuscript (review 1.4).

### Weaknesses / risks
1. **Scope creep → shallowness.** "Coalitional × explainable × graph-based × recommendation" is four
   intersecting literatures. Without a tight scope the paper becomes a shallow annotated bibliography
   that reviewers (and the field) will ignore.
2. **"Thesis recap" risk.** If it reads as a summary of the author's own work, it will be rejected.
   It must be a *field-wide* synthesis in which the author's papers are *case studies among many*.
3. **Survey-methodology expectations.** A credible survey needs a transparent protocol (search
   strategy, inclusion/exclusion criteria, counts, coverage of venues/years) and comparison tables —
   not an informal literature tour. Decide this up front.
4. **Young field → thin "core."** Few papers are *strictly* at the intersection; many are adjacent
   (Shapley feature attribution in KG-based RS; GNN explainers for classification). Inclusion
   criteria must be explicit or reviewers will attack "missing papers."
5. **Scoop risk.** The intersection is hot (2025–2026). Delay invites a competing survey.
6. **DyHuCoG reproducibility flag.** An internal audit found 63 extraction gaps; **the benchmark uses no DyHuCoG code** (authors' decision), and DyHuCoG appears only as a taxonomy worked example described candidly — the audit is treated as an author-side implementation risk, not a published field fact (review 1.3/2).
7. **Venue fit / Review vs. Research.** Discover AI's Review type fits a survey, but the hybrid
   Review+benchmark format must be justified in the cover letter and Introduction; an editor may
   reclassify to `Research` or request a split (review 1.2/2.1).
8. **Predicted-results-as-results integrity risk (blocking, review 1.1/2.1).** Never let predicted
   numbers or a pre-drafted "winner" reach the manuscript; fill result tables only with realized
   numbers; use external pre-registration.
9. **Ethics misclassification (blocking, review 2.7).** MovieLens/Amazon data are human-generated;
   obtain and report an institutional determination rather than asserting "not applicable."

---

## 6. Relationship to the planned method papers — avoid cannibalization

- The survey = **map of the state of the art**; the planned papers = **specific new methods**.
  The adopted design adds a **small empirical case study** that must be kept separate and secondary
  (review 2.5); it is not a vehicle for the author's planned method papers.
- **Order matters** as private strategy (draft the survey first so method papers can cite it), but the
  manuscript must not frame the survey as a "citation home" for the author's own papers (review 1.4).
- Do **not** turn this title into a method paper under a new name (e.g., "we unify the thesis into one
  framework") — that would collide with the planned papers. Keep the survey as a field synthesis.

---

## 7. Recommended scope and framing

Three viable framings, in order of fit:

- **(A) Systematic review + taxonomy with a separately-scoped empirical case study — RECOMMENDED.**
  Deliverable: comprehensive review with a registered PRISMA protocol, taxonomy, comparison tables,
  critical synthesis, and roadmap, plus a separate case-study benchmark planned for external preregistration (realized
  HitRate@K and NDCG@K). This is what the title supports and what the field lacks.
- (B) **Position + research agenda.** Lighter, faster, more opinionated. Viable if the author wants a
  quick, high-impact "call to action" piece, but leaves the field-map unclaimed and invites a
  competitor survey.
- (C) **Method paper** (rename to a specific method). **Not recommended** under this title — it
  conflicts with the planned portfolio.

Within (A), the recommended **tight scope**:
> Focus on **attribution games for top-N graph/hypergraph recommendation** — i.e., games whose
> players are recommendation-relevant entities (interactions, items, users, contexts, signal
> sources) and whose value is a ranking/quality utility — and on the **Shapley family of solution
> concepts** (plus Myerson/structure-aware and Harsanyi interaction variants). Use *explainability*
> (post-hoc, in-training, fairness, actionability) as the organizing lens, not as an afterthought.

This keeps it deep and comparative while still being a survey.

---

## 8. Suggested survey research questions (SRQs) the paper should answer

- **SRQ1 (taxonomy).** What are the distinct ways coalitional games are instantiated for graph-based
  recommendation — across player set, value function, solution concept, role, and graph type?
- **SRQ2 (comparison/synthesis).** What patterns and directions are reported across the coded corpus
  when methods are expressed in one shared vocabulary? (Synthesis: descriptive + structured vote/count —
  not correlation/meta-regression.)
- **SRQ3 (contribution analysis).** What does the game-theoretic lens genuinely add over heuristic
  weighting, and where is it just relabeled reweighting?
- **SRQ4 (validity).** How is attribution validated (faithfulness, stability, actionability,
  reproducibility)? What evaluation gaps recur?
- **SRQ5 (roadmap).** What are the highest-leverage open problems (approximation, online/streaming,
  human-centred evaluation, fairness audits, structure-aware solution concepts, agentic systems)?

---

## 9. Suggested skeleton (if the author proceeds with the survey)

```
Abstract / Keywords
1. Introduction (motivation, scope, contributions, positioning vs. prior surveys)
2. Preliminaries
   2.1 Coalitional (cooperative) game theory: TU games, (N, v)
   2.2 Solution concepts: Shapley, Myerson/communication-graph value, Harsanyi dividends &
       interaction indices, Banzhaf, Weber/Harsanyi sets, core/nucleolus
   2.3 Graph-based recommendation: bipartite, homogeneous GNN, KG/heterogeneous, hypergraph,
       dynamic/temporal; top-N problem
   2.4 Explainability in recommendation: post-hoc / in-training / fairness / actionability; and
       evaluation of explanations
3. Survey methodology (search protocol, sources, inclusion/exclusion, counts, PRISMA 2020 flow + checklist)
4. A taxonomy of coalitional games for explainable graph-based recommendation
   (axes: players · value function · solution concept · role · graph type)
5. Systematic review by category
   (player-centric: features→interactions→items/users→contexts→signal sources→data→providers→agents)
6. Comparative and critical analysis (what game theory buys / does not buy; faithfulness vs.
   actionability; reproducibility; DyHuCoG audit caveat)
7. Separately-scoped empirical case study (planned for external preregistration; realized
   HitRate@K and NDCG@K, BQ answers)
8. Open challenges and research agenda (argued from the literature)
9. Limitations
10. Conclusion
Declarations · Appendices / Online Resources
```

Target length ~10,000–12,000 words, 7 figures and 6 tables in main text (reconciled across files; detailed result tables + prediction register go to supplementary material with distinct labels, review C6/C18).

---

## 10. Concrete recommendations

1. **Commit to framing (A)** — full systematic survey + taxonomy. Write a 1-page scope + protocol
   before drafting.
2. **Publish/draft it before the method papers** (private strategy) so they can cite it; keep the
   portfolio rationale out of the manuscript (review 1.4).
3. **Borrow the thesis** notation (Ch. 2.8) and the Related-Work narrative (Ch. 3), then expand to
   field-wide coverage. Do **not** reuse thesis text verbatim (self-plagiarism / overlap concerns);
   rewrite in survey voice.
4. **Define inclusion criteria explicitly** to protect against "you missed X" reviews, and include a
   transparent paper-count/flow figure.
5. **De-scope the "graph" to recommendation-specific** structures (user–item, KG, hypergraph,
   temporal) and keep GNN node/edge explainers as a clearly-labeled *adjacent* category.
6. **Position DyHuCoG candidly and use no DyHuCoG code** in the benchmark — pin the independent HCCF hypergraph backbone (with a preregistered HGNN fallback); treat the audit as an author-side risk (review 1.3).
7. **Confirm the venue** (Discover AI Review type) and justify the hybrid Review+benchmark format in
   the cover letter; expect possible reclassification (review 1.2/2.1).
8. **Protect research integrity (blocking):** external pre-registration, realized-only result tables,
   neutral hypotheses, per-user analysis unit with a predeclared contrast family (review 1.1/2.1/2.6).
9. **Obtain the ethics determination** for the human-generated data before processing (review 2.7).
10. **Move fast** to mitigate scoop risk; the review draft is feasible on the thesis literature base, but
    budget the case-study benchmark as real empirical work.

---

## 11. Bottom line

Strong "yes" with two structural decisions: (1) do it as a **systematic review + taxonomy**, tightly
scoped, with the **empirical case study kept separate and secondary**; and (2) **protect research
integrity** — external pre-registration, realized-only results, neutral hypotheses, and an ethics
determination. The review is the natural capstone-to-next-step of the thesis and fills a plausible gap;
the case study is real empirical work that must be budgeted and reported honestly. The serious failure
modes are shallowness from scope creep, the predicted-results-as-results integrity risk, ethics
misclassification, and scooping from delay — all controllable by scoping hard, registering early, and
shipping promptly.
