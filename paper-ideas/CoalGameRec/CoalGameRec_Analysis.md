# Analysis — "Coalitional Game Theory for Explainable Graph-Based Recommendation Systems"

**Prepared by:** Arena agent, grounded in `phd-thesis/` and `previous-papers/`
**Date:** 2026-08-03
**Status:** Critical analysis / feasibility memo for a proposed *next paper* (not a manuscript draft)
**Update (2026-08-03):** the benchmark deliverable has been sharpened so the paper reports **concrete results** — the headline benchmark output is now explicit **Recall@K and NDCG@K (K ∈ {5,10,20}) result tables** for every attribution family on both backbones and both datasets (see `Implementation_Spec.md` §B.1a, `spec.md` §7.4, and `Paper_Structure.md` §7.2). The survey remains the primary contribution; the Recall/NDCG tables are the empirical grounding that answers SRQ3 ("what does game theory buy").

---

## 0. One-paragraph verdict

This proposed title reads as a **survey / taxonomy / perspective paper**, not a method paper
(there is no method acronym, unlike DyHuCoG, FairShap, ActionShap, etc.). That is a *strategically
good* choice: it sits at a genuine, currently-unoccupied intersection — coalitional (cooperative)
game theory × explainable AI × graph/hypergraph recommender systems — and the author is uniquely
placed to write it authoritatively (PhD thesis on exactly this spine + 5 prior publications). The
paper is **feasible, low-risk, and high-leverage**: it needs no GPU, no new experiments, and it
creates the citation/framing home for the author's entire planned method portfolio
(ActionShap, FairShap, MHyperShap, SignalShap). The main risks are (1) scope creep into shallowness,
(2) reading as a recap of the author's own thesis rather than a field-wide synthesis, and
(3) getting scooped if publication is delayed. The recommendation is to **publish the survey first,
with a tight, deep taxonomy rather than a broad annotated bibliography.**

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

**Strategic takeaway:** the survey should be the *first* of the next papers (or run in parallel), not
the last. Publishing it first gives every subsequent method paper a ready-made related-work home,
establishes the author as a leading voice in this niche, and makes the whole programme look unified.

---

## 3. The literature gap — why it is genuinely novel (and why now)

Grounding searches (Aug 2026) show a rich but *fragmented* landscape with **no dedicated survey at
this exact intersection**:

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

**So the gap is real:** there is no survey that (a) is scoped to *recommendation ranking* rather than
general XAI or GNN classification, (b) treats the full family of *coalitional solution concepts* used
in graph RS (Shapley + Myerson/communication-graph + Harsanyi interactions + Banzhaf + core/nucleolus),
and (c) provides a comparative, critical synthesis. This is exactly the hole this title fills.

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
3. **A systematic mapping** of methods into the taxonomy, including the author's DyHuCoG *and*
   external work (EdgeSHAPer, GraphSVX, GraphGI, GStarX, Shapley community-CF, Shapley data-pruning
   for RS, TU-bandit creator-incentive games, Myerson-value recommender variants, etc.).
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
- **Defensible gap** (see §3). A well-scoped survey has a genuinely citable contribution.
- **Uniquely positioned author:** 5 prior publications + a thesis on precisely this spine; can frame
  the field authoritatively and honestly situate own work.
- **Cheap to produce:** no GPU, no new datasets, no human subjects, no ethics approval — reuses the
  thesis's literature review and notation. Fits the author's proven "structure mirrors DyHuCoG /
  FairShap blueprint" house style.
- **High citation potential + program-level leverage:** the survey becomes the shared reference that
  all four planned method papers cite.
- **Low novelty risk *if* framed as a survey** (novelty = the taxonomy/synthesis, not a new algorithm).

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
6. **DyHuCoG reproducibility flag.** The repo's own SignalShap audit notes 63 extraction gaps that
   make faithful DyHuCoG reimplementation impossible. If the survey uses DyHuCoG as its flagship
   case study, describe it accurately, do not over-claim, and be ready for scrutiny.
7. **Venue fit.** Survey article types vary by journal; pick a venue that accepts structured reviews
   (e.g., a Springer survey/review track, ACM CSUR-style, or the author's proven Discover AI).

---

## 6. Relationship to the planned method papers — avoid cannibalization

- The survey = **map of the state of the art**; the planned papers = **specific new methods**.
  No substantive overlap *as long as* the survey reports no new results.
- **Order matters:** publish (or at least draft) the survey **before** submitting the method papers
  so they can cite it as "the survey situates our method." Flipping the order invites duplication
  reviewers to ask "why is this not in your survey?"
- Do **not** try to turn this title into a method paper under a new name (e.g., "we unify the thesis
  into one framework") — that would collide with ActionShap/FairShap/MHyperShap/SignalShap and
  weaken them. Keep the survey as pure synthesis.

---

## 7. Recommended scope and framing

Three viable framings, in order of fit:

- **(A) Full systematic survey + taxonomy — RECOMMENDED.** Deliverable: comprehensive review with a
  protocol, taxonomy, comparison tables, critical synthesis, and roadmap. This is what the title
  supports and what the field lacks.
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
- **SRQ2 (comparison).** How do existing methods compare on the same game-formulation vocabulary, and
  what correlates with reported gains?
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
3. Survey methodology (search protocol, sources, inclusion/exclusion, counts, PRISMA-style flow)
4. A taxonomy of coalitional games for explainable graph-based recommendation
   (axes: players · value function · solution concept · role · graph type)
5. Systematic review by category
   (player-centric: features→interactions→items/users→contexts→signal sources→data→providers→agents)
6. Comparative analysis (comparison tables: method × players × v(S) × solution concept × task ×
   graph × role × approximation × evaluation)
7. Critical analysis (what game theory buys / does not buy; faithfulness vs. actionability;
   reproducibility)
8. Open challenges and research agenda
9. Conclusion
Appendix A: notation/glossary · Appendix B: reviewed-works index
```

Target length ~9,000–12,000 words, 6–8 figures, 4–6 comparison tables (per the author's house style).

---

## 10. Concrete recommendations

1. **Commit to framing (A)** — full systematic survey + taxonomy. Write a 1-page scope + protocol
   before drafting.
2. **Publish it first** in the next-paper sequence; cite it from all four planned method papers.
3. **Borrow the thesis** notation (Ch. 2.8) and the Related-Work narrative (Ch. 3), then expand to
   field-wide coverage. Do **not** reuse thesis text verbatim (self-plagiarism / overlap concerns);
   rewrite in survey voice.
4. **Define inclusion criteria explicitly** to protect against "you missed X" reviews, and include a
   transparent paper-count/flow figure.
5. **De-scope the "graph" to recommendation-specific** structures (user–item, KG, hypergraph,
   temporal) and keep GNN node/edge explainers as a clearly-labeled *adjacent* category.
6. **Position DyHuCoG accurately** and flag its reproducibility caveat; do not let an audit failure
   of a flagship case undercut the survey's credibility.
7. **Choose a survey-friendly venue**; confirm survey/article-type acceptance and typical length.
8. **Move fast** to mitigate scoop risk — a focused first draft is feasible in ~3–4 weeks using the
   thesis literature base.

---

## 11. Bottom line

Strong "yes" with one structural decision: do it as a **survey/taxonomy**, tightly scoped, and
publish it **first** in the portfolio. It is the natural capstone-to-next-step of the thesis, fills a
real gap, is cheap and low-risk to produce, and maximizes the impact of the four method papers that
follow. The only serious failure modes are *shallowness from scope creep* and *scooping from delay* —
both are controllable by scoping hard and shipping early.
