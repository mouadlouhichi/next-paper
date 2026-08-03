# CoalGameRec — Discovery AI standards review

**Documents reviewed**

- `CoalGameRec_Analysis.md`
- `Implementation_Spec (1).md`
- `Paper_Structure.md`
- `spec (1).md`

**Review date:** 3 August 2026  
**Review type:** pre-submission scientific, methodological, statistical, citation, ethics, and formatting audit  
**Target journal used for the audit:** *Discover Artificial Intelligence*, Springer Nature

> **Important limitation.** These files are a proposal, paper blueprint, implementation specification, and prediction register—not a completed manuscript, systematic-review corpus, code repository, or results package. Consequently, this review assesses whether the proposed work could support its planned claims and whether the package is submission-ready. It does not validate any claimed empirical result.

---

## Executive verdict

**Recommendation: major redesign and major revision before implementation or submission. Do not submit the current package, and do not put the predicted tables into the manuscript as results.**

The topic is potentially publishable, and a focused review article could fit *Discover Artificial Intelligence*. The strongest idea is a field-wide synthesis of how cooperative-game formulations are used for attribution in graph-based top-N recommendation. However, the current package has several blocking problems:

1. **The abstract and Results plan state unobserved predictions as findings.** The manuscript blueprint already says that Shapley attribution ranks first, although the implementation file says no experiment has yet run.
2. **The review protocol is only a plan.** It lacks a frozen protocol, database-specific queries, dates, deduplication, screening counts, independent reviewers, agreement statistics, a mandatory quality/risk-of-bias procedure, and the final evidence corpus. The phrase “first systematic survey” is therefore unsupported.
3. **The scope conflates different objects.** Post-hoc explanation, in-training edge weighting, training-data valuation, provider-incentive mechanisms, fairness/exposure allocation, and LLM-agent credit are not interchangeable forms of “explainable graph-based recommendation.”
4. **The central game is not operationally defined.** It is unclear whether a coalition is evaluated by masking interactions in a frozen model, retraining on a subset, changing propagation weights, or changing a recommendation list. Those alternatives answer different scientific questions and have very different computational and leakage properties.
5. **The benchmark is not a fair test of “what game theory buys.”** Shapley receives a specially chosen multi-objective value function and a preference term; the baselines do not receive a matched objective, and the proposed reranking/training intervention is mixed with explanation evaluation.
6. **The statistical plan is inadequate for the design as written.** Five random seeds are not five independent datasets; per-user tests, seed aggregation, many correlated cutoffs/metrics/comparisons, and an unspecified Holm family are not coherently specified.
7. **There are material cross-file contradictions.** Most importantly: no experiment/no GPU versus a GPU benchmark; 70/10/20 splitting versus last/second-last leave-one-out; optional Myerson versus “every family on both backbones and datasets”; and several table/figure numbering conflicts.
8. **The current journal information is partly stale or overclaimed.** The official journal page currently reports a median first decision of 16 days, whereas the specification says approximately 23 days. The official pages confirm the `Review` article type, less-than-250-word abstract, cover letter, funding statement, data statement, and current APC, but they do not establish the claimed Q1 category. Q1 status should be cited to a named ranking source and year or removed.
9. **The ethics statement is too categorical.** MovieLens and Amazon reviews are human-generated data. Public/de-identified secondary data may not require consent or full ethics review, but that is an institutional and jurisdictional determination—not something that can be asserted merely because there was no user study.

The project should be reframed as a **systematic review with an explicitly separate, exploratory empirical case study**. If the benchmark is retained, the paper must say that it grounds only one taxonomy slice; it cannot empirically validate the entire five-axis taxonomy or establish a general superiority claim for Shapley methods.

---

## Severity legend

- **P0 — blocking:** must be corrected before implementation, analysis, or submission.
- **P1 — major:** threatens validity, interpretation, reproducibility, or editorial suitability.
- **P2 — moderate:** should be corrected for a credible review and likely reviewer resistance.
- **P3 — minor/editorial:** clarity, style, cross-reference, or polish issue.

---

# 1. Discovery Artificial Intelligence compliance audit

The current official references checked for this review are the [journal submission guidelines](https://link.springer.com/journal/44163/submission-guidelines), [journal home page](https://link.springer.com/journal/44163), [APC page](https://link.springer.com/journal/44163/how-to-publish-with-us), and [Discover editorial policies](https://link.springer.com/brands/discover/policies). The journal guidance should be rechecked immediately before submission because policies, metrics, fees, and article types can change.

| Journal requirement or policy | Current status in the files | Severity | Required action |
|---|---|---:|---|
| Correct article type | The files consistently select `Review`, which is plausible. The journal describes Reviews as critical accounts and comprehensive surveys and says they may be of any length. | P2 | Retain `Review`, but ask the editor in a presubmission inquiry whether a Review may contain a new benchmark of the proposed size. If the benchmark becomes the dominant contribution, reconsider `Research`. |
| Abstract under 250 words | The draft abstract in `Paper_Structure.md` is approximately 256 words by a conventional word-token count, and over 260 by whitespace counting depending on treatment of hyphenated terms. | **P0** | Reduce it below 250 words after removing predictions and citations that are not essential. Check the final Word/LaTeX count. |
| Cover letter | Required by the journal, but only mentioned as a future task. | P1 | Draft a cover letter explaining the review contribution, scope, benchmark status, and fit. Do not market the paper as “Q1” or guaranteed to be accepted. |
| Funding statement | Placeholder only: “declare grant/none.” | P1 | Insert the true funding statement, including grant number/funder or an accurate no-funding statement. |
| Author contributions | A proposed CRediT split is supplied before the work exists. | P2 | Replace with actual contributions after the work is completed; obtain every author’s approval and consent to submit. |
| Competing interests | “The authors declare no competing interests” is prefilled despite the authors’ own DyHuCoG work, thesis, planned portfolio, and benchmark implementation being central to the paper. | **P1** | Make a complete non-financial and financial disclosure. At minimum disclose prior authorship of reviewed methods and any relationship between the benchmark and those methods. |
| Data availability | Public-data statement is planned, but no exact versions, links, hashes, access dates, license/terms, or final custom Amazon split exist. | **P1** | Give dataset citations, URLs, access dates, file checksums, final counts, split-generation code, and a statement that raw data are not redistributed if terms prohibit it. |
| Code/materials availability | Repository and Zenodo DOI are placeholders. | **P1** | Release code, configs, frozen splits, raw result files, environment lockfile, tests, and a permanent archive before submission, or state accurately that they are not yet available. |
| Ethics/human data | The package repeatedly says ethics is not applicable because there is no user study. | **P0** | Ask the institution/ethics committee whether secondary use of public MovieLens/Amazon human data needs an exemption or determination. Report the committee/name/reference if applicable; otherwise explain the institutional determination. |
| AI-use policy | A “standard” group statement is proposed, but the current Discover policy requires transparent, risk-based disclosure and human accountability. | P1 | Disclose actual AI use, including drafting, critique, code assistance, or methodological suggestions if applicable. Authors must verify all scientific content and citations; AI must not be listed as an author. |
| Dual publication/text recycling | The plan says no text/equation/figure will be reused verbatim, but overlap of data, code, experiments, ideas, and thesis material is not mapped. | **P1** | Prepare an overlap table covering the thesis, DyHuCoG, the two clustering papers, code, datasets, figures, equations, and results. Disclose relevant prior work to the editor. |
| Third-party material | No permissions plan for reproduced diagrams, screenshots, dataset figures, or adapted tables. | P2 | Create all figures anew or obtain permission and cite the original source in captions. |
| Reporting standard | “PRISMA-style” is proposed. Discover recommends PRISMA for systematic reviews. | **P0** | Use PRISMA 2020, cite it, complete the checklist and flow diagram, and report all required items. Use a search-reporting supplement such as PRISMA-S if appropriate. |
| Formatting | No final Springer template, 12-point font check, heading-level check, square-bracket numeric citations, or final figure/table compliance. | P1 | Prepare the manuscript in the current Springer format. The journal guidance specifies a consistent font of at least 12 pt, no more than three heading levels, numbered square-bracket citations, sequential Arabic table/figure numbering, captions, and figures/tables in the body. |
| Accessibility | The plan does not specify alt text/descriptive captions, contrast, colour-blind-safe encodings, or pattern use. | P2 | Add descriptive captions/alt text, use patterns as well as colour, verify contrast, and ensure all plot text is legible at final size. |
| Q1/venue claims | “Q1” and “Q1—Information Systems” are repeatedly treated as established facts; the official journal page confirms indexing and CiteScore but not that label. | P2 | Cite the exact Scopus/SCImago/JCR source, category, edition, and year, or remove ranking language. Never use Q1 as a scientific contribution. |
| Journal metrics | The specification says median first decision ≈23 days. The current official journal page checked on 3 August 2026 says 16 days. | P2 | Update to the current official value or omit it; do not use an editorial speed claim to justify scientific feasibility. |
| APC | The stated £1,190 / $1,690 / €1,390 matches the current official APC page checked on the review date, but it is acceptance-date dependent and subject to tax. | P3 | Retain only in private planning notes, with a date and “subject to change”; remove from the manuscript. |

### Discovery-specific policy points the files currently miss

1. The journal’s policy applies to **human data**, not only to direct user studies. The authors must not equate “public” with “not human data.”
2. The current policy requires authors to remain accountable for originality, accuracy, integrity, and editorial judgement. An AI disclosure should be specific and truthful, not merely a generic laboratory sentence.
3. The journal asks that data, materials, software, and custom code support published claims. A future repository placeholder is not an availability statement.
4. The final manuscript must use the journal’s numeric citation convention and include only cited works that are published or accepted for publication. Planned methods without a public manuscript should not be presented as established literature.

---

# 2. P0 blockers that must be resolved first

## 2.1 Predicted results are presented as results

This is the most serious problem.

`Implementation_Spec (1).md` explicitly says Part B contains predictions made before running anything. Nevertheless:

- The draft abstract says that Shapley attribution “ranks first on both metrics across datasets and backbones.”
- `Paper_Structure.md` §7.2 instructs the authors to “highlight that `shapley-mc` ranks 1st” and to lead the Results narrative with that outcome.
- The same section assumes the largest margin occurs on sparse Amazon-Book.
- The planned benchmark tables contain exact predicted means.
- BQ1/BQ4 repeatedly describe the predicted ordering as the “headline prediction.”

This is not a harmless wording issue. It creates confirmation bias, makes the manuscript’s conclusion conditional on a favourable result, and would be unacceptable if the manuscript were submitted before the experiment. A local Markdown file with a date is not a public preregistration.

**Required correction**

- Replace every future-result sentence with a neutral hypothesis: “We test whether…”
- Remove predicted numerical tables from the main manuscript outline and abstract.
- If preregistration is desired, deposit an immutable protocol and prediction register in OSF, AsPredicted, Zenodo, or an equivalent repository before running the benchmark. Record the timestamp, commit hash, code version, and planned deviations.
- Keep predictions in a clearly labelled supplement or registered-report artifact, never as realized results.
- Analyze the primary outcome with a fixed script before inspecting method-specific results if feasible.
- Report every deviation, null result, estimator failure, dropped run, and contingency activation.

## 2.2 The survey is not yet a systematic review

The files specify what a systematic review should contain, but none of the necessary evidence exists. There is no:

- final protocol;
- protocol registration or timestamped repository;
- database-specific query syntax;
- search date and last-update date;
- language/publication-type policy;
- result export;
- duplicate-removal procedure;
- title/abstract/full-text counts;
- independent dual screening;
- disagreement resolution rule;
- inter-rater agreement;
- full-text exclusion log with reasons;
- completed extraction sheet;
- mandatory risk-of-bias/quality assessment;
- final corpus of included works;
- PRISMA 2020 checklist.

The `optional but recommended` wording for quality assessment is too weak for a paper that promises a “critical synthesis,” evaluates validation gaps, and claims to identify what methods genuinely add. Without a quality framework, the paper may merely reproduce the claims of the most promotional papers.

**Required correction**

Adopt and cite PRISMA 2020. Before screening, freeze:

1. databases and platform versions;
2. database-specific search strings;
3. date range and exact search date;
4. language and publication-type rules;
5. core versus adjacent eligibility rules;
6. duplicate handling;
7. two-person screening and conflict resolution;
8. extraction fields and allowed values;
9. quality/risk-of-bias rubric;
10. synthesis method and protocol-deviation policy.

If only one author screens or extracts, say so, justify it, and add an independent audit sample. Do not call the process fully systematic without acknowledging that limitation.

## 2.3 The core scope is conceptually overinclusive

The five axes currently combine at least six distinct scientific settings:

1. post-hoc explanation of a fixed recommender prediction;
2. in-training interaction weighting or edge reweighting;
3. data valuation/pruning;
4. provider/creator exposure or incentive allocation;
5. fairness correction;
6. multi-agent/LLM credit assignment.

A provider-incentive mechanism can use a Shapley value without explaining a model. A training-data Shapley score can value an example without explaining a particular recommendation. An LLM-agent credit game is not automatically graph-based recommendation. A graph neural-network explainer for node classification is not a top-N recommender explanation. These can be useful adjacent literatures, but they must not be silently treated as one population.

The inclusion rule in `spec (1).md`—a method may “define a coalitional game over recommendation-relevant entities **or** use a coalitional solution concept for attribution,” and operate on a graph-like structure—is broad enough to include nearly anything involving Shapley and a network. It also conflicts with the stated exclusion of non-recommendation GNN explainers while `Paper_Structure.md` later presents them as an “adjacent and emerging” review category.

**Required correction**

Define evidence tiers:

- **Core:** top-N recommendation; graph/hypergraph is an input, model, or explicit interaction structure; the game attributes a recommendation-relevant output or training signal.
- **Adjacent A:** ranking/recommendation attribution without a graph model.
- **Adjacent B:** graph/GNN attribution outside recommendation.
- **Adjacent C:** provider/data/agent incentive or valuation games without model explanation.
- **Background only:** general cooperative-game theory and general SHAP surveys.

Include adjacent work for positioning, but do not count it in the core corpus or use it to support claims about explainable graph recommendation. State whether each paper is explanation, optimization, data valuation, or mechanism design.

## 2.4 The estimand for the coalition value is undefined

The files repeatedly write

\[
v(S)=\alpha\,\mathrm{NDCG}(S)+\beta\,\mathrm{Diversity}(S)+\gamma\,\mathrm{Context}(S)
\]

but do not define what `S` changes. At least four incompatible interpretations are possible:

- **Frozen-model mask game:** mask a subset of interactions/edges at inference and measure a fixed user’s ranking.
- **Retraining/data-value game:** train a new model on coalition `S` and evaluate it on validation data.
- **In-training weighting game:** change propagation weights or batch weights and measure the current model.
- **Recommendation-list game:** define the coalition as items or providers and evaluate the list utility directly.

These have different players, baselines, computational costs, axioms, leakage risks, and interpretations. The current notation moves between them without warning.

**Required correction**

State an explicit estimand, for example:

> For user `u` and a frozen recommender `f`, the player set is the set of observed training interactions in `u`’s receptive field. A coalition `S` is applied as an edge mask to the frozen graph. `v_u(S)` is the value of a specified ranking functional on a candidate set constructed without test information.

Or choose a retraining/data-valuation game, but then call it data valuation and budget the retraining cost. Do not use a global `NDCG(S)` without specifying users, candidate items, model state, train/validation/test partition, and whether the coalition contains training or evaluation interactions.

## 2.5 The benchmark cannot support the broad “what game theory buys” claim as designed

The proposed Shapley value receives:

- a multi-objective value function;
- manually selected coefficients;
- a preference-similarity term;
- a refresh/smoothing/clipping pipeline;
- attribution-guided propagation or reranking.

The uniform, attention, and popularity baselines do not receive a matched multi-objective objective or equivalent intervention budget. A gain could therefore be due to the preference prior, the diversity term, the architecture, the learned gate, the reranker, or tuning—not to the Shapley allocation rule.

The term

\[
\lambda_{pref}\sum_{(u,i)\in S}\mathrm{sim}(u,i)
\]

is itself an additive heuristic prior. Its marginal contribution is essentially a per-interaction similarity bonus. A non-game additive-similarity baseline is essential; otherwise the experiment cannot tell whether the game-theoretic aggregation matters.

**Required correction**

Use a factorial ablation:

- same backbone and training loss;
- same scalar objective for all methods;
- Shapley allocation versus a matched additive heuristic;
- with and without the preference term;
- with and without diversity/context terms;
- fixed versus refreshed attribution;
- no-op/uniform propagation;
- learned attention with matched parameter count;
- popularity/degree and random controls;
- standard recommender baselines such as BPR-MF/LightGCN without an attribution module;
- if explanation is claimed, at least one established post-hoc explainer baseline.

State that the benchmark tests one implementation under one protocol; it does not establish that “game theory” generally outperforms non-game-theoretic methods.

## 2.6 The statistical unit and inferential procedure are not coherent

The specification proposes five seeds, per-user paired t-tests, Wilcoxon signed-rank tests, Holm–Bonferroni, Cohen’s `d_z`, and 95% CIs. It does not state:

- whether tests are run on per-user metrics, per-seed metrics, or both;
- whether users are independent after shared model training;
- what comparisons form one Holm family;
- how all three cutoffs, two metrics, two backbones, two datasets, and six families are handled;
- whether the test is primary or exploratory;
- how failed runs are handled;
- how hyperparameter selection is accounted for.

Five seeds are five model-training replicates, not five independent samples from a population. Treating millions of per-user observations as independent can produce extremely small p-values for trivial effects. Conversely, using only five seed-level values gives very low degrees of freedom and makes normality tests uninformative.

**Required correction**

Predeclare a small set of primary contrasts, for example `shapley-mc` versus uniform on NDCG@20 and Recall@20 for each dataset/backbone. Treat all other metrics/cutoffs as secondary or exploratory. Use:

- per-user paired differences for descriptive effect sizes;
- cluster/bootstrap intervals that resample users and preserve seed structure;
- a mixed-effects or hierarchical model if formal inference across seeds and users is claimed;
- randomization/permutation tests with the pairing preserved;
- a correction family defined in advance;
- effect sizes with interpretable confidence intervals;
- a minimum-detectable-effect or precision analysis to justify seed count.

If retaining t-tests/Wilcoxon, state their assumptions and use them as sensitivity analyses rather than the sole basis for conclusions. Report per-seed values, not only pooled per-user rows.

## 2.7 The ethics statement is not defensible as written

The benchmark uses data generated by people: MovieLens ratings/demographic fields and Amazon review/user identifiers, ratings, timestamps, and potentially metadata/text. The absence of direct interaction or a user study does not by itself make the work non-human-data research.

**Required correction**

Before data processing, obtain an institutional determination. The final statement should identify:

- the datasets and fields actually used;
- whether raw review text or demographics are used;
- whether data are public and de-identified/pseudonymous;
- whether an ethics committee determined that approval was not required or granted an exemption;
- why no consent-to-participate/user study was conducted, if applicable;
- how identifiers and raw text are protected and whether they are redistributed.

Do not promise “no ethics approval” as a design decision.

---

# 3. Cross-document consistency audit

| ID | Inconsistency | Locations | Why it matters | Required resolution |
|---|---|---|---|---|
| C1 | Survey is described as requiring “no new experiments/no GPU,” but the final design contains a two-backbone, two-dataset GPU benchmark. | `CoalGameRec_Analysis.md` §§0, 5 versus `Paper_Structure.md` §§1, 7 and `Implementation_Spec (1).md` Part A | Effort, article type, contribution, ethics, and claims are misrepresented. | Mark the benchmark as original empirical work, budget it, and state that the survey remains primary. |
| C2 | The actual abstract reports future results. | `Paper_Structure.md` abstract versus `Implementation_Spec (1).md` status/Part B | It is a fabricated-looking result if copied into a submission. | Use hypotheses now; insert only realized results later. |
| C3 | Split protocol is 70/10/20 temporal with leave-one-out versus last test/second-last validation/rest train. | `spec (1).md` §7.3 versus `Implementation_Spec (1).md` §A.3 and `Paper_Structure.md` §7.1 | Results are not reproducible or comparable across files. | Choose one exact split and update every file and config. |
| C4 | `myerson` is optional and defaulted to MovieLens only, but the paper promises every attribution family on both backbones and both datasets and the abstract lists Myerson as a benchmark family. | `spec (1).md` §§7.2–7.4; `Implementation_Spec (1).md` §§A.5, B.1a; `Paper_Structure.md` abstract/§7.2 | Missing cells invalidate “full factorial” tables and make comparisons selective. | Make it mandatory for all cells, or remove it from headline claims and label it an exploratory partial experiment. |
| C5 | Main table contents conflict. | `Paper_Structure.md` §7.2 says Table 6 is coverage/ILD and Table 7 is stability/cost; planned-table list says Table 6 is full cutoffs and Table 7 contains coverage/ILD/stability/cost. | Readers and scripts will not know what to reproduce. | Freeze a single table map and update all cross-references. |
| C6 | Number of figures/tables conflicts with the actual plan. | `Paper_Structure.md` opening: 7–9 figures/5–7 tables; planned list: 6 figures/8 tables; `spec (1).md` deliverables: 6–8 figures/4–6 comparison tables; `CoalGameRec_Analysis.md` §9: 6–8/4–6 | Layout and acceptance criteria are unstable. | Count only final main-text items; move detailed tables to supplementary material and use one numbering plan. |
| C7 | `Implementation_Spec` says “see §R,” but no §R exists. | `Implementation_Spec (1).md` opening status | Broken reference undermines reproducibility. | Replace with the actual audit/risk-register path or attach the missing section. |
| C8 | Sensitivity analysis is promised at paper §7.7, but the paper structure has no §7.7. | `Implementation_Spec (1).md` §A.5 | Planned analysis can be omitted accidentally. | Add a section or correct the cross-reference. |
| C9 | Statistics are referred to as `§A.8/§B.2`, but A.8 is a test suite and B.2 is a prediction table, not a statistical protocol. | `Implementation_Spec (1).md` §B.1a | Citation points to the wrong method. | Refer to a dedicated statistical-analysis section and include exact formulas. |
| C10 | Runtime and environment are not consistent. | Python 3.12 and broad package ranges in `Implementation_Spec`; Python 3.10+ and PyTorch 2.x in `spec` | DGL/PyTorch/CUDA compatibility and numerical reproducibility are uncertain. | Pin exact versions, CUDA, driver, OS, and lockfile. |
| C11 | “Same datasets/same metrics” continuity with DyHuCoG is overstated. | `Implementation_Spec` §A.0, `spec` §8, `Paper_Structure` planning notes | Amazon is rebuilt, subsampled, and temporally split rather than using the reported canonical protocol. | Say “same source domains and some metrics,” not same experimental setting. |
| C12 | “Five prior publications” does not match the trajectory table. | `CoalGameRec_Analysis.md` §§0–2 | It is an objective biographical inconsistency. | Count publications and thesis separately; supply complete citations. |
| C13 | Core and adjacent works are both excluded and reviewed. | `spec` §3 and §4.4 versus `Paper_Structure` §5.3 | Corpus counts and taxonomy coverage will be ambiguous. | Define core/adjacent labels and report them separately. |
| C14 | The benchmark is said to ground the “main attribution families,” but it only instantiates interaction players and mainly Shapley variants. | `Paper_Structure` §§1, 4, 7; `Implementation_Spec` §§A.5–A.6 | It cannot empirically ground feature/item/provider/agent or all solution-concept cells. | Limit the claim to the interaction-player/ranking-utility slice. |
| C15 | The analysis says “no new results” while the paper structure says concrete Recall/NDCG results are a headline contribution. | `CoalGameRec_Analysis.md` §§0, 6 versus `Paper_Structure` §§1, 7 | Contribution and article-type framing are inconsistent. | Decide whether the benchmark is a genuine secondary empirical contribution. |
| C16 | File names referenced internally differ from attached names. | References to `Implementation_Spec.md`/`spec.md` versus uploaded `(1)` names | Copying the package can break links and scripts. | Use stable repository names and a version manifest. |
| C17 | Amazon “raw interactions” wording is ambiguous. | `Implementation_Spec` §A.3 | The table says Books 5-core plus metadata, while “raw” could imply all Books reviews. | State exact source URL/file (`Books_5.json.gz`), version, and count. |
| C18 | “Table A/B/C” registered predictions and “Tables 5–7” manuscript tables are not clearly separated. | `Implementation_Spec` §B.1a versus `Paper_Structure` §§7.2 and planned list | Prediction tables may be mistaken for results and numbering may be reused. | Put prediction tables in a registered supplement and assign distinct names. |

---

# 4. Document-specific review

## 4.1 `CoalGameRec_Analysis.md`

### Strengths

- Correctly identifies the natural review/taxonomy framing.
- Recognizes scope creep, thesis-recap risk, self-plagiarism, missing-paper risk, and the need for a transparent protocol.
- Recommends placing the author’s work among external work rather than presenting it as the field.
- Identifies value-function arbitrariness, approximation cost, and reproducibility as real critical themes.

### Weaknesses and corrections

1. **Overconfident novelty language.** “No dedicated survey,” “genuinely unoccupied,” “first,” and “the only hole this title fills” are asserted without a search appendix or corpus. Use: “Our search, conducted on [date] using [databases and queries], identified no review meeting the following prespecified criteria.”
2. **The search is called “grounding searches” but no evidence is attached.** Supply exports, search strings, result counts, deduplication, and a date-stamped search log.
3. **The field is represented as more empty than the candidate literature suggests.** A complete search should address the general Shapley-in-ML surveys, ranking-specific Shapley work such as *ShaRP*, graph explainers, data valuation for recommenders, provider-incentive games, and recent 2025–2026 recommender papers. Do not assume that a paper is outside scope from its title.
4. **“No GPU, no new experiments” is stale.** The updated paper package includes a nontrivial benchmark. This memo must be versioned and rewritten rather than left as the strategic basis for a different paper.
5. **“Uniquely positioned,” “leading voice,” and “natural author” are advocacy statements, not scientific evidence.** They belong in private strategy notes, not the manuscript or cover letter in that form.
6. **The trajectory count is inconsistent.** The table shows a thesis, three prior papers, and four planned methods, while the prose says “5 prior publications.” Correct it and provide references.
7. **The portfolio strategy is too prominent.** “Citation home,” “publish the survey first,” and linking every future method to it can look like self-citation planning. In the manuscript, explain the research gap independently of the authors’ publication sequence.
8. **Planned methods are treated as field categories.** `ActionShap`, `FairShap`, `SignalShap`, and `MHyperShap` are planned methods, not established evidence. Use them only as clearly labelled future directions if they are not publicly available.
9. **The venue advice is not current enough.** It mentions ACM CSUR-style and “proven Discover AI” without a formal venue comparison. The package later fixes Discover AI, but the journal claims need the official source and current date.
10. **The recommended survey scope remains too broad.** Even the “tight scope” contains graph types, all player types, five solution families, five attribution roles, fairness, actionability, data valuation, providers, and agents. Use a core/adjacent structure and make the benchmark cover only one slice.
11. **The suggested RQ2 asks what correlates with gains without specifying a synthesis.** Either perform a structured effect-size/meta-regression analysis with enough comparable studies, or change SRQ2 to a descriptive mapping question.
12. **“What game theory buys” is not operationalized.** Define coding rules for a genuine game-theoretic contribution versus a method that merely uses a Shapley label. The codebook should record whether the method defines a coalition value, uses marginal contributions, has a solution concept, and uses the attribution in an intervention.
13. **The memo underestimates the benchmark burden.** A survey plus four-way/five-way benchmark with an unresolved hypergraph backbone is not “cheap,” “low risk,” or a 3–4 week task unless the benchmark is drastically reduced.
14. **The DyHuCoG audit claim needs a source.** “63 extraction gaps” and “faithful reimplementation impossible” are strong statements. Attach the audit, define “extraction gap,” distinguish incomplete reporting from impossible implementation, and avoid using an unpublished internal audit as an objective field-level fact.
15. **The memo recommends no experiments but the benchmark was later added.** Add a revision history with a decision date, rationale, changed claims, and new risk register.

## 4.2 `Implementation_Spec (1).md`

### Major implementation problems

1. **Player-set scale is not feasible as written.** If every user-item interaction is a player, a global coalition value over hundreds of thousands or millions of interactions cannot be recomputed 50 times per player every 10 batches. The specification must say whether the game is local per user, per minibatch, per receptive field, grouped by item/user, or estimated with a single shared permutation. Give asymptotic and measured cost.
2. **The MC estimator is incomplete.** Sampling arbitrary subsets `S_m` does not automatically produce a Shapley estimator. Specify whether samples come from random permutations, from a size-weighted subset distribution, or from an importance-sampling distribution, and provide the estimator’s weights and variance estimator.
3. **`M≈50 at ~99% accuracy` is unsupported.** No error tolerance, confidence level, game class, player count, variance bound, or convergence plot is supplied. A universal 99% claim is not credible. Replace it with an empirical convergence criterion and an error/confidence interval.
4. **Efficiency is misstated for nonzero baselines.** The identity is
   \[
   \sum_j\phi_j(v)=v(N)-v(\varnothing),
   \]
   not `v(N)` unless `v(∅)=0`. Monte-Carlo estimates, smoothing, clipping, and normalization generally do not satisfy exact efficiency. Report the residual and distinguish exact, corrected, and approximate values.
5. **The empty-coalition convention is not enough.** NDCG, diversity, coverage, and context must all be defined for an empty ranking/graph. A convention of zero can create artificial marginal contributions and should be justified.
6. **Myerson is not defined correctly enough.** A Myerson value is normally constructed from a graph-restricted game, often
   \[
   v^g(S)=\sum_{C\in\mathcal C(g[S])}v(C),
   \]
   followed by the Shapley value of the restricted game. “Shapley restricted to connected coalitions” is ambiguous and can describe a different estimator. Define the communication graph, graph restriction, component value, and whether the game is normalized.
7. **Hypergraph-to-graph projection is unspecified.** A 2-section, incidence graph, line graph, or another projection produces different connected coalitions and different attributions. Treat projection choice as a method parameter and analyze sensitivity, or use a hypergraph-native solution concept.
8. **The Harsanyi interaction notation is incomplete.** The pairwise interaction formula has an ellipsis and no exact coefficient; Harsanyi dividends, Shapley interaction indices, and other interaction measures are not synonyms. Give one precise definition per concept or remove formulas not used.
9. **`Weber/Harsanyi allocation sets` are conflated.** The Weber set and Harsanyi set are distinct sets with different constructions. Cite and explain them accurately; do not present both as one solution concept.
10. **The value-function terms are undefined.** `Diversity`, `Context`, and `sim(u,i)` have no formula, units, range, missing-data rule, or data source. If context is absent from the benchmark, `γ` is meaningless. If item metadata or user demographics are used, specify them for both datasets.
11. **Objective scales are not normalized.** A linear combination of NDCG, diversity, context, and similarity is arbitrary unless ranges and normalization are specified. The values of `α=.60`, `β=.25`, `γ=.15`, and `λ=.20` are not justified or independently selected.
12. **The same metric is used as value and outcome.** If the Shapley game includes NDCG and the final headline is NDCG, the benchmark partially rewards an objective it was explicitly designed to optimize. This is not invalid, but it is not independent validation. Add held-out or alternative outcomes and matched non-game objectives.
13. **Potential test leakage is not controlled.** It is unclear whether coalition values use validation/test labels, whether hyperparameters were inherited from prior test results, whether `sim` uses future interactions, and whether graph construction sees held-out edges. The game and all tuning must use training data or a clearly separated validation set.
14. **Attribution-guided reranking is not an explanation evaluation.** It evaluates whether a score manipulation changes ranking metrics. It does not establish faithfulness, sufficiency, comprehensiveness, stability, or human usefulness.
15. **Training/inference mechanics are underspecified.** “Refresh every `f` batches,” “light temporal smoothing,” “clip extremes,” and “normalize” require exact equations, detach/gradient behavior, cache scope, and hyperparameters. These are large researcher degrees of freedom.
16. **The optional estimator has no reproducible identity.** `shapley-ai` is described as an importance/sampling variant but has no algorithm, citation, proposal distribution, correction factor, or pseudocode. It cannot be a named benchmark family until defined.
17. **Baselines are not matched.** Attention changes model capacity; popularity changes an input prior; uniform removes the module. Report parameter counts, optimizer settings, tuning budget, and identical training schedules. Add a matched additive-similarity baseline.
18. **A standard backbone is not fixed.** “DyHuCoG-style” remains unresolved, and the fallback to an independently documented hypergraph GNN changes the model being evaluated. Choose and pin the backbone before the benchmark, and call the result an independent implementation if it is not DyHuCoG.
19. **“Same as DyHuCoG” is not valid under a rebuilt Amazon dataset.** The raw source, user sample, filtering, split, and graph may differ. Published-baseline numbers cannot be used as a calibration target unless the protocol is identical.
20. **The tests are necessary but insufficient.** Add tests for metric definitions, candidate filtering, no future leakage, temporal ties, negative sampling, hypergraph construction, Myerson component decomposition, MC confidence intervals, reproducibility across process restarts, and data-hash validation. The “dummy player” must be mathematically dummy under the implemented value function, not merely called “noise.”
21. **Determinism is overstated.** Same seed cannot guarantee identical CUDA/GNN results without deterministic kernels and fixed library versions. Use numerical tolerances and document nondeterministic operations.
22. **Environment ranges are too broad.** `torch >=2.0`, `dgl >=2.0`, `numpy >=2.4`, and `scipy >=1.18` do not define a reproducible environment and may have compatibility problems. Use an exact lockfile/container and record GPU driver/CUDA versions.
23. **The runtime table is incomplete.** The “five seeds, both datasets” row has no Amazon estimate. It also does not say whether both backbones and all attribution families are included in each estimate.
24. **No failure policy exists for missing metadata, duplicate Amazon products, repeated timestamps, zero-test users, NaNs, OOM, failed seeds, or convergence failures.** Predeclare the rules.

### Dataset and split audit

1. **Filtering order is ambiguous.** Does 5-core apply to all ratings before converting to positive feedback, or to positive interactions after applying rating ≥4? These produce different users, items, densities, and graph structures.
2. **The two split definitions conflict.** Choose either a temporal 70/10/20 split or last-test/second-last-validation leave-one-out. Define the minimum training history and how users with too few positives are handled.
3. **Filtering on the full time range can leak future information.** If 5-core filtering uses future interactions before a temporal split, it uses future information for eligibility. Either filter from the training period or explicitly justify and label the protocol as transductive preprocessing.
4. **Amazon 5-core is already a special subset.** The source page describes `Books_5.json.gz` as a 5-core subset of 27,164,983 reviews. Sampling 50,000 users and re-filtering is a custom dataset, not the standard Amazon-Book benchmark. Release the exact user sample, final graph, code, and hash.
5. **The choice of 2018 rather than the current 2023 Amazon release is not justified.** The 2018 source itself says it is an older version mainly for reproducing past results. Explain continuity, licensing, and why the older version is preferable.
6. **“Density contrast” is confounded.** MovieLens versus Amazon differs in domain, time period, rating process, popularity distribution, catalogue size, metadata, and preprocessing—not only density. Call BQ4 cross-dataset robustness, not a causal sparse-regime test. A controlled density subsampling or more than two datasets is needed for a stronger claim.
7. **The random Amazon user sample has its own sampling variance.** One 50,000-user sample plus five model seeds does not quantify dataset-sampling sensitivity. Use multiple user-sample seeds or make the exact sample the estimand and limit generalization.
8. **Candidate evaluation is not specified.** State full-catalogue versus sampled negatives, candidate-pool construction, exclusion of seen items, whether held-out positives are excluded from negative sampling, tie handling, and whether candidate pools are fixed across methods.
9. **Leave-one-out with one positive makes Recall@K a hit rate.** If there is one test item per user, report HitRate@K or explicitly explain the Recall convention. If multiple positives are used, define the relevant set and test interval.
10. **NDCG edge cases are not specified.** Define users with no test positives, IDCG when no relevant item exists, binary versus graded relevance, and whether ratings are converted before or after splitting.
11. **ILD is not reproducible.** Define `sim(i,j)`, its range, feature source, treatment of missing metadata, and whether the same item features are used for all methods. Do not use learned method-dependent embeddings for the similarity unless that is the estimand.
12. **Coverage denominator is ambiguous.** Use the eligible item catalogue after seen-item filtering or the full item universe, and state which one.
13. **Temporal ties are not fully reproducible.** “Row order” changes if files are re-parsed. Preserve the original line number or use a stable secondary key and release the mapping.
14. **Negative sampling is underspecified.** Define popularity distribution, number of negatives, hard-negative refresh, validation/test exclusion, and random streams.
15. **MovieLens and Amazon feature availability differs.** If `Context` or ILD uses genres, categories, text, or demographics, the cross-dataset comparison is not feature-matched. Report separate feature pipelines or use a common interaction-only similarity.

## 4.3 `Paper_Structure.md`

1. **The abstract is over the word limit and contains future results.** This alone would fail a stated submission criterion.
2. **The manuscript is described as both a survey with no new results and a paper whose “headline” is a concrete benchmark.** Decide how much of the article is review versus original study.
3. **“First systematic survey” and “no synthesis exists” need a completed search, not a blueprint.** The abstract must use a defensible qualified claim until then.
4. **The benchmark’s positive ordering is hard-coded into the Results plan.** Replace “highlight that `shapley-mc` ranks 1st” with a neutral analysis plan.
5. **The five axes are not guaranteed to be exhaustive, mutually exclusive, or independently codable.** A taxonomy needs definitions, allowed values, multi-label rules, missingness, and agreement evaluation.
6. **The taxonomy contains overlapping player labels.** `items`, `nodes`, `data tuples/items`, `interactions`, and `edges/hyperedges` can describe the same entity at different abstraction levels. `features`, `contexts`, and `signal sources` can also overlap.
7. **The benchmark is not aligned to the taxonomy’s breadth.** It covers interactions and a ranking utility, not features, users, contexts, providers, agents, data valuation, Banzhaf, Harsanyi, core, or nucleolus.
8. **The “worked DyHuCoG example” risks making the taxonomy self-serving.** Include multiple external worked examples first and explicitly mark the authors’ method as one coded case.
9. **Section order is inconsistent with `spec (1).md`.** One file places critical analysis before the benchmark and another places it after; choose one order and ensure the RQ mapping is unchanged.
10. **The Results table map is contradictory.** Reconcile §7.2 with the planned figures/tables list.
11. **The proposed length and figure/table counts conflict across files.** Freeze one main-text/supplement plan.
12. **The benchmark is called “small” but the planned factorial design is not small.** Six families × two backbones × two datasets × three cutoffs × two primary metrics × five seeds, plus ablations and statistical tests, is a substantial empirical study.
13. **The “research agenda” is too explicitly mapped to the authors’ planned papers.** This can look like self-promotion and makes the survey’s agenda appear predetermined. Discuss external evidence and state author connections neutrally in a disclosure or final paragraph.
14. **“Actionability” is listed as an evaluation dimension without an operational study.** Either include an actionability proxy/user study or say the survey records that actionability is generally unmeasured.
15. **Fairness/exposure is promised but not benchmarked.** Coverage, ILD, and popularity shift are not sufficient fairness metrics. Include group/provider exposure metrics or remove fairness as an empirical claim.
16. **“State-of-the-art” and “opaque” are broad unsupported claims.** Cite representative recommender and explainability literature and soften absolute language.
17. **The notation list is not sufficient.** It omits or overloads `K`, `k`, `M`, `m`, `f`, `g`, `R_u`, `I`, `U`, `rel`, `C`, `v_pref`, the candidate set, baseline value, and the distinction between model function and refresh frequency.
18. **The manuscript plan has no explicit limitations section.** A systematic review with a custom benchmark needs limitations covering search bias, publication bias, preprints, heterogeneous tasks, custom Amazon preprocessing, no human evaluation, and author conflict.
19. **No meta-analysis/synthesis method is stated.** Comparison tables alone cannot answer which design choices correlate with gains. Add a formal extraction/synthesis plan or narrow SRQ2.
20. **No audit trail is promised for claims not reported in papers.** Code “unknown” rather than infer missing details; distinguish “not reported” from “not used.”

## 4.4 `spec (1).md`

1. **The venue profile is partly accurate but partly stale/unsupported.** The current official page supports open access, Scopus/DOAJ/Ei indexing, CiteScore 2024=6.0, the Review type, and the APC. It currently reports a 16-day median first decision, not ≈23 days. It does not establish the Q1 label.
2. **The “fixed” venue language is too absolute.** Article-type and topical-collection suitability should be confirmed at submission; a hybrid review/benchmark may need editorial guidance.
3. **The inclusion rule is too broad and uses “graph-like interaction structure” without an operational definition.** Define graph role, model input, and recommendation task.
4. **The search string is not reproducible across databases.** Field restrictions, syntax, truncation, query limits, language, date searched, deduplication, and citation chasing are absent. It also lacks several likely terms such as ranking/top-N, attribution, credit assignment, data valuation, edge explanation, network, and graph convolution variants.
5. **Applying the explainability constraint only at screening risks inconsistent selection.** Define it in the query or provide a screening codebook with examples and adjudication.
6. **“Peer-reviewed or stable arXiv” combines different evidence grades.** Label preprints separately and never treat them as peer-reviewed evidence.
7. **Risk-of-bias is optional despite being needed for SRQ4 and critical claims.** Make it required and use a more detailed ML/recommender rubric.
8. **The taxonomy is prescribed before the literature is coded.** It may be useful as a deductive starting framework, but allow new categories and report how the final axes were revised.
9. **The notation says “reuse thesis language.”** The notation may be reused conceptually, but equations and prose need new presentation, citations, and overlap review. Avoid importing thesis definitions that assume a different game.
10. **The Shapley MC formula does not specify the sampling law.** Correct the estimator and add uncertainty/convergence reporting.
11. **The Myerson description is ambiguous/incomplete.** Correct the graph-restricted game formula and distinguish Myerson from other graph-aware indices.
12. **The Harsanyi interaction expression is incomplete and potentially wrong.** Supply a full definition and citation or remove it.
13. **The benchmark protocol contradicts the implementation file.** See C3 above.
14. **The statistical plan is inherited from a thesis without demonstrating suitability for this design.** A prior `stats.py` file is not a statistical justification.
15. **The “primary” hypergraph backbone is unresolved.** No benchmark should be preregistered around a model that the authors say cannot be faithfully reproduced.
16. **The minimum acceptance criterion “≥4 attribution families” conflicts with the stronger headline requirement of every family across every cell.** Make the minimum and the actual paper claim consistent.
17. **The “definition of done” makes the survey depend on the benchmark and future paper sequence.** A null benchmark should not invalidate a rigorous survey; separate survey go/no-go from benchmark go/no-go.
18. **No decision rule exists for changing the scope after the search.** Predefine how unexpected solution concepts or larger adjacent literatures are handled.

---

# 5. Survey methodology audit and required protocol

## 5.1 Search strategy

The current Boolean query is a reasonable starting point but not a complete protocol. It needs:

- database-specific syntax and field tags;
- synonyms and spelling variants: `cooperative game`, `coalition`, `marginal contribution`, `credit assignment`, `data valuation`, `influence`, `SHAP`, `Shapley`, `graph explainer`, `edge attribution`, `ranking explanation`, `top-N`, `retrieval`, `hypergraph neural network`, `heterogeneous graph`, `knowledge graph`, `provider`, `creator`, `exposure`;
- explicit title/abstract/keyword fields;
- language and date filters;
- publication status rules;
- exact date and time zone of each search;
- database export format and deduplication key;
- backward and forward citation chasing from seed papers;
- author/contact method for missing information;
- a final search update immediately before submission.

Do not claim that a single search string is “reproducible” when the databases tokenize and interpret it differently.

## 5.2 Screening

Use at least two independent screeners for a meaningful sample. Record:

- number of records from each source;
- duplicates removed automatically and manually;
- title/abstract exclusions;
- full-text exclusions and one primary reason per paper;
- included core papers and included adjacent papers;
- disagreements, adjudicator, and Cohen’s kappa or percentage agreement.

If only the three authors screen, assign independent roles and keep the screening decisions blind to author identity where practical. The author’s own papers should not be automatically included without applying the same criteria.

## 5.3 Extraction/codebook

The proposed fields are useful but incomplete. Add:

- explanation target and granularity;
- game type: local/frozen-model, training-data, provider-incentive, coalition recommendation, or other;
- coalition semantics and intervention;
- baseline value `v(∅)`;
- whether the value function uses labels, predictions, utility, regret, exposure, or model loss;
- whether the model is retrained for coalitions;
- whether the method is post-hoc or changes training;
- sign and interpretation of negative attribution;
- normalization and baseline/missing-feature policy;
- exact solution concept and estimator;
- estimator uncertainty/convergence;
- explanation evaluation metrics and ground truth;
- recommendation evaluation protocol, candidate pool, split, and leakage controls;
- number of seeds and statistical unit;
- hyperparameter search/tuning budget;
- code/data availability and version;
- reported limitations and author conflicts.

Allowed values must be defined. Use `NR` (not reported), `NA` (not applicable), and `unclear` separately. Do not infer a value function from a paper’s metric table.

## 5.4 Quality/risk-of-bias rubric

A binary “statistics/code/held-out” flag is too coarse. Score or report domains such as:

- task and player-set clarity;
- coalition/value-function clarity;
- model/split/candidate-pool transparency;
- no leakage or proper held-out evaluation;
- baseline adequacy and tuning fairness;
- random seeds and uncertainty;
- explanation-faithfulness evaluation;
- robustness/stability;
- code/data availability;
- completeness of negative results and limitations.

Do not convert the score into an unvalidated total quality number unless you justify the weighting. Use the assessment to qualify synthesis claims, not simply rank papers.

## 5.5 Synthesis

SRQ2 currently asks what correlates with reported gains. Three defensible choices are:

1. **Descriptive synthesis:** report distributions and patterns without causal or correlational claims.
2. **Structured vote/count synthesis:** record direction of results but clearly note heterogeneity and publication bias.
3. **Quantitative synthesis:** extract comparable effect sizes and sample/experiment details, predefine a meta-regression model, handle dependence among multiple metrics, and report uncertainty.

A table of reported gains cannot support “what correlates with gains.” If the quantitative option is not feasible, rewrite SRQ2.

## 5.6 Novelty claim

Replace “the first” with a qualified claim until the search is complete. The positioning table must include, at minimum, the general Shapley/XAI surveys, general Shapley-in-ML surveys, GNN-explanation surveys, recommender explainability reviews, ranking-specific Shapley work, game-theoretic recommender mechanisms, data valuation/pruning for recommender systems, and any direct graph/hypergraph recommender attribution papers.

The review should also explain what it adds beyond simply narrowing a general survey: a reproducible corpus, a coding manual, an operational distinction between game semantics and allocation rules, a taxonomy validated by multiple independent coders, and a critical analysis of evaluation validity.

---

# 6. Taxonomy and theory audit

## 6.1 Taxonomy design

The current five axes are promising but require redesign:

- **Player set** should be separated from graph role. “Edge,” “interaction,” and “user-item tuple” may be the same player. “Node” can mean user, item, or context node. “Feature,” “context,” and “signal source” may be nested.
- **Value function** should distinguish the scientific quantity of interest from a final evaluation metric. A paper may use NDCG for evaluation but define a model-output value for the game.
- **Solution concept/aggregation rule** should distinguish Shapley, Banzhaf, Myerson, interaction indices, Weber set, Harsanyi set, core, and nucleolus; some are point allocations, some are sets, some are stability concepts.
- **Role of attribution** should distinguish explanation, training control, data valuation, provider allocation, and intervention. These are not merely different labels for the same output.
- **Graph structure** should record whether the graph is data, model architecture, communication constraint, or explanation object.

Consider a taxonomy tuple such as:

\[
(\text{task},\ \text{explanation/credit target},\ \text{player granularity},\ \text{coalition semantics},\ \text{value/QoI},\ \text{allocation rule},\ \text{graph role},\ \text{intervention},\ \text{evaluation}).
\]

This is longer but prevents unrelated methods from landing in the same cell. Use multi-label values when a paper has multiple games, and record a primary/secondary game rather than forcing one label.

## 6.2 Game-theoretic claims need qualification

The manuscript should not equate:

- Shapley axioms with moral or causal fairness;
- a model-based attribution with a faithful human explanation;
- efficiency with correctness of the value function;
- a positive attribution with an actionable intervention;
- attention weights with explanation;
- use of a coalition formula with a substantive game-theoretic contribution.

A sound critical section should state that axioms govern allocation **conditional on a chosen game**. They do not select the player representation, missingness/baseline distribution, coalition semantics, or quantity of interest. This is especially important for correlated graph interactions and redundant edges.

## 6.3 Formula corrections

At minimum, the final paper should define:

- normalized TU game and whether `v(∅)=0`;
- Shapley value with consistent player notation;
- exact versus permutation Monte-Carlo estimator;
- baseline and missing-edge semantics;
- Myerson graph-restricted game and communication graph;
- Harsanyi dividends versus pairwise interaction index;
- normalized versus unnormalized Banzhaf;
- Weber set and Harsanyi set as different allocation sets;
- core and nucleolus only if they are relevant to the reviewed corpus.

Use `p` for a generic player and reserve `i` for item, `u` for user, `m` for MC samples, `K` for cutoff, and `g` for a communication graph. Do not use `M` both as player count and sample count or `f` both as model function and refresh frequency.

## 6.4 Scope of solution concepts

The outline promises a full treatment of core/nucleolus, Weber/Harsanyi sets, Banzhaf, Myerson, interactions, and Shapley, but the benchmark only implements one Shapley estimator, one estimator variant, and optional Myerson. If several concepts are theoretical/rare in the corpus, label them as “candidate or adjacent concepts” rather than implying that all are established in graph recommendation.

---

# 7. Benchmark validity audit

## 7.1 Separate three studies that are currently mixed

The current benchmark combines:

1. **Explanation:** attribution for a frozen model’s recommendation.
2. **Intervention:** use attribution to change propagation/ranking and measure accuracy/diversity.
3. **Training control:** refresh attribution during learning.

Each needs a different experiment. A clean design would report:

- **Study A — frozen-model explanation:** compare attribution methods on fidelity, deletion/insertion, sufficiency/comprehensiveness, stability, sparsity, and computational cost.
- **Study B — intervention:** apply a pre-specified reweight/rerank rule without using test labels; measure held-out ranking, coverage, diversity, and fairness.
- **Study C — optional training ablation:** compare fixed attribution versus refreshed attribution, with exact computational accounting.

If only Study B is retained, stop calling it an empirical validation of explainability. It is an attribution-guided recommender intervention study.

## 7.2 Required explanation metrics

For a paper whose title and SRQs emphasize explainability, add or explicitly exclude:

- ranking fidelity to the explained model;
- deletion/insertion or sufficiency/comprehensiveness;
- stability under seed, small graph perturbation, and equivalent representations;
- sparsity/complexity of the explanation;
- sign consistency and negative-attribution analysis;
- sanity checks against model/randomization changes;
- actionability only if a meaningful intervention or user study is supplied.

Recommendation-quality improvement alone can occur while an attribution is a poor explanation.

## 7.3 Matched baselines and ablations

At minimum:

- plain LightGCN;
- plain documented hypergraph backbone;
- uniform/no attribution;
- random weights with matched distribution;
- degree/popularity;
- learned attention with parameter count and regularization reported;
- additive preference-similarity heuristic;
- Shapley without preference term;
- Shapley with ranking-only value;
- Shapley with each additional objective;
- Myerson under a precisely defined graph;
- estimator/refresh-frequency ablations;
- at least one established post-hoc graph explainer if explanation quality is claimed.

Do not call `uniform` an attribution method; call it the no-attribution control.

## 7.4 Hyperparameter and analysis lock

Predefine:

- `α, β, γ, λ_pref` and their selection source;
- objective normalization;
- number of MC samples and stopping/convergence criterion;
- refresh period, smoothing, clipping, and normalization;
- embedding/model dimensions;
- optimizer/learning rate/regularization/epochs;
- negative-sampling parameters;
- candidate pool;
- early stopping and model-selection rule;
- number of seeds and data-sample seeds;
- all primary contrasts and correction family.

Use validation data only for tuning. A parameter borrowed from a previous paper can still be a form of test leakage if it was selected on the same evaluation data; disclose that risk.

## 7.5 Dataset reporting

For each final dataset, report:

- source URL and citation;
- raw file name/version/date/checksum;
- whether ratings or reviews are used;
- threshold and filtering order;
- final users/items/interactions;
- positive-rate and rating distribution;
- temporal range and timestamp resolution;
- per-user/per-item statistics;
- density and long-tail concentration;
- train/validation/test counts;
- candidate-pool size;
- exact sample seed and final user list/hash;
- metadata fields used for similarity/context/ILD;
- license and redistribution constraints.

The official [GroupLens MovieLens 1M page](https://grouplens.org/datasets/movielens/1m/) and [UCSD Amazon 2018 page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) should be cited. The UCSD page reports that Books 5-core contains 27,164,983 reviews and that the source is an older 2018 release mainly retained for reproducing past results; the paper must cite the exact custom preprocessing instead of calling it the canonical Amazon-Book dataset.

---

# 8. Statistical audit and corrected analysis plan

## 8.1 Primary estimand and contrasts

Choose primary outcomes before running:

- NDCG@20 and Recall/HitRate@20;
- one or two prespecified contrasts, such as Shapley-MC versus no-attribution control;
- each dataset/backbone treated as a separate experimental condition;
- all other cutoffs, metrics, families, and ablations secondary/exploratory.

If the scientific question is whether the method is robust across density, predefine the method-by-regime interaction. With only two datasets, call it a comparison, not a general causal regime effect.

## 8.2 Replication structure

Use separate random streams for:

- dataset sampling;
- data split/tie breaking;
- model initialization;
- negative sampling;
- coalition/permutation sampling.

Five model seeds may be adequate for a small exploratory benchmark, but then conclusions should be descriptive and intervals labelled as seed variability. More seeds or a precision analysis is preferable. Multiple Amazon user samples are needed if population-level claims are made.

## 8.3 Inference

A defensible approach is:

1. compute per-user paired metric differences for each seed;
2. report per-seed mean, standard deviation, and distribution of user-level differences;
3. bootstrap users within each seed and aggregate seed estimates, or fit a hierarchical model;
4. report effect size and 95% interval;
5. use paired permutation or Wilcoxon/t-test only as sensitivity analyses;
6. correct the prespecified family of primary comparisons with Holm;
7. label all remaining tests exploratory.

Do not apply a correction without defining the family. Do not report asterisks without the exact unit, test, alternative, correction, and number of comparisons in the caption.

## 8.4 Stability and cost

For BQ3, define:

- convergence failure;
- NaN/OOM rate;
- epochs to early stopping;
- wall-clock measurement boundaries;
- warm-up and data-loading treatment;
- inference latency per user/batch;
- peak CPU/GPU memory measurement method;
- attribution-refresh cost separately from backbone training;
- seed-to-seed metric variance;
- estimator variance and efficiency residual.

A statement such as “variance should be comparable” is a hypothesis, not a result. Use a variance ratio or equivalence margin if making a stability claim.

## 8.5 Numeric prediction inconsistencies

The registered prediction tables are internally inconsistent:

- MovieLens hypergraph `uniform` NDCG@20 is 0.210, below the pipeline-level hypergraph range 0.22–0.28.
- The predicted MovieLens Shapley-MC uplift over uniform is approximately 9.5%, outside the stated +3% to +8% range.
- The predicted Amazon-Book uplift is approximately 18.5%, outside the stated +5% to +12% range.
- The table-level Shapley-AI versus Shapley-MC differences are approximately 3.1% for MovieLens and 6.7% for Amazon, not “within ±1%” if that means relative difference.
- No standard deviations are given despite repeated instructions to report mean ± standard deviation.
- LightGCN Table C has no predicted cells, so the registered prediction is incomplete.

Correct these before registration, or better, do not register exact point predictions unless there is a principled prior model. Register directional hypotheses and effect-size thresholds instead.

---

# 9. Citation and support audit

The four files contain no bibliography and almost no in-text citations. As planning notes this is understandable; as a manuscript basis it is not submission-ready. Every externally verifiable claim below needs a primary or authoritative citation.

| Claim or named item | Problem | What to add |
|---|---|---|
| Shapley 1953, Myerson 1977, Harsanyi 1963, Banzhaf 1965 | Foundational works are named but not bibliographically specified. | Full citations and DOI/URL where available. |
| SHAP and feature attribution | SHAP is not identical to every original TU-game application; baseline/conditional/interventional semantics matter. | Lundberg & Lee and relevant SHAP/feature-dependence literature. |
| General Shapley/XAI surveys | Authors, year, title, volume, and status are inconsistent: Zhao et al. are online/published in a 2026 volume although the outline calls the work 2025; Li et al. need an exact DOI and publication year convention. | Use the final publisher records and DOI links. |
| “No dedicated survey exists” | Novelty claim unsupported. | Search protocol, final date, positioning table, and all near-neighbor reviews. |
| EdgeSHAPer, GraphSVX, GraphGI, GStarX, GISExplainer, GraphEXT | Names are listed without exact references, tasks, publication status, or relevance to recommendation. | Cite the original papers, distinguish graph classification from recommendation, and verify names/spelling. |
| “Myerson/communication-graph Shapley surveys such as Hu, Shan & Li” | Incomplete citation and unclear title/status. | Full reference and explanation of whether it is a survey or method. |
| Shapley community collaborative filtering | Vague label. | Exact paper, venue, year, task, and game definition. |
| Shapley data valuation/pruning for recommenders | Directly relevant recent work is not specified. | Include exact paper and classify it as training-data valuation rather than post-hoc explanation. |
| TU-bandit creator-incentive games | The relevant 2026 work is an AISTATS 2026 oral/arXiv preprint, not simply an unnamed “TU-bandit” paper. | Cite the exact title, authors, arXiv/venue status, and separate peer-reviewed from preprint evidence. |
| DyHuCoG | No full publication citation is supplied. | DOI, venue, version, code/repository, and exact distinction between published results and the new benchmark. |
| “63 extraction gaps” and “faithful reimplementation impossible” | Internal audit claim is not accessible to readers. | Archive the audit or describe it as an author-side implementation risk, not as a published fact. |
| Graph recommenders are “state of the art” and “opaque” | Broad claims. | Representative recommender, graph-recommender, and explainable-recommender surveys/papers. |
| Axiomatic fairness/principled credit/actionability | These are not automatic consequences of Shapley values. | Cite theory and XAI critiques; qualify claims. |
| “Beyond Shapley Values” | The title is mentioned without an exact reference; it is a recent workshop/preprint-style work, not necessarily a journal consensus. | Cite the exact work and describe its publication status. |
| Value-function arbitrariness | Important claim but uncited. | Cite work discussing value-function choice and limitations of Shapley explanations, including relevant XAI critiques. |
| Attention as a non-game baseline/explanation | Attention weights are not automatically faithful explanations. | Cite attention-versus-explanation literature and define what is evaluated. |
| BPR, LightGCN, hypergraph GNN/HCCF | Backbone and loss are named without primary citations. | Cite BPR, LightGCN, the chosen hypergraph model, and the actual code version. |
| NDCG, Recall, coverage, ILD | Definitions are supplied but no metric references or edge-case conventions. | Cite ranking-evaluation sources and define the exact implementation. |
| MovieLens-1M | Source and usage license are not cited. | GroupLens dataset page and recommended dataset citation. |
| Amazon Reviews 2018/Books 5-core | Raw source, version, 5-core count, and required citation are omitted. | UCSD dataset page and Ni, Li & McAuley (2019), plus custom sampling citation/code. |
| “Canonical Amazon-Book split” | Strong assertion about a community split without a primary source. | Cite the exact benchmark/repository and verify the claim before retaining it. |
| Journal Q1, Q2, CiteScore, median decision, APC | Some are current and some are time-sensitive. | Use official pages for journal facts; use a dated external ranking source for quartile claims. |
| “5 prior publications,” “author uniquely placed,” “high citation potential,” “leading voice” | Biographical/marketing claims, not evidence. | Correct counts and remove promotional claims from the manuscript. |
| “M≈50 gives 99% accuracy,” 1.78× runtime, 20–40% memory, coverage gains, NDCG levels | Predictions have no derivation and some conflict with other predictions. | Replace with testable hypotheses and report measured values with uncertainty. |

### Candidate recent literature that the final search should explicitly check

This is not a substitute for the systematic search, but the current plan should not omit obvious near-neighbors such as:

- *Shapley Value: From Cooperative Game to Explainable Artificial Intelligence*;
- *The Shapley Value Contribution to Explainable Artificial Intelligence: A Comprehensive Survey*;
- *The Shapley Value in Machine Learning*;
- *ShaRP: Explaining Rankings and Preferences with Shapley Values*;
- *Shapley Value-driven Data Pruning for Recommender Systems*;
- *A Game-Theoretic Approach to Recommendation Systems with Strategic Content Providers*;
- *Creator Incentives in Recommender Systems: A Cooperative Game-Theoretic Approach for Stable and Fair Collaboration in Multi-Agent Bandits*;
- *Beyond Shapley Values: Cooperative Games for the Interpretation of Machine Learning Models*;
- current graph-Shapley explainers and any direct graph/hypergraph recommendation papers published through the final search date.

Each must be screened against the prespecified scope rather than automatically included.

---

# 10. Formatting, tables, figures, and language audit

## 10.1 Journal formatting

The final manuscript should be checked against the current Discover/Springer instructions:

- minimum consistent font size 12 pt;
- no more than three levels of headings;
- abbreviations defined at first mention and used consistently;
- square-bracket numeric citations;
- references numbered consecutively and limited to cited published/accepted works;
- tables numbered with Arabic numerals and cited in order;
- table captions explain the table and identify reused material;
- figures in the body, sequentially numbered, with captions in the manuscript;
- descriptive accessible captions and colour/pattern accessibility;
- no figure titles embedded in artwork;
- correct resolution and embedded fonts for final artwork.

The current Markdown is a planning document, so these are not failures of the files themselves, but the plan does not yet contain an implementation checklist for them.

## 10.2 Specific editorial issues

- Use one term consistently in the title and body: “cooperative game theory” and “coalitional game theory” can be introduced as synonyms, but do not alternate without reason.
- Use “top-N” consistently; the current files mix top-N, ranking, recommender, and recommendation systems.
- Avoid “state-of-the-art” unless supported by a defined comparison and current citation set.
- Avoid “first,” “only,” “genuinely,” “uniquely,” and “unoccupied” unless the evidence and scope make the claim defensible.
- Replace “game theory buys” in formal prose with “the empirical/theoretical advantages attributed to the game-theoretic formulation.” The phrase is useful as a section title but should not prejudge the result.
- Define `Recall@K` versus `HitRate@K` under leave-one-out evaluation.
- Define whether “NDCG” is per-user averaged first and then averaged across seeds, or computed after pooling recommendations.
- Use a single capitalization and delimiter convention for `Recall@5`, `NDCG@20`, `MovieLens-1M`, `Amazon-Book`, and method labels.
- Avoid using “AI” in `shapley-ai` unless the estimator has a precise published name; it is otherwise misleading.
- Move “Why this is likely to be accepted,” “citation home,” “house style,” “cheap to produce,” and portfolio sequencing out of the manuscript package.
- Add a final limitations section and a data/code artifact statement that contains actual identifiers, not placeholders.

## 10.3 Figure/table numbering correction

A clean map would be:

- **Fig. 1:** PRISMA 2020 flow;
- **Fig. 2:** taxonomy and coding framework;
- **Fig. 3:** primary NDCG@K results;
- **Fig. 4:** primary Recall/HitRate@K results;
- **Fig. 5:** explanation fidelity/stability or cost/diversity, but not an overloaded four-variable plot;
- **Fig. 6:** research agenda.

Main-text tables should be sequential Arabic numbers. Put the complete factorial result tables and prediction register in supplementary material, with distinct labels such as “Online Resource 1” rather than `Table A/B/C` if they could be confused with manuscript results.

---

# 11. Recommended corrected study design

## 11.1 Recommended title and framing

A safer title after the protocol is complete would be:

> **Cooperative-Game Attribution for Explainable Graph-Based Recommendation: A Systematic Review, Taxonomy, and Empirical Case Study**

The word “case study” prevents the benchmark from being mistaken for a comprehensive empirical validation of the field. If the benchmark is removed, use “A Systematic Review and Taxonomy.”

## 11.2 Recommended research questions

- **RQ1:** What game target, player granularity, coalition semantics, value/QoI, allocation rule, graph role, and attribution role occur in the core literature?
- **RQ2:** How transparent and comparable are the reported game definitions, approximation procedures, and recommendation protocols?
- **RQ3:** What evidence exists that the game formulation adds value beyond matched heuristics, attention, or reweighting?
- **RQ4:** Which explanation-validity, stability, fairness, and reproducibility criteria are measured, and which are missing?
- **RQ5:** Which gaps are supported by the evidence, and which are merely plausible future directions?
- **BQ1:** Under one preregistered interaction-player case study, does the specified attribution intervention change held-out ranking outcomes relative to matched controls?
- **BQ2:** What is the effect on coverage/diversity/fairness under the same protocol?
- **BQ3:** What are estimator error, runtime, memory, and seed/data-sample variability?

Do not make the benchmark answer the field-wide RQ3 by itself.

## 11.3 Recommended order of work

1. Freeze core/adjacent scope and RQs.
2. Build and register the systematic-review protocol.
3. Run the search, screen, extract, and quality-audit the corpus.
4. Validate and revise the taxonomy using independent coding.
5. Run an Amazon feasibility spike and a small synthetic Shapley test.
6. Freeze the benchmark estimand, splits, baselines, and statistical plan.
7. Register predictions and analysis script externally.
8. Implement the benchmark with a documented backbone and complete tests.
9. Run the primary analysis without changing the model based on test results.
10. Write the manuscript using realized results only; put deviations and predictions in supplementary material.
11. Complete overlap, ethics, funding, code, data, AI, and formatting checks.

## 11.4 Minimum benchmark specification before coding

The specification should contain, in executable terms:

- exact dataset files and checksums;
- filtering order and train-only/full-data policy;
- exact split function and tie-breaking;
- candidate set and negative sampling;
- graph/hypergraph construction;
- model architecture and parameter count;
- game player set and local/global scope;
- coalition masking/retraining semantics;
- exact `v(S)` and all normalization;
- exact Shapley/Myerson estimator and random sampling law;
- refresh/smoothing/clipping equations;
- all baseline algorithms;
- tuning budget and validation rule;
- primary metrics and edge-case definitions;
- primary contrasts and correction family;
- failure/exclusion rules;
- code/environment/data release plan.

## 11.5 Go/no-go gates

- **Gate 1:** the Amazon sample produces a viable split with released final counts and no hidden test dependence.
- **Gate 2:** the backbone passes a sanity check under the exact temporal/candidate protocol; do not compare to published numbers from a different split as if they were equivalent.
- **Gate 3:** exact/synthetic games pass analytic Shapley tests and MC convergence is measured.
- **Gate 4:** the benchmark has a matched non-game heuristic and a frozen-model explanation evaluation, or the claims are narrowed to intervention.
- **Gate 5:** the full factorial cells promised in the manuscript exist. If Myerson is partial, remove it from headline claims.
- **Gate 6:** the survey corpus and PRISMA audit trail are complete independently of whether Shapley wins the benchmark.

---

# 12. Submission-readiness checklist

| Item | Current status |
|---|---:|
| Final core/adjacent scope | **No** |
| Registered, database-specific review protocol | **No** |
| PRISMA 2020 checklist and flow | **No** |
| Search exports and full-text exclusion log | **No** |
| Independent screening/agreement record | **No** |
| Completed extraction codebook | **No** |
| Mandatory risk-of-bias assessment | **No** |
| Final taxonomy validated against corpus | **No** |
| Complete DOI-checked bibliography | **No** |
| Abstract under 250 words | **No** |
| Neutral abstract with no predicted results | **No** |
| Exact game/coalition estimand | **No** |
| Correct MC/Myerson formulas | **No** |
| Fixed dataset/split/candidate protocol | **No** |
| Resolved DyHuCoG/fallback backbone | **No** |
| Matched baselines and ablations | **No** |
| Explanation-quality evaluation or narrowed claims | **No** |
| Statistical unit and correction family | **No** |
| External prediction registration | **No** |
| Actual benchmark results | **No** |
| Ethics/institutional determination | **No** |
| Accurate funding/COI/AI/overlap statements | **No** |
| Public code, configs, hashes, and archive | **No** |
| Final Springer formatting/accessibility check | **No** |
| Cover letter | **No** |

---

# Final recommendation

**Do not abandon the topic, but do not proceed with the current “Shapley will rank first” paper narrative.** The publishable contribution is a rigorous, bounded systematic review. The benchmark can strengthen it only if it is treated as a separately specified empirical case study with neutral hypotheses, matched controls, a precise estimand, valid uncertainty analysis, and explanation-specific evaluation.

The first concrete deliverable should be a frozen review protocol and a one-page benchmark estimand—not a manuscript abstract or predicted results table. Once those are corrected, the project has a plausible path to a credible *Discover Artificial Intelligence* Review submission. In its current form, it is **not submission-ready and would be vulnerable to desk-return or major-revision criticism for unsupported novelty, pre-stated results, scope conflation, protocol non-reproducibility, statistical overreach, ethics misclassification, and internal inconsistency**.

---

## Sources checked for journal/data/reporting requirements

- [Discover Artificial Intelligence — Submission guidelines](https://link.springer.com/journal/44163/submission-guidelines)
- [Discover Artificial Intelligence — Journal home](https://link.springer.com/journal/44163)
- [Discover Artificial Intelligence — APC and publishing information](https://link.springer.com/journal/44163/how-to-publish-with-us)
- [Discover journals — Editorial policies, including AI, ethics, data, authorship, and reporting](https://link.springer.com/brands/discover/policies)
- [PRISMA 2020 reporting resources](https://www.prisma-statement.org/prisma-2020)
- [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- [UCSD Amazon Review Data 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
