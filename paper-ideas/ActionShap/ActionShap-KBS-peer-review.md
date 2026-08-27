# Peer Review of “ActionShap: Evaluating Recommendation Explanations Beyond Deletion with Bounded Interventions”

**Journal assessed:** *Knowledge-Based Systems* (KBS)  
**Manuscript assessed:** 29-page ACM-formatted PDF dated August 2026  
**Review scope:** Main 29-page PDF plus the separately supplied 30-page “ActionShap: Supplementary Material, Sections S1–S10.” The supplement contains Tables S1–S30, expanded pairwise tests, replications, ablations, convergence diagnostics, and complexity/runtime results. The executable artifact, source revision, raw CSV/JSON matrices, and permanent/reviewer URL remain unavailable.

## PHASE 1 — OVERALL ASSESSMENT

- [ ] Accept
- [ ] Minor Revision
- [x] Major Revision
- [ ] Reject

The paper addresses a real evaluation problem: deletion-based faithfulness and bounded profile modification are different estimands. The manuscript is unusually careful about temporal splitting, fixed candidate sets, tie handling, distinct-user inference, abstention, exact budget-two oracles, and the distinction between statistical difference and practical equivalence. The printed numerical statements are internally consistent within rounding.

Nevertheless, the current manuscript is not ready for KBS. The supplement resolves several documentation gaps: it supplies Holm-adjusted comparisons, effect sizes, success/abstention intervals, complexity and runtime budgets, exact-Shapley subsets, LIME mask ablations, prospective audits, path-matched baselines, and exhaustive `B=3` results. It also strengthens the principal negative conclusion: bounded/path-matched LIME, finite differences, and integrated gradients generally outperform MC Shapley under matched intervention semantics. However, the central empirical evidence is still effectively restricted to ItemKNN because the SASRec-style and LightGCN models remain below popularity. More fundamentally, Eqs. (2)–(3) normalize by total retained weight: downweighting one interaction raises every other interaction’s normalized share. The operation is **relative profile-mass reallocation**, not isolated suppression, and uniformly scaling all positive weights leaves scores unchanged. Table S19 reinforces this concern: Amazon ItemKNN Shapley AIA is exactly `0.708` at every `ρ∈{0,.1,.25,.5,.75,.9}`. The audit remains primarily retrospective and target-conditioned, while the prospective supplement uses only 126–227 selected users per run whose generated top-1 lies in the candidate set. H2 remains confounded because target-margin attributions are judged by NDCG and the manuscript correlates the gap, rather than absolute bounded AIA, with decisions. Amazon full-catalogue evaluation still reverses the sampled-candidate headline. The permanent artifact URL is still a placeholder, and main/supplement version drift introduces new inconsistencies in MDE values, cohort denominators, full-catalogue wording, and the status of exhaustive `B=3`.

| Criterion | Score (/10) |
| --- | ---: |
| Novelty | 7 |
| Technical correctness | 6 |
| Experimental quality | 6 |
| Scientific rigor | 7 |
| Writing quality | 8 |
| Organization | 7 |
| Reproducibility | 6 |
| Impact | 6 |
| References | 7 |
| Overall score | 6 |

### Score justifications

- **Novelty (7/10):** The combination of a declared bounded profile policy, singleton attribution–effect alignment, signed action selection, exact budget-two regret, abstention, and distinct-user null calibration appears meaningfully differentiated from ordinary deletion/insertion audits. However, each ingredient is established, and novelty is primarily protocol integration rather than a new attribution theory, recommender, causal estimator, or optimization method.
- **Technical correctness (6/10):** Most equations are algebraically valid, and the Shapley estimator is unbiased and efficient per prefix walk. However, the normalized intervention reallocates profile mass rather than simply suppressing evidence; Eq. (4) does not unambiguously define the optimized BPR objective; the claimed utility-transformation invariance after Eq. (16) is false in general; the explanatory statement after Eq. (17) mishandles zero predictions; the random-control seed formula is collision-prone; and several analysis denominators are inconsistent or insufficiently labelled.
- **Experimental quality (6/10):** The supplement adds exact-Shapley action validation, mask-design and ridge ablations, prospective audits, path-matched baselines, compute-budget curves, graph/Gowalla cells, and exhaustive `B=3`. The major weakness remains the absence of a competitive, quality-gate-passing neural or graph recommender and the strong dependence on candidate construction.
- **Scientific rigor (7/10):** Hypotheses, missing-value rules, bootstrap intervals, Holm correction, effect sizes, TOST, random controls, and supplementary exact/ablation analyses are commendable. Post hoc/unregistered choices, incomplete uncertainty propagation, H2 confounding, and contradictions between the main paper and supplement prevent a higher score.
- **Writing quality (8/10):** The prose is precise and technically mature. It is also dense, qualification-heavy, and occasionally reads as a rebuttal or audit log rather than a journal article. Several paragraphs combine too many caveats and results.
- **Organization (7/10):** The progression from game to metrics to decisions is logical, but decisive ablations, complexity, statistical validation, and robustness evidence are separated into a very dense 30-page supplement. Main/supplement version drift must be eliminated.
- **Reproducibility (6/10):** The supplement supplies analytical budgets, method timings, full-pipeline ranges, exact-validation tables, and an expanded contract. Reproduction is still impossible because the URL is a placeholder, processor/RAM/RSS are absent, RNG derivation remains underspecified, and code/raw matrices are not supplied.
- **Impact (6/10):** A rigorous intervention-aware audit could influence XAI evaluation practice. Current practical impact is limited by simulator-only feasibility, retrospective target conditioning, and evidence from a narrow primary model.
- **References (7/10):** The bibliography is broad and unusually current through 2026. It needs stronger coverage of explanation-evaluation validity, recommender-explanation user evaluation, adversarial/failure modes of LIME/SHAP, and recent graph recommender counterfactual explanation methods.
- **Overall (6/10):** Promising and technically serious, but substantial experimental and construct-validity revisions are necessary.

### Supplementary-material reassessment

| Prior concern | Supplementary evidence | Updated judgment |
| --- | --- | --- |
| Missing adjusted pairwise tests/effect sizes | Tables S3–S5/S11/S14/S16 | Numerically supplied, but multiplicity families conflict: e.g. MovieLens Shapley–LIME success Holm `p=.0066` in S4 versus `.0216` in S16. The authoritative family is unclear. |
| Missing success/abstention uncertainty | Tables S15–S16 | Resolved numerically, but Table S15 caption says Amazon `n=993` while every Amazon row reports `n=1000`. |
| Missing complexity/runtime | S6.1, S10, Tables S8/S30 | Partially resolved; CPU/RAM/RSS and repeated timing uncertainty remain absent. |
| Unequal compute | Table S19 claims equal-scorer-budget curves | Not resolved: printed Shapley budgets are 4.2k–42k scorer states versus only 128–2,048 LIME masks, differing by roughly 20–33× in paired columns. These are separate budget-response curves, not equal budgets. |
| Incomplete Shapley convergence | Tables S21, S24, S28 | Substantially mitigated for rank/sign/action agreement, but uncertainty is not propagated into final effects/regret. |
| LIME mask duplication | Table S22 | Concern confirmed but not cleanly bounded: Amazon AIA rises from `0.669` to `0.703/0.776`, while valid denominators change from `144/400` to `72/200` and `47/164`; these are not paired same-user comparisons and differ from main AIA `0.827`. |
| No prospective audit | S6, Tables S18–S19 | Partially resolved; prospective AIA is strong but conditioned on only 126–227 users per run whose generated top-1 lies in the candidate set. |
| No intervention-aware baselines | S6, Tables S18–S19/S27 | Resolved conceptually: bounded LIME, finite differences, and integrated gradients generally match/exceed Shapley. These should be primary, not supplementary. |
| No exact `B=3` oracle | Table S29 | Added for 200 users, but conflicts with the main paper’s statement that exhaustive triples are future work. |
| Architecture generality | Tables S18, S20, S26–S27 | Not resolved: SASRec/LightGCN remain below popularity, SASRec exact-estimator rank agreement is only `.395/.688`, and Gowalla lacks sample-size/uncertainty detail. |
| Candidate-universe dependence | Tables S1/S9 and main Table 8 | Confirmed, not resolved: Amazon full-catalogue Shapley bounded AIA is negative. |
| Pure suppression semantics | Eqs. (2)–(3), Table S19 `ρ` sweep | Not resolved and arguably strengthened: Amazon ItemKNN Shapley AIA is invariant (`0.708`) across all six `ρ` values. |
| Power/MDE documentation | Table S17 | Worsened by inconsistency: conventional recomputation from reported differences and `d_z` supports the main values (`≈0.014`, `≈0.051`), not S17’s `0.008/0.032`; S17 uses an undocumented alternative. |
| Reproducibility artifact | S6.1/S10 | Not resolved: artifact URL, source revision, machine-readable records, and manifest remain promised rather than supplied. |

## PHASE 2 — SECTION-BY-SECTION REVIEW

### 1. Title

- **Score (/10):** 7
- **Strengths:** Concise; identifies the central contrast between deletion and bounded intervention.
- **Weaknesses:** “ActionShap” implies a new Shapley method, whereas the paper repeatedly states that it is an explainer-agnostic evaluation protocol and that Shapley is not generally best.
- **Missing information:** The retrospective, target-conditioned, recommender-audit scope is not visible.
- **Logical inconsistencies:** Branding centers Shapley while the empirical conclusion favors LIME/LOO on absolute alignment.
- **Unsupported statements:** “Action” may be read as user-actionability, which the paper explicitly disclaims.
- **Grammar/Formatting issues:** None.
- **Suggestions for improvement:** Consider “ActionAudit: Evaluating Recommendation Attributions Beyond Deletion under Bounded Profile Interventions” or add “An Offline Audit Protocol.”
- **Questions for the authors:** Why should the protocol retain “Shap” in its name if its contribution is method-agnostic?

### 2. Abstract

- **Score (/10):** 8
- **Strengths:** States scope, protocol elements, datasets, cohort size, seeds, main alignment values, decision differences, and equivalence margin.
- **Weaknesses:** “Deployed systems may instead discount” is not demonstrated; the experiments use a simulator-side weighted history. “Exact budget-two oracle” lacks the qualifier “within the declared simulator and candidate set.”
- **Missing information:** No warning that the only quality-gated primary model is ItemKNN; no mention that the graph/sequential robustness models underperform popularity.
- **Logical inconsistencies:** “Shapley and LIME are practically equivalent … on both datasets” is valid under the declared margin, but the margin was not externally preregistered.
- **Unsupported statements:** The implication that bounded weighting reflects deployed operations is anecdotal without a production case study.
- **Grammar/Formatting issues:** Dense but grammatically sound.
- **Suggestions for improvement:** Add “simulator-executable,” state that evidence is conditional on one primary ItemKNN architecture, and report exact rather than rounded NDCG differences.
- **Questions for the authors:** Was the ±0.005 margin fixed before any outcome analysis, and where is that timestamped decision documented?

### 3. Keywords

- **Score (/10):** 8
- **Strengths:** Appropriate coverage of explainable recommendation, intervention evaluation, faithfulness, and cooperative game theory.
- **Weaknesses:** “Counterfactual evaluation” may suggest causal or recourse semantics that are explicitly excluded.
- **Missing information:** “Attribution faithfulness,” “offline evaluation,” “Shapley values,” and “action selection” would improve indexing.
- **Logical inconsistencies:** None.
- **Unsupported statements:** None.
- **Grammar/Formatting issues:** ACM “Additional Key Words and Phrases” is not KBS/Elsevier formatting.
- **Suggestions for improvement:** Use 4–6 KBS keywords, including “Shapley values” and “offline recommender evaluation.”
- **Questions for the authors:** Will terminology be changed from counterfactual to interventional-simulator evaluation where causal meaning is absent?

### 4. Introduction

- **Score (/10):** 8
- **Strengths:** Clearly distinguishes deletion, bounded modification, pointwise alignment, and joint decision quality; avoids causal overclaiming; states hypotheses and contributions.
- **Weaknesses:** The motivating practitioner scenarios are hypothetical. H2 is underspecified: “does not determine” could mean non-monotonic method ordering, low cross-user association, or failure under interactions.
- **Missing information:** A formal estimand table and a concrete deployed weighting example.
- **Logical inconsistencies:** H1 is described as response-surface distinction but is adjudicated mainly through method-specific AIA differences. H2 is later tested partly using gap–regret correlations, although H2 concerns absolute pointwise alignment.
- **Unsupported statements:** “An explanation is often consumed as advice about where to act” requires evidence from user/practitioner studies.
- **Grammar/Formatting issues:** Several paragraphs are overly compressed.
- **Suggestions for improvement:** Define operational hypotheses with explicit test statistics and rejection criteria; distinguish simulator feasibility, platform authority, and user agency at first use.
- **Questions for the authors:** What observation would falsify H2 after accounting separately for utility mismatch and pairwise interactions?

### 5. Related Work

- **Score (/10):** 7
- **Strengths:** Broad coverage of explainable recommendation, counterfactual evaluation, recourse, Shapley methods, graph explainers, and offline recommendation bias.
- **Weaknesses:** Often catalog-like; comparisons are asserted at a high level. Table 11 compresses heterogeneous works into binary cells and may misrepresent method-specific feasibility or inference.
- **Missing information:** Stronger treatment of user-centered explanation evaluation, formal faithfulness metrics, LIME/SHAP failure modes, GREASE and recent graph recommender counterfactual methods, and uncertainty in explanation evaluation.
- **Logical inconsistencies:** The text says ActionShap can audit graph-derived attributions, but no competitive graph scorer supports the main claim.
- **Unsupported statements:** “Non-overlap” in Section 7 is too categorical without a protocol-by-protocol derivation and released implementation comparison.
- **Grammar/Formatting issues:** Citation spacing is visibly malformed in the PDF (e.g., “[ 59 ]”, “[44 ]”).
- **Suggestions for improvement:** Replace catalog paragraphs with a taxonomy of estimand, action space, conditioning, oracle, and inference unit; audit Table 11 with citations to exact definitions.
- **Questions for the authors:** Which closest baseline can be minimally extended to bounded weights, and what does an implementation-level comparison show?

### 6. Background

- **Score (/10):** 8
- **Strengths:** Defines temporal players, retained history, older context, candidate sets, weighted scoring, and target-conditioned utility.
- **Weaknesses:** Excluding older context at scoring time changes the deployed profile relative to the complete fitted history; this is an operating-model choice, not merely game truncation.
- **Missing information:** Formal statement of repeated-item semantics after Amazon deduplication versus MovieLens repeated records; sensitivity to including older context as fixed background.
- **Logical inconsistencies:** Players are described as distinct interaction records, but Amazon duplicates are deduplicated while MovieLens behavior is not discussed comparably.
- **Unsupported statements:** The claim that `n_max` defines an operational profile needs evidence that this window is realistic.
- **Grammar/Formatting issues:** Notation is dense and occasionally overloaded despite Appendix C.
- **Suggestions for improvement:** Add a diagram showing fitted history, fixed background, player window, validation, and target; run a fixed-background game where older context remains scored.
- **Questions for the authors:** Why is older context removed rather than conditioned on as immutable background in every coalition?

### 7. Methodology

- **Score (/10):** 7
- **Strengths:** Clear pipeline; common policy and tie rules; method action is frozen before outcome evaluation; no oracle leakage.
- **Weaknesses:** The additive action policy privileges methods whose attribution scale/sign is locally calibrated; it is not uniquely implied by Shapley theory. Target-margin explanations are evaluated against NDCG decisions, confounding attribution quality with utility mismatch.
- **Missing information:** A formal causal diagram is unnecessary, but a data-flow/conditioning diagram and explicit information sets for attribution, selection, and evaluation are needed.
- **Logical inconsistencies:** “Same policy” does not mean same computational budget: Shapley, LIME, LOO, and greedy use very different numbers of scorer evaluations.
- **Unsupported statements:** “Practitioner would apply” the additive rule is conjectural.
- **Grammar/Formatting issues:** Methodology and experimental protocol repeat several settings.
- **Suggestions for improvement:** Add compute-matched comparisons; directly compare margin-attribution/margin-outcome, NDCG-attribution/NDCG-outcome, and cross-utility cells.
- **Questions for the authors:** How much of H2 survives when attribution and oracle use the same utility?

### 8. Mathematical formulation

- **Score (/10):** 8
- **Strengths:** Coherent weighted-coalition game, deterministic ranking tie rule, exact action space, explicit zero-profile baseline, and valid regret definitions.
- **Weaknesses:** No formal assumptions guarantee that fractional weights are meaningful for each model family. The primary value function is target-conditioned and discontinuous through `TopL`.
- **Missing information:** Formal domains/codomains for all mappings; handling when fewer than `L` non-target candidates exist; fixed-background alternative.
- **Logical inconsistencies:** Calling `(S,w)` a coalition blurs a discrete coalition game and a continuous weighted-input model; Shapley itself is computed only on binary coalitions.
- **Unsupported statements:** “Executable” is mathematical implementability, not demonstrated operational feasibility.
- **Grammar/Formatting issues:** Several symbols are visually crowded; hats and superscripts are difficult to distinguish.
- **Suggestions for improvement:** Separate the discrete game `v(S)=v(S,1)` from the continuous intervention response `v(P,w)` and explicitly state that Shapley axioms apply only to the former.
- **Questions for the authors:** Why is the empty-profile utility fixed by the model’s zero-score tie behavior rather than an empirically meaningful baseline?

### 9. Algorithms

- **Score (/10):** 7
- **Strengths:** Four algorithms cover attribution, selection/oracle evaluation, inference/null testing, and convergence.
- **Weaknesses:** Algorithm 1’s inner loop does not explicitly iterate over both walks; Algorithm 2 mixes two utilities and omits a returned regret; Algorithm 3 leaves bootstrap CI construction and Holm family membership implicit; Algorithm 4 uses a reference that is itself Monte Carlo.
- **Missing information:** Cache key, numerical-tolerance policy, random-generator specification, exact sample reuse, and failure behavior.
- **Logical inconsistencies:** Algorithm 2 “requires” candidate set but not the frozen scorer/model needed to compute effects.
- **Unsupported statements:** “Leakage-free” is stronger than demonstrated because target conditioning remains intentional leakage from a prospective perspective.
- **Grammar/Formatting issues:** “Require/Ensure” lines wrap poorly.
- **Suggestions for improvement:** Adopt the corrected pseudocode in Phase 4 and rename Algorithm 2 “outcome-blind selection with post-selection oracle evaluation.”
- **Questions for the authors:** Are attribution permutations shared across methods or only across seeds, and are caches isolated by utility/weight vector?

### 10. Complexity analysis

- **Score (/10):** 7
- **Strengths:** Supplementary S6.1/S10 and Tables S8/S30 give scorer-call complexity, exact action counts, pipeline runtimes, and method-specific median timings: MovieLens ItemKNN Shapley `0.65 s/user`, LIME `0.043 s`, and LOO `0.0018 s`.
- **Weaknesses:** Peak RSS, processor, RAM, cache-hit rates, and comparable method timings across all models/datasets remain absent. Table S30 mixes complete-pipeline ranges with method-specific medians from a different runner. Table S19’s “equal-scorer-budget” label is false as printed because paired Shapley/LIME columns differ by roughly 20–33× in scorer calls.
- **Missing information:** End-to-end storage cost for cached candidate score vectors, uncertainty over timings, and reproducible hardware specification.
- **Logical inconsistencies:** S10’s `O(m+n)` per-user memory excludes `O(n²)` returned action effects and the `O(min(2^n,T(n+1)))` Shapley scalar cache unless these are counted separately. Table S8 correctly says random uses zero attribution-utility calls, whereas S10 calls random `O(1)` scorer calls; generating `n` random scores is `O(n)` RNG work and zero attribution scorer calls. ItemKNN’s `O(nK+mK̄)` bound is implementation-specific and must be tied to the actual sparse traversal.
- **Unsupported statements:** Practical feasibility remains conditional on an unspecified processor.
- **Grammar/Formatting issues:** The dedicated complexity analysis is present only in the supplement rather than the main methodology.
- **Suggestions for improvement:** Rename S19’s comparison “separate budget-response curves” or compare exactly equal scorer calls. Move a compact cost table into the paper and regenerate timings with processor/RAM/RSS, repeated-run variability, and cache statistics.
- **Questions for the authors:** Are the reported per-user timings inclusive of candidate scoring and cache construction, and on what exact hardware?

### 11. Experimental setup

- **Score (/10):** 7
- **Strengths:** Frozen contract, temporal split, candidate construction, independent seeds, gate, oracle, and uncertainty procedures are detailed.
- **Weaknesses:** Primary claims rest on one effective model architecture and fixed candidates; model-training uncertainty is not propagated.
- **Missing information:** Processor/RAM/RSS, exact repository commit, artifact URL, raw result matrices, and data licences. Runtime ranges and the complete supplement are now supplied.
- **Logical inconsistencies:** Table S30 confirms the 80-run/five-study inventory but the machine-readable inventory remains unavailable.
- **Unsupported statements:** Schema validation does not establish scientific validity.
- **Grammar/Formatting issues:** Environment versions appear implausibly future/new and must be verifiable through a lockfile.
- **Suggestions for improvement:** Release container/lockfile and all raw outputs before review; report hierarchical uncertainty over users, candidates, and trained models.
- **Questions for the authors:** Why are candidate sets held fixed across all seeds, and how sensitive are conclusions to negative-sample redraws?

### 12. Datasets

- **Score (/10):** 6
- **Strengths:** Temporal processing, thresholding, deduplication, 5-core filtering, eligible-pool sizes, and final counts are reported.
- **Weaknesses:** Only two old, entertainment/e-commerce datasets in the primary study; uniform user subsampling and 5-core filtering weaken cold-start and long-tail relevance.
- **Missing information:** Demographics/exposure limitations, item-popularity distribution, timestamp granularity/ties, per-user split audit, and licensing details.
- **Logical inconsistencies:** Table S27 adds Gowalla but gives no sample size, uncertainty, recommendation-quality gate, or preprocessing details for that dataset.
- **Unsupported statements:** Cross-domain generality remains descriptive rather than inferential.
- **Grammar/Formatting issues:** Dataset statistics should be consolidated into one table.
- **Suggestions for improvement:** Add at least one sparse location/social dataset and one recent production-like sequential dataset; report warm/cold and head/tail strata.
- **Questions for the authors:** How many users/items are removed at each preprocessing step, and how does 5-core filtering alter the action-opportunity distribution?

### 13. Baselines

- **Score (/10):** 6
- **Strengths:** Includes Shapley, LIME, LOO, greedy deletion, random control, popularity, KernelSHAP, bounded LIME, finite differences, and integrated gradients. Table S19 supplies equal-scorer-budget curves.
- **Weaknesses:** LOO is an algebraic deletion oracle; strongest intervention-aware baselines remain supplementary and their graph/sequential cells use noncompetitive recommenders.
- **Missing information:** Bounded-LIME, finite differences, integrated gradients, RankSHAP/NDCG-Shapley, GREASE or recommender-specific counterfactual explainers in the main tables.
- **Logical inconsistencies:** The paper’s main question is bounded intervention, yet primary LIME is trained on binary deletion masks rather than bounded masks.
- **Unsupported statements:** “Principal methods” may overstate representativeness.
- **Grammar/Formatting issues:** Baseline definitions are dispersed.
- **Suggestions for improvement:** Make bounded/path-matched baselines primary and compare on equal scorer-call budgets.
- **Questions for the authors:** Why is bounded LIME relegated to a supplement when it is the most direct baseline for the proposed estimand?

### 14. Hyperparameters

- **Score (/10):** 7
- **Strengths:** Most primary settings are explicit in Sections 3–5 and Table B.1.
- **Weaknesses:** Rationale for `rho=.5`, `B=2`, `n_max=20`, LIME width `.25`, ridge `1`, and ItemKNN top-200 neighbours is limited.
- **Missing information:** Search spaces, validation objectives, tuning budgets, and baseline-specific fairness.
- **Logical inconsistencies:** Some values are called “declared” or “frozen,” but no external preregistration is available.
- **Unsupported statements:** Insensitivity of epsilon is asserted without main-text evidence.
- **Grammar/Formatting issues:** **Hyperparameters are merged across Background, Methodology, Experimental Protocol, and Appendix B rather than presented in a dedicated section.**
- **Suggestions for improvement:** Provide a tuning table with source, search range, selected value, and whether selected before test access.
- **Questions for the authors:** Were any hyperparameters changed after viewing Table 4 or Table 5 outcomes?

### 15. Evaluation metrics

- **Score (/10):** 7
- **Strengths:** Separates magnitude AIA, signed alignment, direction accuracy, precision, success, abstention, regret, normalized regret, stability, and null calibration.
- **Weaknesses:** Magnitude AIA can look excellent while directions are wrong; Spearman is unstable with short/tied vectors (Amazon median five players); NRegret conditions on active oracles and can exceed one.
- **Missing information:** Tie-corrected uncertainty per user, calibration error, utility-matched decision metrics, and aggregate coverage–risk curves for abstention.
- **Logical inconsistencies:** The main H2 test uses gap correlations, not direct bounded-AIA–decision associations.
- **Unsupported statements:** The stated Kendall sensitivity is still not numerically tabulated in the supplied supplement.
- **Grammar/Formatting issues:** Metric proliferation burdens interpretation.
- **Suggestions for improvement:** Predeclare primary/secondary metrics, report Kendall with exact small-sample handling, and show coverage–quality trade-offs.
- **Questions for the authors:** How many distinct effect ranks exist per user, especially on Amazon and for NDCG?

### 16. Results

- **Score (/10):** 7
- **Strengths:** Reports negative and null results; avoids claiming Shapley superiority; gives CIs, valid-user counts, equivalence tests, and boundary analyses.
- **Weaknesses:** Main tables omit many statistical comparisons that are deferred to the supplement. Rounded values obscure near-zero differences (Table 9). Full-catalogue Shapley becomes negatively aligned, which substantially weakens the headline.
- **Missing information:** Complete per-user distributions/raw matrices and robust failure-case analyses. The supplement now supplies pairwise tests, compute costs, and one worked example.
- **Logical inconsistencies:** “Central alignment finding generalizes” is too broad when full-catalogue Shapley AIA is −0.05 and neural/graph models underperform popularity.
- **Unsupported statements:** S6/S9/S25 claims are now inspectable, but the underlying CSV/JSON archive remains unavailable.
- **Grammar/Formatting issues:** Results are dense and occasionally mix primary and sensitivity claims in one paragraph.
- **Suggestions for improvement:** Move decisive robustness and statistical tables into the paper; report exact values beyond three decimals where differences are near zero.
- **Questions for the authors:** What fraction of users changes selected action when replacing estimated Shapley by exact Shapley on enumerable profiles?

### 17. Discussion

- **Score (/10):** 7
- **Strengths:** Generally careful; distinguishes audit from recourse and simulator action from user action; discusses responsible-AI limits.
- **Weaknesses:** Attributes LIME/LOO strength to locality without a direct mechanism test. Practical recommendations are not supported by a deployment or expert study.
- **Missing information:** Failure-case taxonomy and implications of fixed target/candidate conditioning.
- **Logical inconsistencies:** A positive gap is said to flag settings where deletion conclusions are fragile, but full-catalogue Shapley demonstrates that a positive gap can mean “less negative,” not useful bounded alignment.
- **Unsupported statements:** “When a bounded audit is worth running” requires a cost-benefit analysis.
- **Grammar/Formatting issues:** Several paragraphs exceed a reasonable cognitive load.
- **Suggestions for improvement:** Center absolute bounded alignment and action regret; demote the gap to a secondary sensitivity statistic throughout.
- **Questions for the authors:** Would any practitioner act on a method with positive gap but negative bounded AIA?

### 18. Ablation study

- **Score (/10):** 7
- **Strengths:** S6 and Tables S19, S22, and S29 now provide forced-action, magnitude-only, path-matching, `rho`, LIME kernel/mask/ridge, equal-budget, and exact-`B=3` ablations.
- **Weaknesses:** Most ablations lack CIs, paired tests, sample-size detail, or a common factorial design; fixed-background and candidate-redraw ablations remain absent.
- **Missing information:** Direct utility-matched target-margin/NDCG factorial analysis and interaction-aware versus additive decision outcomes with inferential uncertainty.
- **Logical inconsistencies:** Table S29 states exhaustive `B=3` “replaces” the greedy lower bound, while the main paper and Table B.1 still describe exhaustive triples as future work and the archived `B=3` result as greedy.
- **Unsupported statements:** Prospective/path-matched claims are supported only by compact aggregate tables, not raw records.
- **Grammar/Formatting issues:** **Ablation study is not present as a dedicated main-manuscript section. This is a critical omission for Knowledge-Based Systems.**
- **Suggestions for improvement:** Add a main-paper factorial ablation table and report paired effects with corrected tests.
- **Questions for the authors:** Which component contributes most to regret reduction, and are interactions or sign filtering responsible?

### 19. Explainability analysis

- **Score (/10):** 6
- **Strengths:** Faithfulness, direction, stability, decision quality, and null calibration are conceptually separated.
- **Weaknesses:** Only one worked example is added; there is still no user/expert evaluation, comprehensibility measure, or model-randomization sanity test.
- **Missing information:** Diverse success/failure cases, especially positive-gap/negative-AIA examples and unstable actions.
- **Logical inconsistencies:** The protocol evaluates attribution behavior but not whether outputs constitute useful explanations to humans.
- **Unsupported statements:** Broader explanation-quality implications exceed the fidelity/decision audit.
- **Grammar/Formatting issues:** **Explainability analysis is merged into metrics/results/discussion rather than presented as a dedicated section.**
- **Suggestions for improvement:** Add representative and adversarial cases; explicitly limit conclusions to functional faithfulness under the simulator.
- **Questions for the authors:** How would a user or operator interpret a negative Shapley attribution attached to a historical interaction?

### 20. Limitations

- **Score (/10):** 9
- **Strengths:** Exceptionally candid about construct, metric, internal, external, selective-reporting, and statistical-conclusion validity.
- **Weaknesses:** Some limitations remain severe despite supplementary experiments, especially failed neural/graph quality gates and selection-conditioned prospective audits.
- **Missing information:** Candidate-sampling uncertainty and compute-budget fairness deserve explicit treatment.
- **Logical inconsistencies:** “Central alignment finding replicates” sits uneasily with the stated quality failure of neural/graph models.
- **Unsupported statements:** Claims of broad generality exceed the compact, noncompetitive replication cells.
- **Grammar/Formatting issues:** Long and dense.
- **Suggestions for improvement:** Separate remediable threats from irreducible scope conditions.
- **Questions for the authors:** Which validity threat most likely reverses a method ranking?

### 21. Threats to validity

- **Score (/10):** 8
- **Strengths:** Integrated into Section 7.1 with six explicit categories.
- **Weaknesses:** Fixed-candidate uncertainty, target selection, dependence induced by a shared fitted similarity matrix, and exchangeability of the null shuffle need deeper treatment.
- **Missing information:** External artifact audit and independent replication.
- **Logical inconsistencies:** User-bootstrap intervals are sometimes discussed as if they cover the broader data-generating process, while the paper later correctly says they are conditional on the fitted model.
- **Unsupported statements:** Build validation cannot rule out selective design choices made before schema v2.
- **Grammar/Formatting issues:** **Threats to validity are merged with Limitations rather than independently signposted at top level; acceptable scientifically but structurally less clear.**
- **Suggestions for improvement:** Add a table mapping threat, direction of bias, affected claim, and mitigation.
- **Questions for the authors:** How would uncertainty change under resampled training data and candidate sets?

### 22. Conclusion

- **Score (/10):** 8
- **Strengths:** Accurately states that gap is not sufficient for validity and avoids claiming a universal best explainer.
- **Weaknesses:** “Fully specified protocol” is premature while the executable artifact and raw archive remain absent and main/supplement versions conflict.
- **Missing information:** Explicit restatement that evidence is conditional on a target-conditioned simulator and one primary model.
- **Logical inconsistencies:** “System-executable” remains stronger than the demonstrated simulator-executable interface.
- **Unsupported statements:** None if appropriately qualified.
- **Grammar/Formatting issues:** Clear.
- **Suggestions for improvement:** Replace “fully specified” with “specified in the manuscript and accompanying artifact,” once the artifact exists.
- **Questions for the authors:** What is the minimum evidence required before ActionShap should influence a production explanation audit?

### 23. References

- **Score (/10):** 7
- **Strengths:** 78 references; good coverage of classical recommendation, Shapley, XAI, counterfactual explanation, and work through 2026.
- **Weaknesses:** Several references are workshop/arXiv items where archival alternatives may exist; DOI/style consistency requires checking; future-dated 2026 citations must be verifiable at submission.
- **Missing information:** See Phase 11.
- **Logical inconsistencies:** ACM reference style conflicts with a KBS submission.
- **Unsupported statements:** Table 11 classifications require more precise citation support.
- **Grammar/Formatting issues:** In-text citation spacing is inconsistent; journal names and proceedings formatting should be converted to Elsevier style.
- **Suggestions for improvement:** Verify every DOI, publication status, author spelling, page range, and 2026 bibliographic record.
- **Questions for the authors:** Are references [7], [48], and [51] formally published by the manuscript’s submission date, or accepted/preprint versions?

## PHASE 3 — MATHEMATICAL VALIDATION

### Global mathematical assessment

The discrete Shapley game is well formed: `P_u` is finite, `v_u^attr(S)` is real-valued, and `v(∅)=0.5` is explicit. Exact Shapley therefore satisfies efficiency, symmetry, dummy, and additivity for that declared game. These axioms do **not** transfer causal validity, feasibility, or optimality to the continuous intervention policy. The most important construct issue is the denominator in Eqs. (2)–(3): for a full profile, reducing one weight from `1` to `ρ` changes its normalized coefficient from `1/n` to `ρ/(n−1+ρ)` while increasing every other coefficient from `1/n` to `1/(n−1+ρ)`. Uniform scaling by any positive constant leaves the score unchanged. The intervention is consequently relative reweighting, not pure attenuation. Other weaknesses are that `(S,w)` conflates a discrete coalition with continuous weights, LIME is fitted on the deletion surface while audited on a bounded surface, and target-margin attribution is used for NDCG decisions.

The supplement introduces no additional numbered equations or formal proofs. Its unnumbered interaction, complexity, and action-count expressions are algebraically consistent with the main definitions, except that its exhaustive-`B=3` status conflicts with the main experimental contract and its memory bound must account separately for the Shapley cache.

| Eq. Number | Correct? (YES/NO/PARTIAL) | Reason | Suggested Correction |
| --- | --- | --- | --- |
| (1) | YES | Finite recent-player window is well defined. | State whether repeated-item records can coexist after each dataset’s preprocessing. |
| (2) | PARTIAL | The weighted mean is algebraically and dimensionally valid, but normalization means downweighting one player reallocates mass to all others; it is not isolated evidence suppression. Uniform positive scaling has no effect. | Compare with an unnormalized or fixed-denominator scorer and rename the current operation “relative profile reweighting” unless this exact normalized interface is operationally justified. |
| (3) | PARTIAL | The profile mean and dot product are dimensionally valid, but they inherit Eq. (2)’s relative-reallocation semantics. | Define embedding normalization and evaluate a fixed-denominator/unnormalized intervention in addition to the zero-vector convention. |
| (4) | NO | The displayed regularizer’s scope appears outside the triple sum, while prose says it is applied per sampled gradient step. The context-mean denominator, record-versus-item exclusion, duplicate-embedding multiplicity, and clipping derivative are unspecified. These alternatives define different objectives/updates. | Write the per-triple loss and SGD gradients explicitly; define `r_{u,-i}=|C|^{-1}Σ_{h∈C}q_h`; specify whether clipping only stabilizes sigmoid evaluation or zeroes gradients; normalize or justify context-length-dependent regularization. |
| (5) | PARTIAL | Sigmoid target-versus-top-`L` mean margin is valid and dimensionless only if score scale/temperature is understood. `TopL` makes it nonsmooth. | Define behavior when fewer than `L` competitors exist and report temperature/score-scale sensitivity. |
| (6) | YES | Deterministic one-indexed rank with global tie priority is correct. | State that `τ` is a strict total order over the full candidate universe. |
| (7) | YES | One-relevant-item NDCG@10 is correctly `1/log2(r+1)` inside cutoff. | Typeset the multiplication clearly and call it sampled-candidate NDCG where applicable. |
| (8) | YES | Weight intervention vector is correctly defined. | Use one symbol consistently for player (`p`) versus interaction/item. |
| (9) | YES | Signed effect is post-intervention minus full-profile utility. | Add the utility and candidate-set conditioning to the notation when comparing conditions. |
| (10) | YES | Deletion and bounded singleton effects follow Eq. (9). | Retain exact `ρ` in superscript/subscript to avoid hard-coding `.5`. |
| (11) | PARTIAL | Multiplicative downweighting is executable in the simulator. Repeated application semantics are not relevant but are unstated. | Define it as a one-shot map from baseline weight 1: `w'_p=ρ`, not a potentially compounding assignment. |
| (12) | YES | Budgeted power-set action space including no action is valid. | Add heterogeneous-cost generalization if practical actionability is retained as motivation. |
| (13) | YES | Average marginal contribution over uniform forward/reverse permutations is unbiased; dependence affects variance, not expectation. | Make the sum index explicitly range over both walks and report a paired variance estimator or empirical MC standard error. |
| (14) | YES | Standard exact Shapley formula; axioms hold for the declared game. | Explicitly state the four axioms in notation and emphasize they apply to `v(S,1)`, not bounded effects. |
| (15) | PARTIAL | Weighted ridge objective is valid with unpenalized intercept. Heavy mask duplication for short profiles induces random multiplicity weights. | Enumerate unique coalitions for small `n`, or use weighted unique rows/without-replacement masks; report conditioning of the normal equations. |
| (16) | PARTIAL | Spearman magnitude correlation is defined for nonconstant finite vectors but discards sign and is coarse for short/tied profiles. The claim that AIA is invariant to monotone re-expression of utility is false in general: a nonlinear monotone transform applied before baseline differencing can reorder absolute effect magnitudes. | Restrict the invariance claim to monotone transformations of the already-formed magnitude vector; define tie handling/minimum `n`; pair AIA with signed alignment as co-primary. |
| (17) | PARTIAL | The formula is valid, but the prose says `sign(0)` never contributes because zero effects are excluded. A predicted benefit can be zero while the realized effect is nonzero, in which case `sign(0)` contributes as a mismatch. | Correct the prose or explicitly exclude zero predictions; add uncertainty and report coverage `|I_u|/n_u`. |
| (18) | PARTIAL | Formula is arithmetically valid, but top-`k` may include predicted-negative actions when fewer than `k` positive benefits exist; sign condition is redundant. | Define a coverage-aware precision over eligible positive predicted benefits or report both fixed-`k` and abstention-aware variants. |
| (19) | PARTIAL | Mean pairwise seed Spearman is a reasonable diagnostic, but the stated missingness condition “fewer than two such pairs” excludes a user with exactly one valid pair without clear reason. | Require at least one valid pair, or justify the stricter threshold; report the number of valid pairs. |
| (20) | PARTIAL | Plus-one dataset-level permutation p-value is computed correctly. Validity requires an exchangeable label-shuffle null across players, questionable with temporal/item heterogeneity. | State the randomization null precisely; add stratified permutations by recency/popularity or a conditional model-based null. |
| (21) | YES | Additive predicted-benefit score is coherent as a heuristic. | Do not imply it follows from Shapley; calibrate or normalize attribution scales when comparing policies. |
| (22) | YES | Exact budget-two utility-specific oracle is correctly defined. | Include the deterministic tie rule in the equation or immediately adjacent definition. |
| (23) | YES | Regret is oracle effect minus selected effect and is nonnegative if action spaces match exactly. | Report numerical tolerance and validation failure threshold. |
| (24) | YES | General budget oracle is valid. | For `B>2`, call approximate search an approximate lower-bound oracle, not an oracle. |
| (25) | YES | Normalized regret is valid on positive-oracle users and can exceed one for harmful actions. | Make conditional estimand explicit in the symbol and accompany means with medians/quantiles. |
| (A.1) | YES | Prefix marginal contributions telescope exactly to `v(P)-v(∅)`. | Note that this proves per-walk efficiency but says nothing about convergence of individual attributions. |

### Shapley axioms

| Axiom | Status | Validation |
| --- | --- | --- |
| Efficiency | Satisfied for exact Shapley and every complete prefix walk | Eq. (A.1) telescopes to `v(P)-v(∅)=v(P)-0.5`. This is not a convergence test. |
| Symmetry | Satisfied by exact value | Players with equal marginal contributions for every coalition receive equal values. Finite MC estimates need not be exactly equal. |
| Dummy | Satisfied by exact value | A player with zero marginal contribution to every coalition receives zero. Numerical MC/cache tolerances should be reported. |
| Additivity | Satisfied by exact value | Holds for linear combinations of characteristic functions. It does not imply additive realized effects under joint bounded intervention. |

### Gradients and uncertainty

- No analytic gradient derivation is required for permutation Shapley or LIME. The supplement reports finite-difference/integrated-gradient outcomes but still does not provide enough implementation detail to audit those paths fully.
- The BPR objective requires correction, not merely clarification. For context `C`, `c=|C|`, `r=c⁻¹Σ_{h∈C}q_h`, and `z=rᵀ(q_i−q_j)`, the declared per-triple loss implies `∇_{q_i}ℓ=(σ(z)−1)r+λq_i`, `∇_{q_j}ℓ=(1−σ(z))r+λq_j`, and `∇_{q_h}ℓ=((σ(z)−1)/c)(q_i−q_j)+λq_h`, with multiplicity accumulation for repeated embeddings. The implementation must state whether these are the actual updates.
- Bootstrap percentile intervals are mathematically conventional but condition on one fitted model and fixed candidate samples. They do not represent training, preprocessing, or negative-sampling uncertainty.
- TOST arithmetic is coherent: both reported 90% CIs lie inside `[-.005,.005]`; MovieLens can be both statistically different from zero and practically equivalent.
- The stated random seed `model_seed + 1,000,000 + user_id` is collision-prone: `(u,s+1)` and `(u+1,s)` can map to the same integer. Use tuple-based seed derivation such as `SeedSequence(stream_tag, experiment_seed, stable_user_index)`.
- Analysis populations are not consistently labelled: Table 2 gives 196 active Amazon NDCG oracles among 1,000 users, later text uses `196/993`, the NDCG TOST excludes seven constant-AIA users without a decision-specific reason, and Section 6.4 reports normalized-regret correlations with `n=1000/987`, which cannot be NDCG NRegret based on active-oracle counts `339/196`. Every statistic must name its utility and denominator.

## PHASE 4 — ALGORITHM REVIEW

No additional numbered algorithms or pseudocode blocks are present in the supplement. Supplementary S6.1/S10 add cost formulas and experiment descriptions but do not repair the main algorithms’ missing RNG, null, and gradient specifications.

### Algorithm 1 — Paired Monte Carlo attribution

**Purpose:** Estimate Shapley values with uniform base permutations and their reverses.

**Flaws identified**

- The pseudocode says “for each consecutive pair” without explicitly nesting over both forward and reverse walks.
- Inputs omit cache-key semantics and random-generator definition.
- Output diagnostics do not include Monte Carlo standard error or per-user convergence.
- A complete prefix walk needs `n_u+1` utility values; scorer-call complexity is not stated.

**Corrected pseudo-code**

```text
INPUT: player list P_u, characteristic function v_u, base count M_pair,
       seeded RNG, cache keyed by (u, coalition, utility)
OUTPUT: attribution vector phi_hat, efficiency residual, MC diagnostics

set accumulator[p] = 0 for every p in P_u
for m = 1,...,M_pair:
    sample a uniform permutation pi of P_u
    for walk in [pi, reverse(pi)]:
        S = empty set
        previous = value(u, S, cache)
        for p in walk:
            current = value(u, S union {p}, cache)
            accumulator[p] += current - previous
            S = S union {p}
            previous = current
T = 2 * M_pair
phi_hat[p] = accumulator[p] / T for every p
residual = sum_p phi_hat[p] - (v_u(P_u) - v_u(empty))
return phi_hat, residual, finite/constant flags, optional batchwise MCSE
```

**Complexity:** `O(M_pair n_u C_v)` value evaluations without cache sharing across walks, with up to `2M_pair(n_u+1)` coalition requests; memory is `O(min(2^{n_u}, M_pair n_u))` cached coalition values plus scorer outputs. `C_v` includes candidate scoring and top-`L` selection.

### Algorithm 2 — Outcome-blind action selection and exact budget-two oracle

**Purpose:** Select an action from attribution only, freeze it, then evaluate realized effects and exact oracles.

**Flaws identified**

- “Leakage-free” is misleading because held-out-target conditioning is intentionally used.
- Required inputs omit frozen scorer/model, baseline scores, and numerical tolerance.
- The algorithm returns effects but not regret/NRegret despite being the decision-quality algorithm.
- Attribution scale is not calibrated; only sign/order matters for unit-cost top-`B`.

**Corrected pseudo-code**

```text
INPUT: phi_hat, P_u, B=2, rho, frozen scorer f_u, candidates E_u,
       utilities Z, strict tie order tau, epsilon
OUTPUT: selected action A_hat, utility-specific oracle actions,
        realized effects, regret and normalized regret

benefit[p] = -phi_hat[p]
eligible = players p with benefit[p] > epsilon
sort eligible by decreasing benefit, then increasing player index
A_hat = first min(B, |eligible|) players; if eligible is empty, A_hat = empty
freeze A_hat

actions = {empty} union all singletons union all unordered pairs from P_u
for each A in actions:
    construct one-shot weights w_p = rho if p in A else 1
    for each utility z in Z:
        effect[z,A] = z(P_u,w_A,rho) - z(P_u,1)
for each utility z:
    A_star[z] = argmax_A effect[z,A],
                breaking ties by smaller cardinality then lexicographic order
    regret[z] = effect[z,A_star[z]] - effect[z,A_hat]
    if effect[z,A_star[z]] > epsilon:
        nregret[z] = regret[z] / effect[z,A_star[z]]
    else:
        nregret[z] = missing
return all outputs
```

**Complexity:** Selection is `O(n_u log n_u)`. Exact `B=2` evaluation uses `1+n_u+n_u(n_u-1)/2 = O(n_u^2)` actions and `O(|Z|n_u^2 C_z)` time. Storing all candidate-score vectors costs `O(n_u^2|E_u|)` unless effects are streamed.

### Algorithm 3 — Distinct-user inference and AIA null

**Purpose:** Aggregate seed records within users, construct within-user null calibration, bootstrap users, and perform corrected paired comparisons.

**Flaws identified**

- The randomization null and its exchangeability assumption are not stated.
- “Corrected comparisons” does not define the exact family in pseudocode.
- Bootstrap resampling, CI quantiles, and missingness intersections are abbreviated.
- Fixed candidates and fitted model are not resampled.

**Corrected pseudo-code**

```text
INPUT: per-user/per-seed attribution and effect vectors, R_null, R_boot,
       R_perm, predeclared comparison families, seeded RNG streams
OUTPUT: user summaries, null p-values, bootstrap CIs, Holm-adjusted tests

for each user u and seed s:
    if both vectors are finite and nonconstant:
        compute observed AIA[u,s]
        for r = 1,...,R_null:
            apply the declared within-user label permutation
            compute nullAIA[u,s,r]
    else mark (u,s) invalid

for each user u:
    V = valid seeds
    if |V| < 3: mark user missing
    else:
        obs[u] = mean_s in V observed AIA[u,s]
        null[u,r] = mean_s in V nullAIA[u,s,r]

compute the dataset statistic as the mean over valid distinct users
compute plus-one upper-tail null p from matched r-level dataset null means
bootstrap distinct users R_boot times for percentile 95% CIs
for each predeclared paired comparison:
    intersect users finite under both methods
    compute paired user differences and effect size
    perform R_perm paired label swaps/sign flips and plus-one two-sided p
within each predeclared family, apply Holm correction
return estimates, denominators, intervals, raw and adjusted p-values
```

**Complexity:** Null generation is `O(U R_seed R_null n_u log n_u)` if each Spearman correlation is recomputed naïvely. Bootstrap and paired permutation cost `O((R_boot+R_perm)U)` per statistic/comparison.

### Algorithm 4 — Independent convergence selection

**Purpose:** Select an MC budget using rank agreement and action Jaccard against an independent high-budget MC reference.

**Flaws identified**

- The reference is noisy, not ground truth.
- Population-mean thresholds permit poor per-user convergence; coverage is only 56–69% at selected budgets.
- “Smallest” budget selection can be unstable without monotonicity or uncertainty around agreement.
- Reusing nested samples versus independently drawing each candidate budget is unstated.

**Corrected pseudo-code**

```text
INPUT: candidate budgets M, independent reference samples,
       thresholds for rank, action Jaccard, valid fraction and user coverage
OUTPUT: selected budget or unconverged status

construct a reference estimate from an RNG stream disjoint from every candidate
for each candidate M in increasing order:
    estimate attributions using a declared independent or nested sample design
    for each user with valid vectors:
        compute rank agreement and signed-action Jaccard to the reference
    compute means, bootstrap CIs, valid fraction, and fraction of users
        satisfying both per-user thresholds
    candidate qualifies only if lower confidence bounds on mean criteria pass
        and predeclared per-user coverage passes
select the first qualifying candidate
if none qualifies:
    report "unconverged"; do not silently relabel the largest budget converged
validate the selected budget against exact Shapley on every enumerable user
return all diagnostics
```

**Complexity:** Sum of Algorithm 1 costs over candidate budgets plus the reference. Exact validation is `O(2^{n_u}C_v)` per enumerable user.

## PHASE 5 — FIGURES

| Fig. Number | Caption Quality | Readability | Scientific Usefulness | Supports Claims? | Suggestions for Improvement |
| --- | --- | --- | --- | --- | --- |
| Fig. 1 | High | Moderate | High as a protocol overview | Conceptually, not empirically | Increase type size; distinguish data unavailable prospectively; label simulator-only intervention. |
| Fig. 2 | High | Low–moderate due to many rows/methods/panels | High | Yes for alignment decomposition | Split primary and robustness conditions; use shapes/line styles; add a zero reference to every panel; enlarge labels. |
| Fig. 3 | Adequate | High | Moderate | Partially; shows only MovieLens | Add Amazon panel and exact values; identify interval type in-axis note; include zero line. |
| Fig. 4 | High | Moderate | High for MC diagnostics | Partially, because reference is MC and per-user coverage remains low | Show uncertainty bands and per-user coverage; identify exact-reference subset; avoid log ticks rendered only as `10^2,10^3` without intermediate labels. |

### Figure-specific issues

- **Supplementary figures:** No supplementary figures are present. All supplementary visual evidence is tabular (Tables S1–S30).
- **Axes and units:** Figures 2–4 identify correlation, NDCG change, or agreement, but “permutation count,” candidate condition, and interval unit should be more explicit. Figure 3 should say “mean ΔNDCG@10 per user.”
- **Legends:** Figure 2’s legend is long and visually overloaded. Figure 4 mixes dataset, model, and utility in one legend; faceting would reduce decoding cost.
- **Color accessibility:** Captions state that tables provide numeric values, but figures must still be independently interpretable. Use colorblind-safe colors plus distinct markers and line styles.
- **Resolution:** Vector figures are claimed, which is appropriate. The PDF’s dense labels remain too small at normal review zoom.
- **Selection:** Figure 3 presents only MovieLens, although equivalence and reversed ordering on Amazon are central. This is selective visual emphasis even though Table 5 contains Amazon.
- **Uncertainty:** Figure 4 lacks uncertainty around convergence estimates. Agreement to an MC reference should not be shown as a noise-free curve.

## PHASE 6 — TABLES

| Table Number | Formatting | Completeness | Missing Statistics (CI/SD) | Significance Indicators | Possible Errors | Suggestions |
| --- | --- | --- | --- | --- | --- | --- |
| Table 1 | Compact | Low | Not applicable | Not applicable | Categories are oversimplified binaries | Add citations per row and footnotes for “usually/method-specific.” |
| Table 2 | Clear | Moderate | IQR/CI absent for mean history | None needed | Minimum-player text elsewhere says possible 2 but observed 3 | Add min, IQR, eligible pool, active-oracle percentage. |
| Table 3 | Clear | Good | Popularity uncertainty and paired difference CI absent | No test indicators | “Popularity constants” may hide user-level paired variability | Add paired model-minus-popularity CI/test and candidate count per row. |
| Table 4 | Clear | Good | Deletion CIs deferred to Table S3 | CIs in main; adjusted tests/effect sizes in S3 | Rounded Amazon gap `.414-.286=.128`, while shown `.129` from unrounded values; acceptable but footnote needed | Include the most important adjusted comparisons in the main table. |
| Table 5 | Clear | Moderate | CIs absent in table body; only prose | No significance/equivalence markers | Rounded values hide method differences | Put 95% CIs, TOST result, paired adjusted p, and active-oracle rate in table. |
| Table 6 | Clear after caption | Good | Dataset-level null uncertainty not shown | Unadjusted p explicitly labeled | Plus-one minimum p `.0010` is correct for 1,000 draws | Add stratified/conditional-null sensitivity and standardized distance from null. |
| Table 7 | Clear | Low | No uncertainty; only selected rows | None | “Coverage” 56–69% contradicts broad per-user convergence; S6/S21/S24/S28 partly mitigate this | Include final-floor, exact-reference, and cap-user diagnostics in the main paper. |
| Table 8 | Dense | Good | CIs present | No adjusted paired tests | `0.74 [0.53,1.04]` regret is very uncertain; Shapley bounded AIA is negative despite positive gap | Emphasize negative absolute alignment and low NDCG active count; separate target-margin and NDCG tables. |
| Table 9 | Misleading rounding | Low | CIs present | No p/effect size | Differences shown as `-0.000` are uninterpretable and can conceal sign | Report 5–6 decimals, adjusted p, effect size, and practical margin. |
| Table 10 | Clear | Moderate | Descriptive CIs present | No multiplicity adjustment | Tiny MovieLens strata (`n=9`) unstable | Add IQR/profile composition and interaction test; avoid cross-dataset causal language. |
| Table 11 | Dense/wrapped | Moderate | Not applicable | Not applicable | Binary taxonomy may misclassify nuanced prior work | Provide explicit criterion definitions and evidence citations; avoid “No” where “not evaluated” is accurate. |
| Table B.1 | Useful | Good | Not applicable | Not applicable | S29’s exhaustive `B=3` conflicts with B.1/main text’s greedy-only status; primary vs sensitivity columns misclassify some settings | Synchronize the contract with S7/S29 and add software commit, hardware, candidate redraw policy, and artifact URL. |

### Supplementary tables S1–S30

| Table Number | Formatting | Completeness | Missing Statistics (CI/SD) | Significance Indicators | Possible Errors | Suggestions |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Very long, split over 3 pages | Good condition coverage | CIs/missing counts deferred to CSV | None | “All declared conditions” omits budgets without explanation; AIA/gap not locally defined; 250-user catalogue values differ from main 1,000-user Table 8 | Add cohort/condition scope, definitions, and CIs. |
| S2 | Extremely long, split over 10 landscape pages | Broad | Most means lack CIs | None | Sideways running headers; `ndcg` does not say NDCG@10; changing regret `n` is not identified as active-oracle denominator | Split by condition/utility, fix orientation, define units/denominators, and add CIs. |
| S3 | Dense | Good | CIs present, but level/method unstated | Holm `p`, `d_z` present | Δ direction and adjustment family unstated; CIs differ slightly from S14; uses misleading “Actionability Gap” | State inferential contract, reconcile S14, and rename gap. |
| S4 | Dense | Good | CIs present but unadjusted | Holm `p`, `d_z` present | Family unstated; adjusted success `p` conflicts with S16; an unadjusted CI can exclude zero while Holm `p` is nonsignificant | Name/cross-reference the confirmatory family and clearly separate descriptive CIs. |
| S5 | Clear | Good | CIs present, but level/method unstated | Holm `p`, `d_z` present | Δ direction/family unstated; conditional-only estimand may be mistaken for full-cohort quality | Define inference and add unconditional regret/effect. |
| S6 | Clear | Good budget coverage | No CIs | Selection flags | Reference construction and user count omitted; mean agreement masks low coverage | Add uncertainty, denominator/reference seeds, and final-budget marker. |
| S7 | Clear | Moderate | Not applicable | Not applicable | Caption says contract used by “every final run,” false for candidate/utility/budget/rho sensitivities; RNG/software details absent | Restrict scope and expand contract. |
| S8 | Clear | Good scorer-state counts | No runtime variability | Not applicable | Evaluated states versus unique cached scorer calls are not distinguished | Add unique-call/cache-hit counts and measured distributions. |
| S9 | Dense/redundant columns | Good | CIs absent | None | Caption says “absolute values shown for all,” yet negative correlations are retained; repeated LIME/LOO/random columns add redundancy | Correct caption, normalize layout, and foreground sign reversal. |
| S10 | Compact | Moderate | CIs absent | None | `B=1/B=3` rows omit dataset; dashes replace available S2 outcomes; “exploratory” is asserted but not visibly labelled; greedy/exhaustive `B=3` versions conflict | Synchronize with S29, identify dataset/cohort, and print available values. |
| S11 | Clear | Selective headline subset | CIs present | Holm `p`, `d_z` present | Omits `n`; “will be included” is stale because S3–S5 already contain pairwise tests | Add `n`, clarify purpose, remove future tense. |
| S12 | Clear | Selected rows only | CIs absent | None | Omits `T`, user count, and reference details; duplicates S6; NDCG remains unconverged | Merge with S6 or add complete metadata. |
| S13 | Dense | Good signed metrics | CI only for signed alignment | Descriptive only | One `n` may conceal metric-specific validity; deterministic LOO/greedy stability of 1 is trivial | Add CIs/valid counts per metric and separate estimator/pipeline stability. |
| S14 | Very long | Complete bounded-AIA family | CIs present | Holm `p`, `d_z` present | CIs differ slightly from S3; “primary conditions” includes non-primary profile models below popularity | Reconcile bootstrap provenance and label boundary models. |
| S15 | Dense | Good quantities | CIs present | None | Caption says Amazon `n=993`, but all Amazon rows show `n=1000`; identical NDCG−popularity repeated for every explainer | Correct caption and de-duplicate model-quality rows. |
| S16 | Clear | Complete 12-test family | No effect-size CI | Holm `p` present | Family count is correct, but adjusted values differ from S4 (`.0216` vs `.0066` for MovieLens Shapley–LIME success); sign-flip assumptions remain unstated | Identify which family is primary and add effect sizes/test assumptions. |
| S17 | Clear | Partial power analysis | Hierarchical CIs present; MDE CIs not applicable | None | MDE values `0.008/0.032` conflict with main `0.014/0.051`; “hierarchical” does not include candidate/model retraining | Reconcile values and give formula, alpha, SD, hierarchy levels. |
| S18 | Extremely compressed | Moderate | No CIs, sample sizes mostly absent | None | Estimator budget omitted; SASRec exact-rank agreement `.395/.688`; Amazon ItemKNN Shapley `.669` differs from main `.414` without reconciliation | Add per-run `n`, uncertainty, budgets, condition labels, and estimator-error caveats. |
| S19 | Overloaded multi-block table | Broad sensitivity | No `n`, CIs, or seed variability | None | Uses `B` for scorer budget although `B` already means action budget; “equal” budgets are not equal; Amazon AIA is `0.708` for all `ρ`; unconditional regret undefined | Rename symbols/curves, compare equal calls, split table, and define quantities. |
| S20 | Clear | Good quality boundary | No CIs | Masking-gate counts | “Gate” may be mistaken for recommendation-quality passage even though both SASRec variants are below popularity | Rename column “masking gate”; keep as negative boundary. |
| S21 | Clear | Strong exact validation | Quantiles but no CI | None | Model, utility, MC budget, and intervention are omitted; exact subset is profile-length selected | Add full configuration, effect/regret loss, and `n_u` strata. |
| S22 | Clear | Useful ablation | SD present | None | Caption says 200 users while rows reach 400; changing valid subsets prevents a causal design comparison; Amazon `.669` differs from main `.827` | Correct the unit/condition, use paired same-user subsets, and report paired differences/CIs/tests. |
| S23 | Clear | Minimal | No uncertainty/sample decomposition detail | None | Variance estimator, scale, utility, and observational unit are undefined; model-seed variance exceeds attribution-seed variance but is not propagated | Fully specify and expand the variance-component analysis. |
| S24 | Clear | Good quantiles | No CIs | None | Rank-valid counts absent; coverage remains 74%/83% at `M_pair=500`; main text incorrectly locates this table in S8 | Correct cross-reference and add counts/uncertainty. |
| S25 | Clear | Good KernelSHAP check | Difference CIs present | `d_z` present; adjusted `p` absent | Amazon `n=810` is unexplained; main text cites Pearson agreement `0.78–0.98`, but S25 does not report Pearson correlations | Explain missingness, add the cited correlations, and report paired adjusted tests. |
| S26 | Clear | Good negative quality check | CIs absent | Masking-gate passes | Title says “recommendation-quality gate,” but “Gate passes” refers to masking while NDCG remains below popularity | Separate recommendation quality from masking and avoid competitive validation claims. |
| S27 | Clear | Low | No `n`, seeds, CIs, or tests; Gowalla preprocessing/quality absent | None | Utility is not restated; calls result “generalization” without competitive model or inferential support | Add complete inferential and dataset/model protocol. |
| S28 | Clear | Useful cap-user MC check | No CI | None | Only 50 users/dataset; labels `M=250` “selected” although Amazon selected 50 and final floor is 500; prose says every correlation `≥.95` although minimum is `.948` | Correct budget labels and strict claim; repeat on larger samples. |
| S29 | Clear | Useful exact `B=3` check | No CIs | None | Contradicts main paper/S10 status; utility is not explicit; only 200 users | Synchronize documents, label utility, and add paired uncertainty. |
| S30 | Clear | Good summary | Timing uncertainty absent | Not applicable | Memory statement separates working arrays and Shapley cache ambiguously; hardware missing | Add processor/RAM/RSS, repeated timings, and cache memory. |

### Units and text-to-table mapping

- Correlations are unitless; NDCG effects are absolute changes, not percentages. Captions should say this consistently.
- Table 6’s null mean uses `10^-4` units while AIA and p95 are raw units. The caption explains this, but mixed units invite errors; report all in raw units.
- Table 9’s `−0.000` formatting is unacceptable for claims about differences at the `10^-4` to `10^-3` scale.
- Table 3 supports the stated quality values exactly.
- Table 4 supports the rounded gaps; exact values in prose use more precision than the table.
- Table 5 supports the rounded decision results; prose gives additional digits not displayed.
- The separate supplement provides these tables, but decisive evidence should still be summarized in the main paper and synchronized with it.

## PHASE 7 — RESULTS VALIDATION

The paper does not claim conventional percentage improvements for ActionShap. It primarily reports **absolute correlation differences** and **absolute ΔNDCG differences**. Treating these as percentages would be misleading. The table below recomputes relative changes only as an audit aid.

| Metric | Dataset | Baseline | Proposed | Claimed % | Actual % (Calculated) | Arithmetic Correct? | Statistically Meaningful? |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| NDCG@10, ItemKNN vs popularity | MovieLens | 0.158 | 0.284 | Not claimed; “exceeds” | +79.75% | YES | Clearly separated by ItemKNN CI, but paired popularity-difference test not reported. |
| HR@10, ItemKNN vs popularity | MovieLens | 0.320 | 0.504 | Not claimed; “exceeds” | +57.50% | YES | Same qualification. |
| NDCG@10, ItemKNN vs popularity | Amazon | 0.090 | 0.181 | Not claimed; “exceeds” | +101.11% | YES | Same qualification. |
| HR@10, ItemKNN vs popularity | Amazon | 0.155 | 0.285 | Not claimed; “exceeds” | +83.87% | YES | Same qualification. |
| Shapley AIA, bounded vs deletion | MovieLens | 0.762 | 0.779 | +0.017 absolute, not % | +2.23% relative | YES | Bootstrap CI `[+.013,+.022]` excludes 0; practical importance is modest. |
| Shapley AIA, bounded vs deletion | Amazon | 0.286 | 0.414 | +0.129 absolute, not % | +44.76% from rounded values | YES, allowing unrounded inputs | CI `[+.110,+.148]`; large but short/tied profiles complicate rank interpretation. |
| LIME AIA, bounded vs deletion | MovieLens | 0.951 | 0.933 | −0.018 absolute | −1.89% relative | YES | CI excludes 0 according to Table 4. |
| LOO AIA, bounded vs deletion | MovieLens | 1.000 | 0.978 | −0.022 absolute | −2.20% relative | YES | CI excludes 0; deletion value 1 is algebraic. |
| LIME AIA, bounded vs deletion | Amazon | 0.930 | 0.827 | −0.103 absolute | −11.08% relative | YES | CI excludes 0. |
| LOO AIA, bounded vs deletion | Amazon | 1.000 | 0.851 | −0.149 absolute | −14.90% relative | YES | CI excludes 0. |
| Maximum principal-method ΔNDCG difference | MovieLens | Shapley 0.0403 | LOO 0.0431 | At most 0.003 absolute | +6.95% relative to Shapley | YES (`.0028`) | Small effect; LIME–Shapley is statistically detectable but within ±.005 equivalence margin. |
| Maximum principal-method ΔNDCG difference | Amazon | LOO 0.0313 | Shapley 0.0333 | At most 0.003 absolute | +6.39% relative to LOO | YES (`.0020`) | Pairwise uncertainty overlaps; Shapley–LIME TOST supports equivalence under declared margin. |
| TOST margin fraction of typical effect | Both | 0.040 | 0.005 | “Approximately one eighth” | 12.5% | YES | Margin justification remains subjective and not externally preregistered. |
| Exact `B=2` action count at `n=20` | N/A | N/A | 211 | Not a % | `1+20+190=211` | YES | Deterministic combinatorial identity. |
| Exact `B=3` action count at `n=20` | N/A | N/A | 1351 | Not a % | `1+20+190+1140=1351` | YES | Deterministic combinatorial identity. |

### Statistical significance of numerical claims

Statistical significance is reported more carefully than in most XAI papers. Tables S3–S5 now expose Holm-adjusted paired comparisons and `d_z`; Tables S15–S17 add success/abstention intervals, paired tests, hierarchical user×seed intervals, and Shapley–LIME MDE values. These support the main primary-condition arithmetic. However, the raw matrices remain unavailable and several inconsistencies remain: the main text reports bounded-AIA MDEs of `±0.014` (MovieLens) and `±0.051` (Amazon), whereas Table S17 reports `0.008` and `0.032` for Shapley–LIME; TOST procedure/provenance remains under-specified; and multiple analysis populations conflict across captions and tables.

## PHASE 8 — STATISTICAL REVIEW

### Strengths

- Distinct users, not seed–user rows, are the unit of inference.
- Five seed records are averaged within user.
- User-bootstrap intervals use 10,000 draws.
- Paired randomization tests use 10,000 plus-one draws and Holm correction.
- Paired Cohen’s `d_z`, a Friedman omnibus test, valid-user intersections, and active-oracle denominators are specified.
- TOST is used rather than interpreting a nonsignificant difference as equivalence.
- Constant vectors are missing rather than imputed as zero.

### Missing or inadequate validation

1. **Training-data/model uncertainty:** One fitted similarity structure per condition is treated as fixed. User bootstrap CIs do not cover model retraining variability.
2. **Candidate-sampling uncertainty:** The same sampled candidates are fixed across methods and seeds. Results are conditional on one negative sample.
3. **Null exchangeability:** Shuffling effects across temporally ordered, popularity-varying players assumes a label-exchangeable chance model that is not justified.
4. **Short-vector Spearman behavior:** Amazon’s median `n=5` yields a coarse, tie-heavy statistic. No exact/tie-stratified uncertainty is shown.
5. **Per-user MC error:** Tables S21, S24, and S28 materially improve validation: exact-vs-MC action Jaccard is `0.969/0.963` on enumerable Amazon/MovieLens users, sign error is `0.002/0.019`, and cap-user cross-budget rank minima are `0.982/0.948`. Nevertheless, primary inference still treats estimated actions as fixed, and threshold coverage at the production budget remains incomplete.
6. **TOST margin provenance:** No external preregistration exists; “independently declared” must be documented with a timestamped protocol or softened.
7. **Power analysis:** Table S17 identifies the Shapley–LIME contrast but conflicts with main-text MDE values and still omits the formula, alpha/multiplicity convention, paired SD, and temporal provenance.
8. **Multiple robustness analyses:** Holm correction is described within narrow families, but the manuscript reports many datasets, models, utilities, histories, candidates, budgets, and strengths. Selective emphasis across families is not controlled.
9. **Conditional NRegret selection:** Comparing methods only on active-oracle users is coherent, but uncertainty in the active set and zero-inflated unconditional utility should also be modeled.
10. **Distributional summaries:** Means dominate despite right-skewed regret and zero-inflated effects. Medians/quantiles are in prose only for selected cases.
11. **Cluster dependence:** Users share item similarities and items; ordinary user bootstrap ignores dependence induced by popular shared items.
12. **No independent external replication:** The supplement adds internal replication families, but schema validation and same-team replications are not independent confirmation.
13. **Conflicting multiplicity families:** S4 and S16 report different Holm-adjusted p-values for the same success contrasts without clearly identifying which family is confirmatory.
14. **Permutation-resolution anomaly:** The protocol specifies 10,000 paired permutations, for a minimum plus-one p-value near `0.0001`, yet many strong tests are reported as exactly `0.0010`. This suggests 1,000 draws, censoring, or undocumented rounding.

### Mandatory statistical tests/analyses

- Hierarchical or repeated-split bootstrap over **training interactions/model fit, users, and candidate samples**.
- Candidate-redraw sensitivity with at least 20–30 independently sampled candidate sets.
- Exact or stratified randomization null preserving recency and item-popularity strata.
- Extend Tables S21/S28 from agreement diagnostics to direct propagation of attribution-estimation error into realized action effect and regret.
- Direct paired association analysis between **absolute bounded AIA** and decision quality, not only gap and regret.
- Utility-matched factorial analysis: attribution utility × outcome utility × method × dataset.
- Friedman or aligned-rank omnibus followed by Holm-adjusted paired tests, with all effect sizes and CIs in the main package.
- Publish one authoritative multiplicity map and exact raw permutation counts/p-values; explain the repeated `0.0010` floor.
- Coverage–risk analysis for abstention and unconditional zero-inflated decision outcomes.
- Sensitivity of TOST conclusions across a justified range of equivalence margins.
- Cluster-robust or item-block bootstrap sensitivity for shared-item dependence.

## PHASE 9 — EXPERIMENTAL VALIDATION

### Sufficiency assessment

- **Dataset diversity:** Insufficient for a high-impact general claim. Two primary datasets are old and domain-limited.
- **Bias:** Thresholding, 5-core filtering, uniform user sampling, and sampled negatives favor warm users/items. Exposure bias is discussed but not corrected because the target is an audit estimand; this still limits practical interpretation.
- **Temporal split:** Strong and clearly specified.
- **Cross-validation:** Not necessary in the classical sense, but multiple temporal cutoffs are needed to test stability.
- **Cold start:** Not evaluated; filtering actively removes the coldest users/items.
- **Hyperparameter fairness:** Table S19 adds equal-budget Shapley/LIME curves and parameter sensitivities, but primary headline methods still use unequal budgets and tuning provenance remains incomplete.
- **Runtime/memory:** Supplementary S6.1/S10 adds useful method and complete-pipeline timings; hardware-normalized runtime/RSS remains unavailable.
- **Hardware reproducibility:** Inadequate; CPU, RAM, and peak RSS are absent.
- **Architectural diversity:** Inadequate in the validated primary evidence. Reported sequential and graph models are below popularity or fail a gate.

### Mandatory additional experiments

1. **Competitive model replication:** Run the complete primary protocol on at least one tuned, quality-gate-passing sequential recommender and one tuned graph recommender that both outperform popularity and standard collaborative-filtering baselines.
2. **Utility-matched factorial experiment:** Compare target-margin and NDCG attributions against target-margin and NDCG oracles, with additive and interaction-aware selection. This isolates utility mismatch from nonadditivity.
3. **Prospective/non-target-conditioned audit in the main paper:** Explain actual top-`K` recommendations without using the held-out target during attribution or action selection; compare with the retrospective audit.
4. **Candidate and training uncertainty:** Repeat with many independently redrawn negative sets and independently retrained models/splits; report hierarchical CIs.
5. **Expanded compute-matched comparison:** Table S19 compares Shapley and bounded LIME budgets; extend the frontier to LOO/greedy, finite differences, integrated gradients, and KernelSHAP with wall-clock uncertainty.
6. **Complete exact-reference validation:** Tables S21/S28 report attribution/action agreement; add realized effect loss and regret inflation by profile length and convergence status.
7. **Fixed-background intervention:** Keep older history `B_u` in the score while treating only recent interactions as players; compare with the current truncated operating profile.
8. **Cold/long-tail strata:** Include low-history users before 5-core filtering where technically possible, and stratify by item popularity/exposure.
9. **Qualitative failure cases:** Show cases with positive gap but negative bounded AIA, high AIA but poor pair selection, and abstention failures.

## PHASE 10 — SUGGEST TARGET JOURNAL

### Primary recommendation: ACM Transactions on Recommender Systems

The manuscript is already formatted and framed as an ACM TORS article, and its main contribution is a recommender-specific evaluation protocol. TORS is a more natural audience than KBS in the current form.

### Alternatives

1. **ACM Transactions on Information Systems:** Appropriate if the authors expand ranking/explanation evaluation beyond recommendation and strengthen general IR relevance.
2. **User Modeling and User-Adapted Interaction:** Appropriate only after adding human/practitioner evaluation and user-centered actionability.
3. **Information Sciences:** Plausible for a broadened methodological treatment with stronger cross-model validation.
4. **Knowledge-Based Systems:** In scope because KBS explicitly includes recommender systems and decision support, but the paper must strengthen knowledge-based decision support, competitive AI methodology, and practical validation.

For KBS submission, convert from ACM to Elsevier format, remove “ACM Reference Format” and “Manuscript submitted to ACM,” provide the required title-page declarations and data statement, and submit 3–5 highlights of at most 85 characters each. The KBS guide encourages highlights and limits the abstract to 250 words.

## PHASE 11 — RELATED WORK & CITATIONS

### Coverage assessment

The review is broad, but the paper needs deeper comparison rather than more citation volume. Table 11 should be audited line-by-line, and recent methods should be implemented where they target the same estimand.

### Important additions

1. **Yeh et al., “On the (In)fidelity and Sensitivity of Explanations,” NeurIPS 2019** — formal fidelity/sensitivity evaluation.
2. **Slack et al., “Fooling LIME and SHAP,” AIES 2020** — fragility of local post-hoc explanation audits.
3. **Kumar et al., “Problems with Shapley-value-based explanations as feature importance measures,” ICML 2020** — limits of Shapley interpretation.
4. **Jeyakumar et al., “How Can I Explain This to You? An Empirical Study of Deep Neural Network Explanation Methods,” NeurIPS 2020** — human and functional evaluation.
5. **Balog and Radlinski, “Measuring Recommendation Explanation Quality: The Conflicting Goals of Explanations,” SIGIR 2020** — recommender-explanation evaluation objectives.
6. **Gedikli et al., “How Should I Explain? A Comparison of Different Explanation Types for Recommender Systems,” IJHCI 2014** — user-centered evaluation.
7. **Tintarev and Masthoff’s explanation-quality framework for recommender systems** — transparency, scrutability, trust, effectiveness, persuasiveness, efficiency, satisfaction.
8. **GREASE: counterfactual explanations for GNN-based recommender systems** — needed for graph-recommender positioning.
9. **Recent work on counterfactual explanation robustness and plausibility under distributional constraints** — to separate executable weights from realistic interventions.
10. **Recent uncertainty-aware XAI evaluation work** — confidence intervals and estimator uncertainty for feature attribution.
11. **Conditional/interventional Shapley literature** beyond “many Shapley values” — baseline semantics and dependence.
12. **Off-policy/recommender evaluation work on candidate sampling variability** — relevant to fixed sampled ranking.

### Citation-quality corrections

- Verify archival status and exact bibliographic metadata for every 2026 reference.
- Replace arXiv citations with archival versions where available.
- Correct inconsistent spaces around bracketed references throughout.
- Avoid using a citation to support a broader claim than its evaluated scope.
- Table 11 should use “not evaluated/reported” rather than “No” unless absence is proven.

## PHASE 12 — NOVELTY ANALYSIS

The paper’s novelty is **compositional and protocol-level**:

- bounded rather than deletion-only intervention semantics;
- explainer-independent common action policy;
- signed, abstention-aware budgeted selection;
- exact `B≤2` utility oracle and regret;
- within-user null calibration and distinct-user inference;
- explicit separation of alignment and decision quality.

No single component is fundamentally new. Weighted interventions, Shapley attribution, LIME/LOO, deletion audits, counterfactual recommender evaluation, abstention, exact small-budget search, bootstrap inference, and regret are established. The credible contribution is the disciplined integration of these pieces into one recommender-audit contract.

### Overlap/incrementality risks

- Recent counterfactual-explanation evaluation frameworks already compare explanation perturbations and recommendation changes.
- Refined fidelity metrics already question deletion conventions and contradictory features.
- RankSHAP and related work define ranking-aware characteristic functions.
- Recourse literature already emphasizes feasible action sets and costs.
- The paper does not introduce a new Shapley value or prove a new approximation guarantee.

### KBS novelty judgment

**Conditionally meets KBS novelty expectations, but not yet at the required evidential level.** The protocol combination is potentially publishable if the authors demonstrate that it changes scientific conclusions on competitive modern recommenders and if they release a complete reusable artifact. In the current evidence, the contribution risks appearing as a rigorous evaluation wrapper around ItemKNN rather than a broadly validated AI methodology.

## PHASE 13 — WRITING QUALITY

### Global issues

- Prose is precise but excessively dense.
- Parenthetical qualifications and long enumerations reduce readability.
- The text often anticipates reviewer objections, producing a defensive tone.
- “Executable,” “feasible,” “actionable,” “counterfactual,” and “intervention” need stricter separation.
- Primary, robustness, replication, and supplement-only evidence should be visually and verbally separated.
- Citation spacing and ACM boilerplate require correction.

### Three most problematic paragraphs and proposed rewrites

#### Problematic paragraph 1: Introduction hypothesis paragraph (page 2)

**Issue:** H1 mixes estimand distinction with method ranking; H2’s falsification condition does not map cleanly to the later analyses.

**Rewritten version**

> We test two preregistered-style hypotheses with explicit estimands. **H1** states that singleton deletion effects and singleton bounded-downweighting effects induce different player rankings within users. We test H1 using paired within-user differences between deletion AIA and bounded AIA, with confidence intervals and a random-attribution control. **H2** states that singleton bounded alignment is insufficient to predict the quality of a budget-two joint action. We test H2 by comparing method orderings under bounded AIA and oracle regret and by estimating, within each method, the association between bounded AIA and realized joint-action outcomes. Because attribution and outcome utilities may differ, we report utility-matched and cross-utility analyses separately. A positive bounded-minus-deletion difference is treated only as evidence of perturbation sensitivity; it is not interpreted as explanation quality.

#### Problematic paragraph 2: Convergence paragraph (pages 19–20)

**Issue:** “Mean-converged” language obscures low per-user coverage and the final-budget choice.

**Rewritten version**

> The population means pass the rank and Jaccard thresholds at `M_pair=50–250`, depending on dataset and model. This result does not establish reliable convergence for individual users: only 56–69% of rank-valid users satisfy both thresholds at the selected budgets. We therefore use `M_pair=500` as a conservative common budget, for which the corresponding per-user coverage is 74–85% in the supplementary quantile table. Because 15–26% of users still fail the joint criterion, all Shapley decision results should be interpreted as estimates with non-negligible action-selection error. We quantify this error against exact Shapley on enumerable profiles and report both attribution agreement and selected-action regret.

#### Problematic paragraph 3: Practitioner-use paragraph (pages 22–23)

**Issue:** Long, promotional, and too confident about operational value without deployment evidence.

**Rewritten version**

> The protocol is relevant when a recommender exposes a validated inference-time mechanism for reducing the influence of selected historical interactions. In that setting, deletion and bounded downweighting need not rank interactions identically, so a deletion audit may not predict the effects of the available operation. ActionShap evaluates three distinct properties: singleton alignment with bounded effects, the quality of a budgeted joint action relative to an in-simulator oracle, and population-level evidence beyond a declared chance model. These quantities remain conditional on the frozen scorer, candidate set, target, and intervention simulator. They do not establish user agency, causal effects, legal recourse, or net benefit in deployment; those claims require authority, consent, cost, and online feedback analyses.

## PHASE 14 — KNOWLEDGE-BASED SYSTEMS COMPLIANCE

| Criterion | Score (/10) | Justification |
| --- | ---: | --- |
| Scientific contribution | 7 | Clear evaluation problem and coherent integrated protocol; no new recommender or attribution theory. |
| AI methodology | 6 | Sound audit mechanics, but effective primary evidence is ItemKNN and modern models fail quality gates. |
| Experimental rigor | 7 | Strong user-level statistics, ablations, and exact checks; weak competitive architecture diversity and unresolved cross-document inconsistencies. |
| Explainability | 7 | Multiple functional faithfulness dimensions; no human comprehensibility/usefulness evidence. |
| Ethics & Responsible AI | 7 | Careful non-causal disclaimers, conflict/funding/AI-use statements; no deployment harm or consent study. |
| Reproducibility & Open science | 6 | Supplement supplies extensive generated results and costs, but placeholder URL and missing code/raw archive prevent reproduction. |
| Novelty | 7 | Novel protocol combination, incremental components. |
| Practical impact | 5 | Plausible audit use, but only simulator-executable and not validated in production. |

### Format compliance

- The manuscript is in ACM format, not KBS/Elsevier format.
- “ACM Reference Format” and repeated “Manuscript submitted to ACM” text must be removed.
- KBS highlights are not supplied.
- The permanent/reviewer artifact URL is a placeholder.
- The abstract appears within KBS’s 250-word limit, but authors should verify exact count after revision.
- Data and code statements are present but not operational until links and deposits exist.
- AI-assisted language/formatting disclosure is present and should be adapted to Elsevier’s declaration format.

## PHASE 15 — CRITICAL ISSUES

### Issue #1

- **Severity:** Critical
- **Location:** Section 3.2, Eqs. (2)–(3), page 6
- **Explanation:** Scores divide by the sum of retained weights. Downweighting one interaction therefore raises every other interaction’s normalized share; uniform positive scaling leaves the score unchanged.
- **Impact:** The intervention is relative profile-mass reallocation, not isolated “discounting” or “suppression.” The headline deletion-versus-bounded contrast may substantially reflect normalization.
- **Recommended Fix:** Compare normalized reweighting with an unnormalized or fixed-denominator intervention; justify which corresponds to an operational scorer; rename the current operation “relative profile reweighting.”

### Issue #2

- **Severity:** Critical
- **Location:** Sections 5–7; Supplementary S6/S9 claims
- **Explanation:** Only ItemKNN provides quality-gated primary evidence. Profile, SASRec-style, and LightGCN models reportedly underperform popularity and/or fail a gate.
- **Impact:** General claims about recommendation explanations, sequential models, graph recommenders, and architecture robustness are not established.
- **Recommended Fix:** Complete the full protocol on competitive, gate-passing sequential and graph recommenders and move core results into the main paper.

### Issue #3

- **Severity:** Critical
- **Location:** Section 6.6, Tables 8–9, pages 19–21
- **Explanation:** In the 1,000-user Amazon full-catalogue audit, Shapley bounded AIA is negative (`−0.05 [−0.09,−0.01]`). Its positive gap (`+0.16`) occurs only because deletion AIA is more negative (`−0.21`).
- **Impact:** The strongest robustness result reverses the sampled-candidate headline. A positive gap can mean “less anti-aligned,” not valid bounded prediction.
- **Recommended Fix:** Promote this result to the abstract, results summary, discussion, and conclusion; center absolute bounded AIA rather than the gap.

### Issue #4

- **Severity:** Critical
- **Location:** Sections 3.3–4.4 and 6.4
- **Explanation:** Target-margin attributions select actions evaluated under NDCG. This changes singleton to joint action and target-margin to NDCG simultaneously.
- **Impact:** H2 does not isolate interaction/nonadditivity from utility mismatch.
- **Recommended Fix:** Run a full attribution-utility × outcome-utility × additive/interaction-aware factorial analysis.

### Issue #5

- **Severity:** Critical
- **Location:** Sections 1, 3.3, 7.1.3
- **Explanation:** The audit is retrospective and uses the held-out target to define the explanation, even when that target is not in the top 10.
- **Impact:** It may evaluate how to promote a future item rather than explain an actual recommendation.
- **Recommended Fix:** Add a prospective audit of actual top-ranked outputs as a co-primary experiment and explicitly separate explanation from recommendation correction.

### Issue #6

- **Severity:** Critical
- **Location:** Sections 1, 3.3, 7
- **Explanation:** “Executable” means executable in a custom simulator. No deployed system, API constraint, latency requirement, persistence rule, consent mechanism, or empirical operating range validates fractional history weights.
- **Impact:** The central actionability motivation may be a reparameterized perturbation rather than an operational intervention.
- **Recommended Fix:** Use “simulator-executable” throughout unless a real interface/case study is demonstrated; justify feasible `rho`, costs, budget, and authority.

### Issue #7

- **Severity:** High
- **Location:** Section 3.2, Eq. (4), pages 6–7
- **Explanation:** The displayed BPR regularizer’s scope conflicts with the prose’s per-gradient-step implementation; context averaging, duplicate embeddings, record/item exclusion, and clipping derivatives are unspecified.
- **Impact:** The profile model’s objective and gradients are not mathematically reproducible.
- **Recommended Fix:** Rewrite the per-triple loss and exact SGD updates, including clipping semantics and regularization normalization.

### Issue #8

- **Severity:** High
- **Location:** Sections 3.1, 5.2, 6.6
- **Explanation:** Primary ranking uses one fixed target-plus-199-negative sample. Five attribution/model seeds do not capture candidate-set uncertainty, and full-catalogue evaluation changes Shapley bounded AIA’s sign.
- **Impact:** Main conclusions are highly conditional on candidate construction.
- **Recommended Fix:** Use repeated independent candidate samples, report between-candidate variability, and make full-catalogue evaluation primary where feasible.

### Issue #9

- **Severity:** High
- **Location:** Sections 6.5; Table 7; Algorithm 4
- **Explanation:** Population-threshold coverage is 56–69% at selected budgets and 74–83% for primary ItemKNN at `M_pair=500`. Tables S21/S28 show strong exact/cross-budget action-rank agreement, which mitigates but does not propagate estimator uncertainty into outcomes.
- **Impact:** Aggregate conclusions are more credible than the main PDF alone suggested, but individual decision uncertainty remains understated.
- **Recommended Fix:** Propagate MC uncertainty into effect/regret and stratify outcomes by convergence; retain adaptive stopping for unstable users.

### Issue #10

- **Severity:** High
- **Location:** Sections 5.3, 6.3–6.4; Tables 2 and 5
- **Explanation:** Analysis populations are inconsistent or ambiguous: 196 Amazon NDCG-active users are alternately referenced against 1,000 and 993; the NDCG TOST excludes constant-AIA users without clear necessity; normalized-regret correlations use `n=1000/987`, incompatible with NDCG active-oracle counts `339/196`; S22 calls its sample 200 users while rows reach 400; and S25’s Amazon `n=810` is unexplained.
- **Impact:** The utility and denominator of key inferential claims cannot be determined.
- **Recommended Fix:** Label every statistic by utility and exact analysis population; provide an inclusion-flow table and all-user sensitivity.

### Issue #11

- **Severity:** High
- **Location:** Sections 5.3, 7.1.6
- **Explanation:** Uncertainty is conditional on one fitted scoring structure, one cohort draw, and fixed candidate/tie samples.
- **Impact:** CIs and p-values understate total experimental uncertainty and may not generalize to candidate or retraining variation.
- **Recommended Fix:** Use hierarchical resampling/retraining, cohort resampling, and candidate redraws.

### Issue #12

- **Severity:** High
- **Location:** Eq. (20), Algorithm 3, Table 6
- **Explanation:** Within-user effect shuffling assumes an exchangeable chance assignment across temporally and semantically heterogeneous interactions.
- **Impact:** Null p-values may be anti-conservative or test an unrealistic null.
- **Recommended Fix:** Define the null formally and add recency/popularity-stratified or conditional randomization controls.

### Issue #13

- **Severity:** High
- **Location:** Section 4.2; baselines
- **Explanation:** Primary methods have different scorer budgets, and Table S19 incorrectly labels separate curves as “equal-scorer-budget”: its paired Shapley/LIME columns use about 20–33× different scorer calls.
- **Impact:** The supplement does not provide a valid compute-matched comparison, although bounded LIME is stronger across its own budget curve.
- **Recommended Fix:** Compare identical scorer-call counts and extend the frontier to all baselines with wall-clock uncertainty.

### Issue #14

- **Severity:** High
- **Location:** Sections 4.3 and 6.4
- **Explanation:** The manuscript claims Spearman AIA is invariant to monotone utility re-expression, which is false after absolute baseline differencing, and tests the gap rather than absolute bounded AIA against decision quality.
- **Impact:** Metric justification and H2 adjudication are incomplete.
- **Recommended Fix:** Correct the invariance claim and report utility-matched, within-method associations between absolute bounded AIA and joint outcomes.

### Issue #15

- **Severity:** High
- **Location:** Sections 4.2 and 5.3
- **Explanation:** The stated random-control integer seed can collide across adjacent user/experiment-seed pairs. In addition, the sign-flip test requires exchangeability/symmetry assumptions that are not stated.
- **Impact:** Random-control dependence and inferential validity cannot be fully assessed.
- **Recommended Fix:** Use tuple-based seed derivation and document/validate the paired randomization assumptions or use a studentized robust alternative.

### Issue #16

- **Severity:** High
- **Location:** Sections 5.4, 6, 10; whole manuscript
- **Explanation:** The supplement is now supplied, but the artifact URL, source revision, code, manifest, and raw matrices remain absent; main/supplement claims conflict; the document is explicitly ACM rather than KBS format.
- **Impact:** Printed tables can be reviewed, but results cannot be regenerated and the authoritative experimental version is unclear.
- **Recommended Fix:** Reconcile both PDFs, provide an immutable artifact, and convert to Elsevier/KBS format.

### Issue #17

- **Severity:** High
- **Location:** Main Sections 5.3/6.6/7.1 and Supplement Tables S10, S15, S17, S29
- **Explanation:** The two PDFs are not version-consistent. Main-text bounded-AIA MDEs are `0.014/0.051`, while S17 gives `0.008/0.032`; S15’s caption says Amazon uses 993 users while every Amazon row uses 1,000; the main paper says exhaustive `B=3` is future work while S29 reports it completed; and “smaller full-catalogue subset” language conflicts with the main 1,000-user Amazon table.
- **Impact:** Readers cannot identify the authoritative analysis plan, cohort, or result version.
- **Recommended Fix:** Generate both documents from one versioned result manifest, add a cross-document consistency test, and state the exact revision/hash in each PDF. Recompute S17: conventional 80%-power calculations from S3’s differences and `d_z` reproduce the main values (`≈.014/.051`), so S17 must document or correct its alternative.

### Issue #18

- **Severity:** High
- **Location:** Supplement Tables S3–S4, S11, S14, S16; main Section 5.3
- **Explanation:** The same success contrast receives different Holm-adjusted p-values under undocumented competing families (MovieLens Shapley–LIME `.0066` in S4 versus `.0216` in S16). Moreover, many tests equal `.0010` despite a declared 10,000-permutation procedure whose minimum plus-one p is approximately `.0001`.
- **Impact:** Confirmatory significance and multiplicity control are not reproducible from the reported protocol.
- **Recommended Fix:** Publish the exact family membership and raw exceedance count for every test, identify the sole confirmatory family, and explain any p-value censoring/rounding or actual draw count.

### Issue #19

- **Severity:** High
- **Location:** Supplement Tables S18–S20 and S26–S27
- **Explanation:** Modern-model cells are both noncompetitive and estimator-unstable: SASRec exact-vs-MC Shapley rank agreement is only `.395` on Amazon and `.688` on MovieLens; LightGCN remains below popularity; tables lack CIs/sample sizes; and “gate pass” refers to masking rather than recommendation quality.
- **Impact:** Claims of architectural transfer/generalization may reflect weak recommenders and Shapley estimation error.
- **Recommended Fix:** Re-run on competitive models, use exact/adaptively converged attribution where possible, add complete inference, and separate masking responsiveness from recommendation-quality passage.

## PHASE 16 — MINOR ISSUES

- Title: consider removing “Shap” from the protocol name to avoid implying a new Shapley algorithm.
- Abstract: replace “deployed systems may” with a qualified, cited statement.
- Abstract: write “absolute ΔNDCG@10” when reporting `.003`.
- Throughout: standardize “full catalogue,” “full-catalogue,” and “full unseen catalogue.”
- Throughout: standardize `ItemKNN` capitalization; Table 7 uses `itemknn`.
- Throughout: standardize `n_max`, `M_pair`, and Greek `rho` typography in figures.
- Throughout: remove spaces inside citation brackets.
- Section 2: distinguish explanation generation, explanation evaluation, and recourse more consistently.
- Table 1: define “Method-spec.” and “Optim. target.”
- Section 3.1: reconcile “minimum possible player count is two” with “observed minimum is three” whenever strata begin at 3–5.
- Section 3.1: explain whether MovieLens repeated user–item events exist.
- Eq. (2): state similarity range and how zero co-occurrence is stored.
- Eqs. (2)–(3): do not describe normalized reallocation as isolated suppression; add the coefficient-change derivation.
- Eq. (3): define item embedding initialization notation as `N(0,0.05^2)` rather than potentially ambiguous `N(0,0.05²)` if the second parameter is variance.
- Eq. (4): clarify whether `λ/2` applies to the entire bracket.
- Eq. (4): specify whether margin clipping is differentiated through or used only for stable sigmoid evaluation.
- Eq. (5): define `TopL` when candidate count is below 11.
- Eq. (6): define `τ` as injective.
- Eq. (7): improve typesetting of the indicator-times-discount expression.
- Section 3.3: use “simulator intervention” consistently instead of alternating with feasible action.
- Algorithm 1: explicitly return diagnostics.
- Algorithm 2: include scorer/model among required inputs.
- Algorithm 3: specify percentile endpoints and RNG stream.
- Algorithm 4: state whether candidate budgets use nested samples.
- Table 2: add IQR and active-oracle percentage.
- Section 5.1: provide per-step filtering counts in the paper or supplement.
- Section 5.2: explain why the masking gate thresholds are scientifically meaningful.
- Table 3: label candidate-set size directly in each row.
- Section 5.3: “post-hoc minimum-detectable-effect” should not be presented as prospective power.
- Section 5.3: define the exact TOST procedure, pair-specific SD/MDE calculation, alpha after correction, and analysis population.
- Section 4.2: replace additive integer seed construction with collision-resistant tuple-based seed derivation.
- Section 5.4: processor model, RAM, and peak RSS should be recorded in regenerated runs.
- Section 5.4: verify all future-version package numbers against a lockfile.
- Figure 2: enlarge row labels and use marker shapes.
- Table 4: include deletion AIA CIs rather than deferring them.
- Main Figure 2/Table 4: correct the claim that deletion-AIA CIs are in S7; Table S9 contains no CIs.
- Table 5: include CIs and equivalence annotations in the table.
- Section 6.3: report exact adjusted p-values for MovieLens LIME–Shapley.
- Section 6.3: distinguish statistical superiority from practical equivalence in one sentence, not two separated clauses.
- Figure 3: add Amazon.
- Section 6.4: explain why Amazon gap–effect correlation `p=.002` is not emphasized after multiplicity correction.
- Table 6: avoid mixed raw and `10^-4` units.
- Table 7: report final `M_pair=500` rows, not only selected lower budgets.
- Main Section 6.5: change the convergence-quantile cross-reference from S8 to S9/Table S24.
- Figure 4: add uncertainty and per-user coverage.
- Table 8: visually flag negative bounded AIA.
- Table 8: clarify whether the 883 target-margin or 54 NDCG active-oracle denominator applies to each statistic.
- Table 9: replace `−0.000` with sufficient precision.
- Table 10: mark `n<30` strata directly.
- Table 11: use “not reported” instead of “No” where appropriate.
- Section 7: shorten the practitioner scenario and avoid promotional phrasing.
- Section 7.1.4: do not call below-popularity models evidence of generalization.
- Section 7.1.4: reconcile “full-catalogue subset is smaller” with the reported 1,000-user Amazon full-catalogue audit.
- Section 7.1.6: reconcile “one frozen trained recommender per model” with five model seeds and seed-specific gate outcomes.
- Section 7.1.5: schema validation cannot prove absence of earlier analytic selection.
- Section 8: qualify “fully specified” until the artifact is public.
- Section 9: state dataset licences/terms with URLs.
- Section 10: replace placeholder artifact sentence before any submission.
- Acknowledgments: adapt funding, interests, CRediT, and AI-assistance declarations to KBS/Elsevier forms.
- References: verify author names “Amin” versus “Amir Reza” Mohammadi where intended.
- References: replace available arXiv records with archival citations.
- Appendix C: `R_null=10^3` is consistent with 1,000; ensure superscripts render correctly.
- Table B.1: move complete pre-test exclusion, no retraining, and Holm correction into the primary-setting column; state that `B=3` uses a greedy lower bound, not an exact oracle.
- Supplement S6: replace “Appendix S6” with “Section S6.”
- Supplement S3–S5: remove redundant heading labels immediately preceding full captions.
- Supplement throughout: standardize MovieLens/ML-1M/MovieLens-1M, ItemKNN/itemknn, Shapley abbreviations, and human-readable condition labels.
- Supplement tables: increase the approximately 5.5–6-point text size; S2’s sideways landscape running headers are especially difficult to read.
- PDF accessibility: add document tagging, semantic table structure, figure alt text, and correct math-glyph mappings in the text layer.
- PDF: remove line-number debris and ACM footer/header artifacts in the KBS version.

## PHASE 17 — FINAL DECISION

### Major strengths

- Important distinction between deletion faithfulness and bounded intervention validity.
- Explicit temporal, candidate, tie, seed, and action-space controls.
- Correct separation of singleton alignment and joint decision regret.
- Exact budget-two oracle including no action.
- Distinct-user inference, missingness rules, random control, bootstrap CIs, Holm correction, effect sizes, and TOST.
- Honest negative results: Shapley is not claimed to be universally superior.
- Strong limitations section and explicit non-causal scope.

### Major weaknesses

- One effective primary recommender architecture.
- Normalized weighting reallocates profile mass and does not isolate evidence suppression.
- Full-catalogue Amazon results reverse Shapley bounded alignment from positive to negative.
- Simulator-only actionability.
- Retrospective target conditioning rather than direct explanation of actual recommendations.
- Utility mismatch confounds the claimed pointwise-versus-joint distinction.
- Ambiguous/inconsistent utility-specific analysis denominators.
- Eq. (4) and the random-control seeding scheme are not reproducible as written.
- Residual per-user MC uncertainty despite strong supplementary exact/cross-budget diagnostics.
- Fixed-model and fixed-candidate uncertainty.
- Questionable player-exchangeability null.
- Bounded/path baselines and equal-budget evidence are relegated to the supplement.
- Missing executable/raw artifact and placeholder URL.
- Main/supplement version drift, including MDE, cohort, full-catalogue, and `B=3` inconsistencies.
- ACM rather than KBS submission format.

### Mandatory revisions

- Validate on competitive, gate-passing neural and graph recommenders.
- Compare normalized relative reweighting with unnormalized/fixed-denominator suppression.
- Make the full-catalogue reversal central to the abstract and conclusions.
- Add prospective actual-recommendation audits.
- Run utility-matched and interaction-aware factorial experiments.
- Propagate candidate, training, and Shapley-estimation uncertainty.
- Correct Eq. (4), seed derivation, metric-invariance claims, and all utility-specific denominators.
- Replace or validate the shuffle null.
- Move compute-matched, intervention-aware baselines into the primary analysis.
- Extend exact-subset validation to realized effect/regret uncertainty.
- Supply an immutable reproducibility artifact with raw records.
- Add full complexity/runtime/memory analysis.
- Convert to KBS format and moderate actionability terminology.

### Optional improvements

- Human or practitioner study.
- Heterogeneous action costs and authority constraints.
- Online/off-policy extension.
- Additional target items per user and multiple temporal cutoffs.
- Interactive audit visualization and qualitative case studies.

### Publication recommendation

**Major Revision.** The conceptual protocol is promising and much of the mathematics is sound, but the empirical and construct-validity gaps are too substantial for acceptance in KBS in the current form.

### Confidence in review (1–10)

**9/10.** Confidence is high after reviewing both PDFs. It remains below 10 because machine-readable results, code, and the authoritative content-addressed artifact are unavailable.

### Prioritized Revision Checklist

#### Critical

- [ ] Test unnormalized/fixed-denominator suppression and accurately name the normalized intervention.
- [ ] Demonstrate the full protocol on competitive, quality-gate-passing sequential and graph recommenders.
- [ ] Promote the negative full-catalogue Shapley AIA result into the abstract and principal conclusion.
- [ ] Add utility-matched attribution/outcome experiments and isolate interaction effects.
- [ ] Add a prospective, non-target-conditioned audit of actual recommendations.
- [ ] Replace “executable/actionable” with “simulator-executable” unless operational feasibility is empirically demonstrated.
- [ ] Reconcile the main paper and supplement, then provide an immutable artifact URL, code, configs, raw outputs, and regeneration instructions.

#### High

- [ ] Propagate uncertainty over retraining, users, candidate redraws, and Shapley estimation.
- [ ] Correct the BPR objective/gradients, collision-prone seed derivation, and analysis-population inconsistencies.
- [ ] Extend the strong S21/S28 convergence checks to action-effect/regret uncertainty and adaptive handling of unstable users.
- [ ] Validate or redesign the within-user null to respect temporal/item heterogeneity.
- [ ] Move the compute-matched bounded/path-aware comparisons from S18/S19/S27 into the primary analysis.
- [ ] Directly test bounded AIA versus decision quality.
- [ ] Enumerate or multiplicity-correct LIME masks for short profiles.
- [ ] Add end-to-end complexity, runtime, scorer-call, cache, and memory results.
- [ ] Move all confirmatory adjusted tests and decisive ablations into the reviewed package.

#### Medium

- [ ] Add fixed-background history sensitivity.
- [ ] Expand dataset diversity and cold/long-tail analyses.
- [ ] Add qualitative success/failure cases and actual recommendation examples.
- [ ] Audit Table 11 and deepen related-work comparisons.
- [ ] Report distributional summaries and coverage–risk curves.
- [ ] Document equivalence-margin provenance and test margin sensitivity.

#### Low

- [ ] Simplify dense prose and shorten defensive caveats.
- [ ] Improve figure legibility and add the Amazon decision panel.
- [ ] Increase numerical precision in near-zero tables.
- [ ] Standardize notation, capitalization, hyphenation, and citation spacing.
- [ ] Convert ACM boilerplate, declarations, references, and highlights to KBS/Elsevier requirements.
