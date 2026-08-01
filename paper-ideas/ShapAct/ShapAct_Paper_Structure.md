# ShapAct — Full Paper Structure, TOC & Embedded Content

**Target journal:** *Discover Artificial Intelligence* (Springer Nature, open access, Q1 — Information Systems)
**Article type:** Research article
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco
**Corresponding author:** mouad_louhichi@um5.ac.ma

**Structure:** six-section layout of *SignalShap* and of *Game Theory Meets Explainable AI* (IJACSA 2025), not the nine-section DyHuCoG layout. This is a companion paper to **SignalShap** and is built on its pipeline.
**Target:** ≈ 8,500–9,500 words, 7 figures, 8 tables.

> **Why this is the "easy-to-accept" paper.** It invents **no new architecture, no new dataset, no new estimator, and no new game**. It reuses the SignalShap exact source-attribution game (five players, 32 coalitions, laptop CPU) and its entire preprocessing pipeline, and adds one new experimental layer — a *construction-level intervention audit* that executes the decommissioning decisions the attribution recommends and measures whether the realized change matches the predicted change. The novelty is in *what is measured*, not in how anything is computed. It answers the question the group's own thesis declares open (thesis Definition 1.1 defines "actionable insight" and concedes in §1, §8.3, §9.4 that it is "a framing concept rather than a separately measured endpoint") — but **after the publication of the group's ActionShap**, which now claims that gap at the factor/interaction level. ShapAct's remaining niche is the *construction level*: ActionShap's interventions perturb a factor inside a trained model; ShapAct's interventions remove a whole pipeline from the system's construction. Those are different decisions, measured differently, and ActionShap's own codebase does not contain a never-built counterfactual.

> **Critical dependency note (read before starting).** ShapAct is a **companion to two papers, sequenced after both**. (1) **SignalShap** supplies the exact source game: consume its pipeline's outputs (cached per-source score columns, the 32 coalition values, per-user decompositions) and do **not** start until its gates (Implementation Spec §B.7, milestones 1–4) have passed. (2) **ActionShap** (now in `paper-ideas/ActionShap/`) supplies the actionability-measurement paradigm (AIA, top-$k$ precision, regret, A-Shapley reranking) and the reference implementation of it (`code/actionshap/`: `metrics.py`, `attribution.py`, `stats.py`, the modifiability freeze machinery); ShapAct reuses that paradigm at the source level and must be written as its explicit companion, not as a competitor. **Nothing from the DyHuCoG codebase is used** — and this is now *load-bearing twice over*: SignalShap's extraction audit already documented 63 gaps in the DyHuCoG paper, and the full audit lives in `paper-ideas/ActionShap/code/docs/dyhucog_spec.md`, whose verdict is that "this paper cannot be reimplemented faithfully from its own contents." ActionShap's own plan to "reuse both existing codebases" therefore inherits an unresolved risk on the dynamic side; ShapAct avoids it entirely by hosting the audit on SignalShap's vetted, reproducible exact pipeline. The `stats.py` and clustering diagnostics that SignalShap's spec assumed external to this repository are now actually present at `paper-ideas/ActionShap/code/actionshap/` and port directly.

---

## Working Title (primary + alternates)

- **Primary (selected):** *ShapAct: A Construction-Level Audit of Exact Shapley Source Attribution — Fidelity, Order, and Reflexivity of Decommissioning Decisions in Hybrid Recommenders*
- Alt 1: *After ActionShap: What the Attribution Predicts About Retiring a Source vs. What Retiring It Does*
- Alt 2: *Attributions That Survive Their Own Execution: Construction Counterfactuals for Exact Source Credit*
- Alt 3: *From Factor-Level Actionability to Construction-Level Decisions: An Audit of Source Retirement*

*(The primary title names the actual angle — the **construction-level** audit (never-built counterfactual), which ActionShap's factor-level feasible-intervention paradigm does not cover — rather than restating the subject area. **ShapAct** is retained as the method/audit-framework name and the codebase name. The name is deliberately distinct from the group's existing **ActionShap** paper and codebase (`paper-ideas/ActionShap/`); the two must never be conflated in text or code comments, and ShapAct's manuscripts must cite ActionShap as the paradigm paper it extends.)*

## One-paragraph thesis (the spine)

The group's own ActionShap paper has now claimed the general gap this line of work opened: it operationalizes the thesis's unmeasured "actionable insight" definition into computable metrics (Actionability Score, Attribution–Intervention Alignment, top-$k$ intervention precision, regret) and tests, across four datasets and six attribution methods, whether high attribution predicts the realized effect of a *feasible factor-level intervention* — perturbing a feature or an interaction by an amount a decision-maker can realize. What no paper in the folder — including ActionShap — measures is the *construction level*: what happens when the recommended action is not "modify a factor" but "retire a whole signal pipeline," which is the decommissioning decision the group's own SignalShap draft already names as the practical use of source credit (SignalShap §5.1). Retiring a source is a different intervention class: the source is never trained, never scored, never contributes candidates — a counterfactual about the system's *construction*, not about its inputs. ShapAct is the audit for that class. On SignalShap's exact five-source game (which ActionShap's MC-based dynamic regime does not have), it measures the **fidelity gap** between the in-game masked marginal and the realized never-built effect, the **order validity** of the credit ranking as a decommissioning rule, the **reflexivity** of the attribution after its own recommendation is executed (a quantity no file in the folder defines), and the **realized quality** of Shapley-based vs. leave-one-out vs. feature-SHAP decommissioning decisions. This is the intervention axis ActionShap's paradigm leaves to the construction level, made measurable on the one pipeline in the group that is exact and reproducible.

## Research questions of *this paper* (and their link to the thesis)

| RQ | Question | Thesis link |
|---|---|---|
| **RQ1** | Does exact source-level attribution predict the **realized** effect of actually retiring a source (the never-built counterfactual), and how large is the fidelity gap relative to the in-game masked marginal? | Operationalizes Definition 1.1's two clauses: "identifies a modifiable factor" (the source) and "change associated with a specifiable change in model output" (the predicted vs. realized ΔNDCG@10). ActionShap measures the same clauses at *factor level*; ShapAct is the *construction-level* instance, which ActionShap's intervention set (reweight an interaction, alter a context signal, adjust exposure — ActionShap §6.3) does not contain |
| **RQ2** | Is the Shapley credit ranking of sources **order-valid** as a decision rule — does it rank realized retirement effects correctly? | The thesis-level "actionable insight" claim (§8.1) made testable at decision level; complements ActionShap's AIA (which scores an *ordering*) with a *choice* test over the exact game (ActionShap's regret is computed over factors, not over source-retirement outcomes) |
| **RQ3** | Does the attribution remain valid **after** its own recommendation is executed (reflexivity), or does it describe a system state that acting on it destroys? | Extends the thesis's attribution→intervention loop (RQ3, §9.1) in a direction ActionShap does not define: ActionShap's stability (§4.4) is dispersion across repeated *attribution* runs; ShapAct's reflexivity is the *recomputed attribution* of the intervened system |
| **RQ4** | Which explanation design — exact source Shapley, leave-one-out, or feature-level SHAP on the fusion — produces decisions with the highest **realized** quality? | Decision-level companion to ActionShap's method comparison: ActionShap compares attribution *methods* by the intervention their top factor induces; ShapAct compares *decision rules* by the realized outcome of the source they retire — and only ShapAct tests the LOO-vs-Shapley divergence SignalShap Proposition 2 predicts at the decision level |

---

# TABLE OF CONTENTS

```
Abstract / Keywords
1. Introduction
   1.1 Background and motivation
   1.2 Attribution explains what is, not what happens when you act
   1.3 Why an intervention audit (and why now)
   1.4 From SignalShap to ShapAct
   1.5 Contributions
   1.6 Organization
2. Literature Review
   2.1 Cooperative-game attribution in recommendation
   2.2 Source-level attribution and hybrid fusion
   2.3 Evaluation of explanations: faithfulness, stability, usefulness
   2.4 Counterfactual and causal intervention evaluation
   2.5 Positioning and differentiation (comparison table)
3. Methodology
   3.1 Notation and problem formulation
   3.2 The reused source-attribution game
   3.3 Three counterfactual levels (masked / regenerated / never-built)
   3.4 Actionability measures: fidelity, order validity, reflexivity
   3.5 Decision rules under test
   3.6 Theoretical justification
   3.7 Complexity analysis
   3.8 Practical implementation
4. Experimental Results
   4.1 Datasets and preprocessing
   4.2 Protocol, metrics, and baselines
   4.3 RQ1 — intervention fidelity
   4.4 RQ2 — order validity of the credit ranking
   4.5 RQ3 — post-intervention reflexivity
   4.6 RQ4 — realized quality under alternative decision rules
   4.7 Sensitivity, stability, and ablations
   4.8 Statistical significance
5. Discussion and Broader Implications
   5.1 When attribution misleads: the measured conditions
   5.2 The decision-support contract for recommendation explanations
   5.3 Relation to feature-level, population-level, and temporal explanation
   5.4 Limitations and threats to validity
6. Conclusion and Future Work
Declarations
References
Appendix A — Proofs
Appendix B — Full counterfactual tables (L0 / L1 / L2, per source, per dataset)
Appendix C — Hyperparameters
Appendix D — Full-retraining robustness (never-built for all five sources)
Appendix E — Decision-rule results in full
Notation list
```

---

# ABSTRACT (draft, ~215 words)

Shapley-based explanations of recommender systems are routinely described as actionable, but "actionable" is asserted, never tested: no evaluation executes the decision an attribution recommends and checks whether the realized change matches the predicted change. This is exactly the gap the literature itself names — our prior work defines actionable insight as an explanation that identifies a modifiable factor whose change produces a specifiable change in model output, then concedes that this criterion is used as a framing concept rather than a measured endpoint. We introduce **ShapAct**, an intervention audit built on an exact five-source Shapley game over the signal sources of a hybrid recommender. For each source we evaluate three counterfactual levels — masking the source at fusion time, regenerating the candidate set without it, and never building it — and measure the fidelity gap between predicted and realized retirement effects, the order validity of the credit ranking as a decision rule, and the reflexivity of the attribution after its own recommendation is executed. On MovieLens-1M and Amazon-Book we find that masking systematically overstates the value of sources with close substitutes (the fidelity gap is largest for a redundant popularity/recency pair), that exact Shapley decisions realize higher ranking quality than leave-one-out or feature-level rules, and that attribution is largely stable under its own execution. ShapAct converts "actionable" from a claim into a measured, falsifiable property.

**Keywords:** Shapley value · Cooperative game theory · Explainable AI · Recommender systems · Actionable insight · Intervention evaluation · Credit assignment · Counterfactual evaluation

---

# 1. INTRODUCTION

## 1.1 Background and motivation

The group's research programme, as consolidated in the thesis, argues that Shapley-value attribution is "a mathematically founded and operational mechanism for providing actionable insight" in complex AI systems (thesis abstract; resume p. 1). The thesis even defines the term precisely:

> **Thesis Definition 1.1 (Actionable insight).** An explanation is actionable when it identifies at least one modifiable factor whose change is associated with a specifiable change in model output, and when that factor is accessible to the relevant decision-maker.

Then it immediately bounds the claim: "In the present thesis, however, actionable insight is used primarily as a framing concept rather than as a separately measured endpoint" (thesis §1; repeated in §8.3 and §9.4, where "whether the explanations developed in this thesis measurably improve analyst judgement, user trust, intervention quality, or perceived fairness" is listed as *future work*). Every recommendation paper the group has produced since inherits this framing: DyHuCoG's Shapley weights "provide an interpretable lens" (DyHuCoG §3.2, §4.7.2), SignalShap's source shares are "decision-relevant" (SignalShap §3.2, §5.1), FairShap's exposure attribution drives fairness corrections. None of them — nor, to our knowledge, the evaluated literature — runs the experiment Definition 1.1 calls for: take the recommended action, and check whether the specifiable change materializes.

ShapAct is that experiment, made into a paper.

## 1.2 Attribution explains what is, not what happens when you act

The conceptual gap is precise and already acknowledged inside the group's own drafts. SignalShap's limitations section states: "**Masking a source at fusion time is not equivalent to never having built it**, since the candidate set is generated once from the grand coalition" (SignalShap §5.4; same point in the thesis §7.14 discussion of removal-based counterfactuals). What SignalShap's exact game measures — the marginal value of source $g$ when its score column is withheld from an already-trained fusion — is a *system state that never exists on the road to the decision it recommends*. If a system owner acts on the attribution by decommissioning a source, the world that results is one where the source was never trained, never scored, and never contributed candidates: a different candidate set, a different fusion, possibly a different load for the remaining sources. Attribution describes the current system; the decision is about a system that does not exist yet. The gap between the two is precisely what the audit measures.

**Where the group's own ActionShap stands on this.** ActionShap operationalizes the same thesis definition, but its interventions are *factor-level*: $\Delta_j = \mathbb{E}_x[f(x \mid do(x_j \leftarrow x_j + \tau_j)) - f(x)]$ (ActionShap Definition 2) — a bounded perturbation of an input or interaction *inside a trained model*, with modifiability elicited per factor. Its recommendation-regime interventions are "reweight or remove a specific interaction, inject/suppress a context signal, or adjust an item's exposure" (ActionShap §6.3). None of these is a construction-level intervention: nothing in ActionShap asks what the system would look like if a *signal source had never been built*, and its recsys game is DyHuCoG's Monte-Carlo interaction game rather than an exact source game. ShapAct is the construction-level complement: same measurement paradigm (predict → intervene → measure realized effect), different intervention class, exact game, and one quantity — reflexivity — that ActionShap does not define at all.

We note the same structure at the other two levels of the group's work. DyHuCoG uses Shapley values *in training*, so its attribution never faces a "what happens if I act" question — the action (reweighting propagation) is the training loop itself. Feature-level SHAP (thesis C1/C2, IJACSA 2025) explains clusters, where the modifiable factor is a physicochemical variable the analyst may not control. ActionShap handles factor-level actionability (perturbation within a feasible budget). Source-level exact attribution is the one setting where the modifiable factor is a real *architecture* decision (build/keep/decommission a pipeline) — which is why the construction-level audit belongs here.

## 1.3 Why an intervention audit (and why now)

Three things make the audit timely and cheap. First, the machinery exists: SignalShap has already validated the exact five-source game, and its Appendix D already specifies the regenerate-candidates-per-coalition variant — the audit simply promotes that appendix variant to a first-class condition and adds the one genuinely new condition (never-built). Second, exactness makes the audit *interpretable*: with $2^5 = 32$ coalition values and no Monte-Carlo noise, any discrepancy between predicted and realized effects can be attributed to the intervention semantics, not to estimator variance. Third, the field's own evaluation vocabulary is converging on intervention-style tests (deletion/insertion curves for feature attribution, causal removal frameworks — see §2.3–2.4), but none has been applied at the *source/component* level in recommendation, where the modifiable factor is an engineering pipeline rather than an input feature.

## 1.4 From SignalShap and ActionShap to ShapAct

ShapAct reuses, without modification: the five players (CF, CB, POP, REC, SEQ — SignalShap §3.2), the characteristic function $v(C) = \mathrm{NDCG@10}(f^C_\theta) - \mathrm{NDCG@10}(\pi)$ with $f^\emptyset_\theta \equiv \pi$ (SignalShap Definition 1), the exact Shapley computation over $2^5 = 32$ coalitions (SignalShap Definition 2), the per-user decomposition by linearity (SignalShap Proposition 3), and the entire data pipeline including the rebuilt Amazon-Book split (SignalShap §4.1 and Implementation Spec §A.3). From **ActionShap** it reuses the *measurement paradigm* and its reference implementation: the AIA/top-$k$-precision/regret metric family (`code/actionshap/metrics.py`), the attribution-wrapper interface (`code/actionshap/attribution.py`), the significance machinery (`code/actionshap/stats.py`), and the pre-registration discipline for anything elicited (the freeze-hash mechanism in `code/actionshap/modifiability.py` — ShapAct has no modifiability elicitation, since every source is modifiable by construction, but the audit trail pattern carries over for its intervention protocols). What ShapAct adds is the measurement layer ActionShap does not contain: the three **construction-level** counterfactual levels of §3.3, the fidelity gap, order validity, **reflexivity**, and the decommissioning decision-rule comparison. SignalShap asks *which source earns the accuracy*; ActionShap asks *which factor-level intervention an attribution predicts well*; ShapAct asks *whether the source-level attribution survives the construction-level decision it recommends*.

## 1.5 Contributions

Numbered, bold lead-ins, IJACSA style. Four substantive plus one artifact.

1. **The intervention audit as an evaluation paradigm.** We formalize the three-level counterfactual ladder — *masked* (in-game, what attribution measures), *regenerated* (candidate set rebuilt without the source, SignalShap's Appendix D variant), *never-built* (the source is never trained, scored, or fused; what the recommended decommissioning decision means) — and define the **fidelity gap** $F_g$ between the predicted and realized effect of retiring source $g$. This converts the thesis's actionable-insight definition into two measurable numbers per source.

2. **Two short results that make the audit meaningful.** Proposition 1 decomposes the realized retirement effect into the in-game marginal and the fidelity gap, showing the gap is exactly the "retraining bonus" of the never-built system (a one-line identity that makes the measurement bookkeeping exact). Proposition 2 gives a sufficient condition (co-monotone fidelity) under which the Shapley credit ranking is order-valid as a decision rule, and commits the paper to reporting *where* the condition fails — because that is precisely where acting on the attribution misleads.

3. **Reflexivity as a property of explanations.** Proposition 3 bounds the post-intervention shift of the surviving sources' attributions, and we measure it: after retiring the lowest-credit source, do the remaining credits move (meaning the original explanation described a state that acting on it destroyed) or hold (meaning the explanation is a stable description of the system's structure)? DyHuCoG showed attribution can *drive* learning; ShapAct measures whether attribution *survives* its own outcome.

4. **A decision-level comparison of explanation designs.** We implement four decision rules — retire the lowest-Shapley-credit source, retire the lowest-leave-one-out source, retire the lowest feature-SHAP source, and a random baseline — execute each on both datasets, and report *realized* NDCG@10 after each decision. This is the first time in the group's line of work that explanation designs are compared by the outcome of the action they recommend rather than by internal plausibility.

5. **A reproducible artifact.** Everything runs on a laptop CPU with the SignalShap codebase; the audit adds at most five base-scorer retrainings (one per source, for the never-built condition) plus cheap fusion refits. Code, configurations, and the full L0/L1/L2 counterfactual tables are released.

## 1.6 Organization

One short roadmap paragraph naming sections 2–6 (standard house style).

---

# 2. LITERATURE REVIEW

Thematic subsections, each closing with a one-sentence gap statement. Target ~1,300 words.

## 2.1 Cooperative-game attribution in recommendation

Cover the group's own lineage first, since it defines the boundary of what is claimed: the thesis's Shapley-for-black-box spine (thesis C1/C2), DyHuCoG's dynamic hypergraph game with preference-aware Monte-Carlo Shapley embedded in message passing (DyHuCoG §3.4–3.5; thesis C3), the real-time Shapley adjustment line [Nesmaoui et al., 2025, cited in the group's bibliography], Shapley-driven data pruning for recommenders (Zhang et al., KDD 2025, already cited in DyHuCoG and the thesis), SignalShap's exact source-level game, and — most directly relevant — **ActionShap**, the group's intervention-grounded evaluation framework, which operationalizes the thesis's actionability definition into the Actionability Score, Attribution–Intervention Alignment, top-$k$ intervention precision, and intervention regret, and tests six attribution methods across four datasets (ActionShap §1.5, §4). The precise relation to state: ActionShap evaluates whether an attribution predicts the realized effect of *feasible factor-level interventions* (perturbations of inputs or interactions within an elicited budget); it does not contain construction-level interventions (never-built components), an exact source game, or post-intervention re-attribution. *Gap: the construction level of "acting on an attribution" — decommissioning a whole signal pipeline — remains unevaluated in the group's own framework, and ShapAct is that evaluation.*

## 2.2 Source-level attribution and hybrid fusion

Burke's hybridization taxonomy; score-level fusion as the production-dominant pattern; SignalShap's five-source game and its redundancy-collapse result (SignalShap Proposition 2: under perfect redundancy leave-one-out reports zero for both members while Shapley splits the credit). Note the specific consequence the audit exploits: for a redundant pair, LOO and Shapley *recommend different decisions*, so the decision-level comparison of §4.6 is not a re-run of the attribution-level comparison — it is its downstream, decision-level test (and it is a test ActionShap does not run, since its method comparison scores the intervention induced by a *factor*, not the retirement of a *source*). *Gap: fusion-weight optimization and source attribution both stop at description; the decommissioning decision the description supports is never executed.*

## 2.3 Evaluation of explanations: faithfulness, stability, usefulness

The thesis's own multi-criterion position (§2.7.5): faithfulness (deletion/insertion performance), stability (perturbation sensitivity; the group's Explanation Drift metrics TAD/ESS/EHL [Nesmaoui et al., 2026 preprint]), sparsity, and causability/usability. SignalShap adds population heterogeneity (per-segment profiles) as a stability axis. ActionShap adds the *intervention* axis at factor level: feasibility-constrained modification (with a modifiability-held-out control to avoid construction-induced claims, ActionShap Remark 1) and decision-level metrics. *Gap: every criterion in this literature, including ActionShap's, is computed on the explained system or on a perturbed version of its inputs; none is computed on the system that results from removing a component from its construction — the reflexivity axis is missing entirely.*

## 2.4 Counterfactual and causal intervention evaluation

Causal Shapley and the do-operator critique of removal-based attribution (Janzing et al., 2020 — already in the group's bibliography and in MHyperShap's masking design); counterfactual explanation in recommendation (thesis §3.4.4: "counterfactuals reveal sensitivity, whereas Shapley values allocate contribution"); deletion/insertion curves for feature attribution; ActionShap's feasible-intervention effect $\Delta_j$ under a declared budget (its Definition 2, explicitly contrasted with deletion baselines). The bridge to ShapAct: counterfactual reasoning is *about the explained system* (what would change the output); ActionShap's interventions are *about feasible input changes* (what a decision-maker could realize); ShapAct's never-built condition is *about the system's construction* (what the output would be if a component had never existed). *Gap: construction-level counterfactuals — the only ones that match architecture decisions — have no evaluation protocol in recommendation; ShapAct supplies one, and it is the natural third rung after deletion-based and factor-level-feasible intervention tests.*

## 2.5 Positioning and differentiation

**Table 1.** Rows: representative prior methods. Columns: *unit of attribution*, *exact or approximate*, *recsys?*, *use in training?*, *actionability measured (intervention executed)?*, *fidelity/reflexivity reported?*.

| Method family | Unit | Exact | Recsys | In-training | Actionability measured | Fidelity/reflexivity |
|---|---|---|---|---|---|---|
| Feature SHAP (thesis C1/C2; Louhichi 2023/2025) | feature | no (surrogate) | no | no | no (framing only) | no |
| Data Shapley (Ghorbani & Zou 2019) | datum | no | partial | no | partial (pruning) | no |
| DyHuCoG (ours, 2026) | user–item–context entity | no (MC, M=50) | yes | **yes** | no (diagnosis only) | no |
| Shapley data pruning (Zhang et al. 2025) | interaction | no | yes | no | partial (drop low-value) | no |
| Explanation drift (ours, 2026 preprint) | feature | no | yes | no | no (stability only) | no |
| SignalShap (companion, this line) | **source** | **yes** | yes | no | framed, not measured | no |
| **ActionShap (ours, in repo)** | **factor / interaction** | no (MC) | yes | no | **yes — feasible factor-level interventions (AIA, P@k, regret)** | fidelity at factor level; **no construction counterfactual, no reflexivity** |
| Ablation / leave-one-out | source | yes | yes | no | no (collapses under redundancy) | no |
| **ShapAct (this work)** | **source** | **yes** | yes | no | **yes — construction-level (never-built) interventions, realized effect measured** | **yes — fidelity gap, order validity, reflexivity (re-attribution after acting)** |

The differentiation is over the last two columns, and against **ActionShap** specifically it is over the *intervention class*: ActionShap perturbs a factor or interaction inside a trained model; ShapAct removes a component from the system's construction and re-measures. ActionShap is cited as the paradigm paper and the metric family is reused (AIA, P@k, regret adapted to retirement outcomes); the two are positioned as companion papers on the factor-level vs. construction-level axes of the same operationalization of thesis Definition 1.1.

---

# 3. METHODOLOGY

## 3.1 Notation and problem formulation

| Symbol | Meaning |
|---|---|
| $\mathcal{U}, \mathcal{I}$ | user and item sets |
| $\mathcal{G} = \{g_1,\dots,g_K\}$ | set of signal sources, the players; $K = 5$ (CF, CB, POP, REC, SEQ) |
| $s_g(u,i)$ | score assigned to $(u,i)$ by source $g$ |
| $C \subseteq \mathcal{G}$ | a coalition of sources |
| $f_\theta^{C}$ | fusion model refitted using only the sources in $C$; $f_\theta^{\emptyset} \equiv \pi$ |
| $v(C)$ | characteristic function: $\mathrm{NDCG@10}(f_\theta^{C}) - \mathrm{NDCG@10}(\pi)$ |
| $\varphi_g$ | exact Shapley value of source $g$ (reused from SignalShap) |
| $v^{\text{reg}}(C)$ | characteristic function with the candidate set **regenerated** from the sources in $C$ (L1) |
| $v^{\text{nb}}_{-g}(C)$ | characteristic function of the recommender in which source $g$ was **never built** (untrained, unscored, unfused) (L2) |
| $P_g$ | predicted retirement effect: $P_g = v(\mathcal{G}) - v(\mathcal{G}\setminus\{g\})$ (the masked marginal) |
| $R_g$ | realized retirement effect: $R_g = v^{\text{nb}}_{-g}(\mathcal{G}\setminus\{g\}) - v(\mathcal{G})$ |
| $F_g$ | fidelity gap: $F_g = P_g - R_g$ |
| $\tau$ | Kendall rank correlation between credit order and realized-effect order |
| $\rho_g$ | reflexivity shift: relative change of remaining sources' credits after retiring $g$ |
| $\pi$ | null ranker (uniform random, fixed seed) |

Keep the notation table in the manuscript — house convention since the IJACSA paper.

## 3.2 The reused source-attribution game

State, cite, and stop. The game, its players, the characteristic function, the exact computation, the per-user decomposition, and the z-normalization/fusion conventions are **not re-derived here**: they are SignalShap Definitions 1–3, Propositions 1 and 3, and §3.3–§3.5, and the group's proofs already live in SignalShap Appendix A and thesis Appendix A. Restate the two equations that the audit reads off the game, since they define the predictions:

$$\varphi_g = \sum_{C \subseteq \mathcal{G}\setminus\{g\}} \frac{|C|!\,(K-|C|-1)!}{K!}\bigl[v(C \cup \{g\}) - v(C)\bigr], \qquad v(C) = \mathrm{NDCG@10}\bigl(f_\theta^{C}\bigr) - \mathrm{NDCG@10}(\pi).$$

The audit's recommended decisions are derived from $\varphi_g$: the *retirement rule* retires the source with the smallest Shapley credit (equivalently, the smallest average marginal), and the *investment rule* (discussion only, §5.1) ranks sources by credit divided by the production-cost column of SignalShap's Table 3.

**Calibration note (house convention).** This paper carries **three short propositions and one remark**, matching the SignalShap calibration (three propositions + one remark) and the group's published level (the Explanation Drift paper carries zero; ActionShap carries one proposition + one corollary). No efficiency proof is restated (SignalShap Proposition 1), no redundancy-collapse theorem is restated (SignalShap Proposition 2), no alignment-under-local-linearity result is restated (ActionShap Proposition 1), and no MC-convergence or Shapley-uniqueness material appears (thesis Appendices A.1, A.3). What is new is only what the construction-level audit needs, and each proposition is a few lines.

## 3.3 Three counterfactual levels

The core design. A *retirement* of source $g$ is the decision the attribution supports (SignalShap §5.1: "a source with a small share and a high maintenance cost is a decommissioning candidate"). It can be instantiated at three levels of faithfulness to the decision's real meaning:

| Level | What is done | What it measures | Status in prior work |
|---|---|---|---|
| **L0 — masked** | Withhold $g$'s score column; refit fusion; keep scorers and candidate set fixed | In-game marginal $v(\mathcal{G}) - v(\mathcal{G}\setminus\{g\})$; what SignalShap already reports | SignalShap main results |
| **L1 — regenerated** | Withhold $g$'s scores **and** regenerate each user's candidate set from the remaining sources (round-robin across sources, same $N$) | Adds the candidate-set effect; removes the "ceiling contamination" SignalShap Appendix D quantifies | SignalShap Appendix D (robustness only) |
| **L2 — never-built** | Do **not train** $g$; regenerate candidates; refit fusion — i.e., run the pipeline as if $g$ had never existed | The realized effect of the decision: $R_g$ | **No prior work in the folder — not SignalShap (masking only), not ActionShap (factor-level perturbations only). This is ShapAct's addition.** |

> **Pin the semantics before the build starts, and contrast them with ActionShap's in the paper.** ActionShap's intervention effect $\Delta_j = \mathbb{E}_x[f(x \mid do(x_j \leftarrow x_j + \tau_j)) - f(x)]$ (its Definition 2) perturbs an *input* of a *fixed* model within a feasible budget $\tau_j$; the model's construction is untouched. ShapAct's L2 removes a *component* from the construction: the scorer is never trained, the candidates are regenerated, the fusion is refit. The two are different intervention classes on different objects (factors vs. components), which is the one-paragraph differentiation the paper must state in §1.2 and again in §2.5. L2 is the only level that matches what a decommissioning decision means, and it is the only level where the *prediction* (an in-game marginal from the current system) and the *outcome* (a system that never contained $g$) are genuinely different objects. L1 is a necessary intermediate: it isolates the candidate-set effect from the retraining effect, so that $F_g$ can be decomposed as $F_g = [\text{candidate-set effect}] + [\text{retraining effect}]$, and only the latter requires the expensive L2 run. Report all three levels per source — the L0/L1/L2 ladder in Appendix B is the paper's transparency showpiece, exactly as the full coalition table is SignalShap's.

**Cost control.** L2 requires retraining the base scorer of every *other* source? No — it does not. Under the SignalShap design, base scorers are trained once and cached; a never-built world for source $g$ still uses the *same* trained CF, CB, etc., because those sources are not the ones being retired. Only the retired source's scorer is absent (never trained), the candidate set is regenerated, and the fusion is refit. The genuinely expensive variant — retraining the *surviving* collaborative scorer in a world where $g$ never existed, to capture load-redistribution effects inside CF itself — is reserved for the sensitivity section (§4.7) and Appendix D, applied to the top-ranked retirement only, because it is the one place where "never built" could mean "the other sources would have been trained differently."

## 3.4 Actionability measures: fidelity, order validity, reflexivity

**Relation to ActionShap's metric family (state this in the methodology).** ActionShap defines Attribution–Intervention Alignment (rank agreement between attribution and measured intervention effect, its Definition 5), top-$k$ intervention precision and intervention regret (Definitions 6–7). ShapAct adapts the same family to *retirement outcomes*: order validity (Definition 2 below) is AIA's decision-level cousin computed over realized retirement effects; the fidelity gap (Definition 1) is a pointwise prediction–realization comparison that ActionShap does not define (its $\Delta_j$ is the *measured effect* itself, not a prediction to be checked against a construction counterfactual); reflexivity (Definition 3) is new. Cite ActionShap for the family, reuse its `metrics.py` implementations where they apply, and state that the additions are the construction-level quantities.

**Definition 1 (Fidelity gap).** For source $g$, the fidelity gap is $F_g = P_g - R_g$, where $P_g = v(\mathcal{G}) - v(\mathcal{G}\setminus\{g\})$ is the masked marginal (L0, the *predicted* loss of removal) and $R_g = v(\mathcal{G}) - v^{\text{nb}}_{-g}(\mathcal{G}\setminus\{g\})$ is the *realized loss* of never building $g$ (L2); $R_g > 0$ means retiring $g$ harms the system, $R_g < 0$ means it improves it. $F_g > 0$ means the attribution *overstates* the harm of removing $g$ (the system adapts to its absence); $F_g < 0$ means the attribution *understates* it (removing $g$ costs more than the current-system marginal suggests). *Sign note (corrected against the first draft of this blueprint): with $R$ defined as the realized loss, $F_g = P_g - R_g = v^{\text{nb}}_{-g}(\mathcal{G}\setminus\{g\}) - v(\mathcal{G}\setminus\{g\})$ holds identically — the fidelity gap is exactly the value difference between the never-built system and the masked system — which is the identity Proposition 1 and the implementation-spec test 9 rely on.*

**Definition 2 (Order validity).** The credit ranking is order-valid at level $k$ if the set of the $k$ lowest-credit sources coincides with the set of the $k$ smallest realized retirement losses (equivalently, the $k$ most-justified retirements). Report Kendall's $\tau$ over the full ordering and top-1/top-2 agreement, per dataset. (This is the construction-level analogue of ActionShap's AIA, and the two must be reported in relation to each other in §4.4, not silently renamed.)

**Definition 3 (Reflexivity shift).** After executing the retirement of source $g^*$ (the lowest-credit source), recompute the exact game on the never-built system and define $\rho_{g^*} = \frac{1}{K-1}\sum_{h \neq g^*} \bigl|\varphi_h^{\text{nb}} - \varphi_h\bigr| \big/ \bigl(\frac{1}{K-1}\sum_{h\neq g^*}\varphi_h\bigr)$, the mean relative change of the surviving sources' credits. Low $\rho$ means the original explanation is a stable description of the system's structure; high $\rho$ means the explanation was a description of a state that acting on it destroyed. (Contrast explicitly with ActionShap's stability, its Definition 3: that is dispersion of one attribution across repeated runs; reflexivity is the *change of the attribution itself* when the system it described is altered by its own recommendation.)

All three quantities are exact in the same sense as the underlying game: they are computed from exact coalition values, so their reported values carry no sampling error. This is the single most important sentence of the methodology for a reviewer — and it is the sentence that differentiates the audit from ActionShap's dynamic regime, whose Monte-Carlo estimates (DyHuCoG $M=50$) carry sampling variance that must be reported alongside AIA.

## 3.5 Decision rules under test

Four rules, each stated as a policy: *retire the source with the smallest score according to the explanation*, executed on the test set, evaluated by realized NDCG@10 after the decision (L2 semantics).

| Rule | Score used | Represented claim |
|---|---|---|
| **Shapley rule** | $\varphi_g$ (exact source Shapley) | Shapley credit ranks sources by expected marginal contribution; the audit's main object |
| **LOO rule** | $v(\mathcal{G}) - v(\mathcal{G}\setminus\{g\})$ per source (leave-one-out) | The standard ablation-based answer, known to collapse under redundancy (SignalShap Prop. 2) |
| **Feature-SHAP rule** | mean |SHAP| of the fused ranker's five features (KernelSHAP on the fusion layer) | Feature-level attribution applied to a source-level decision — a category error the paper quantifies |
| **Random rule** | uniform | Calibration floor: what realized quality a decision-maker achieves with no attribution at all |

The comparison answers RQ4 and doubles as the *decision-level* test of SignalShap's Proposition 2: on the engineered-redundant pair POP↔REC, LOO reports near-zero for both (and therefore treats them as equally retireable), while Shapley splits their joint credit — so the two rules recommend *different* retirements, and the realized outcomes decide which explanation was right.

## 3.6 Theoretical justification

**Proposition 1 (Intervention-fidelity decomposition).** For every source $g$,
$$R_g = P_g - F_g = \bigl[v(\mathcal{G}) - v(\mathcal{G}\setminus\{g\})\bigr] - \bigl[v^{\text{nb}}_{-g}(\mathcal{G}\setminus\{g\}) - v(\mathcal{G}\setminus\{g\})\bigr],$$
i.e., the fidelity gap equals the value difference between the regenerated-but-masked system and the never-built system. *Proof: rearrangement of the definitions; the parenthesized term is the "regeneration + never-trained" adjustment. One line.* The value of the proposition is interpretive: it pins the audit's measurement to a single, checkable quantity per source, and it makes the L0/L1/L2 ladder a bookkeeping identity rather than a design choice.

**Proposition 2 (Order preservation under co-monotone fidelity).** Order sources by masked retirement effect $P_g$. If the fidelity gaps satisfy $F_g \ge F_{g'}$ whenever $P_g \ge P_{g'}$ (fidelity is co-monotone with $P$), then the realized ordering by $R_g$ coincides with the masked ordering by $P_g$; hence the Shapley-credit retirement rule is realized-optimal whenever (i) the Shapley-credit ordering agrees with the $P$-ordering and (ii) co-monotonicity holds. *Proof sketch: $R_g = P_g - F_g$ with $F$ co-monotone preserves pairwise order. A few lines.* The empirical content is the *violation* report: §4.4 tabulates, for each dataset, the pairs of sources on which co-monotonicity fails — these are exactly the decisions where acting on the attribution misleads.

**Proposition 3 (Reflexivity bound).** Let $\varphi^{\text{nb}}$ be the Shapley allocation of the never-built game after retiring $g^*$. Then $\sum_{h \neq g^*} \varphi_h^{\text{nb}} = v^{\text{nb}}_{-g^*}(\mathcal{G}\setminus\{g^*\}) = v(\mathcal{G}) - R_{g^*}$, so the total credit of the surviving sources shifts by exactly the realized loss of the retirement; the per-source shifts $\rho_{g^*}$ are therefore bounded in aggregate by $|R_{g^*}|$ and are measured per source. *Proof: efficiency applied to the never-built game; the identity is the reflexivity analogue of SignalShap Proposition 1.* This is what makes RQ3 a quantitative claim rather than a hand-wave: the surviving system's total credit *must* move by the realized loss; what is informative is how that total loss is distributed across the surviving sources (concentrated on the redundant sibling → the explanation correctly identified a structural substitute; spread uniformly → the explanation misdescribed the system's dependence structure).

**Remark 1 (Cost-adjusted credit).** The decision-relevant quantity for decommissioning is credit per unit of production cost, $\varphi_g / c_g$, with the cost column taken from SignalShap's Table 3 (CF: interaction store + retraining; CB: metadata ingestion; REC: timestamps; SEQ: session logging + low-latency state; POP: negligible). The audit reports raw-credit decisions as the primary result (costs are deployment-specific) and cost-adjusted decisions in Appendix E. This remark, like SignalShap's Remark 1, is deliberately not promoted to a proposition.

## 3.7 Complexity analysis

Reuse the SignalShap cost table (its §3.10) and add the audit's increments. Let $B$ be the cost of training one base scorer, $\Phi$ one fusion refit, $P = |\mathcal{U}| \cdot N$ the cached score-matrix size.

| Stage | Cost | Note |
|---|---|---|
| Base scorers | $O(KB)$ | paid once, not per coalition (SignalShap) |
| Coalition sweep (L0) | $O(2^K \Phi)$ | 32 convex refits, sub-second each (SignalShap) |
| Shapley + per-user decomposition | $O(K2^K)$ + free | SignalShap Prop. 3 |
| L1 regeneration | $O(K \cdot |\mathcal{U}| \cdot N)$ per source | candidate union recomputed from remaining sources' top lists; no model training |
| L2 never-built | $O(\Phi + K \cdot |\mathcal{U}| \cdot N)$ per source | the retired source's scorer is simply not trained — zero extra $B$ for the primary result |
| L2 full-retraining variant (§4.7, Appendix D) | $O(B)$ per source, restricted to the top-1 retirement | the only expensive increment; budget-bound by design |

Headline sentence: **the audit's primary results cost less than one extra base-scorer training**, because the never-built world for source $g$ reuses the other four trained scorers; the full-retraining variant is deliberately restricted to one source. Contrast with the MC route the group's prior work required (DyHuCoG's $O((M/f)m)$ amortized Shapley cost): here the audit inherits exactness from the small player set, and the whole intervention study is reproducible on a laptop CPU.

## 3.8 Practical implementation

Library versions, seeds, hardware, wall-clock, and the caching scheme, mirroring SignalShap §3.11 and its Implementation Spec §A.2 (Python 3.12, numpy/scipy/sklearn/implicit, no GPU). Pin the three conventions that can silently drift: the L0/L1/L2 semantics (§3.3), the $f_\theta^{\emptyset} \equiv \pi$ identification (SignalShap), and the $\sigma_{u,g} = 0$ normalization fallback (SignalShap). Follow the house convention of two numbered algorithm blocks — one for the counterfactual ladder (L0→L1→L2), one for the decision-rule executor and reflexivity recomputation.

---

# 4. EXPERIMENTAL RESULTS

Setup and results combined, per the IJACSA/SignalShap structure. **EVERY NUMBER IS A PLACEHOLDER; registered predictions are in the Implementation Spec Part B and must be reported against it, including misses.**

## 4.1 Datasets and preprocessing

MovieLens-1M and Amazon-Book — the exact pair and protocol of SignalShap §4.1 / Implementation Spec §A.3, chosen for the same density-contrast reason (two orders of magnitude apart) and for continuity with the group's published recsys line (DyHuCoG). **The Amazon-Book split is rebuilt from the raw Amazon Reviews 2018 Books corpus** per the SignalShap spec (subsampled to ~50k users, iterative 5-core, temporal leave-one-out with deterministic tie-breaking), for the reasons documented there: the canonical split carries no timestamps, no metadata, and no order, which would make REC, SEQ, and CB — three of the five players — unimplementable. State plainly that the resulting counts do not reproduce the canonical 52,643/91,599/2,984,108 figures used in DyHuCoG and the thesis; that is expected and disclosed (see "Conventions and discrepancies" below). Candidate recall @200 is the ceiling on every number in the paper and is reported first, per house convention.

## 4.2 Protocol, metrics, and baselines

Metrics: NDCG@10 (primary; defines $v$), Recall@20, MRR; the null ranker's NDCG reported so the uplift subtraction is auditable. Baselines for the *recommender*: single sources, uniform fusion, global fusion (from SignalShap's runs, reused), plus the strong single-model reference already established in SignalShap (§4.2). Baselines for the *explanation* (the actual comparison): the four decision rules of §3.5.

## 4.3 RQ1 — intervention fidelity

**Table 4 / Figure 2:** per source, per dataset: $P_g$ (L0 marginal), L1 regenerated marginal, $R_g$ (L2 realized), and $F_g$. The decomposition identity of Proposition 1 is verified numerically as a correctness row. The registered expectation (Implementation Spec §B.2): **$F_g > 0$ for the redundant pair POP↔REC on both datasets** (the never-built world suffers less than masking predicts, because the sibling covers the load — the decision-level face of SignalShap's redundancy result), and **$F_g < 0$ for CB on Amazon-Book** (content metadata has no substitute; removing it costs more than the current-system marginal suggests). Report the actual values, including any sign surprises — a wrong-signed $F_g$ is a finding about the system's substitutability structure, not an error.

## 4.4 RQ2 — order validity of the credit ranking

**Table 5 / Figure 3:** Kendall's $\tau$ between the $\varphi_g$-ordering and the $R_g$-ordering; top-1 and top-2 agreement; the co-monotonicity violation pairs of Proposition 2 flagged explicitly. Expected: high agreement at the top of the order, degradation near the bottom (null players), lower overall $\tau$ on Amazon-Book where the CB anomaly of RQ1 sits. Report the violations even if they contradict the expectation — this table is the paper's honesty instrument.

## 4.5 RQ3 — post-intervention reflexivity

**Table 6 / Figure 4:** after retiring the lowest-credit source per dataset, the surviving sources' credit vectors before/after, and $\rho_{g^*}$; plus the aggregate identity $\sum_{h\neq g^*}\varphi_h^{\text{nb}} = v(\mathcal{G}) - R_{g^*}$ as a verification row. Expected: $\rho$ concentrated on the redundant sibling of the retired source (structural-substitute identification), modest elsewhere; if $\rho$ is large and diffuse, the explanation is self-defeating and the paper says so.

## 4.6 RQ4 — realized quality under alternative decision rules

**Table 7 / Figure 5:** realized NDCG@10, Recall@20, MRR after executing each rule's recommended retirement (L2 semantics), per dataset; the random rule as the floor. Expected: Shapley rule ≥ LOO rule ≥ feature-SHAP rule ≥ random, with the Shapley-vs-LOO gap concentrated on the POP↔REC decision — directly predicted by SignalShap Proposition 2, which this table tests at the decision level. **If the Shapley rule does not win, report it: the decision-level claim of the paper softens to "exact attribution is decision-competitive," and the falsification table (Implementation Spec §B.6) says how to re-frame.**

## 4.7 Sensitivity, stability, and ablations

- L2 full-retraining variant for the top-1 retirement (does retraining the surviving CF change $R_{g^*}$ materially? bounds the "load-redistribution" caveat).
- Seed stability of all audit quantities across five seeds (exact game ⇒ only base-scorer training contributes variance).
- Sensitivity of the decision outcomes to candidate size $N$ and to the L1 regeneration round-robin rule.
- Cost-adjusted decisions (Remark 1) in the appendix.
- Nonlinear monotone rescaling of source scores, re-running the whole audit, per SignalShap §4.7's precedent (percentile rank applied uniformly).

## 4.8 Statistical significance

Paired tests over *users*, Holm–Bonferroni across the decision-rule family, Wilcoxon signed-rank as non-parametric companion, Cohen's $d_z$ effect sizes — the exact house protocol of DyHuCoG Appendix A / thesis Chapter 4, applied to realized NDCG@10 per user under each rule. Reuse the ported `stats.py` from the ActionShap codebase, per SignalShap's spec. State the unit of analysis explicitly (users).

---

# 5. DISCUSSION AND BROADER IMPLICATIONS

## 5.1 When attribution misleads: the measured conditions

Lead with whatever the fidelity table actually shows. The candidate headline: masking overstates the value of substitutable sources and understates the value of un-substitutable ones, so the decision-relevant reading of an attribution is *not* the raw marginal but the marginal adjusted by measured substitutability — which is exactly what the audit's $F_g$ column provides. If the CB-on-Amazon prediction holds, the discussion gets its sharpest case: the source that looks least valuable under L0 is the one whose removal hurts most, and only L2 sees it. This is the "decision-support contract" made concrete.

## 5.2 The decision-support contract for recommendation explanations

State the contract explicitly: an explanation of a recommender system is actionable iff (i) it identifies a modifiable component (source, weight, stage), (ii) it specifies the predicted change, (iii) the prediction survives execution — order-valid and approximately fidelity-tight, and (iv) the attribution does not destroy the basis of its own validity when acted upon (reflexivity). Map the group's outputs onto the contract: DyHuCoG satisfies (i)–(ii) in training but is never audited post-hoc; SignalShap satisfies (i)–(ii) at the source level and the audit shows to what degree (iii)–(iv) hold. This is the paper's contribution to the thesis narrative: it turns the thesis's framing concept into a measured quantity on the group's own flagship setting.

## 5.3 Relation to feature-level, population-level, temporal, and factor-level-actionability explanation

Feature-level SHAP (thesis C1/C2) answers a different decision (which variable to investigate); population-level segmentation (SignalShap RQ3) asks whether attribution is stable across users; temporal stability (Explanation Drift) asks whether it is stable across time; ActionShap asks whether attribution predicts the realized effect of *factor-level feasible interventions* (its AIA/regret, across static and dynamic regimes). ShapAct adds the remaining axis: whether attribution is stable *across the construction-level execution of its own recommendation* — the decommissioning of a source. The axes compose: a source-level attribution that is population-stable, temporally stable, factor-level-actionable (ActionShap), and construction-faithful (ShapAct) is the strongest possible actionable claim in this line of work. The two-paper story to write: ActionShap is the paradigm paper (factor-level intervention axis, four datasets, six methods); ShapAct is the construction-level companion (source-level exact game, never-built counterfactual, reflexivity) — the same orthogonality framing the group already uses for SignalShap vs. Explanation Drift.

## 5.4 Limitations and threats to validity

Be forthright, in the SignalShap §5.4 voice: the audit measures retirement decisions, not investment decisions (adding a new source has no natural never-built counterpart in this design — flag as future work); L2 reuses the surviving sources' scorers as trained in the original world, so it captures fusion and candidate-level adaptation but not retraining-level adaptation (bounded by the §4.7 variant for the top-1 case); offline ranking quality remains a proxy for deployed value; two datasets, however deliberately contrasted, are two datasets; the never-built semantics assume the source is removed wholesale rather than replaced by a cheaper substitute; and the decision rules are single-step (one retirement), so multi-step or portfolio decisions are out of scope. Also record the dependence on SignalShap's pipeline and its disclosed Amazon-Book rebuild.

---

# 6. CONCLUSION AND FUTURE WORK

Restate the four contributions against the four RQs, with the measured headline numbers once available. Future work: (a) investment-side audit — measuring the realized gain of *adding* a candidate source or of retraining an existing one, which would require a controlled intervention protocol on a synthetic or semi-synthetic recommender; (b) multi-step decision paths (retire then re-audit, iterated to a fixed point — the reflexivity measure makes this natural, since $\rho_{g^*}$ tells you when re-auditing is necessary); (c) linking the audit to the temporal axis (Explanation Drift metrics on the never-built system); (d) a small analyst study where humans make budget decisions with and without the audit's fidelity report, connecting to the thesis's human-centred future-work direction — the automated audit is the prerequisite that makes such a study cheap to score; (e) extending the ladder to FairShap's exposure-level actions, where the modifiable factor is an exposure budget rather than a source.

---

# DECLARATIONS (required by Discover AI)

Funding · Competing interests · Ethics approval (not applicable, public secondary data) · Consent · Data availability (both datasets public; the rebuilt Amazon-Book split and preprocessing scripts released, per SignalShap's disclosure) · Code availability (repository link; ShapAct lives in the SignalShap repository) · Author contributions (CRediT, reuse the DyHuCoG/SignalShap split) · Use of AI tools (state the same declaration the group's Explanation Drift paper carries, per SignalShap's convention).

# APPENDICES

- **A.** Proofs of Propositions 1–3 and Remark 1 — three short arguments, roughly one page total.
- **B.** Full counterfactual tables: $v(C)$ for all 32 coalitions at L0, plus the L1 and L2 marginal and $R_g$, $F_g$ per source per dataset. The transparency showpiece — feasible only because the game is exact and small.
- **C.** Hyperparameter grids and selected values (reuse SignalShap's table, plus the audit's additions: $N$, round-robin rule, L2 scope).
- **D.** Full-retraining robustness: never-built with the surviving CF retrained, for the top-1 retirement per dataset.
- **E.** Decision-rule results in full: raw-credit and cost-adjusted (Remark 1) rules, per dataset, with per-user significance tables.

# NOTATION LIST

Reproduce the §3.1 table as a standalone appendix-adjacent list, matching the IJACSA/SignalShap convention.

---

# PLANNED FIGURES & TABLES

| # | Type | Content |
|---|---|---|
| Fig 1 | Diagram | The L0/L1/L2 counterfactual ladder |
| Fig 2 | Bar | $P_g$, $R_g$, $F_g$ per source per dataset (fidelity) |
| Fig 3 | Scatter | Credit order vs. realized-effect order, annotated at co-monotonicity violations |
| Fig 4 | Paired bars | Surviving credits before/after retirement (reflexivity) |
| Fig 5 | Bar | Realized NDCG@10 under the four decision rules |
| Fig 6 | Line | $N$-sensitivity of decision outcomes |
| Fig 7 | Heatmap | L0/L1/L2 marginal matrix, all sources × both datasets |
| Tab 1 | Comparison | Positioning against prior work (last two columns are the claim) |
| Tab 2 | Descriptive | Dataset statistics (rebuilt Amazon-Book counts) |
| Tab 3 | Descriptive | The five sources and their production costs (reused from SignalShap) |
| Tab 4 | Results | RQ1: fidelity table with decomposition verification row |
| Tab 5 | Results | RQ2: order-validity table with violation pairs |
| Tab 6 | Results | RQ3: reflexivity table with aggregate identity row |
| Tab 7 | Results | RQ4: realized quality under decision rules |
| Tab 8 | Results | Significance tests and effect sizes |

---

# PLANNING NOTES (NOT part of the manuscript)

## Why this is likely to be accepted

The claim is narrow, fully supported, and — unusually for this literature — *about measurement rather than about a new model*. There is no new estimator to attack (exact Shapley, 32 coalitions), no new dataset to defend (the pair and the rebuild are already disclosed and vetted in SignalShap), and no oversold theorem (three short propositions, each a few lines, proofs in the appendix). The framing directly answers the group's own declared limitation, which is the strongest possible internal consistency story: the thesis defines actionable insight and says it was never measured; **ActionShap measures it at factor level; ShapAct measures the construction level that ActionShap's own intervention set excludes**. The decision-level comparison (RQ4) is the kind of result reviewers can re-derive in an afternoon, which is a feature, not a risk.

**Positioning discipline — the ActionShap overlap must be addressed head-on, in the paper and in the cover letter.** A reviewer will know (or the group will tell them) that ActionShap already claims "intervention-grounded evaluation of cooperative attribution." The differentiation is one paragraph, to be stated in §1.2 and §2.5 in the same words: (a) *intervention class* — ActionShap perturbs factors/interactions inside a trained model within an elicited budget; ShapAct removes a component from the system's construction (never-built counterfactual), which ActionShap's intervention set does not contain; (b) *game* — ActionShap's dynamic regime is DyHuCoG's Monte-Carlo interaction game; ShapAct's is the exact five-source game, so every reported gap is free of sampling error; (c) *reflexivity* — re-attribution after acting, defined by no file in the folder; (d) *decision level* — ShapAct compares decommissioning decision rules by realized outcome, including the LOO-vs-Shapley test of SignalShap Proposition 2. If the group prefers to submit only one paper, the smallest reframing is to fold §3.3–§3.5 of this blueprint into ActionShap as a construction-level study in its dynamic regime — but ActionShap's recsys host (DyHuCoG) is not faithfully reimplementable (see the guardrail record), which is itself the strongest argument for keeping ShapAct on the SignalShap pipeline.

**Sequencing.** ShapAct is the paper that comes *after* both SignalShap and ActionShap: it consumes SignalShap's pipeline and ActionShap's metric family/paradigm. If SignalShap's RQ3 (segment heterogeneity) fails and that paper thins, ShapAct is unaffected: it does not depend on the segmentation contribution at all. If SignalShap's redundancy prediction (POP↔REC) fails, ShapAct's RQ1 expectation shifts but the audit remains well-defined — the falsification table says what to do. If ActionShap is submitted first, ShapAct's framing must cite it as the paradigm paper (as drafted here); if ActionShap's dynamic regime stalls on the DyHuCoG reimplementation problem, ShapAct becomes the only recsys-side intervention paper in the group and should say so explicitly.

## Build order (each step is independently checkable)

1. Verify SignalShap gates 1–4 have passed (data, scorers, fusion, coalition sweep, tests 1–7 of its spec).
2. Implement the L1 regeneration harness; **assert the L0≡L1 no-op check** (if the retired source contributed zero unique candidates for every user, L1 must equal L0 — this validates the regeneration code itself).
3. Implement L2 never-built for all five sources (cheap: no extra base training); verify the Proposition 1 decomposition identity in a unit test.
4. Compute $P_g, R_g, F_g$; check the RQ1 registered predictions (§B.2) — **run this before writing any prose**.
5. Order-validity statistics ($\tau$, top-k agreement, co-monotonicity violations).
6. Reflexivity recomputation (exact game on the never-built system).
7. Decision-rule executor (four rules, L2 evaluation) + significance tests.
8. Full-retraining variant (top-1), cost-adjusted results, LaTeX emitters.

## What can be reused from existing work

The SignalShap pipeline wholesale (data, five scorers, normalization, fusion, game, per-user decomposition); the **ActionShap codebase, now in the repository** (`paper-ideas/ActionShap/code/actionshap/`) — `stats.py` (paired tests, Holm–Bonferroni, Cohen's $d_z$), `metrics.py` (AIA, top-$k$ precision, regret — adapt to retirement outcomes), `attribution.py` (the uniform Attributor interface over TreeSHAP/KernelSHAP/LIME/permutation, useful for the feature-SHAP decision rule), and the freeze-hash pattern in `modifiability.py` for pre-registering the intervention protocols; the DyHuCoG protocol conventions (seeds {42..46}, paired tests, Holm–Bonferroni, Wilcoxon, Cohen's $d_z$, mean ± std, 95% CIs, cross-validation table format); SignalShap Proposition 2 and its Appendix D variant; ActionShap Proposition 1 and its appendix proofs as the citation target for the factor-level alignment result this paper must not re-derive; the thesis's proofs (Appendix A) as the citation target for any game-theoretic background. **Nothing is re-derived**; the guardrail against re-proving efficiency, redundancy collapse, MC convergence, or alignment-under-local-linearity applies here as it does to the rest of the group's drafts.

## Estimated effort

Roughly two to three weeks of implementation on top of a passing SignalShap codebase, dominated by the L1 regeneration harness and the L2 bookkeeping, not by anything game-theoretic. The L0 game and its verification are already done by SignalShap.

## Decisions taken

| Decision | Choice | Consequence |
|---|---|---|
| Host game | SignalShap's exact five-source game, reused unmodified | The audit inherits exactness, cost, and vetting; novelty is confined to measurement; explicitly NOT DyHuCoG's MC game (un-rebuildable per `dyhucog_spec.md`) and NOT ActionShap's factor-level game |
| Relation to ActionShap | ShapAct is the **construction-level companion**: reuses its metric family (AIA/P@k/regret adapted to retirement outcomes) and cites it as the paradigm paper | Required for a defensible gap after ActionShap's addition to the folder; the one-paragraph differentiation (§1.2, §2.5) is the submission-critical text |
| Primary action studied | Single-source retirement (decommission) | Matches the decision SignalShap §5.1 already names; investment decisions flagged as future work; complements ActionShap's factor-modification interventions |
| L2 semantics | Never-built: source untrained, candidates regenerated, fusion refit; surviving scorers kept as trained | The faithful reading of the decision at bounded cost; retraining-level adaptation bounded by the §4.7 variant; the never-built condition is the single quantity ActionShap's intervention set does not contain |
| Full-retraining variant | Restricted to top-1 retirement per dataset | Keeps the laptop-CPU claim; the only expensive increment |
| Dataset pair | MovieLens-1M + rebuilt Amazon-Book (SignalShap pair) | Continuity with the recsys line; LastFM-2K held in reserve; overlaps ActionShap's dynamic pair by design — the differentiation is the intervention class, not the data |
| Decision rules | Shapley / LOO / feature-SHAP / random | The comparison that makes RQ4 a claim about explanation *designs*; LOO-vs-Shapley is the decision-level test of SignalShap Prop. 2 that ActionShap does not run |
| Method name | **ShapAct** | Deliberately distinct from the group's **ActionShap** paper/codebase now in the repo; never conflate in text or code comments |

## Remaining open questions

- Whether the reflexivity recomputation should be done for the top-1 retirement only (cheap, primary) or for all five retirements (five more exact-game runs, still minutes). Default: all five — the reflexivity table is stronger with the full matrix, and the cost is trivial.
- Whether the "regenerated candidate set" should drop below the round-robin truncation $N=200$ when a source is removed (the union shrinks). Default: keep $N=200$ and let the round-robin rebalance among the remaining four sources; report the union size as a diagnostic.
- Whether to include the cost-adjusted rule (Remark 1) in the main text or only Appendix E. Default: appendix; the cost column is deployment-specific and a reviewer fight is not worth the headline.

## Submission checklist (Discover AI)

- [ ] Abstract ≤ ~250 words, no citations.
- [ ] All declarations present (including the group's standard AI-tools declaration).
- [ ] Data + code availability with working links (SignalShap repo + rebuilt split).
- [ ] Figures ≥ 300 dpi; vector where possible.
- [ ] Reproducibility: seeds, hyperparameters, hardware reported.
- [ ] Self-citation lineage included (thesis, DyHuCoG, SignalShap, Explanation Drift).
- [ ] Statistical tests with multiple-comparison correction (house protocol).
- [ ] Falsification: every registered prediction in the Implementation Spec §B.2–B.5 reported against, misses flagged in the text.
- [ ] Check Morocco APC discount eligibility (Research4Life) before acceptance.

---

# CONVENTIONS AND DISCREPANCIES (guardrail record — keep in the planning notes, do not delete)

Every claim about prior work in this blueprint is traceable to a specific file and section. The following discrepancies between documents in the folder were identified and resolved; flag them wherever the manuscript touches them rather than silently picking one.

1. **DyHuCoG paper vs. thesis, equations 11–12.** The published DyHuCoG paper gives $y_{ui} = (1+a_{ui})\cdot e_i^\top$ (its Eq. 11, dimensionally incomplete — a bare vector transpose) and $f(u,i,c_{u,i}) = \langle e_u, e_i\rangle + \lambda_c \cdot g(c_{u,i}) + a_{ui}$ (its Eq. 12, mixing an embedding inner product with a context-encoder term). The thesis (§7.5.4) corrects these to $y_{ui} = (1+a_{ui})\langle e_u, e_i\rangle$ and $f = y_{ui} + \lambda_c\langle g(c_{u,i}), e_{c_{u,i}}\rangle$, which are dimensionally consistent. **Use the thesis versions when citing DyHuCoG's scoring, and cite both.**
2. **Amazon-Book provenance.** The DyHuCoG paper (Table 2) and the thesis (Table C.1) use the canonical split counts (52,643/91,599/2,984,108) with no discussion of its construction; the SignalShap Implementation Spec (§A.3) documents that this split has no ratings/timestamps/metadata and must be rebuilt from Amazon Reviews 2018, and it is the most recent and most detailed convention. **Adopt the SignalShap rebuild; disclose the divergence in §4.1.**
3. **Implicit-positive threshold.** DyHuCoG paper and thesis use "rating > 3"; SignalShap's spec writes "rating ≥ 4". These coincide for the integer ratings of both datasets; state one (use ≥ 4, the SignalShap spec's wording) and note the coincidence in a sentence.
4. **Theory: asserted vs. proved.** The DyHuCoG paper asserts the MC estimator's $O(1/M)$ variance decay inline; the thesis proves it (Appendix A.3) and also proves Shapley uniqueness (Appendix A.1) and the hierarchical-consistency proposition (Prop 6.1 / Appendix A.2), which the published IJACSA paper states without proof. **Cite the thesis for the proofs; never re-prove them.**
5. **DyHuCoG codebase trustworthiness.** SignalShap's structure doc ("Critical dependency note") documents an extraction audit finding 63 gaps that preclude faithful DyHuCoG reimplementation. The full audit now lives in `paper-ideas/ActionShap/code/docs/dyhucog_spec.md` (added to the repository on main, commit 8b64599) and reaches the same verdict with the full catalogue: 22 blocking gaps (hypergraph construction, all loss weights, $sim(u,i)$, $ContextScore(S)$, ...), 8 broken equations as printed (Eqs. 11, 12 dimensionally broken and mutually contradictory; Eqs. 6 vs. 7 two propagation rules), 14 protocol contradictions (the 70/10/20-vs-leave-one-out clause, the unfiltered Table 2 statistics, the canonical Amazon-Book split with no ratings/timestamps, the Table 4 fold values that are exact arithmetic progressions, the misassigned Holm thresholds in Table 9, ...), and 12 total silences (no code, no data, no hyperparameter table). **ShapAct therefore depends on the SignalShap codebase only, never on DyHuCoG code.** This also means: (a) FairShap's assumption of reusing DyHuCoG code is no longer safe — recorded for the group's planning; (b) **ActionShap's own plan to "reuse all four thesis datasets and both existing codebases" inherits the same risk on its dynamic regime** — its static regime (Wine, Beijing) is implemented and tested in `code/`, but its dynamic regime (ML-1M, Amazon-Book, DyHuCoG Shapley vs. attention) has no implemented code in the repository and cannot be built faithfully from the DyHuCoG paper; this is the single strongest argument for hosting ShapAct's recsys audit on SignalShap's exact pipeline instead.
6. **ActionShap (added to the folder 2026-08-01, commit 8b64599 on main) — the paper this blueprint must be sequenced against.** Files: `ActionShap_Paper_Structure.md`, `actionshap.tex` (full draft), `actionshap-bibliography.bib`, and `code/` (static-regime implementation: `actionshap/{metrics,intervention,modifiability,attribution,decomposition,rerank,stats,data,models/static}.py`, pytest suite, extraction-audit docs `docs/{clustering_spec,dyhucog_spec}.md`, annotation rubric + provisional wine annotation, and `results/raw/wine_static.json`). Verified facts: (i) the claim "no one has measured actionability" is **no longer available** — ActionShap claims the capstone framing and operationalizes thesis Definition 1.1 into AS/AIA/P@k/regret; (ii) its interventions are factor-level feasible perturbations (Definition 2) and its recsys interventions are interaction/context/exposure perturbations on DyHuCoG's MC game — **no construction-level never-built counterfactual, no exact source game, no post-intervention re-attribution (reflexivity)**; (iii) the code is static-regime only; `requirements.txt` lists torch/captum for the dynamic regime but no dynamic implementation exists; (iv) the wine result in `results/raw/wine_static.json` is explicitly marked provisional (annotation file header: "PROVISIONAL — NOT AN ELICITATION ... a smoke test, not a finding") and must not be cited; (v) the clustering_spec audit confirms the published clustering pipeline is internally inconsistent (PCA computed but never consumed; missing-value "imputation vs omission" contradiction; wind direction categorical and unhandled) — the static implementation resolves these by documented implementer choice, which is the house pattern ShapAct should follow for its own protocol decisions.
7. **Discrepancy between ActionShap's plan and its own audit.** ActionShap's structure doc says "reuse all four thesis datasets and both existing codebases, needs no GPU beyond the existing RTX 4090"; its own `docs/dyhucog_spec.md` concludes the DyHuCoG codebase cannot be faithfully reimplemented. When ShapAct's related work cites ActionShap's dynamic regime, state the plan as a plan and the audit as the evidence that the source-level exact pipeline is the reproducible host. Do not silently pick one side.
8. **Modifiability elicitation exists in ActionShap and is not needed in ShapAct.** ActionShap's rubric (three levels 1.0/0.5/0.0, pre-registered freeze with payload-hash enforcement in `code/actionshap/modifiability.py`) applies to *factors*. ShapAct's sources are all modifiable by construction (each is a real pipeline a team could build or remove), so no elicitation is required — but the *intervention protocols* (L0/L1/L2 budgets, $N$, the round-robin rule) should be frozen with the same discipline; reuse the freeze-hash pattern for the protocol YAML.
