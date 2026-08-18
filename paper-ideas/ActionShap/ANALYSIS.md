# ActionShap — Implementation, Results, and Paper Analysis

*Working notes produced 2026-08-05 from a read-through of `paper-ideas/ActionShap` (branch `arena/019fd335-next-paper`), including the schema-v2 release tables, the `paper-v3` manuscript, the revision-4 spec/audit trail, and the full test-suite run (95 passed, 1 legacy skip).*

**Canonical artifacts:** spec = `ActionShap_Recommendation_Spec.md` (rev 4); manuscript = `paper-v3/actionshap.tex`; evidence = `paper-v3/final/` (validation `PASS`, 0 errors / 0 warnings / 34 disclosed notes); executable contract = `code/configs/final.yaml` + `code/scripts/run_final_suite.py`.

---

## 1. What ActionShap is (current form)

ActionShap started as a cross-domain proposal (clustering + recommendation, Wine/Beijing/MovieLens/Amazon, author-elicited modifiability, AS/AIA/H1–H3 decomposition). The Q1 audit invalidated the schema-v1 pilot, and revision 4 replaced it with a **recommendation-only** evaluation protocol:

> Given a frozen, history-conditioned recommender, does a user-specific attribution ranking predict the effect of a *feasible, bounded* intervention on the interaction history, and does it lead to good joint intervention decisions under a fixed budget?

Core objects:

- **Players:** the user's `n_max = 20` most recent *training* interactions (validation/test excluded). Older context `B_u` is used for fitting but deliberately **not** scored.
- **Models:** ItemKNN (primary; scores = weighted mean of frozen item–item cosine similarities over the *retained* history, so masking/downweighting moves outputs at inference) and a leave-one-out-BPR profile aggregator (robustness-only, disclosed as weaker than popularity).
- **Attribution game:** continuous **target-margin** utility `v(S) = σ(s_y − mean top-10 competitor scores)` (chosen by an independent convergence preflight because discrete NDCG coalition values never converged); **NDCG@10** remains the separately labelled operational outcome.
- **Interventions:** deletion `ρ=0` (faithfulness diagnostic) vs bounded downweighting `w_p ← 0.5·w_p` (primary feasible action; `ρ=0.25` sensitivity). No retraining.
- **Methods:** Monte-Carlo Shapley (antithetic prefix walks, `M_pair=500`, `T=1000`), locally weighted LIME, leave-one-out (LOO), greedy sequential deletion, random control.
- **Metrics:** deletion AIA, bounded AIA, their difference = **Actionability Gap**; signed alignment; top-k precision; success/abstention; NDCG and target-margin regret against an **exact B≤2 oracle** over {∅, singletons, pairs}; within-user permutation null (R=1000); stability; independent convergence selection vs an `M=1000` reference.
- **Inference:** distinct users (5 seeds averaged *within* user), user bootstrap, paired plus-one sign permutation, Holm correction.

## 2. Implementation review (`code/`, ~10.4k LoC, 95/95 non-legacy tests pass)

Module map:

| Module | Role | Notable detail |
|---|---|---|
| `recommendation_data.py` | temporal splits | deterministic `(timestamp, record_index)` ordering; last=test, penultimate=validation |
| `candidates.py` | fixed candidate sets | negatives exclude the **complete** pre-test history incl. validation; global seeded tie priority reused everywhere |
| `models/itemknn.py` | primary model | sparse cosine co-occurrence, top-200 non-zero neighbours, non-negative by construction; `score_downweighted_batch` vectorizes all intervention profiles |
| `models/profile.py` | robustness model | LOO-BPR item embeddings frozen; profile recomputed at inference |
| `recommendation.py` | game core | `UserGame` validation; target-rank/NDCG with explicit tie-break; cached antithetic `mc_shapley`; **two separate action rules** (`select_joint_action` for magnitude diagnostics vs `select_downweight_action` for signed benefit with abstention) |
| `evaluation.py` | effects/oracles/metrics/nulls/gate | batched prefix-walk Shapley; `single_player_effects(rho=0)` evaluated as exact set-difference (not weight-zeroing) so LOO ≡ deletion to 1e-12; exhaustive B≤2 oracle with per-utility oracles and no-action; within-user null with plus-one p; masking gate **with inert static control** |
| `baselines.py` | LOO / LIME / greedy / random | LIME = full+empty+all-LOO masks + random masks, Hamming kernel (width .25), weighted ridge — a genuine local surrogate |
| `stats.py` | user-level inference | seeds averaged within distinct user before any resampling; Holm; Cohen's dz |
| `scripts/` | orchestration | `run_final_suite.py` freezes the 2×2×5 matrix + sensitivities; `make_paper_assets.py` (1,980 LoC) is a validator+builder that refuses to emit assets unless the schema-v2 contract holds |

**Engineering strengths (these are the load-bearing parts of the paper's credibility):**

1. **Degeneracy prevention.** The masking-sensitivity gate (§7.1.1) with an exactly-inert static control kills the worst failure mode (static-embedding models giving flat results mistaken for null findings). ItemKNN passes all gates; the profile model's one failed ML seed is retained and disclosed.
2. **Circularity traps avoided by construction.** LOO is labelled an oracle at B=1 (its deletion AIA is 1.0 *by algebra*, enforced to 1e-12 by the exact-set-difference deletion path); the scientific comparison lives at B=2 where joint effects break the LOO identity. Prefix-walk efficiency (telescoping, ~1e-18) is demoted to a cache/arithmetic check; convergence is judged against an independently seeded M=1000 reference on rank + signed-B=2 action Jaccard, never on efficiency.
3. **Leakage discipline.** Models fit on complete histories; candidates exclude the complete pre-test history; candidate/user/tie/model/attribution seeds are independent and frozen; tie-breaking is a seeded catalogue-wide priority, not argsort order.
4. **Decision realism.** Action space includes no-action; abstention allowed and reported; signed benefit rule (`−φ > 0`) separates magnitude prediction from beneficial-action claims; NRegret conditional on positive oracle, ε=1e-12, non-negativity validated.
5. **Chance calibration and correct statistical unit.** Within-user matched nulls (null means ≈ 0, e.g. p95 ≈ 0.006 ML / 0.013 Amazon), random control, plus-one p-values, distinct-user inference (never 5×1000 = 5000 pseudoreplicates).
6. **Release gating.** `validation_report.json` = PASS with 0 errors/0 warnings; 34 notes are *disclosed* robustness limitations (profile underperforms popularity, etc.), not hidden.

**Minor implementation observations:**

- `permutation_importance` is an alias of LOO; for single-player deletion this is mathematically the same object, but the spec's intent (an independent baseline) collapses to 4 real methods + random. Acceptable, should just never be presented as two baselines.
- LIME's kernel width 0.25 on normalized Hamming makes random masks (distance ≈ 0.5) weigh ≈ e⁻⁴ ≈ 0.02, i.e. LIME effectively fits the full profile + its LOO neighbourhood. That *explains* its ≈ deletion-level alignment; fine as a frozen contract, but a one-line note in the discussion would preempt a reviewer deriving it.
- Some CSV rows have truncated/joined columns only in my `cut` views, not in the files themselves — the `.csv` sources are complete; the `.tex` exports are intentionally compact.

## 3. Results analysis (schema-v2, `paper-v3/final/tables`)

**Primary ItemKNN, sampled candidates, distinct users (ML n=1000; Amazon n=993):**

| Method | ML deletion AIA | ML bounded AIA | ML gap | AMZ deletion | AMZ bounded | AMZ gap |
|---|---|---|---|---|---|---|
| MC Shapley | 0.762 | 0.779 | **+0.017** | 0.286 | 0.414 | **+0.129** |
| LIME | 0.951 | 0.933 | −0.018 | 0.930 | 0.827 | −0.103 |
| LOO | 1.000 | 0.978 | −0.022 | 1.000 | 0.851 | −0.149 |
| Greedy | 0.321 | 0.318 | −0.002 | 0.191 | 0.131 | −0.060 |
| Random | −0.001 | 0.000 | +0.001 (CI∋0) | −0.008 | −0.010 | −0.002 |

Interpretation:

1. **The headline is real but narrow.** Shapley is the only method whose alignment *improves* when the estimand moves from deletion to bounded downweighting, in all nine singleton ItemKNN conditions (e.g. ML full-catalogue gap +0.046, Amazon full-catalogue +0.169). The random control's gap straddles zero in the primary conditions, and the within-user nulls centre at ~0 with every real method at the plus-one floor p=0.001. So the gap is a genuine, non-chance perturbation-robustness signal.
2. **But absolute alignment says the opposite.** LIME/LOO dominate bounded AIA everywhere (0.93/0.98 vs 0.78 on ML; 0.83/0.85 vs 0.41 on Amazon). As a *predictor* of bounded intervention effects, Shapley loses to a local ridge surrogate. Part of this is mechanical: LOO's gap is ≤0 by algebra; LIME is fitted essentially on the deletion neighbourhood. The paper states this plainly ("the gap measures a change of evaluation target; it is not an explanation-validity score") — good, because the Amazon full-catalogue row shows a positive gap (+0.169) arising from **two negative alignments** (−0.288 → −0.119). A reader skimming the abstract could still misread "+gap = Shapley better"; the three-panel Fig. 2 helps.
3. **Decision quality is a wash with a local-method edge.** B=2 NDCG outcomes: ML ΔNDCG 0.0431 (LOO) / 0.0425 (LIME) / 0.0403 (Shapley), success 28.9/28.8/27.4%, conditional NRegret (339 active users) 0.217/0.225/0.268. Amazon: 0.0313/0.0331/0.0333, NRegret (196 active) 0.232/0.197/0.206. Greedy (NRegret 1.47 ML) and random (1.20 AMZ, negative ΔNDCG) fail as required. Only ~20–34% of users have an improving B=2 action at all (active-oracle counts), i.e. the operational headroom is small and the NDCG landscape is flat — the honest read is "explanations barely move deployed outcomes, and no method wins consistently."
4. **Convergence is the quiet weak spot.** Target-margin selects M_pair=50–250, but only 43–89% of *individual* users meet both per-user thresholds even at M=1000 (selection is on aggregate means, reported as a diagnostic); NDCG attribution never converges at M=1000 and is demoted to a stress test. The conservative floor M=500 is defensible, but the utility choice is thereby *forced* rather than free — disclosed, correctly.
5. **Robustness.** The positive Shapley gap persists across candidate size (100/500), history caps (50/100), ρ=0.25, and full-catalogue; longer histories show reduced masking sensitivity (supporting n_max=20 as an operating boundary); the profile model supplies a useful negative boundary (changes rankings yet underperforms popularity; one gate failure). Sensitivity rows in `aia_components.csv` are consistent with the headline.

Bottom line: the data support exactly the paper's safe claim — *Shapley's target-margin alignment changes favourably relative to deletion across ItemKNN conditions; local methods retain higher absolute alignment and slightly better ML decisions; a positive gap is not standalone validity.* They do **not** support any "Shapley is more actionable" reading, and the paper (to its credit) never makes one.

## 4. Paper assessment (`paper-v3/actionshap.tex`)

**Strengths.**
- Contribution framing matches evidence: the protocol (leakage-safe, abstention-aware, null-calibrated, distinct-user, budget-exact) *is* the contribution; method rankings are explicitly empirical and bounded. This "evaluation instrument" genre is publishable at Discover AI and similar venues.
- Exceptional internal consistency: every trap the audit identified (LOO circularity, vacuous efficiency, static-model degeneracy, label conflation, pseudoreplication, forced harmful actions) has a corresponding mechanism in code, a corresponding table, and matching wording in Limitations/Declarations. The three algorithms boxes + contract table (Appendix A/B) make the paper unusually auditable.
- Honest negative controls and boundaries (random control near null; profile model kept despite failing quality/gate; NDCG non-convergence disclosed).

**Risks / weaknesses (reviewer-facing).**
1. **Impact vs. rigor imbalance.** The empirical deltas are small (ML gap +0.017) or negative-absolute (Amazon full catalogue). Reviewers may see a very careful protocol wrapped around a modest finding. The paper's own discipline ("gap ≠ validity") shrinks the headline further; the contribution argument must lean on the protocol and the faithfulness-vs-bounded *separation* result, which it mostly does.
2. **Retrospective, target-conditioned audit.** The characteristic function conditions on the held-out target; nothing here is prospective. Disclosed, but the word "actionability" in the title lineage still invites a stronger reading than the design supports. The title chosen ("Beyond Deletion Faithfulness toward Intervention-Robust…") is appropriately modest.
3. **Architecture coverage.** "Architecture-agnostic protocol" is demonstrated on exactly one healthy architecture (ItemKNN). The robustness model is a disclosed failure case. No attention/sequential/differentiable-utility model is included (justified: no attention mechanism; NDCG non-differentiable), but a reviewer may still ask for one.
4. **Sampled-ranking scope.** 200-item candidate sets; full-catalogue only 250 users with sparse NDCG correlations. Correctly labelled, but the Amazon negative bounded alignments under full catalogue remind that margin-AIA on tiny candidate pools is the strongest evidence.
5. **Hygiene.** Multiple parallel generations coexist (`actionshap-overleaf/`, `paper-v2/`, `paper-v3/`, legacy docs). `REVIEW_RESPONSE.md` and `RUN_RECOMMENDATION.md` still reference old paths (`paper/paper.tex`, `overleaf-springer/`, `paper/legacy_pilot/`). The overleaf tex is numerically current but duplicates paper-v3. Before submission: pick one canonical manuscript directory, fix stale path references, make the repo public, and mint the Zenodo DOI the release section promises.

## 5. Verdict

The implementation is among the most self-critical I have seen at this scale: the spec pre-commits write-ups, the code enforces the spec, the asset builder gates the manuscript, and the tests (95 passing) pin the algebra (LOO identity, gate inertness, abstention, oracle non-negativity). The results are honest, null-calibrated, and correctly scoped: they establish that **deletion faithfulness and bounded-intervention alignment are different estimands**, that **Shapley is the most perturbation-robust but not the most locally accurate explainer**, and that **decision-level gains are small and dataset-dependent**. The residual risk is novelty/impact perception and the retrospective scope, not rigor.

Suggested follow-ups, in priority order: (1) consolidate versioned directories and stale path pointers; (2) add one sentence on the effective locality of LIME under the frozen kernel width; (3) report the per-user fraction with positive gap alongside the mean gap (readers currently have to dig into CSVs for dispersion); (4) if one more experiment is ever run, a single differentiable sequential model with its own intervention semantics would answer the architecture-coverage objection without disturbing the frozen matrix.
