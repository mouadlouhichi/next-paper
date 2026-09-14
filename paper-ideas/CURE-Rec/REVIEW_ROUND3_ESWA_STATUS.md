# Round-3 ESWA review — fix status

Status snapshot taken on 2026-09-01 against branch `arena/01a05ca6-next-paper`,
commit `29d3f319c72b69ac63dda4984fe7f45b6d1e1bcb` ("latex fixes", 2026-08-28);
**item 2 was implemented and executed on 2026-09-02**. Every item below was
checked against files on disk, not against the previous round's notes.

Manuscript under review: `paper/cure-rec-eswa/cure-rec-eswa.tex`
(1,263 lines, ~17,343 words). Companion evidence:
`code/results/reviewer_phase_assets/` (337 checksummed entries in
`SHA256SUMS.txt`).

**Bottom line (updated 2026-09-02): item 2 -- the review's second
non-negotiable -- is implemented, executed, numerically verified against the
archived evidence, and written into the manuscript. Remaining: 18 of 29 items
not started, 7 partially covered by previous-round evidence, 1 already
satisfied, 2 summary rows. 1 of the 5 non-negotiables is complete.**

## Item-by-item status

| # | Review item | Status | Evidence on disk |
|---|---|---|---|
| 1 | Large predeclared portfolio-decision challenge benchmark (100–300 envs) | **Not started** | No `challenge` token anywhere in `code/cure_rec/`, `code/scripts_review/`, `code/configs/`. Largest existing design is the 25-configuration LHS study recorded in `REVIEWER_REVISION_PHASES.md`. |
| 2 | Seed+scenario robust (or chance-constrained) planner to fix held-out feasibility | **DONE 2026-09-02** | `cure_rec/seed_robust.py` + `scripts_review/phase_h_seed_robust.py`. Held-out feasibility rises 0.70→1.00 (lhs-012) and 0.47→1.00 (lhs-009) for one frozen portfolio, at a measured cost of 0.031746 / 0.053897 utility (paired $d_z=-137.1$ / $-81.2$, n=20 evaluation seeds); chance-constrained α=0.2 is intermediate (0.55, cost 0.008973). Published protocol reproduced exactly (−0.089270/0.70 and +0.198583/0.47). Assets: `results/reviewer_phase_assets/seed_robust_planner/`. Manuscript: new §`sec:seed-robust` + `tab:seed-robust`. |
| 3 | Paper-to-paper comparison table (8–12 closest methods) | **Not started** | `tab:capability-gap` (tex L87–105) still has 4 approach-family rows + CURE-Rec; caption reads "representative formulations of related approach families". |
| 4 | Update literature to 2025–2026 / recent ESWA work | **Not started** | `cure-rec-bibliography.bib` year histogram: 8×2024, 1×2025, 1×2026. Both recent entries are the authors' own prior work (`louhichi2025`, `louhichi2026`). No ESWA 2025/2026 citation added. |
| 5 | New theory: decision stability under value-estimation error; margin-preserving repair theorem | **Not started** | `\begin{theorem}` count = 0. Only the 4 existing propositions (`prop:efficiency`, `prop:well-defined`, `prop:repair`, `prop:admissibility`, tex L434–446). |
| 6 | Scalability into the main paper | **Partial** | n=8 integrated game archived (`integrated_scalability/players_8/integrated_summary.csv`: 7,505.2 s, 113.93 MB, Shapley efficiency gap 0.0, selection regret 0.0, decision = `repeat_cap`) but reported in **Appendix F** (`tab:integrated-scalability`, tex L1129). n=10 integrated game **not run**: no `players_10/` directory; notebook 13 cell 9 is commented out; Appendix I states "the integrated $n=10$ run is not presented as completed". |
| 7 | Retention construct (behavioural mechanism or drop from utility) | **Not started** | tex L269 still `Ret = clip(1 − Fat + 0.25 Sat, 0, 1)`; `w_r = 0.30` still in the primary utility; L271 and L861 still only disclose the coupling. |
| 8 | Alternative provider-fairness metric (Jain / exposure deviation) | **Not started** | No `jain`, `exposure_deviation`, or alternative-fairness token in code or manuscript. |
| 9 | Alternative cost semantics (activity-based, hybrid) | **Not started** | Only the membership charge (`Definition~\ref{def:cost}`); the sole no-op handling in code is the counter `stats["injection_noop"]` in `interventions.py` L352. |
| 10 | Canonical-order / injection-capacity `q` / no-op-charge sensitivity | **Not started** | Single canonical order asserted in tex §3.3; `injection_capacity: 2` fixed in both `curesim_full.yaml` and `curesim_quickstart.yaml` L27; no order-permutation experiment exists. |
| 11 | Uncertainty attitudes: nominal → mean → CVaR → maximin | **Partial** | Maximin-vs-mean × hard-vs-penalty is implemented (`revision_suite.objective_ablation`) and archived (`tables/phase_b_objective_constraint_ablation.csv`, 24 rows), reported as a null result at tex L1123 ("selects repeat cap for all tested penalty coefficients {0,…,10}"). **CVaR is absent from code and manuscript**, and the comparison is offline selection on one archived game, not the disjoint-seed held-out version the review asks for. |
| 12 | Learned base rankers inside CURE-Sim, in the main paper | **Partial** | BPR-MF learned base executed and archived (`semireal_integration/semireal_comparison.csv`: `learned_bpr_logged_feedback` → repair/`repeat_cap`, lower improvement 0.33538 vs 0.29851 handcrafted) but reported in **Appendix F** (`tab:semireal`, tex L1149). No SASRec base inside CURE-Sim: `semireal.py` contains only `LearnedBPRPolicy`/`fit_logged_bpr`; `sasrec.py` serves the MovieLens audit only. |
| 13 | Evaluate all eligible MovieLens users | **Not started** | tex L485: "Evaluation then proceeds in ascending user-identifier order over the first 1,000 warm-target users"; L490 and Table 9 caption report `n=1,000`. `models.py` defaults `max_users: int = 1_000` (L57, L192, L207). |
| 14 | One more recommender family (LightGCN / BERT4Rec) | **Not started** | `models.py` provides `PopularityRecommender`, `BPRMFRecommender`, `PopularityHybrid`; SASRec in `sasrec.py`/`torch_models.py`. No graph or transformer baseline. |
| 15 | Explanation-fidelity evaluation (removal effect) + decision card in main paper | **Partial** | `explanation_card.json` fields are decision / interpretation / selected+scenario attributions / interactions / rejected-or-deferred — **no removal-effect or drop-one field**; no `removal_effect`/`drop_one` code exists. A `figure_06_decision_card.png` exists inside archived run assets, but the ESWA figure set is only `figure_02`–`figure_05`. |
| 16 | Decision-quality statistics (optimal-selection rate, feasibility-failure rate, abstention accuracy, …) | **Partial** | Only `selection_regret_vs_oracle` / `selection_regret_vs_best_feasible` in `phase_g_integrated_scalability.py` L184–185, for a single n=8 configuration. No cross-environment decision-quality metrics. |
| 17 | Rename/reframe the disagreement section as mechanistic failure cases | **Not started** | RQ3 is still "when portfolio reasoning changes the decision" (`sec:divergent-selectors`, tex L625); Table caption still "Two configurations selected by the disagreement-screening rule". |
| 18 | Move scalability + learned-base diagnostics into the main article | **Partial** | Deliberately the opposite today: tex L765 "Approximation, common-random-number, scalability, and learned-base-policy diagnostics are retained in Appendix~\ref{app:additional-diagnostics}". |
| 19 | Fix float placement (Table 2 near §3.3, etc.) | **Not addressed** | `placeins` is loaded but all 6 `\FloatBarrier` calls are at L1085–1162 (appendices only). All 13 main-body floats use `[t]`, 12 of them `table*` (L87, 213, 522, 547, 575, 594, 631, 648, 681, 728, 746, 773, 799). **Rendered page position not verified here — no `pdflatex` in this sandbox.** |
| 20 | Shorten the manuscript 15–20% | **Not started** | Still 1,263 lines / ~17,343 words (comments stripped). |
| 21 | Retitle toward the application problem | **Not started** | Title unchanged: "CURE-Rec: Constrained Portfolio Selection for Sequential Recommendation via Fully Enumerated Cooperative Games". |
| 22 | Rewrite the abstract around the new benchmark | **Not started** | Abstract still leads with "In a 20-seed behavioural study, repeat control is selected in every seed, with mean robust lower improvement $0.29371\pm0.00258$". |
| 23 | Add the "why this is an ESWA paper" framing paragraph | **Not started** | No occurrence of "decision-support", "intelligent system", or "expert system" in the manuscript. |
| 24 | Reorganize around the decision problem | **Not started** | Layout unchanged: 1 Intro, 2 Literature, 3 Background, 4 Methodology, 5 Results (RQ1–RQ6), 6 Discussion, 7 Conclusion, Appendices A–I. |
| 25 | Full experiment matrix | **Summary row** | See items 1, 2, 6, 8, 9, 10, 12, 13. |
| 26 | Keep the limitations, use them to motivate new experiments | **Satisfied (nothing to do)** | Limitations section intact (tex L857–871); construct-validity, external-validity, and Gini disclosures unchanged. |
| 27 | Reproducibility box/table | **Partial** | Prose subsection "Reproducibility and reporting checklist" (tex L871) + prose Appendix I artifact inventory; no per-artifact ✓ table. |
| 28 | Restructured RQs (RQ1–RQ6 with MovieLens demoted) | **Not started** | RQ list at tex L514 unchanged; MovieLens still occupies RQ5 as a full research question. |
| 29 | The five non-negotiables | **1 of 5 complete** | (1) not started; **(2) done 2026-09-02**; (3) evidence exists but sits in Appendix F; (4) not started; (5) not started. |

## Previous-round work that is already done and reusable

Archived and checksummed under `code/results/reviewer_phase_assets/` — these are
round-1/round-2 assets, and they are the raw material for items 6, 11, 12, 16:

- `objective_constraint_sweeps/` — 14 predeclared utility-weight vectors
  (13/14 select `repeat_cap`; `cost_insensitive` switches to mask 5 =
  repeat_cap + tail_slot) and a 135-point constraint frontier
  (90 repair / 30 improve / 15 abstain).
- `divergent_selector_holdout/lhs-012`, `/lhs-009` — selection seeds 42–46,
  disjoint evaluation seeds 200–219, 8 selectors each.
- `semireal_integration/` — learned BPR base inside CURE-Sim (single seed 42).
- `integrated_scalability/players_8/` — exact n=8 integrated game.
- `scalability/` — exact vs sampled Shapley at n=6/8/10, budgets 32–2048
  (timing/fidelity on the value table; not full n=10 rollouts).
- `crn_simulator/`, `crn_click_feedback/`, `movielens_1m_paired/`,
  `second_dataset/` (MovieLens-25M negative-transfer audit).

## Still open from the *previous* round

- Integrated **n=10** game (`phase_g_integrated_scalability.py 10 42`) — ~6–8 h,
  notebook 13 cell 9 commented out.
- **Run D** Amazon-2014 (Toys & Games) second-domain audit — no
  `results/reviewer_phase_assets/second_domain/` directory exists; only
  `second_dataset/` (MovieLens-25M) is archived.

## What could not be verified in this sandbox

- **Manuscript compilation / rendered float positions (item 19):** no `pdflatex`
  or `latexmk` on PATH; `paper/cure-rec-eswa/` contains no built PDF. Source-level
  float placement is reported above, but the "Table 2 lands on page 39" symptom
  could not be reproduced or confirmed as fixed.
- **Any experiment re-run:** `numpy`/`pandas` are not importable here
  (`ModuleNotFoundError: No module named 'numpy'`), so no code path in
  `cure_rec/` was executed. All numbers above were read from archived CSV/JSON
  artifacts, not recomputed.

## Suggested order of attack (matches the review's own priority list)

1. ~~Item 2 — multi-seed/chance-constrained planner.~~ **Done 2026-09-02.**
2. Item 1 — challenge benchmark. Reuse `phase_e_offline_sweeps.py` recombination
   machinery so most environments need no new rollouts; `seed_robust.py` now
   supplies the per-configuration decision metrics it needs.
3. Items 6 + 18 + 12 — move the existing n=8 scalability and learned-base tables
   from Appendix F into the main paper (manuscript-only edit, no new compute),
   then run n=10 unattended.
4. Items 7, 8, 9, 10 — construct-sensitivity block; all four are offline
   recombinations of archived coalition outcomes except the retention mechanism,
   which needs new rollouts.
5. Items 3, 4, 19, 20, 21, 22, 23, 24, 28 — manuscript work, no compute.
