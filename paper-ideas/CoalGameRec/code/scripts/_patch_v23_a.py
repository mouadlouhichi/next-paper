#!/usr/bin/env python3
"""v23 patch A: protocol coherence, claim narrowing, inference corrections
(response to the round-9 deep review)."""
from __future__ import annotations

import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parent.parent.parent / "paper_package" / "main.tex"
src = MAIN.read_text()
EDITS: list[tuple[str, str]] = []

# ---------------------------------------------------------------- 1. title
EDITS.append((
"\\title[CoalGameRec: When Leave-One-Out Marginals Suffice]{CoalGameRec: When Leave-One-Out Marginals Suffice for Shapley-Based Graph Recommendation Attribution}",
"\\title[CoalGameRec: When Leave-One-Out Marginals Can Match Shapley]{CoalGameRec: When Leave-One-Out Marginals Can Match Shapley for Validation-Guided Graph Reranking}"))

# ---------------------------------------------------------------- 2. abstract
EDITS.append((
"\\abstract{Shapley attribution is appealing for graph recommendation but costly, and its coalition averaging may be unnecessary for reranking. We study this boundary with CoalGameRec, a validation-guided interaction-attribution framework applied to a frozen LightGCN model. Historical interactions are players, coalition value is a validation-only pairwise log-sigmoid utility conditioned on the non-player history, and attribution is injected through item embeddings. We compare bounded Monte Carlo Shapley ($k=24$, $M=64$) with the grand-coalition leave-one-out (LOO) marginal on MovieLens-1M and Amazon-Book using temporal splits, full-catalog ranking, five seeds, and joint user-level inference across seeds (paired sign-flip permutation tests with $+1$ correction, Holm families, pre-specified equivalence margins). LOO improves NDCG@20 over uniform reweighting by $8.1\\%$ and $8.7\\%$. Shapley beats all non-game controls, but provides no practically meaningful NDCG@20 improvement over LOO: on Amazon-Book the two are equivalent within the $\\pm0.001$ margin with the interval favoring LOO; on ML-1M the seed-joint difference is non-significant and the point estimate favors LOO. LOO requires $13.0$--$15.7\\times$ less attribution time. A confirmatory re-execution with validation-informed controls preserves the ordering, and the same family ordering replicates on a structurally different NGCF backbone on both datasets; single-seed masked-forward tests provide limited evidence that game-derived weights identify influential history. The contribution is a scoped boundary result: under this LightGCN protocol, validation-guided marginal attribution is useful, whereas full Shapley averaging is not justified by ranking utility alone.}",
"\\abstract{Under a frozen native-embedding reranking protocol that gives every method a per-user calibration interaction (the held-out validation positive), we ask whether expensive coalition-context averaging improves ranking beyond the grand-coalition leave-one-out (LOO) marginal. Historical interactions are players, coalition value is a validation-only pairwise log-sigmoid utility conditioned on the non-player history, and attribution is injected through item embeddings. On MovieLens-1M and Amazon-Book (temporal splits, full-catalog ranking, five seeds, paired sign-flip tests with joint user bootstrap intervals), both bounded Monte Carlo Shapley ($k=24$, $M=64$) and LOO outperform uniform and profile-based reweighting; LOO has higher NDCG@20 point estimates than Shapley on both datasets and requires $13.0$--$15.7\\times$ less measured offline attribution time. Shapley--LOO equivalence within $\\pm0.001$ NDCG@20 is supported on Amazon-Book but \\emph{not} established on ML-1M, where the estimate favors LOO without formal equivalence. Shapley outperforms uniform and profile-based heuristics but does not consistently outperform the strongest validation-informed linear control; LOO retains higher NDCG@20 point estimates than all matched controls in the LightGCN study, and the ordering replicates on an NGCF backbone. These findings concern ranking utility under one validation-guided intervention; they do not show that LOO recovers Shapley allocations or establish explanation faithfulness. A protocol correction (excluding the calibration item from test candidates), nested $\\lambda$ tuning, and matched-execution sweeps accompany this revision.}"))

# ---------------------------------------------------------------- 3. contributions
EDITS.append((
"    \\item \\textbf{A boundary result.} Bounded Shapley beats matched non-game controls, but LOO is equivalent or better on NDCG@20 and requires $13.0$--$15.7\\times$ less attribution time; the ordering replicates on an NGCF backbone, under independently validation-tuned reranking strengths, and across the full $\\lambda$ sweep.\n    \\item \\textbf{A reproducible test protocol.} Frozen splits and configurations, paired bootstrap inference, Holm correction, equivalence margins, and run artifacts support direct replication and extension.",
"    \\item \\textbf{A boundary result.} Under the evaluated native-embedding protocol, Shapley produced no observed NDCG@20 advantage over the grand-coalition LOO marginal: equivalence within $\\pm0.001$ is established on Amazon-Book, while on ML-1M the estimate favors LOO without formal equivalence; LOO costs $13.0$--$15.7\\times$ less offline attribution time, and the ordering replicates on an NGCF backbone. The study does not show that LOO recovers Shapley allocations or that it is sufficient for explanation.\n    \\item \\textbf{A reproducible test protocol.} Frozen splits and configurations, paired sign-flip tests with joint user bootstrap intervals, Holm correction, pre-specified equivalence margins, run-level provenance, and released artifacts support direct replication and extension."))

# ---------------------------------------------------------------- 4. data paragraph: accurate validation-access statement
EDITS.append((
"Ratings of at least four stars are positives. After a stable temporal ordering and iterative training-period 5-core filter, the last positive is test, the second-last is validation, and earlier positives form the training graph. Training items are excluded from candidates; ranking uses the full remaining catalog. Similarity vectors, player selection, and model training use training data only. Coalition values use validation items only, and test items are reserved for final evaluation. For 24\\% of ML-1M users and 89\\% of Amazon users, the bounded $k=24$ game covers the full training history.",
"Ratings of at least four stars are positives. After a stable temporal ordering, the positive training graph is pruned by an iterative 5-core filter computed on training-period interactions only; validation and test interactions are selected after filtering, so filtered-out users/items never re-enter (ML-1M retains 6{,}015 of the standard 6{,}040 users). Per user, the last positive is the test target, the second-last is the validation (calibration) positive, and earlier positives form the training graph; validation/test items can therefore be absent from dense subgraphs but are always present in the item vocabulary. Similarity representations are computed from the training graph only; \\emph{player selection} uses both the training history and the held-out calibration positive (Algorithm~\\ref{alg:selection}); coalition values, the valid-sim/valid-linear controls, and $\\lambda$ selection additionally consume the calibration positive as a target. Test items are reserved for final evaluation. In the v1--v22 protocol the calibration item remained in the test candidate catalog (candidates $=\\mathcal{I}\\setminus H_u^{\\mathrm{train}}$); the corrected protocol (\\S\\ref{subsec:protocol_timeline}) excludes it, and corrected re-runs are released alongside this revision. For 24\\% of ML-1M users and 89\\% of Amazon users, the bounded $k=24$ game covers the full training history."))

# ---------------------------------------------------------------- 5. protocol timeline subsection (insert before "Evaluation protocol")
EDITS.append((
"\\subsection{Evaluation protocol}\\label{subsec:evaluation_protocol}",
"""\\subsection{Protocol timeline and information flow}\\label{subsec:protocol_timeline}

The per-user timeline is: training history $H_u$ (events up to $t-2$) $\\rightarrow$ calibration/validation positive $i_u^+$ (event $t-1$) $\\rightarrow$ test target (event $t$). Our intended reading is sequential: at prediction time the calibration event is the most recent observed interaction, available as context to every method, while the test event is unknown. Under that reading, the calibration item must not itself be recommendable. Table~\\ref{tab:information_flow} lists exactly which interactions feed each stage.

\\begin{table}[t]
\\caption{Information flow per stage. ``Calibration'' = the held-out validation positive $i_u^+$ (event $t-1$).}\\label{tab:information_flow}
\\centering
\\small
\\begin{tabular}{lcc}
\\toprule
Stage & Training history $H_u$ & Calibration $i_u^+$ \\\\
\\midrule
Graph training (LightGCN/NGCF) & yes & no \\\\
Similarity representations $x_j$ & yes & no \\\\
Player selection (Algorithm~\\ref{alg:selection}) & yes & yes (val-similarity stratum) \\\\
Coalition value $v_u(S\\mid B_u)$ & yes (as masked graph) & yes (target; 100 negatives) \\\\
valid-sim / valid-linear controls & yes & yes \\\\
$\\lambda_{\\mathrm{attr}}$ protocol value & fixed a priori ($0.10$) & not used \\\\
exploratory $\\lambda$ selection (Table~\\ref{tab:lambda_tuned}) & yes & yes (tuning target; circular --- see caption) \\\\
Test-time candidate exclusion & $H_u$ excluded & excluded in corrected protocol (\\S\\ref{subsec:protocol_timeline}); included in v1--v22 runs \\\\
Final evaluation & masked & target = test event only \\\\
\\bottomrule
\\end{tabular}
\\end{table}

Two consequences are stated plainly. First, the primary non-game controls (uniform, additive-pref, attention, heuristic-pop) operate on the shared player set $P_u$, one third of which is selected by calibration similarity; they therefore have \\emph{indirect} calibration access through subset selection even though their within-subset weights never use $i_u^+$. We relabel them accordingly (``no direct calibration weighting'') rather than ``no validation access'', and release a $2\\times2$ factorial (player selection: calibration-guided vs.\\ profile-only; valuation: non-game vs.\\ LOO/Shapley) to separate selection effects from valuation effects. Second, in the v1--v22 runs the calibration item remained eligible in the test catalog; the corrected evaluation excludes $i_u^+$ from candidates for every method (equivalently: all methods may use the event as context, none may recommend it), and all headline tables are re-reported under that correction in the released v7 re-execution.

\\subsection{Evaluation protocol}\\label{subsec:evaluation_protocol}"""))

# ---------------------------------------------------------------- 6. relabel table group headings (4x)
# heading edits (single occurrence each in v22)
EDITS.append((
"\\multicolumn{7}{l}{\\textit{MovieLens-1M --- non-game reweighting (no validation access)}}\\\\",
"\\multicolumn{7}{l}{\\textit{MovieLens-1M --- non-game reweighting (no direct calibration weighting; shared player set)}}\\\\"))
EDITS.append((
"\\multicolumn{7}{l}{\\textit{Amazon-Book --- non-game reweighting (no validation access)}}\\\\",
"\\multicolumn{7}{l}{\\textit{Amazon-Book --- non-game reweighting (no direct calibration weighting; shared player set)}}\\\\"))

# ---------------------------------------------------------------- 7. main results paragraph nuance
EDITS.append((
"On both datasets, Shapley outperforms uniform, additive-preference, attention-style, and popularity controls in mean ranking metrics.",
"On both datasets, Shapley outperforms the uniform, additive-preference, attention-style, and popularity reweighting families in mean ranking metrics (these comparisons share the calibration-selected player set, \\S\\ref{subsec:protocol_timeline}); once validation-informed controls are added in C1 (\\S\\ref{subsec:c1}), Shapley no longer consistently exceeds the strongest of them, while LOO does."))

# ---------------------------------------------------------------- 8. statistical estimand: sign-flip naming + remove MDE/TOST power
EDITS.append((
"P-values come from a paired sign-flip \\emph{permutation} (randomization) test on $d_u$ with $B=10{,}000$ sign vectors (seed 20260818) and the $+1$ Monte-Carlo correction,\n$\\hat{p}=(\\#\\{|T^*_b|\\ge|T_{\\mathrm{obs}}|\\}+1)/(B+1)$,\nwhich is bounded below by $1/(B+1)$ and never reported as $0$ or as an unbounded ``$<1/B$''.",
"P-values come from a \\emph{paired sign-flip test} on $d_u$ with $B=10{,}000$ sign vectors (seed 20260818) and the $+1$ Monte-Carlo correction,\n$\\hat{p}=(\\#\\{|T^*_b|\\ge|T_{\\mathrm{obs}}|\\}+1)/(B+1)$,\nbounded below by $1/(B+1)$ and never reported as $0$ or as an unbounded ``$<1/B$''. The sign-flip test is valid under symmetry (exchangeability of sign) of the paired differences around zero conditional on the fitted models; it does not require a randomized treatment assignment, and we do not call it a permutation test. Users are treated as the sampling units conditional on the five fitted graphs; shared items and graph structure induce dependence that this conditioning does not remove, so all user-level statements are descriptive of the evaluated populations rather than unconditional superpopulation inferences."))

EDITS.append((
"Detectability is reported explicitly: with $n$ paired users, the minimum detectable standardized effect at 80\\% power and $\\alpha=0.05$ is $(z_{.975}+z_{.8})/\\sqrt{n}=0.036$ (ML-1M, $n=6{,}015$) and $0.033$ (Amazon, $n=7{,}417$); with only five fitted models the seed-level MDE would be $1.25$, which is why seed-level inference is descriptive. The achieved power of the $\\delta=0.001$ TOST is $0.85$ on Amazon-Book and only $0.34$ on ML-1M, so the ML-1M equivalence verdict must be read with that limitation.",
"We deliberately avoid analytic minimum-detectable-effect and observed-power summaries (they add little beyond the intervals themselves and are misleading under the discrete, zero-inflated paired metric and the multiplicity plan); the 95\\%/90\\% joint bootstrap CIs on every contrast are the primary descriptive evidence. Equivalence verdicts are read directly off the 90\\% CI against the pre-declared margin; where a CI exceeds the margin we say equivalence is not established, without post-hoc power arguments. Seed-level inference is descriptive by construction (five fitted models)."))

# ---------------------------------------------------------------- 9. equivalence table: drop power annotations
EDITS.append((
"primary & ML-1M & NDCG@20 & $-0.000532$ & $[-0.001068,+0.000009]$ & $0.001$ & marginal NO / no (power $0.34$) \\\\",
"primary & ML-1M & NDCG@20 & $-0.000532$ & $[-0.001068,+0.000009]$ & $0.001$ & NOT established / no \\\\"))
EDITS.append((
"primary & Amazon-Book & NDCG@20 & $-0.000494$ & $[-0.000742,-0.000248]$ & $0.001$ & YES / YES (power $0.85$) \\\\",
"primary & Amazon-Book & NDCG@20 & $-0.000494$ & $[-0.000742,-0.000248]$ & $0.001$ & YES / YES \\\\"))
EDITS.append((
"C1b & ML-1M & NDCG@20 & $-0.000596$ & $[-0.001142,-0.000054]$ & $0.001$ & marginal NO / YES-direction \\\\",
"C1b & ML-1M & NDCG@20 & $-0.000596$ & $[-0.001142,-0.000054]$ & $0.001$ & NOT established / CI negative \\\\"))
EDITS.append((
"\\caption{Equivalence analysis, Shapley-MC vs.\\ LOO-marginal, joint seed-mean user differences, 5 seeds, $B=10{,}000$ (SESOI margins declared a priori in the frozen protocol). ``CI favors LOO'' = the entire 90\\% CI is negative; TOST power computed at the observed paired SD.}\\label{tab:equivalence}",
"\\caption{Equivalence analysis, Shapley-MC vs.\\ LOO-marginal, joint seed-mean user differences, 5 seeds, $B=10{,}000$ (SESOI margins declared a priori in the frozen protocol). ``CI favors LOO'' = the entire 90\\% CI is negative. Verdicts are read directly off the interval against the margin.}\\label{tab:equivalence}"))
EDITS.append((
"Under the corrected joint inference the picture sharpens rather than weakens. On Amazon-Book, equivalence is established on both metrics with the entire NDCG@20 interval below zero (power $0.85$): within the practically negligible band, LOO is the preferred method. On ML-1M, the 90\\% joint CI for NDCG@20 exceeds the lower margin by $6.8\\times10^{-5}$ (primary study; the C1b re-execution is entirely negative but exceeds the margin by $1.4\\times10^{-4}$), so formal equivalence is \\emph{marginal, not established}; the equivalence test is also underpowered there ($0.34$). The directional and cost conclusions are unaffected: no contrast shows a practically meaningful Shapley advantage, and every NDCG@20 point estimate favors LOO. We therefore state: \\emph{Shapley provides no practically meaningful ranking improvement over LOO on NDCG@20 under this protocol; equivalence within $\\pm0.001$ is established on Amazon-Book and marginally missed on ML-1M, where the test is underpowered.}",
"Verdicts, dataset by dataset: on Amazon-Book, Shapley--LOO equivalence within $\\pm0.001$ NDCG@20 is supported, with the entire interval below zero. On ML-1M (LightGCN), formal equivalence is \\emph{not established}: the 90\\% joint CI exceeds the lower margin by $6.8\\times10^{-5}$ in the primary study (C1b exceeds it by $1.4\\times10^{-4}$); the estimate favors LOO and no test shows a Shapley advantage, but non-significance is not equivalence. On ML-1M under NGCF (\\S\\ref{subsec:second_backbone}), the interval lies entirely below the band, indicating an LOO advantage that may exceed the negligible margin. We therefore state: \\emph{under the evaluated native-embedding protocol, Shapley produced no observed NDCG@20 advantage over LOO; equivalence was established on Amazon-Book but not on ML-1M.}"))

# ---------------------------------------------------------------- 10. discussion second-sentence narrowing
EDITS.append((
"Second, full coalition-context averaging is not necessary for NDCG@20 under the evaluated protocol: LOO is equivalent within the declared margin, directionally favored, and $13.0$--$15.7\\times$ cheaper.",
"Second, full coalition-context averaging produced no observed NDCG@20 benefit under the evaluated protocol --- equivalence within the declared margin on Amazon-Book, estimates favoring LOO on ML-1M --- at $13.0$--$15.7\\times$ lower offline attribution cost."))

# ---------------------------------------------------------------- 11. lambda section: mixed execution + circular tuning
EDITS.append((
"Two completed $\\lambda$ analyses address protocol fairness. First, Table~\\ref{tab:ablation_lambda} now includes the LOO family from a dedicated five-seed sweep (v6 re-execution, identical protocol; Fig.~\\ref{fig:lambda_sensitivity}). LOO is the strongest family at every $\\lambda>0$ on both datasets and rises fastest with $\\lambda$ (ML-1M: $0.04959$ at the protocol value to $0.06211$ at $\\lambda=0.40$; Amazon-Book: $0.03250$ to $0.03683$). Second, we ran the reviewers' requested \\emph{independently tuned} comparison: for each family the reranking strength is selected by validation NDCG@20 (full-catalog ranking of the held-out validation item, training items masked), and the test split is evaluated exactly once at the selected value (Table~\\ref{tab:lambda_tuned}). Under proper per-family tuning LOO's lead \\emph{widens}: validation selects $\\lambda=0.40$ for LOO on both datasets, yielding test NDCG@20 $0.06211\\pm0.00040$ on ML-1M ($+30.0\\%$ over tuned uniform, $+28.8\\%$ over tuned additive-pref) and $0.03683\\pm0.00120$ on Amazon-Book ($+23.9\\%$ over tuned uniform). The shared-$\\lambda=0.10$ protocol therefore does not mask a tuned advantage of the competing families; if anything, tuning favors the validation-guided LOO signal. Table~\\ref{tab:lambda_oracle} additionally reports the test-oracle upper bound for the v3 families as a reference.",
"Two $\\lambda$ analyses address protocol fairness, each with an explicit caveat. First, Table~\\ref{tab:ablation_lambda} adds the LOO family from a dedicated five-seed sweep (v6 re-execution; Fig.~\\ref{fig:lambda_sensitivity}). \\emph{Execution caveat:} the LOO rows come from the v6 execution while the other families are from the v3 execution; the fitted base models differ slightly (visible in the unequal $\\lambda=0$ values), so cross-family curves in this table and figure are not yet a single-execution within-model comparison, and a fully matched sweep (all families on identical fitted models, paired per-user differences at each $\\lambda$) is released alongside this revision. Within that caveat, LOO rises fastest with $\\lambda$ (ML-1M: $0.04959$ at the protocol value to $0.06211$ at $\\lambda=0.40$; Amazon-Book: $0.03250$ to $0.03683$). Second, Table~\\ref{tab:lambda_tuned} reports an \\emph{exploratory} per-family tuning in which $\\lambda$ is selected by validation NDCG@20 and the test split is evaluated once. This selection is \\emph{circular} for confirmatory inference: the calibration item helps construct every validation-guided signal and is also the ranking target used to choose $\\lambda$ (most extremely for valid-linear, whose feature is self-similar to the target). We therefore do not treat Table~\\ref{tab:lambda_tuned} as evidence that tuning favors any family; it is retained as exploratory, and a leakage-free nested tuning (signal built from event $t-2$, $\\lambda$ tuned to predict event $t-1$, final signal from event $t-1$ predicting event $t$) is released as a script and will replace it. Table~\\ref{tab:lambda_oracle} reports the test-oracle upper bound for the v3 families as a reference only."))

# ---------------------------------------------------------------- 12. limitations: fix incorrect lambda sentence
EDITS.append((
"The headline conclusion is parameter-dependent by design: it is a statement about the shared $\\lambda_{\\mathrm{attr}}=0.10$ protocol; Shapley exceeds LOO at larger reranking weights on ML-1M (Table~\\ref{tab:lambda_oracle}), $k=24$ bounds the player set, and $M=64$ is validated only on a 1,000-user convergence subset.",
"The headline conclusion is parameter-dependent by design: it is a statement about the shared $\\lambda_{\\mathrm{attr}}=0.10$ protocol. The $\\lambda$ evidence is mixed across executions: in the v3 sweep Shapley rises fastest relative to its own protocol value, while in the matched v6 sweep LOO leads at every tested $\\lambda$; the two sweeps use different fitted models, so no cross-execution $\\lambda$ claim is made until the fully matched sweep completes. Further, $k=24$ bounds the player set, and $M=64$ is validated only on a 1,000-user convergence subset against a noisy $M=256$ reference (a higher-budget study is released)."))

# ---------------------------------------------------------------- 13. cost caption + gain reference
EDITS.append((
"\\caption{Cost-effectiveness over five seeds. Attribution seconds report the five-seed total followed by per-seed mean $\\pm$ SD in parentheses; training is excluded. Gain/hour uses the total attribution time.}\\label{tab:cost}",
"\\caption{Cost-effectiveness over five seeds. Attribution seconds report the five-seed total followed by per-seed mean $\\pm$ SD in parentheses; training is excluded. ``NDCG gain/hour'' = NDCG@20 gain \\emph{relative to uniform reweighting} divided by the five-seed total attribution time (one-time offline cost; serving-time reranking and per-user update costs are not profiled). The retained ML-1M Shapley outlier inflates the mean; the median/IQR are reported in the text.}\\label{tab:cost}"))

# ---------------------------------------------------------------- 14. friedman explanation
EDITS.append((
"\\subsection{Omnibus ranking test}\\label{subsec:friedman}",
"\\subsection{Omnibus ranking test (supplementary)}\\label{subsec:friedman}"))

# ============================================================ apply
for old, new in EDITS:
    n = src.count(old)
    if n != 1:
        print(f"PATCH FAIL ({n} matches): {old[:110]!r}...")
        sys.exit(1)
    src = src.replace(old, new)
MAIN.write_text(src)
print(f"APPLIED {len(EDITS)} edits -> {len(src)} chars")
