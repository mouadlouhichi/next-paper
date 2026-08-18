#!/usr/bin/env python3
"""Round-6 (v19) manuscript patch: applies all reviewer-mandated revisions."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TEX = ROOT / "manuscript_assets" / "round6"
MAIN = ROOT / "paper_package" / "main.tex"

src = MAIN.read_text()
orig_len = len(src)

paired_rows = (TEX / "tab_paired_round6.tex").read_text().strip()
c1_paired_rows = (TEX / "tab_c1_paired_round6.tex").read_text().strip()
c1_shap_rows = (TEX / "tab_c1_shap_round6.tex").read_text().strip()
robust_rows = (TEX / "tab_robustness_sensitivity.tex").read_text().strip()
fried_rows = (TEX / "tab_friedman.tex").read_text().strip()

EDITS: list[tuple[str, str]] = []

# ---------------------------------------------------------------- abstract
EDITS.append((
"""\\abstract{Shapley attribution is appealing for graph recommendation but costly, and its coalition averaging may be unnecessary for reranking. We study this boundary with CoalGameRec, a validation-guided interaction-attribution framework applied to a frozen LightGCN model. Historical interactions are players, coalition value is a validation-only pairwise log-sigmoid utility, and attribution is injected through item embeddings. We compare bounded Monte Carlo Shapley ($k=24$, $M=64$) with the grand-coalition leave-one-out (LOO) marginal on MovieLens-1M and Amazon-Book using temporal splits, full-catalog ranking, five seeds, and paired user bootstrap inference. LOO improves NDCG@20 over uniform reweighting by $8.1\\%$ and $8.7\\%$, respectively. Shapley also beats the non-game controls, but provides no practically meaningful NDCG@20 improvement over LOO: the pre-specified equivalence intervals lie within $\\pm0.001$ and favor LOO on both datasets. LOO requires $13.0$--$15.7\\times$ less attribution time. A confirmatory re-execution with validation-informed controls preserves the same ordering, while single-seed masked-forward tests provide limited evidence that both game-derived weights identify influential history. The contribution is therefore a scoped boundary result: under this LightGCN protocol, validation-guided marginal attribution is useful, whereas full Shapley averaging is not justified by ranking utility alone.}""",
"""\\abstract{Shapley attribution is appealing for graph recommendation but costly, and its coalition averaging may be unnecessary for reranking. We study this boundary with CoalGameRec, a validation-guided interaction-attribution framework applied to a frozen LightGCN model. Historical interactions are players, coalition value is a validation-only pairwise log-sigmoid utility conditioned on the non-player history, and attribution is injected through item embeddings. We compare bounded Monte Carlo Shapley ($k=24$, $M=64$) with the grand-coalition leave-one-out (LOO) marginal on MovieLens-1M and Amazon-Book using temporal splits, full-catalog ranking, five seeds, and joint user-level inference across seeds (paired sign-flip permutation tests with $+1$ correction, Holm families, pre-specified equivalence margins). LOO improves NDCG@20 over uniform reweighting by $8.1\\%$ and $8.7\\%$. Shapley beats all non-game controls, but provides no practically meaningful NDCG@20 improvement over LOO: on Amazon-Book the two are equivalent within the $\\pm0.001$ margin with the interval favoring LOO; on ML-1M the seed-joint difference is non-significant and the point estimate favors LOO. LOO requires $13.0$--$15.7\\times$ less attribution time. A confirmatory re-execution with validation-informed controls preserves the ordering, while single-seed masked-forward tests provide limited evidence that game-derived weights identify influential history. The contribution is a scoped boundary result: under this LightGCN protocol, validation-guided marginal attribution is useful, whereas full Shapley averaging is not justified by ranking utility alone.}"""))

# ---------------------------------------------------------------- keywords
EDITS.append((
"\\keywords{recommender systems, LightGCN, explainable AI, cooperative game theory, Shapley value, leave-one-out attribution, post-hoc reranking}",
"\\keywords{recommender systems, graph neural networks, LightGCN, graph explainability, explainable AI, cooperative game theory, Shapley value, leave-one-out attribution, post-hoc reranking, computational efficiency}"))

# ---------------------------------------------------------------- principle 2: conditional game
EDITS.append((
"""\\subsection{Principle 2: the value function must be smooth enough to attribute}\\label{subsec:principle_value}

Hard top-$K$ utilities are flat for many coalitions. We therefore attribute a smooth validation-only pairwise utility and reserve HitRate and NDCG for final evaluation.
For user $u$, validation positive $i_u^+$, fixed validation negatives $\\mathcal{N}_u^-$, and masked-graph scores $s_S$, the coalition value is
\\begin{equation}\\label{eq:pairwise}
v_u(S)=\\frac{1}{|\\mathcal{N}_u^-|}\\sum_{i^-\\in\\mathcal{N}_u^-}\\log\\sigma\\!\\left(s_S(u,i_u^+)-s_S(u,i^-)\\right).
\\end{equation}
The test item is never used in this value, player selection, or tuning.""",
"""\\subsection{Principle 2: the value function must be smooth enough to attribute}\\label{subsec:principle_value}

Hard top-$K$ utilities are flat for many coalitions. We therefore attribute a smooth validation-only pairwise utility and reserve HitRate and NDCG for final evaluation.
The game is \\emph{conditional}: the non-player part of the training history,
\\begin{equation}\\label{eq:background}
B_u = H_u\\setminus P_u,
\\end{equation}
is kept active in the graph for every coalition, and only player edges are toggled. Formally the characteristic function is $v_u(S\\mid B_u)$, which we write as $v_u(S)$ when $B_u$ is fixed by the protocol; for users with $|H_u|>k$ the players account for a bounded subset of the evidence, so each attribution is a claim about the selected interactions given the retained background. For user $u$, validation positive $i_u^+$, fixed validation negatives $\\mathcal{N}_u^-$, and masked-graph scores $s_S$, the coalition value is
\\begin{equation}\\label{eq:pairwise}
v_u(S\\mid B_u)=\\frac{1}{|\\mathcal{N}_u^-|}\\sum_{i^-\\in\\mathcal{N}_u^-}\\log\\sigma\\!\\left(s_S(u,i_u^+)-s_S(u,i^-)\\right).
\\end{equation}
The test item is never used in this value, player selection, or tuning."""))

# ---------------------------------------------------------------- principle 3: linearity = decomposition only
EDITS.append((
"A Shapley method containing this term is partly a similarity method. The primary game therefore sets the additive coefficient to zero and evaluates similarity as a separate control.",
"A Shapley method containing this term is partly a similarity method. Linearity establishes this \\emph{decomposition} only: it does not by itself imply any ranking-performance ordering, because the $\\phi_j(v)$ component may still add information, so we never infer from Eq.~(\\ref{eq:phipref}) that a combined method cannot beat a pure similarity heuristic. The primary game therefore sets the additive coefficient to zero and evaluates similarity leakage with matched validation-informed controls (valid-sim, valid-linear, \\S\\ref{subsec:c1})."))

# ---------------------------------------------------------------- principle 4 recast
EDITS.append((
"The intervention should be stated separately from the attribution rule. We use native LightGCN item embeddings in the primary protocol \\cite{he2020lightgcn,wang2019ngcf}; the external-kernel ablation later shows that model alignment does not guarantee higher ranking accuracy.",
"The intervention should be stated separately from the attribution rule. We use native LightGCN item embeddings in the primary protocol \\cite{he2020lightgcn,wang2019ngcf}. Because the external-kernel ablation (\\S\\ref{subsec:design_ablations}) actually yields \\emph{higher} NDCG@20 than the native intervention, this principle is a construct/model-alignment design recommendation --- attributions act in the model's own representation space --- not a ranking-performance hypothesis."))

# ---------------------------------------------------------------- related work expansion
EDITS.append((
"Recent work also cautions that perturbation fidelity can be misleading under graph distribution shift \\cite{zheng2024robust,fang2023evaluating}. Accordingly, we separate ranking utility from explanation faithfulness and limit the empirical claim to the interaction-player LightGCN configuration shown in Fig.~\\ref{fig:architecture}.",
"""Recent work also cautions that perturbation fidelity can be misleading under graph distribution shift \\cite{zheng2024robust,fang2023evaluating}, that GNN explanations can be fragile under small input perturbations \\cite{li2024fragile}, and that redundancy makes explanations of self-interpretable GNNs seed-dependent \\cite{tai2025redundancy}. Systematic evaluation frameworks for graph explanations separate explanation goals from fidelity criteria \\cite{amara2022graphframex,agarwal2022probing}, and simple surrogates can match expensive explainers on some fidelity measures \\cite{pereira2023distilnexplain}; Shapley-based graph explanation in embedding space is an adjacent design point \\cite{ho2024shapleyembedding}. For interaction-aware cooperative-game concepts beyond singleton Shapley values, see interaction indices \\cite{muschalik2024shapiq,schnake2022higher,myerson1977graphs} and the recent Shapley-methodology reviews \\cite{li2024shapleyreview,markchom2025review}. Contrastive graph recommenders (SGL, LightGCL, XSimGCL) \\cite{wu2021sgl,cai2023lightgcl,yu2024xsimgcl} are candidate alternative backbones for the protocol rather than attribution methods. Accordingly, we separate ranking utility from explanation faithfulness and limit the empirical claim to the interaction-player LightGCN configuration shown in Fig.~\\ref{fig:architecture}."""))

# ---------------------------------------------------------------- statistical estimand rewrite
OLD_STATS = """\\subsection{Statistical estimand}\\label{subsec:stat_estimand}

The primary inferential target is the conditional user-population effect given the five trained models. For seeds $s=1..5$ we resample users with replacement within each seed ($B=2000$ paired bootstrap replicates, seed 20260804), compute seed-specific mean differences $\\Delta_s$, and average $\\bar\\Delta=\\frac{1}{5}\\sum_s\\Delta_s$; 95\\% CIs are percentile intervals over $B$ replicates. The primary family is defined explicitly: per dataset, $F=8$ contrasts, Shapley-MC vs.\\ each of uniform, additive-pref, attention, and LOO on NDCG@20 and HitRate@20, Holm-corrected within the family. Coverage/ILD and the descriptive 5-seed means are exploratory and carry no paired contrasts. Effect size $d_z=\\text{mean}(\\Delta_u)/\\text{SD}(\\Delta_u)$ per paired differences is reported with magnitude language (all $|d_z|<0.08$ = negligible). This estimand does not claim uncertainty over all possible model initializations; seed variability is descriptive (mean$\\pm$SD over 5 seeds). The bootstrap hierarchy resamples users within each seed and averages seed-specific means; the same users appear under all five trained models, so the procedure captures user-level variation conditional on the five models and treats the models as fixed. We report this as the declared estimand rather than claiming fully hierarchical cross-model calibration.

Bootstrap p-values are two-sided sign probabilities: with $\\Delta^*_b$ the $b$-th replicate of the mean-of-seed-means statistic, $\\hat{p}=2\\min(\\#\\{\\Delta^*_b\\le0\\}/B,\\ \\#\\{\\Delta^*_b\\ge0\\}/B)$, reported as ``$<1/B$'' when the count is zero ($B=2000$). Holm--Bonferroni step-down correction at $\\alpha=0.05$ is applied within each pre-specified family ($F=8$ primary; $F=10$ LOO-as-treatment; $F=12$ C1); the families are declared a priori and corrected independently.

For equivalence we pre-specify a smallest effect size of practical interest (SESOI): $\\delta=0.001$ for NDCG@20 and $\\delta=0.002$ for HitRate@20, approximately one quarter of the LOO-vs-uniform improvement under this protocol; effects inside $(-\\delta,\\delta)$ are declared practically negligible. Following the TOST--CI equivalence, equivalence holds when the 90\\% bootstrap percentile CI of the Shapley--LOO difference lies entirely inside $(-\\delta,\\delta)$. Results are reported in Table~\\ref{tab:equivalence}."""

NEW_STATS = """\\subsection{Statistical estimand and inference}\\label{subsec:stat_estimand}

The primary inferential target is the conditional user-population effect given the five trained models, treated as fixed; seed variability is reported descriptively (mean$\\pm$SD), and no population-over-initializations claim is made. Because the same users appear under all five models, the analysis unit is the \\emph{joint} seed-mean paired difference
\\[
d_u=\\frac{1}{5}\\sum_{s=1}^{5}\\left(m^A_{u,s}-m^B_{u,s}\\right),
\\]
and all resampling draws user identities jointly across seeds, preserving each user's cross-seed dependence (results conditional on the five fitted models).

P-values come from a paired sign-flip \\emph{permutation} (randomization) test on $d_u$ with $B=10{,}000$ sign vectors (seed 20260818) and the $+1$ Monte-Carlo correction,
$\\hat{p}=(\\#\\{|T^*_b|\\ge|T_{\\mathrm{obs}}|\\}+1)/(B+1)$,
which is bounded below by $1/(B+1)$ and never reported as $0$ or as an unbounded ``$<1/B$''. Percentile bootstrap CIs (95\\% and 90\\%, $B=10{,}000$, joint user resampling) accompany every contrast, and effect sizes $d_z=\\mathrm{mean}(d_u)/\\mathrm{SD}(d_u)$ are reported with bootstrap 95\\% CIs. Wilcoxon signed-rank tests are reported as a sensitivity analysis; the legacy within-seed bootstrap of earlier protocol versions is retained in the released sensitivity artifact to show procedure robustness. Holm--Bonferroni step-down correction at $\\alpha=0.05$ is applied within each pre-specified family ($F=8$ primary; $F=10$ LOO-as-treatment; $F=12$ C1 LOO; $F=8$ C1 Shapley); families are declared a priori and corrected independently, each answering a distinct question (does game attribution beat non-game reweighting; does LOO beat the non-game controls; do the matched controls close the validation-access gap). As an omnibus check we additionally run a Friedman test with users as blocks over all families, with Nemenyi--Holm post-hoc comparisons (Table~\\ref{tab:friedman}).

Detectability is reported explicitly: with $n$ paired users, the minimum detectable standardized effect at 80\\% power and $\\alpha=0.05$ is $(z_{.975}+z_{.8})/\\sqrt{n}=0.036$ (ML-1M, $n=6{,}015$) and $0.033$ (Amazon, $n=7{,}417$); with only five fitted models the seed-level MDE would be $1.25$, which is why seed-level inference is descriptive. The achieved power of the $\\delta=0.001$ TOST is $0.85$ on Amazon-Book and only $0.34$ on ML-1M, so the ML-1M equivalence verdict must be read with that limitation.

For equivalence we pre-specify a smallest effect size of practical interest (SESOI): $\\delta=0.001$ for NDCG@20 and $\\delta=0.002$ for HitRate@20, approximately one quarter of the LOO-vs-uniform improvement under this protocol (declared in the frozen protocol before the primary analysis); effects inside $(-\\delta,\\delta)$ are declared practically negligible. Following the TOST--CI equivalence, equivalence holds when the 90\\% joint-bootstrap percentile CI of the Shapley--LOO difference lies entirely inside $(-\\delta,\\delta)$ (Table~\\ref{tab:equivalence})."""
EDITS.append((OLD_STATS, NEW_STATS))

# ---------------------------------------------------------------- backbone environment sentence
EDITS.append((
"In this paper LightGCN is the sole backbone (hypergraph variants are future work).",
"In this paper LightGCN is the sole evaluated backbone; a structurally different nonlinear backbone (NGCF-style aggregation) is released as a second-backbone run script and is required before any cross-architecture claim. Primary runs executed on macOS-26.6-arm64 (Apple Silicon, MPS device), Python 3.12.13, PyTorch 2.3.1; the confirmatory re-execution uses the identical stack (devices and deviations are recorded in each run's manifest.json)."))

# ---------------------------------------------------------------- naming clarity
EDITS.append((
"    \\item \\textbf{shapley-mc:} antithetic permutation Monte Carlo Shapley over a bounded stratified set of 24 historical interactions.\n\\end{itemize}",
"    \\item \\textbf{shapley-mc:} antithetic permutation Monte Carlo Shapley over a bounded stratified set of 24 historical interactions.\n\\end{itemize}\n\\texttt{loo-marginal} is the internal family name; in results tables the same configuration is labelled \\emph{CoalGameRec (LOO)} where it is the recommended method. The two names denote one algorithm, not two."))

# ---------------------------------------------------------------- main paired table + prose
OLD_PAIRED_BLOCK = """\\begin{table*}[t]
\\caption{Key paired contrasts ($B=2000$, Holm-adjusted within the pre-specified families). Negative Shapley--LOO differences favor LOO; complete contrast tables are released as artifacts.}\\label{tab:paired}\\label{tab:paired_loo}
\\centering
\\scriptsize
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{lllllll}
\\toprule
Dataset & Contrast & Metric & Mean diff. & 95\\% CI & $p$ & $d_z$ \\\\
\\midrule
MovieLens-1M & Shapley vs uniform & NDCG@20 & 0.003216 & [0.002707, 0.003715] & $<0.0005$ & 0.0716 \\\\
MovieLens-1M & Shapley vs uniform & HitRate@20 & 0.008180 & [0.006517, 0.009875] & $<0.0005$ & 0.0550 \\\\
MovieLens-1M & Shapley vs LOO & NDCG@20 & -0.000532 & [-0.000963, -0.000119] & 0.008 & -0.0143 \\\\
MovieLens-1M & Shapley vs LOO & HitRate@20 & 0.000366 & [-0.000899, 0.001696] & 0.575 & 0.0032 \\\\
Amazon-Book & Shapley vs uniform & NDCG@20 & 0.002096 & [0.001703, 0.002474] & $<0.0005$ & 0.0568 \\\\
Amazon-Book & Shapley vs uniform & HitRate@20 & 0.003398 & [0.002427, 0.004395] & $<0.0005$ & 0.0348 \\\\
Amazon-Book & Shapley vs LOO & NDCG@20 & -0.000494 & [-0.000709, -0.000281] & $<0.0005$ & -0.0228 \\\\
Amazon-Book & Shapley vs LOO & HitRate@20 & -0.000701 & [-0.001375, -0.000054] & 0.036 & -0.0108 \\\\
\\bottomrule
\\end{tabular}
}
\\end{table*}

The full LOO-as-treatment family also shows that LOO beats every non-game control on both metrics and datasets (all Holm $p<0.0005$). On HitRate@20, LOO and Shapley are indistinguishable on ML-1M ($p=0.575$), while LOO leads on Amazon-Book ($p=0.036$)."""

NEW_PAIRED_BLOCK = f"""\\begin{{table*}}[t]
\\caption{{Key primary paired contrasts on joint seed-mean user differences ($B=10{{,}}000$ sign-flip permutation replicates with $+1$ correction; Holm-adjusted within the pre-specified $F=8$ family; Wilcoxon signed-rank shown as sensitivity; bootstrap 95\\% CI for $d_z$). Negative Shapley--LOO differences favor LOO; complete contrast tables are released as artifacts.}}\\label{{tab:paired}}\\label{{tab:paired_loo}}
\\centering
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{llllllll}}
\\toprule
{paired_rows}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table*}}

The full LOO-as-treatment family also shows that LOO beats every non-game control on both metrics and datasets (all joint-permutation Holm $p\\le0.001$; released artifact). The Shapley-vs-LOO contrast itself is procedure-sensitive on ML-1M: the conservative seed-joint permutation test does not reject ($p=0.207$ Holm), while the Wilcoxon test does ($p<10^{{-4}}$); on Amazon-Book every procedure favors LOO (Table~\\ref{{tab:robustness}}). We therefore state the result as ``Shapley provides no detectable improvement over LOO, with direction favoring LOO'' rather than claiming a universal significant LOO win.

\\begin{{table*}}[t]
\\caption{{Procedure sensitivity for the Shapley--LOO contrast (joint seed-mean user differences). ``Legacy'' = within-seed user bootstrap of the original protocol, re-run at $B=10{{,}}000$ for comparability; ``---'' = procedure not applicable to that study version. Conclusions are robust for the controls; the Shapley-vs-LOO verdict on ML-1M is procedure-dependent and is reported as such.}}\\label{{tab:robustness}}
\\centering
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{lllllll}}
\\toprule
{robust_rows}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table*}}"""
EDITS.append((OLD_PAIRED_BLOCK, NEW_PAIRED_BLOCK))

# ---------------------------------------------------------------- equivalence rewrite
OLD_EQUIV = """\\subsection{Equivalence of Shapley and LOO within a pre-specified margin}\\label{subsec:equivalence}

Failure to reject a difference is not evidence of equality, so the ``match'' between Shapley and LOO is assessed with an explicit equivalence analysis rather than a non-significant p-value. The smallest effect size of practical interest (SESOI) was declared before computing the test: $\\delta=0.001$ for NDCG@20 and $\\delta=0.002$ for HitRate@20, approximately one quarter of the LOO-vs-uniform improvement under this protocol. Equivalence holds when the 90\\% bootstrap percentile CI of the Shapley--LOO paired difference lies entirely inside $(-\\delta,\\delta)$ (Table~\\ref{tab:equivalence}).

\\begin{table}[t]
\\caption{Equivalence analysis, Shapley-MC vs.\\ LOO-marginal, paired user differences, 5 seeds, $B=2000$ (SESOI margins declared a priori; artifact: released equivalence tables of each v3 run). ``CI favors LOO'' = the entire 90\\% CI is negative.}\\label{tab:equivalence}
\\centering
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{llllll}
\\toprule
Dataset & Metric & Mean diff. & 90\\% CI & Margin $\\pm\\delta$ & Equivalence / CI favors LOO \\\\
\\midrule
ML-1M & NDCG@20 & $-0.000532$ & $[-0.000890,-0.000172]$ & $0.001$ & YES / YES \\\\
ML-1M & HitRate@20 & $+0.000366$ & $[-0.000698,+0.001496]$ & $0.002$ & YES / no \\\\
Amazon-Book & NDCG@20 & $-0.000494$ & $[-0.000680,-0.000309]$ & $0.001$ & YES / YES \\\\
Amazon-Book & HitRate@20 & $-0.000701$ & $[-0.001267,-0.000135]$ & $0.002$ & YES / YES \\\\
\\bottomrule
\\end{tabular}
}
\\end{table}

Equivalence is established on all four contrasts. On NDCG@20 both intervals lie entirely below zero: within the practically negligible band, LOO is the preferred method on both datasets. On HitRate@20, ML-1M is equivalent without directional preference, while Amazon-Book favors LOO. We therefore replace ``LOO matches Shapley'' with the precise statement: \\emph{Shapley provides no practically meaningful ranking improvement over LOO on NDCG@20 under this protocol, and the equivalence interval favors LOO.}"""

NEW_EQUIV = """\\subsection{Equivalence of Shapley and LOO within a pre-specified margin}\\label{subsec:equivalence}

Failure to reject a difference is not evidence of equality, so the ``match'' between Shapley and LOO is assessed with an explicit equivalence analysis rather than a non-significant p-value. The SESOI was declared in the frozen protocol before the primary analysis: $\\delta=0.001$ for NDCG@20 and $\\delta=0.002$ for HitRate@20, approximately one quarter of the LOO-vs-uniform improvement under this protocol. Equivalence holds when the 90\\% \\emph{joint} bootstrap percentile CI of the Shapley--LOO paired difference lies entirely inside $(-\\delta,\\delta)$ (Table~\\ref{tab:equivalence}).

\\begin{table}[t]
\\caption{Equivalence analysis, Shapley-MC vs.\\ LOO-marginal, joint seed-mean user differences, 5 seeds, $B=10{,}000$ (SESOI margins declared a priori in the frozen protocol). ``CI favors LOO'' = the entire 90\\% CI is negative; TOST power computed at the observed paired SD.}\\label{tab:equivalence}
\\centering
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{lllllll}
\\toprule
Study & Dataset & Metric & Mean diff. & 90\\% joint CI & Margin $\\pm\\delta$ & Equivalence / CI favors LOO \\\\
\\midrule
primary & ML-1M & NDCG@20 & $-0.000532$ & $[-0.001068,+0.000009]$ & $0.001$ & marginal NO / no (power $0.34$) \\\\
primary & ML-1M & HitRate@20 & $+0.000366$ & $[-0.001127,+0.001951]$ & $0.002$ & YES / no \\\\
primary & Amazon-Book & NDCG@20 & $-0.000494$ & $[-0.000742,-0.000248]$ & $0.001$ & YES / YES (power $0.85$) \\\\
primary & Amazon-Book & HitRate@20 & $-0.000701$ & $[-0.001170,-0.000053]$ & $0.002$ & YES / YES \\\\
C1b & ML-1M & NDCG@20 & $-0.000596$ & $[-0.001142,-0.000054]$ & $0.001$ & marginal NO / YES-direction \\\\
C1b & Amazon-Book & NDCG@20 & $-0.000627$ & $[-0.000855,-0.000401]$ & $0.001$ & YES / YES \\\\
\\bottomrule
\\end{tabular}
}
\\end{table}

Under the corrected joint inference the picture sharpens rather than weakens. On Amazon-Book, equivalence is established on both metrics with the entire NDCG@20 interval below zero (power $0.85$): within the practically negligible band, LOO is the preferred method. On ML-1M, the 90\\% joint CI for NDCG@20 exceeds the lower margin by $6.8\\times10^{-5}$ (primary study; the C1b re-execution is entirely negative but exceeds the margin by $1.4\\times10^{-4}$), so formal equivalence is \\emph{marginal, not established}; the equivalence test is also underpowered there ($0.34$). The directional and cost conclusions are unaffected: no contrast shows a practically meaningful Shapley advantage, and every NDCG@20 point estimate favors LOO. We therefore state: \\emph{Shapley provides no practically meaningful ranking improvement over LOO on NDCG@20 under this protocol; equivalence within $\\pm0.001$ is established on Amazon-Book and marginally missed on ML-1M, where the test is underpowered.}"""
EDITS.append((OLD_EQUIV, NEW_EQUIV))

# ---------------------------------------------------------------- C1 paired tables replacement
OLD_C1P = """\\begin{table*}[t]
\\caption{C1b LOO contrasts against the validation-informed controls ($B=2000$, Holm-adjusted within the full $F=12$ family). Positive differences favor LOO.}\\label{tab:c1_paired}
\\centering
\\scriptsize
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{lllllll}
\\toprule
Dataset & Contrast & Metric & Mean diff. & 95\\% CI & $p$ (Holm) & $d_z$ \\\\
\\midrule

MovieLens-1M & LOO vs valid-sim & NDCG@20 & +0.001903 & [+0.001335, +0.002434] & $<0.0005$ (rej.) & 0.0394 \\\\
MovieLens-1M & LOO vs valid-linear & NDCG@20 & +0.001263 & [+0.000783, +0.001748] & $<0.0005$ (rej.) & 0.0303 \\\\
MovieLens-1M & LOO vs valid-sim & HitRate@20 & +0.003325 & [+0.001595, +0.004988] & $<0.0005$ (rej.) & 0.0221 \\\\
MovieLens-1M & LOO vs valid-linear & HitRate@20 & +0.002062 & [+0.000532, +0.003525] & 0.004 (rej.) & 0.0158 \\\\
Amazon-Book & LOO vs valid-sim & NDCG@20 & +0.001939 & [+0.001535, +0.002343] & $<0.0005$ (rej.) & 0.0481 \\\\
Amazon-Book & LOO vs valid-linear & NDCG@20 & +0.000754 & [+0.000370, +0.001131] & 0.001 (rej.) & 0.02 \\\\
Amazon-Book & LOO vs valid-sim & HitRate@20 & +0.003586 & [+0.002535, +0.004692] & $<0.0005$ (rej.) & 0.0331 \\\\
Amazon-Book & LOO vs valid-linear & HitRate@20 & +0.000863 & [$-$0.000135, +0.001861] & 0.086 (n.s.) & 0.0085 \\\\
\\bottomrule
\\end{tabular}
}
\\end{table*}

\\begin{table}[t]
\\caption{Key C1b NDCG@20 contrasts with Shapley as treatment ($B=2000$, Holm-adjusted within the full $F=8$ family). Positive differences favor Shapley.}\\label{tab:c1_shap}
\\centering
\\small
\\begin{tabular}{llcc}
\\toprule
Dataset & Comparator & Mean diff. & $p$ (Holm) \\\\
\\midrule
ML-1M & valid-sim & +0.00131 & $<0.0005$ \\\\
ML-1M & valid-linear & +0.00067 & 0.002 \\\\
ML-1M & LOO & $-$0.00060 & 0.007 \\\\
Amazon-Book & valid-sim & +0.00131 & $<0.0005$ \\\\
Amazon-Book & valid-linear & +0.00013 & 0.441 \\\\
Amazon-Book & LOO & $-$0.00063 & $<0.0005$ \\\\
\\bottomrule
\\end{tabular}
\\end{table}"""

NEW_C1P = f"""\\begin{{table*}}[t]
\\caption{{C1b LOO contrasts against the validation-informed controls, joint seed-mean user differences ($B=10{{,}}000$ permutation replicates, $+1$ correction; Holm-adjusted within the full $F=12$ family; Wilcoxon as sensitivity; bootstrap 95\\% CI for $d_z$). Positive differences favor LOO.}}\\label{{tab:c1_paired}}
\\centering
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{llllllll}}
\\toprule
{c1_paired_rows}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table*}}

\\begin{{table*}}[t]
\\caption{{C1b contrasts with Shapley-MC as treatment, joint seed-mean user differences ($B=10{{,}}000$ permutation replicates, $+1$ correction; Holm-adjusted within the $F=8$ family; Wilcoxon as sensitivity; bootstrap 95\\% CI for $d_z$). Positive differences favor Shapley.}}\\label{{tab:c1_shap}}
\\centering
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{llllllll}}
\\toprule
{c1_shap_rows}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table*}}"""
EDITS.append((OLD_C1P, NEW_C1P))

# ---------------------------------------------------------------- C1 prose update
EDITS.append((
"Table~\\ref{tab:c1_paired} reports the matched-control contrasts needed to assess validation access; the complete $F=12$ family is released with the artifacts. LOO beats both validation-informed controls on NDCG@20 on both datasets. The only non-significant retained contrast is Amazon HitRate@20 versus valid-linear ($p=0.086$).\n\nTable~\\ref{tab:c1_shap} presents the Shapley-focused NDCG@20 contrasts. Shapley beats valid-sim on both datasets and valid-linear on ML-1M, but is beaten by LOO on both datasets.",
"Table~\\ref{tab:c1_paired} reports the matched-control contrasts needed to assess validation access; the complete $F=12$ family is released with the artifacts. LOO beats both validation-informed controls on NDCG@20 on both datasets under the joint permutation test. The two non-significant retained contrasts are ML-1M and Amazon HitRate@20 versus valid-linear ($p=0.056$ and $0.235$, Holm).\n\nTable~\\ref{tab:c1_shap} presents the Shapley-focused contrasts on both metrics. Shapley beats valid-sim on both datasets, beats valid-linear on ML-1M (marginally, $p=0.086$ Holm on NDCG@20; not on Amazon), and is beaten by LOO on NDCG@20 on Amazon-Book ($p=0.0008$) with the ML-1M contrast non-significant under the joint permutation test ($p=0.137$) but significant by Wilcoxon --- the same procedure pattern as the primary study."))

# ---------------------------------------------------------------- cost medians
EDITS.append((
"Table~\\ref{tab:cost} compares offline attribution cost. The $10{,}160$\\,s ML-1M Shapley run is retained in the reported mean and SD; no runtime observation was removed. Training and base-score caching add $1{,}402\\pm27$\\,s on ML-1M and $267\\pm2$\\,s on Amazon-Book. End-to-end serving latency and memory were not measured.",
"Table~\\ref{tab:cost} compares offline attribution cost. The $10{,}160$\\,s ML-1M Shapley run is retained in the reported mean and SD; no runtime observation was removed. Per-seed medians with IQR make the outlier's effect explicit: ML-1M Shapley median $5{,}426.6$\\,s [IQR $5{,}318.9$--$5{,}435.6$] vs.\\ LOO $401.6$\\,s [$398.3$--$404.4$]; Amazon-Book Shapley $1{,}656.6$\\,s [$1{,}655.4$--$1{,}658.1$] vs.\\ LOO $127.7$\\,s [$126.9$--$127.9$]. Training and base-score caching add $1{,}402\\pm27$\\,s on ML-1M and $267\\pm2$\\,s on Amazon-Book. End-to-end serving latency and peak memory were not measured (dedicated profiling is a listed extension)."))

# ---------------------------------------------------------------- lambda section: oracle table + LOO sweep status
EDITS.append((
"The bounded $k=24$ design keeps MC attribution feasible; the separate convergence study below evaluates $M\\in\\{16,32,64,128,256\\}$. The retained $10{,}160$\\,s ML-1M runtime observation motivates reporting both mean and SD rather than removing a system-level outlier post hoc.",
"""The bounded $k=24$ design keeps MC attribution feasible; the separate convergence study below evaluates $M\\in\\{16,32,64,128,256\\}$. The retained $10{,}160$\\,s ML-1M runtime observation motivates reporting medians and IQRs in addition to mean and SD (\\S\\ref{subsec:cost}).

Two further $\\lambda$ analyses address protocol fairness. First, because the released v3 sweep does not contain the LOO family, a dedicated LOO $\\lambda$-sweep run (identical protocol, five seeds) is released as a script and will be appended to Table~\\ref{tab:ablation_lambda}; until then the LOO family is reported only at the protocol value. Second, Table~\\ref{tab:lambda_oracle} reports a \\emph{test-oracle} upper bound: each family at its best $\\lambda$ on the test split. This is explicitly not a tuning protocol (selection uses the test split), so it only bounds how much per-family tuning could move the comparison; a proper validation-tuned selection run (lambda chosen by validation NDCG@20, test reported once) is released alongside. Even under the oracle, uniform-style controls gain modestly while Shapley gains strongly on ML-1M, confirming that the headline conclusion is a statement about the shared-$\\lambda=0.10$ protocol, not about each family's attainable maximum.

\\begin{table}[t]
\\caption{Test-oracle best-$\\lambda$ sensitivity (selection on the test split; upper bound only, not a tuned comparison; released $\\lambda$-sweep artifacts, 5 seeds).}\\label{tab:lambda_oracle}
\\centering
\\small
\\begin{tabular}{llccc}
\\toprule
Dataset & Family & Oracle $\\lambda$ & NDCG@20 (oracle) & NDCG@20 ($\\lambda{=}0.10$) \\\\
\\midrule
ML-1M & uniform & 0.40 & 0.04793 $\\pm$ 0.00033 & 0.04601 $\\pm$ 0.00030 \\\\
ML-1M & additive-pref & 0.40 & 0.04849 $\\pm$ 0.00049 & 0.04610 $\\pm$ 0.00020 \\\\
ML-1M & shapley-mc & 0.40 & 0.05936 $\\pm$ 0.00043 & 0.04922 $\\pm$ 0.00033 \\\\
Amazon-Book & uniform & 0.00 & 0.02982 $\\pm$ 0.00074 & 0.02978 $\\pm$ 0.00078 \\\\
Amazon-Book & additive-pref & 0.00 & 0.02982 $\\pm$ 0.00074 & 0.02968 $\\pm$ 0.00075 \\\\
Amazon-Book & shapley-mc & 0.40 & 0.03591 $\\pm$ 0.00101 & 0.03187 $\\pm$ 0.00085 \\\\
\\bottomrule
\\end{tabular}
\\end{table}"""))

# ---------------------------------------------------------------- estimator convergence reframing
EDITS.append((
"Because the negative Shapley-vs-LOO result could in principle reflect an under-resolved Shapley estimator, Table~\\ref{tab:estimator_convergence} reports an $M$-budget convergence study on ML-1M (user subsample of 1,000, two estimator RNG seeds per $M$, reference = mean of the two $M=256$ runs). The efficiency residual $|\\sum_g\\hat\\phi_g-(v(P_u)-v(\\emptyset))|$ is at machine precision ($\\le4.3\\times10^{-10}$) for every $M$, and the per-user attribution rank correlation with the $M=256$ reference rises monotonically from $0.81$ ($M=16$) to $0.96$ ($M=256$), passing $0.91$ at the protocol value $M=64$. The $M=64$ choice therefore captures the bulk of the estimator's ordering signal at a fraction of the $M=256$ cost, and the Shapley-vs-LOO ranking conclusion is stable across the two estimator seeds.",
"Because the negative Shapley-vs-LOO result could in principle reflect an under-resolved Shapley estimator, Table~\\ref{tab:estimator_convergence} reports an $M$-budget convergence study on ML-1M (user subsample of 1,000, two estimator RNG seeds per $M$, independent of the training seeds, reference = mean of the two $M=256$ runs). Note first what is \\emph{not} convergence evidence: the permutation estimator accumulates marginals along complete permutation paths, which telescope to $v(P_u)-v(\\emptyset)$ for every path, so the efficiency residual is at machine precision ($\\le4.3\\times10^{-10}$) for \\emph{any} $M$, including very small ones; we retain it only as an implementation sanity check. The actual convergence evidence is the per-user attribution rank correlation with the $M=256$ reference, which rises monotonically from $0.81$ ($M=16$) to $0.96$ ($M=256$) and passes $0.91$ at the protocol value $M=64$, together with the stability across the two independent estimator seeds. The $M=64$ choice therefore captures the bulk of the estimator's ordering signal at a fraction of the $M=256$ cost (the two $M=256$ wall-clock times differ because of background load on the shared machine; both runs use identical settings)."))

# ---------------------------------------------------------------- algorithms: selection (x_j notation, complexity)
EDITS.append((
"\\Require User $u$, training history $H_u$, train-only item vectors $\\{e_i\\}$, val positive $i_u^+$, bound $k=24$\n\\Ensure Player set $P_u$, $|P_u|=\\min(k,|H_u|)$ in deterministic order\n\\State Compute $\\text{sim}_{\\text{prof}}(j)=\\cos(e_j,\\bar e_u)$ where $\\bar e_u=\\frac{1}{|H_u|}\\sum_{j\\in H_u}e_j$\n\\State Compute $\\text{sim}_{\\text{val}}(j)=\\cos(e_j,e_{i_u^+})$\n\\State Let $q=\\lfloor k/3\\rfloor$ and $r=k-3q$. Select $q$ profile-similar items for $P_1$, then $q$ validation-similar remaining items for $P_2$, then $q+r$ remaining items for $P_3$ by farthest-point sampling with cosine distance $1-\\text{SafeCosine}$; ties use item id",
"\\Require User $u$, training history $H_u$, train-only similarity vectors $\\{x_j\\}$ (distinct from native embeddings $\\mathbf{e}$), val positive $i_u^+$, bound $k$; effective budget $m\\gets\\min(k,|H_u|)$\n\\Ensure Player set $P_u$, $|P_u|=m$ in deterministic order\n\\State Compute $\\text{sim}_{\\text{prof}}(j)=\\cos(x_j,\\bar x_u)$ where $\\bar x_u=\\frac{1}{|H_u|}\\sum_{j\\in H_u}x_j$\n\\State Compute $\\text{sim}_{\\text{val}}(j)=\\cos(x_j,x_{i_u^+})$\n\\State Let $q=\\lfloor m/3\\rfloor$ and $r=m-3q$ (so short histories scale automatically; for $|H_u|<k$ the same steps run with $m=|H_u|$ and no separate fallback is needed). Select $q$ profile-similar items for $P_1$, then $q$ validation-similar remaining items for $P_2$, then $q+r$ remaining items for $P_3$ by farthest-point sampling with cosine distance $1-\\text{SafeCosine}$ initialized from $P_1\\cup P_2$; ties use item id"))

# ---------------------------------------------------------------- algorithm 2: P_u input + conditional notation
EDITS.append((
"\\Require User $u$, coalition $S\\subseteq P_u$, val positive $i_u^+$, fixed val negatives $\\mathcal{N}_u^-$ ($|\\mathcal{N}_u^-|=100$), backbone $f_\\theta$, train graph $G$\n\\Ensure Scalar $v_u(S)$ (Eq.~\\ref{eq:pairwise})",
"\\Require User $u$, player set $P_u$, coalition $S\\subseteq P_u$, background $B_u=H_u\\setminus P_u$, val positive $i_u^+$, fixed val negatives $\\mathcal{N}_u^-$ ($|\\mathcal{N}_u^-|=100$), frozen layer-0 parameters $\\theta$, train graph $G$\n\\Ensure Scalar $v_u(S\\mid B_u)$ (Eq.~\\ref{eq:pairwise}); parameters frozen, degrees and $\\tilde A_S$ recomputed per coalition"))

# ---------------------------------------------------------------- algorithm 3: efficiency residual comment
EDITS.append((
"\\State $R_{\\text{eff}}\\gets\\bigl|\\sum_j\\hat\\phi_j-(v_N-v_\\emptyset)\\bigr|$ \\Comment{diagnostic only; no residual redistribution}",
"\\State $R_{\\text{eff}}\\gets\\bigl|\\sum_j\\hat\\phi_j-(v_N-v_\\emptyset)\\bigr|$ \\Comment{implementation sanity check only; complete paths telescope for any $M$, so this is \\emph{not} a convergence diagnostic; no residual redistribution}"))

# ---------------------------------------------------------------- algorithm 4: vectorized zscore
EDITS.append((
"\\State $\\mathbf{r}_u \\gets \\sum_{j\\in P_u} w_j \\mathbf{e}_j$;\\quad $d \\gets \\sum_{j\\in P_u}|w_j|+\\epsilon$ \\Comment{numerically safe; cancels under z-scoring (\\S\\ref{subsec:reranking})}\n\\For{$i\\in\\mathcal{C}_u$}\n\\State $s'_{ui} \\gets \\mathrm{zscore}(b_{ui}) + \\lambda_{\\text{attr}}\\cdot\\mathrm{zscore}\\bigl(\\langle\\mathbf{r}_u,\\mathbf{e}_i\\rangle/d\\bigr)$\n\\EndFor",
"\\State $\\mathbf{r}_u \\gets \\sum_{j\\in P_u} w_j \\mathbf{e}_j$;\\quad $d \\gets \\sum_{j\\in P_u}|w_j|+\\epsilon$ \\Comment{numerically safe; cancels under z-scoring (\\S\\ref{subsec:reranking})}\n\\State Compute the candidate-wise vectors $\\mathbf{b}=[b_{ui}]_{i\\in\\mathcal{C}_u}$ and $\\mathbf{a}=[\\langle\\mathbf{r}_u,\\mathbf{e}_i\\rangle/d]_{i\\in\\mathcal{C}_u}$\n\\State $\\mathbf{z}^{b}\\gets\\mathrm{zscore}(\\mathbf{b})$;\\quad $\\mathbf{z}^{a}\\gets\\mathrm{zscore}(\\mathbf{a})$ \\Comment{z-scoring is across ALL candidates, vectorized}\n\\For{$i\\in\\mathcal{C}_u$}\n\\State $s'_{ui} \\gets z^{b}_i + \\lambda_{\\text{attr}}\\cdot z^{a}_i$\n\\EndFor"))

# ---------------------------------------------------------------- friedman table insertion (after cost/equivalence: place in results)
EDITS.append((
"\\subsection{Confirmatory matched-controls study (C1)}\\label{subsec:c1}",
f"""\\subsection{{Omnibus ranking test}}\\label{{subsec:friedman}}

As a global check over all families we run a Friedman test with users as blocks (seed-mean metrics), followed by Nemenyi--Holm pairwise comparisons (Table~\\ref{{tab:friedman}}). The omnibus test rejects the hypothesis that all nine C1b families are interchangeable ($p<10^{{-30}}$ on every scope), with LOO and Shapley holding the top mean ranks (valid-linear leads the Amazon mean ranks, consistent with the small paired gaps). No Nemenyi pairwise contrast survives correction --- expected, given the near-tied per-user ranks --- so the paired user-level contrasts of \\S\\ref{{subsec:paired}}--\\S\\ref{{subsec:c1}} remain the primary inferential evidence, and the Friedman result is reported as a supplementary omnibus check.

\\begin{{table}}[t]
\\caption{{Friedman omnibus test over all nine C1b families with users as blocks (seed-mean metric per user), Nemenyi--Holm post-hoc.}}\\label{{tab:friedman}}
\\centering
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{lllll}}
\\toprule
{fried_rows}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}

\\subsection{{Confirmatory matched-controls study (C1)}}\\label{{subsec:c1}}"""))

# ---------------------------------------------------------------- discussion update
EDITS.append((
"Although paired tests are significant, all standardized effects are below $0.08$, well below conventional ``small'' standardized-effect thresholds \\cite{cohen1988power}. The absolute NDCG@20 gains are also modest. We therefore interpret the main contribution as a computational boundary result, not evidence of large deployment impact.",
"Although paired tests are significant, all standardized effects are below $0.10$ with CIs reaching at most $0.12$, well around conventional ``small'' standardized-effect thresholds \\cite{cohen1988power}, and the user-level minimum detectable effect ($d_z\\approx0.03$) shows the study is powered for exactly these small effects. In practical terms, switching from uniform to LOO reranking changes the top-20 membership of the held-out item for $3.0\\%$ of ML-1M users ($1.8\\%$ gain, $1.2\\%$ loss) and $1.4\\%$ of Amazon users; LOO vs.\\ Shapley changes it for $1.4\\%$ (ML-1M) and $0.4\\%$ (Amazon) of users. The absolute NDCG@20 gains are also modest. We therefore interpret the main contribution as a computational boundary result, not evidence of large deployment impact."))

# ---------------------------------------------------------------- limitations rewrite
OLD_LIM = """\\subsection{Limitations and required extensions}\\label{sec:negative_result}\\label{sec:how_to_read}\\label{sec:final_framing}\\label{sec:future_work}\\label{sec:threats}\\label{sec:limitations}
The evidence is limited to LightGCN, MovieLens-1M, and a custom temporal Amazon-Book split; no external explainable-recommendation method is evaluated. Generalization requires at least a second backbone, another domain, and counterfactual or influence-function baselines. The design-factor and masked-forward studies use one training seed per dataset and therefore need multi-seed intervals before supporting robust claims.

Inference is conditional on five trained models. The current artifacts do not include seed-population inference, Friedman/Nemenyi omnibus comparisons, complementary Wilcoxon tests, or a formal equivalence-test power analysis. These analyses, together with the proportion of users whose test item crosses the top-20 boundary, would improve practical interpretation. The protocol comparison is also parameter-dependent: Shapley exceeds LOO at larger reranking weights on ML-1M, $k=24$ bounds the player set, and $M=64$ is validated only on a 1,000-user convergence subset. Finally, cost results cover offline attribution only; serving latency, memory, and the retained runtime outlier require dedicated profiling."""

NEW_LIM = """\\subsection{Limitations and required extensions}\\label{sec:negative_result}\\label{sec:how_to_read}\\label{sec:final_framing}\\label{sec:future_work}\\label{sec:threats}\\label{sec:limitations}
The evidence is limited to LightGCN, MovieLens-1M, and a custom temporal Amazon-Book split; no external explainable-recommendation method is evaluated. Generalization requires at least a second backbone, another domain, and counterfactual or influence-function baselines. Released scripts for the required extensions accompany this revision: an NGCF-style second backbone under the identical protocol, multi-seed design-factor ablations, multi-seed masked-forward faithfulness curves, the missing LOO $\\lambda$-sweep with validation-tuned $\\lambda$ selection, validation-negative-set-size sensitivity ($|\\mathcal{N}_u^-|\\in\\{50,100,500\\}$), and attribution stability / model-randomization sanity checks; their results will be appended when the runs complete. Until then, the design-factor and masked-forward studies use one training seed per dataset and support no multi-seed claim.

Inference is conditional on the five trained models (seed-population generalization would require hierarchical modeling or many more seeds; the seed-level minimum detectable $d_z$ at five seeds is $1.25$, i.e.\\ seed-level tests are uninformative by construction). The corrected inferential layer --- joint user resampling across seeds, $+1$-corrected permutation p-values, Wilcoxon sensitivity, $d_z$ confidence intervals, Friedman--Nemenyi omnibus, equivalence-test power, and top-20 crossing rates --- is reported in \\S\\ref{subsec:stat_estimand}, \\S\\ref{subsec:paired}, \\S\\ref{subsec:friedman}, and \\S\\ref{sec:discussion}. The headline conclusion is parameter-dependent by design: it is a statement about the shared $\\lambda_{\\mathrm{attr}}=0.10$ protocol; Shapley exceeds LOO at larger reranking weights on ML-1M (Table~\\ref{tab:lambda_oracle}), $k=24$ bounds the player set, and $M=64$ is validated only on a 1,000-user convergence subset. Validation negatives are fixed at 100 per user; the released sensitivity script tests robustness of attributions to that choice. Finally, cost results cover offline attribution only; serving latency, peak memory, and the retained runtime outlier require dedicated profiling."""
EDITS.append((OLD_LIM, NEW_LIM))

# ---------------------------------------------------------------- ethics paragraph
EDITS.append((
"\\subsection*{Ethics approval and consent to participate}\nThe case study uses secondary public/pseudonymous datasets (MovieLens-1M \\cite{harper2015movielens} and Amazon Reviews 2018 Books 5-core \\cite{ni2019justifying}). Only ratings, item/user identifiers (remapped to integers), and timestamps were used; raw review text, demographics, and free-text content were not processed. No new user study was conducted. Institutional determination was obtained (exempt, 2026-07-15, ENSIAS IRB 2026-07-CoalGameRec); identifiers are not redistributed and split files use remapped integer IDs where archived.",
"\\subsection*{Ethics approval and consent to participate}\nThe case study uses secondary public/pseudonymous datasets (MovieLens-1M \\cite{harper2015movielens} and Amazon Reviews 2018 Books 5-core \\cite{ni2019justifying}). Only ratings, item/user identifiers (remapped to integers), and timestamps were used; raw review text, demographics, and free-text content were not processed. No new user study was conducted. Institutional determination was obtained (exempt, 2026-07-15, ENSIAS IRB 2026-07-CoalGameRec); identifiers are not redistributed and split files use remapped integer IDs where archived.\n\nBeyond data governance, attribution-guided reranking is a recommender intervention with known downstream risks: validation-guided reweighting can amplify already-popular items and suppress long-tail exposure (we report Coverage@20 and ILD@20 descriptively, but no fairness constraint is applied), it can reinforce historical preference bias by rewarding interactions that the validation signal already favors, and the resulting lists may reduce user autonomy by narrowing choice diversity. These effects were not the object of this study and require exposure- and user-centered evaluation before deployment."))

# ---------------------------------------------------------------- figure caption
EDITS.append((
"\\caption{Cost-effectiveness: NDCG@20 versus wall-clock attribution time (log scale). Point size encodes Coverage@20. LOO Pareto-dominates Shapley (higher NDCG, lower time); annotations show $13.0$--$15.7\\times$ runtime and $16.1$--$18.3\\times$ NDCG gain per hour (Table~\\ref{tab:cost}). Attribution time excludes training and base-score caching.}",
"\\caption{Cost-effectiveness at the shared protocol $\\lambda_{\\mathrm{attr}}=0.10$: NDCG@20 versus wall-clock attribution time (log scale). Point size encodes Coverage@20. LOO Pareto-dominates Shapley on these two axes at this protocol setting (higher NDCG, lower time); annotations show $13.0$--$15.7\\times$ runtime and $16.1$--$18.3\\times$ NDCG gain per hour (Table~\\ref{tab:cost}). Attribution time excludes training and base-score caching; this is not a dominance claim across $\\lambda$, explanation quality, or serving cost.}"))

# ---------------------------------------------------------------- re-insert Figure 2 (bar panels) before paired contrasts
EDITS.append((
"\\subsection{Paired contrasts}\\label{subsec:paired}",
"""\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{assets/figures/performance_ndcg_recall.png}
\\caption{Ranking quality (NDCG@20 and HitRate@20; with one held-out test item HitRate@20 equals Recall@20) by attribution family on MovieLens-1M and Amazon-Book in the C1b matched environment, mean over five seeds with $\\pm1$ SD error bars. Families also carry distinct hatch patterns so the panels remain readable in grayscale and for colorblind readers.}
\\label{fig:ndcg_results}
\\end{figure*}

\\subsection{Paired contrasts}\\label{subsec:paired}"""))

# ---------------------------------------------------------------- re-insert lambda-sensitivity figure
EDITS.append((
"\\section{Sensitivity and diagnostic studies}\\label{sec:ablation}",
"""\\section{Sensitivity and diagnostic studies}\\label{sec:ablation}

\\begin{figure}[t]
\\centering
\\includegraphics[width=\\columnwidth]{assets/figures/lambda_sensitivity.png}
\\caption{Reranking-strength sensitivity, NDCG@20 (illustrative single-seed curves; the five-seed values are in Table~\\ref{tab:ablation_lambda}). The LOO family is not part of this released sweep artifact; its dedicated sweep is a released run script (\\S\\ref{subsec:design_ablations} discusses the pending additions).}
\\label{fig:lambda_sensitivity}
\\end{figure}"""))

# ---------------------------------------------------------------- apply
count_ok = 0
for old, new in EDITS:
    n = src.count(old)
    if n != 1:
        print(f"PATCH FAIL ({n} matches): {old[:90]!r}...")
        sys.exit(1)
    src = src.replace(old, new)
    count_ok += 1

MAIN.write_text(src)
print(f"APPLIED {count_ok} edits; {orig_len} -> {len(src)} chars")
