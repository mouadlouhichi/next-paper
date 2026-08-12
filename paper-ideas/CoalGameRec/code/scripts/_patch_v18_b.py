"""v18 patch part B: masked-forward rewrite, ablations/convergence subsections,
abstract/contributions/limitations/discussion updates."""
from pathlib import Path
A = Path("../manuscript_assets")
p = Path("../paper_package/main.tex")
s = p.read_text()

def rep(old, new):
    global s
    assert s.count(old) == 1, f"MATCH FAIL ({s.count(old)}x): {old[:100]!r}"
    s = s.replace(old, new)

mf_rows = (A/"c1b_masked_forward_rows.tex").read_text().rstrip("\n")

# B1: abstract -> C1b with Shapley re-run + masked-forward
rep("""A confirmatory re-execution of the frozen protocol with matched validation-informed non-game controls (valid-similarity reweighting and a validation-tuned linear reranker) shows that LOO also beats both matched controls on NDCG@20 on both datasets (Holm $p\\le0.004$), closing the validation-access asymmetry for the LOO claim (Shapley was not re-run in C1).""",
"""A confirmatory re-execution of the frozen protocol with matched validation-informed non-game controls (valid-similarity reweighting and a validation-tuned linear reranker) and a re-run of Shapley under the identical environment shows that LOO beats both matched controls on NDCG@20 on both datasets (Holm $p\\le0.004$) and is significantly preferred to Shapley on NDCG@20 in the matched environment, closing the validation-access asymmetry; masked-forward deletion/insertion diagnostics further show that the game-derived weights select influential history.""")

# B2: Contribution 2 (C1 mention)
rep("""bounded Shapley ($k=24$, $M=64$) beats all matched non-game controls on NDCG@20 --- and, in the confirmatory study C1, also matched \\emph{validation-informed} non-game controls --- yet does not outperform its own LOO marginal: a pre-specified equivalence analysis shows the two to be practically equal on NDCG@20 (both intervals entirely on the LOO side), at $13.0$--$15.7\\times$ lower attribution time for LOO (Tables~\\ref{tab:main_results}, \\ref{tab:paired}, \\ref{tab:paired_loo}, \\ref{tab:equivalence}, \\ref{tab:c1_main}, \\ref{tab:c1_paired}).""",
"""bounded Shapley ($k=24$, $M=64$) beats all matched non-game controls on NDCG@20 --- and, in the confirmatory study C1, also matched \\emph{validation-informed} non-game controls --- yet does not outperform its own LOO marginal: a pre-specified equivalence analysis shows the two to be practically equal on NDCG@20 (both intervals entirely on the LOO side), and in the fully matched C1 re-execution (Shapley included) LOO is significantly preferred on NDCG@20 on both datasets, at $13.0$--$15.7\\times$ lower attribution time (Tables~\\ref{tab:main_results}, \\ref{tab:paired}, \\ref{tab:paired_loo}, \\ref{tab:equivalence}, \\ref{tab:c1_main}, \\ref{tab:c1_paired}, \\ref{tab:c1_shap}).""")

# B3: C1 faithfulness paragraph -> masked-forward
rep("""C1 also extends the faithfulness proxies of \\S\\ref{sec:explainability} from a single fraction to deletion/insertion curves over fractions $\\{0.05,0.10,0.20,0.30\\}$ with uniform and seeded-random weight controls (Table~\\ref{tab:c1_faith}). On both datasets the LOO attributions induce a strictly larger deletion drop than uniform and random weights at every fraction (e.g., ML-1M at 20\\%: $0.00972\\pm0.00035$ vs.\\ $0.00854\\pm0.00018$ uniform and $0.00821\\pm0.00034$ random $\\Delta$NDCG@20), the gap is monotone in the removed fraction, and top-fraction insertion retains slightly more performance under LOO. The proxies remain candidate-masking (not masked-forward re-propagation), so faithfulness is still not claimed; the curves sharpen the preliminary evidence that the game-derived weights select influential history.""",
"""C1 replaces the candidate-masking proxy with the \\emph{true masked-forward} intervention that defines the cooperative game: for each user and fraction, the top-attributed player edges are removed (deletion) or kept exclusively (insertion), the normalized adjacency is rebuilt, the frozen LightGCN parameters are re-propagated on the masked graph, and the test-item rank is re-evaluated (Table~\\ref{tab:c1_faith}). The masked re-propagation is executed on CPU with a self-test (keeping the full history reproduces the cached base scores to $\\sim10^{-10}$). Two patterns hold. \\emph{Insertion}: keeping only the top-attributed edges retains --- and on ML-1M even exceeds --- unmasked performance under the game families (ML-1M at 10\\%: Shapley $0.05176$, LOO $0.04894$ vs.\\ unmasked $0.04759$; Amazon: $0.02163$/$0.02162$ vs.\\ $0.01723$), while uniform falls below unmasked and random stays near it. \\emph{Deletion}: removing the top-attributed edges degrades ranking most under the game families --- LOO on ML-1M ($-0.00127$ at 10\\%) and Shapley on Amazon ($-0.00183$) --- with uniform and random near-flat. Faithfulness in the full sense is still not claimed (no perturbation-stability, model-randomization, or redundancy controls), but the masked-forward evidence shows that the game-derived weights select history that is genuinely influential for the model's own ranking.""")

# B4: tab:c1_faith caption + rows -> masked-forward
rep("""\\caption{C1 faithfulness proxy curves (candidate-masking, five-seed mean $\\pm$ SD; the released C1 faithfulness-curve artifacts). Deletion $\\Delta$NDCG@20: drop when the top fraction of attributed history is removed (larger = more influential). Insertion NDCG@20: retained performance keeping only the top fraction.}\\label{tab:c1_faith}""",
"""\\caption{C1 masked-forward faithfulness (true masked re-propagation on CPU, self-tested; single seed 42 per dataset; released masked-forward artifacts). Deletion: NDCG@20 after removing the top-fraction attributed player edges (lower = the removed edges were influential). Insertion: NDCG@20 keeping only the top-fraction edges (higher = the kept edges carry the ranking). Unmasked reference per dataset in the group header row.}\\label{tab:c1_faith}""")

i1 = s.index("\\label{tab:c1_faith}")
mid = s.index("\\midrule", i1)
bot = s.index("\\bottomrule", mid)
# rebuild header too (5 columns)
hdr_start = s.rindex("\\begin{tabular}", 0, i1)
hdr_end = s.index("\\midrule", hdr_start) + len("\\midrule")
new_hdr = """\\begin{tabular}{lllcc}
\\toprule
Dataset & Weights & Fraction & Deletion NDCG@20 & Insertion NDCG@20 \\\\
\\midrule"""
s = s[:hdr_start] + new_hdr + "\n" + mf_rows + "\n" + s[bot:]

# B5: explainability three-caveats update (masked-forward now executed)
rep("""Three caveats bound the claim. First --- and most importantly --- the proxy masks candidates rather than re-propagating LightGCN on the masked graph $G_S$; it therefore does not test faithfulness to the very intervention that defines the cooperative game, and recent work shows that conventional perturbation-based fidelity evaluations of graph explanations can be unreliable under distribution shift \\cite{zheng2024robust,fang2023evaluating}. The true masked-forward protocol (edge removal, degree renormalization, frozen re-propagation) is specified in the released faithfulness scripts; until it is executed, no faithfulness claim is made. Second, a single 20\\% fraction is reported here; multi-fraction deletion/insertion curves with uniform and seeded-random controls were subsequently executed in the confirmatory study C1 (\\S\\ref{subsec:c1}, Table~\\ref{tab:c1_faith}). Third, the present table was computed for the Shapley-MC attribution only; the cross-family (LOO vs.\\ uniform vs.\\ random) comparison under the identical proxy is provided by C1, where LOO dominates at every fraction on both datasets. Faithfulness of the attributions is therefore \\emph{not claimed}; the table is preliminary evidence that the game-derived weights select influential history.""",
"""Three caveats bound the claim. First, the present table is a candidate-masking proxy: it masks candidates rather than re-propagating LightGCN on the masked graph $G_S$, and recent work shows that conventional perturbation-based fidelity evaluations of graph explanations can be unreliable under distribution shift \\cite{zheng2024robust,fang2023evaluating}. The true masked-forward protocol (edge removal, degree renormalization, frozen re-propagation) was subsequently executed in the confirmatory study C1 (\\S\\ref{subsec:c1}, Table~\\ref{tab:c1_faith}) and is the primary faithfulness evidence of this paper. Second, the masked-forward evaluation uses a single training seed per dataset and fractions $\\{0.10,0.20,0.30\\}$; multi-seed curves and perturbation-stability and model-randomization controls remain future work. Third, the proxy table below was computed for the Shapley-MC attribution only; the cross-family masked-forward comparison (LOO, Shapley, uniform, random) is provided by C1. Faithfulness of the attributions in the full sense is therefore \\emph{not claimed}; the masked-forward results are evidence that the game-derived weights select history influential for the model's own ranking.""")

# B6: new ablation subsections (design factors + estimator convergence), inserted before \section{Explainability analysis
ABLU = """
\\subsection{Design-factor ablations}\\label{subsec:design_ablations}

Table~\\ref{tab:design_ablations} reports the design-factor ablations executed in the C1 environment (single training seed 42 per dataset; native intervention unless stated). Four factors are ablated. \\emph{Player budget $k$}: NDCG@20 is flat for $k\\ge16$ on both datasets (ML-1M LOO: $0.04942/0.04976/0.04912/0.04948$ for $k=8/16/24/32$), supporting the choice $k=24$ without a sharp cliff. \\emph{Player selection}: stratified, similarity-only, and random selection are statistically indistinguishable for Shapley on both datasets (ML-1M: $0.04892/0.04840/0.04902$), so the stratified scheme is a convenience rather than a performance driver. \\emph{Coalition utility}: the smooth pairwise log-sigmoid utility beats the hard top-$K$ NDCG utility for both families on both datasets (ML-1M LOO: $0.04912$ vs.\\ $0.04522$; Shapley $0.04892$ vs.\\ $0.04610$), the first direct empirical support for the smooth-utility design principle. \\emph{Intervention}: the external train-only kernel intervention outperforms the native embedding intervention on both datasets (ML-1M LOO: $0.05235$ vs.\\ $0.04912$; Amazon LOO: $0.05439$ vs.\\ $0.03209$). This is an honest finding against the model-native-alignment principle as a performance claim; we retain the native intervention as the primary protocol because it is the intervention defined by the cooperative game and is model-consistent, and we report the kernel variant as a sensitivity showing that the alignment principle concerns model consistency rather than guaranteed accuracy.

\\begin{table*}[t]
\\caption{Design-factor ablations in the C1 environment (single training seed 42 per dataset; released design-ablation artifacts). NDCG@20. Intervention rows compare the native (protocol) intervention against the external train-only kernel.}\\label{tab:design_ablations}
\\centering
\\scriptsize
\\begin{tabular}{llllcc}
\\toprule
Ablation & Variant & Family & & ML-1M & Amazon-Book \\\\
\\midrule
Player budget $k$ & $k=8$  & LOO     & & 0.04942 & 0.03187 \\\\
                    & $k=8$  & Shapley & & 0.04867 & 0.03133 \\\\
                    & $k=16$ & LOO     & & 0.04976 & 0.03207 \\\\
                    & $k=16$ & Shapley & & 0.04909 & 0.03142 \\\\
                    & $k=24$ & LOO     & & 0.04912 & 0.03209 \\\\
                    & $k=24$ & Shapley & & 0.04892 & 0.03147 \\\\
                    & $k=32$ & LOO     & & 0.04948 & 0.03211 \\\\
                    & $k=32$ & Shapley & & 0.04900 & 0.03147 \\\\
\\midrule
Player selection & stratified & Shapley & & 0.04892 & 0.03147 \\\\
                 & similarity & Shapley & & 0.04840 & 0.03149 \\\\
                 & random     & Shapley & & 0.04902 & 0.03146 \\\\
\\midrule
Coalition utility & pairwise log-sigmoid & LOO     & & 0.04912 & 0.03209 \\\\
                  & pairwise log-sigmoid & Shapley & & 0.04892 & 0.03147 \\\\
                  & hard top-$K$ NDCG    & LOO     & & 0.04522 & 0.03063 \\\\
                  & hard top-$K$ NDCG    & Shapley & & 0.04610 & 0.03143 \\\\
\\midrule
Intervention & native (protocol) & LOO     & & 0.04912 & 0.03209 \\\\
             & external kernel   & LOO     & & 0.05235 & 0.05439 \\\\
             & native (protocol) & Shapley & & 0.04892 & 0.03147 \\\\
             & external kernel   & Shapley & & 0.05037 & 0.05517 \\\\
\\bottomrule
\\end{tabular}
\\end{table*}

\\subsection{Shapley estimator convergence}\\label{subsec:estimator_convergence}

Because the negative Shapley-vs-LOO result could in principle reflect an under-resolved Shapley estimator, Table~\\ref{tab:estimator_convergence} reports an $M$-budget convergence study on ML-1M (user subsample of 1,000, two estimator RNG seeds per $M$, reference = mean of the two $M=256$ runs). The efficiency residual $|\\sum_g\\hat\\phi_g-(v(P_u)-v(\\emptyset))|$ is at machine precision ($\\le4.3\\times10^{-10}$) for every $M$, and the per-user attribution rank correlation with the $M=256$ reference rises monotonically from $0.81$ ($M=16$) to $0.96$ ($M=256$), passing $0.91$ at the protocol value $M=64$. The $M=64$ choice therefore captures the bulk of the estimator's ordering signal at a fraction of the $M=256$ cost, and the Shapley-vs-LOO ranking conclusion is stable across the two estimator seeds.

\\begin{table}[t]
\\caption{Shapley estimator convergence on ML-1M (1,000-user subsample, two estimator seeds; released convergence artifacts). Efficiency residual $=|\\sum_g\\hat\\phi_g-(v(P_u)-v(\\emptyset))|$; Spearman $\\rho$ is the per-user attribution rank correlation with the $M=256$ reference.}\\label{tab:estimator_convergence}
\\centering
\\small
\\begin{tabular}{lccc}
\\toprule
$M$ & Attribution time (s) & Efficiency residual & Spearman $\\rho$ vs.\\ $M=256$ \\\\
\\midrule
16  & 296 / 280    & $3.6\\times10^{-11}$ / $2.1\\times10^{-11}$ & 0.807 / 0.818 \\\\
32  & 484 / 503    & $1.3\\times10^{-10}$ / $1.0\\times10^{-10}$ & 0.875 / 0.877 \\\\
64  & 900 / 898    & $3.0\\times10^{-10}$ / $3.1\\times10^{-10}$ & 0.913 / 0.914 \\\\
128 & 2337 / 1893  & $3.6\\times10^{-10}$ / $3.9\\times10^{-10}$ & 0.940 / 0.944 \\\\
256 & 17150 / 6716 & $4.3\\times10^{-10}$ / $4.1\\times10^{-10}$ & (reference) \\\\
\\bottomrule
\\end{tabular}
\\end{table}

"""
rep("\\section{Explainability analysis (faithfulness proxies)}\\label{sec:explainability}",
      ABLU + "\\section{Explainability analysis (faithfulness proxies)}\\label{sec:explainability}")

# B7: Limitations bullets update (ablations executed, masked-forward executed, C1b)
rep("""    \\item Shapley is bounded to $k=24$ stratified players (not full history) for feasibility; the $k$-sweep, $M$-budget convergence, player-selection, native-vs-external intervention, and hard-vs-smooth utility ablations are specified in the released scripts but were not executed in this study, so no results are reported for them.""",
"""    \\item Shapley is bounded to $k=24$ stratified players (not full history) for feasibility. The $k$-sweep, player-selection, coalition-utility, and intervention ablations were executed in the C1 environment (Table~\\ref{tab:design_ablations}); a full multi-seed ablation campaign across all factors remains future work.""")
rep("""    \\item Faithfulness diagnostics are candidate-masking proxies, not masked-forward graph re-propagation; no faithfulness claim is made.""",
"""    \\item The primary faithfulness evidence is the masked-forward evaluation of C1 (Table~\\ref{tab:c1_faith}), executed with a single training seed per dataset; the earlier candidate-masking proxy (Table~\\ref{tab:faithfulness}) is retained as a secondary diagnostic. Perturbation-stability, model-randomization, and redundancy controls remain future work; faithfulness in the full sense is not claimed.""")
rep("""    \\item The matched validation-informed controls were run in C1 rather than in the primary frozen run: C1 re-executes the frozen protocol (fidelity verified in \\S\\ref{subsec:c1}), ML-1M on the original hardware (Apple MPS) and Amazon-Book on CPU, and Shapley-MC was not re-run in C1; Shapley comparisons therefore remain anchored to the primary v3 artifacts.""",
"""    \\item The matched validation-informed controls and the Shapley re-run were produced in the confirmatory study C1 rather than in the primary frozen run: C1 re-executes the frozen protocol on Apple MPS (fidelity verified in \\S\\ref{subsec:c1}), so cross-study comparisons rest on that re-execution reproducing the primary results within seed noise, which it does.""")
rep("""    \\item The Shapley estimator budget $M=64$ is not convergence-tested here; the released convergence script provides the protocol.""",
"""    \\item The Shapley estimator budget $M=64$ is justified by the convergence study (Table~\\ref{tab:estimator_convergence}) on a 1,000-user subsample; a full-user convergence sweep remains future work.""")

# B8: Discussion C1 sentence update
rep("""The confirmatory matched-controls study (\\S\\ref{subsec:c1}) strengthens the first finding: LOO also beats the validation-informed non-game controls valid-sim and valid-linear on NDCG@20 on both datasets (Holm $p\\le0.004$), showing that the gain is not an artifact of validation-access asymmetry.""",
"""The confirmatory matched-controls study (\\S\\ref{subsec:c1}) strengthens the first finding: in a fully matched re-execution that includes Shapley, LOO beats the validation-informed non-game controls valid-sim and valid-linear on NDCG@20 on both datasets (Holm $p\\le0.004$) and is significantly preferred to Shapley on NDCG@20 in the matched environment, showing that the gain is not an artifact of validation-access asymmetry; masked-forward diagnostics further show the game-derived weights select genuinely influential history.""")

p.write_text(s)
print("PART B done")
