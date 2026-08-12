"""v18 patch part A: C1 section -> v4b (Shapley re-run included)."""
from pathlib import Path
A = Path("../manuscript_assets")
p = Path("../paper_package/main.tex")
s = p.read_text()

def rep(old, new):
    global s
    assert s.count(old) == 1, f"MATCH FAIL ({s.count(old)}x): {old[:100]!r}"
    s = s.replace(old, new)

c1b_main = (A/"c1b_main_rows.tex").read_text().rstrip("\n")
c1b_loo  = (A/"c1b_paired_loo_rows.tex").read_text().rstrip("\n")
c1b_shap = (A/"c1b_paired_shap_rows.tex").read_text().rstrip("\n")

# P1: deviations sentence -> Shapley re-run, both datasets on MPS
rep("""Both share the a-priori $\\lambda_{\\text{attr}}=0.10$ and neither uses coalition values or attribution. C1 re-executes the frozen protocol (identical hyperparameters, seeds 42--46, archived splits; a scripted, notebook-driven pipeline; complete execution logs accompany each run). Deviations from the v3 runs --- hardware (ML-1M re-executed on the original v3 machine with Apple MPS; Amazon-Book re-executed on CPU, whereas its v3 reference run trained on MPS), torch re-execution, one shared propagation per batch (equivalent gradients), vectorized negative sampler, and Shapley-MC not re-run for compute reasons --- are recorded in each run's manifest. As a fidelity check, C1 reproduces the primary study within seed noise: uniform NDCG@20 on ML-1M is $0.04601\\pm0.00022$ (identical to v3), unreranked $0.04493\\pm0.00018$ vs.\\ v3 $0.04482\\pm0.00021$, and LOO $0.04968\\pm0.00038$ vs.\\ v3 $0.04976\\pm0.00041$.""",
"""Both share the a-priori $\\lambda_{\\text{attr}}=0.10$ and neither uses coalition values or attribution. C1 re-executes the frozen protocol (identical hyperparameters, seeds 42--46, archived splits; a scripted, notebook-driven pipeline; complete execution logs accompany each run) and, in its final version (C1b), also re-runs Shapley-MC under the identical matched environment, so all nine families are compared on the same re-executed models. Deviations from the v3 runs --- torch re-execution (both datasets on Apple MPS, torch 2.3.1), one shared propagation per batch (equivalent gradients), and the vectorized negative sampler --- are recorded in each run's manifest. As a fidelity check, C1 reproduces the primary study within seed noise: uniform NDCG@20 on ML-1M is $0.04601\\pm0.00022$ (identical to v3), unreranked $0.04493\\pm0.00018$ vs.\\ v3 $0.04482\\pm0.00021$, and LOO $0.04968\\pm0.00038$ vs.\\ v3 $0.04976\\pm0.00041$.""")

# P2: C1 main-results paragraph
rep("""Table~\\ref{tab:c1_main} reports the C1 main results. Validation access is itself valuable --- valid-sim and valid-linear significantly beat uniform on both datasets and both metrics (Holm $p<0.0005$; the released validation-access contrast artifacts) --- but the game signal adds on top of it: CoalGameRec (LOO) attains the best NDCG@20, HitRate@20 (ML-1M), and Coverage@20 among all eight families.""",
"""Table~\\ref{tab:c1_main} reports the C1b main results. Validation access is itself valuable --- valid-sim and valid-linear significantly beat uniform on both datasets and both metrics (Holm $p<0.0005$; the released validation-access contrast artifacts) --- but the game signal adds on top of it: on both datasets the ordering is LOO $>$ Shapley $>$ valid-linear $>$ valid-sim $>$ heuristics on NDCG@20, and CoalGameRec (LOO) attains the best NDCG@20 among all nine families.""")

# P3: tab:c1_main caption + rows
rep("""\\caption{Confirmatory study C1: LightGCN test performance, mean $\\pm$ SD over five seeds (frozen protocol re-executed with matched validation-informed controls; the released C1 run artifacts). ML-1M executed on Apple MPS (original v3 hardware); Amazon-Book on CPU. Bold = best per column among reranked families (per dataset); on Amazon-Book, valid-linear attains the highest Coverage@20 and ILD@20.}\\label{tab:c1_main}""",
"""\\caption{Confirmatory study C1b: LightGCN test performance, mean $\\pm$ SD over five seeds (frozen protocol re-executed on Apple MPS with matched validation-informed controls \\emph{and} Shapley-MC under the identical environment; the released C1b run artifacts). Bold = best per column among reranked families (per dataset); on Amazon-Book, valid-linear attains the highest Coverage@20 and ILD@20.}\\label{tab:c1_main}""")

# replace the old 8-family rows (between tab:c1_main midrule and bottomrule) with v4b rows
i1 = s.index("\\label{tab:c1_main}")
mid = s.index("\\midrule", i1)
bot = s.index("\\bottomrule", mid)
s = s[:mid+9] + "\n" + c1b_main + "\n" + s[bot:]

# P4: tab:c1_paired paragraph + rows
rep("""Table~\\ref{tab:c1_paired} reports the paired contrasts with LOO as treatment against all six matched controls (Holm family $F=12$ per dataset; $B=2000$, seed 20260804). On ML-1M all twelve contrasts are significant (Holm $p\\le0.004$): LOO beats valid-sim by $+0.00190$ NDCG@20 ($+3.98\\%$) and valid-linear by $+0.00126$ ($+2.61\\%$), with the same ordering on HitRate@20. On Amazon-Book, eleven of twelve contrasts are significant; the single non-significant contrast is HitRate@20 vs.\\ valid-linear ($p=0.077$), while NDCG@20 favors LOO ($+0.00076$, $p=0.001$, $+2.39\\%$). Across the two studies, LOO is never significantly beaten by any matched control: the validation-access asymmetry of the primary design is therefore closed in favor of the game attribution, not of the non-game validation-informed rerankers.""",
"""Table~\\ref{tab:c1_paired} reports the paired contrasts with LOO as treatment against all six matched controls (Holm family $F=12$ per dataset; $B=2000$, seed 20260804). On ML-1M all twelve contrasts are significant: LOO beats valid-sim by $+0.00190$ NDCG@20 ($+3.98\\%$) and valid-linear by $+0.00126$ ($+2.61\\%$), with the same ordering on HitRate@20. On Amazon-Book, eleven of twelve contrasts are significant; the single non-significant contrast is HitRate@20 vs.\\ valid-linear ($p=0.086$), while NDCG@20 favors LOO ($+0.00075$, $p=0.001$, $+2.38\\%$). Across both datasets, LOO is never significantly beaten by any matched control: the validation-access asymmetry of the primary design is therefore closed in favor of the game attribution, not of the non-game validation-informed rerankers.

Table~\\ref{tab:c1_shap} reports the same matched environment with Shapley as treatment. Shapley significantly beats uniform and both validation-informed controls on NDCG@20 on ML-1M (valid-sim $+0.00131$, $p<0.0005$; valid-linear $+0.00067$, $p=0.002$), confirming that the game signal is not an artifact of the LOO estimator. On Amazon-Book, Shapley beats valid-sim ($+0.00131$, $p<0.0005$) but is statistically indistinguishable from valid-linear ($p=0.441$). Crucially, in this fully matched environment Shapley is significantly \\emph{beaten} by LOO on NDCG@20 on both datasets (ML-1M $-0.00060$, $p=0.007$; Amazon-Book $-0.00063$, $p<0.0005$) --- the LOO-over-Shapley boundary holds under identical re-execution, not only under the original v3 protocol.""")

rep("""\\caption{C1 paired contrasts, LOO-marginal (CoalGameRec) as treatment ($B=2000$, Holm $F=12$ per dataset). Artifacts: the released C1 paired-contrast tables with Holm corrections. Positive mean difference = LOO wins.}\\label{tab:c1_paired}""",
"""\\caption{C1b paired contrasts, LOO-marginal (CoalGameRec) as treatment ($B=2000$, Holm $F=12$ per dataset). Artifacts: the released C1b paired-contrast tables with Holm corrections. Positive mean difference = LOO wins.}\\label{tab:c1_paired}""")

i1 = s.index("\\label{tab:c1_paired}")
mid = s.index("\\midrule", i1)
bot = s.index("\\bottomrule", mid)
s = s[:mid+9] + "\n" + c1b_loo + "\n" + s[bot:]

# P5: insert Shapley-as-treatment table after tab:c1_paired's \end{table*}
i1 = s.index("\\label{tab:c1_paired}")
endtab = s.index("\\end{table*}", i1) + len("\\end{table*}")
SHAP_TABLE = """

\\begin{table*}[t]
\\caption{C1b paired contrasts with Shapley-MC as treatment ($B=2000$, Holm $F=8$ per dataset; released C1b paired-contrast tables). Positive mean difference = Shapley wins; the Shapley-vs-LOO rows test the boundary result under the matched environment.}\\label{tab:c1_shap}
\\centering
\\scriptsize
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{lllllll}
\\toprule
Dataset & Contrast & Metric & Mean diff. & 90\\% CI & $p$ (Holm) & $d_z$ \\\\
\\midrule
""" + c1b_shap + """
\\bottomrule
\\end{tabular}
}
\\end{table*}"""
s = s[:endtab] + SHAP_TABLE + s[endtab:]

p.write_text(s)
print("PART A done")
