#!/usr/bin/env python3
"""v24 patch: integrate v7 corrected-protocol results (ML-1M) into the manuscript."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MAIN = ROOT / "paper_package" / "main.tex"
src = MAIN.read_text()
main_rows = (ROOT / "manuscript_assets" / "round9" / "tab_v7_main_rows.tex").read_text().strip()
contrast_rows = (ROOT / "manuscript_assets" / "round9" / "tab_v7_contrast_rows.tex").read_text().strip()

anchor = "\\subsection{Second backbone: NGCF (cross-architecture replication)}\\label{subsec:second_backbone}"
block = f"""\\subsection{{Corrected temporal protocol (v7): calibration item excluded}}\\label{{subsec:v7_protocol}}

The v1--v22 runs left the calibration item $i_u^+$ in the test candidate catalog (\\S\\ref{{subsec:protocol_timeline}}). The corrected protocol (v7) excludes $i_u^+$ from candidates for every family and computes reranking z-scores over the corrected candidate set; everything else is byte-identical to the frozen matched-controls protocol (seeds 42--46, $k=24$, $M=64$, 100 validation negatives, $\\lambda_{{\\mathrm{{attr}}}}=0.10$, native intervention). ML-1M is complete (Tables~\\ref{{tab:v7_main}}--\\ref{{tab:v7_contrasts}}); Amazon-Book is executing under the identical script.

\\begin{{table*}}[t]
\\caption{{Corrected protocol (v7), ML-1M, mean $\\pm$ SD over five seeds. Candidates exclude training items \\emph{{and}} the per-user calibration item. The family ordering of the v1--v22 runs is preserved.}}\\label{{tab:v7_main}}
\\centering
\\small
\\begin{{tabular}}{{llcc}}
\\toprule
Dataset & Method & HitRate@20 & NDCG@20 \\\\
\\midrule
{main_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}

\\begin{{table*}}[t]
\\caption{{Corrected protocol (v7), ML-1M: paired contrasts on joint seed-mean user differences ($B=10{{,}}000$ sign-flip replicates, $+1$ correction; Holm within the declared families $F=12$ (LOO) and $F=8$ (Shapley); Wilcoxon sensitivity; bootstrap 95\\% CI for $d_z$).}}\\label{{tab:v7_contrasts}}
\\centering
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{llllllll}}
\\toprule
Dataset & Contrast & Metric & Mean diff. & 95\\% CI & $p$ (perm., Holm) & $p$ (Wilcoxon, Holm) & $d_z$ [95\\% CI] \\\\
\\midrule
{contrast_rows}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table*}}

Findings under the corrected protocol. First, excluding the calibration item \\emph{{raises}} absolute NDCG@20 for every family (the calibration item was a strong competitor for top ranks but is never the scored target), and the complete family ordering is preserved: LOO $>$ Shapley $>$ valid-linear $>$ valid-sim $>$ attention $\\approx$ heuristic-pop $\\approx$ additive-pref $\\approx$ uniform $>$ unreranked. Second, the Shapley--LOO NDCG@20 equivalence verdict on ML-1M improves from ``not established'' (v1--v22) to \\emph{{established}}: the 90\\% joint CI is $[-0.000937,+0.000143]$, inside $\\pm0.001$ (point estimate $-0.000399$ favors LOO; the interval straddles zero, so no directional claim is made). HitRate@20 is equivalent within $\\pm0.002$ with a small point estimate favoring Shapley ($+0.00106$, n.s.). Third, LOO beats all six comparators on NDCG@20 (Holm $p\\le0.007$), including both validation-informed controls; on HitRate@20 the two validation-informed contrasts are non-significant. Shapley beats uniform and valid-sim on both metrics, is non-significant against valid-linear on NDCG@20 (Holm $p=0.062$), and is non-significant against LOO on both metrics under the sign-flip test (the Wilcoxon sensitivity again disagrees on NDCG@20, $p=0.0009$ --- the same procedure divergence observed in every previous execution, reported as such). The Friedman omnibus rejects ($p<10^{{-100}}$) with LOO and Shapley holding the top mean ranks.

{anchor}"""
assert src.count(anchor) == 1
src = src.replace(anchor, block)

EDITS = [
("A protocol correction (excluding the calibration item from test candidates), nested $\\lambda$ tuning, and matched-execution sweeps accompany this revision.}",
 "A protocol correction excluding the calibration item from test candidates has been re-executed on ML-1M: the family ordering is preserved and Shapley--LOO NDCG@20 equivalence on ML-1M is established under the correction; nested $\\lambda$ tuning and matched-execution sweeps accompany this revision.}"),
("and all headline tables are re-reported under that correction in the released v7 re-execution.",
 "and the headline results are re-reported under that correction in the v7 re-execution (\\S\\ref{subsec:v7_protocol}; ML-1M complete, Amazon-Book executing)."),
("In the v1--v22 protocol the calibration item remained in the test candidate catalog (candidates $=\\mathcal{I}\\setminus H_u^{\\mathrm{train}}$); the corrected protocol (\\S\\ref{subsec:protocol_timeline}) excludes it, and corrected re-runs are released alongside this revision.",
 "In the v1--v22 protocol the calibration item remained in the test candidate catalog (candidates $=\\mathcal{I}\\setminus H_u^{\\mathrm{train}}$); the corrected protocol (\\S\\ref{subsec:protocol_timeline}) excludes it, and the corrected re-execution is reported in \\S\\ref{subsec:v7_protocol} (ML-1M complete; Amazon-Book executing)."),
("second-backbone tables from \\texttt{*\\_ngcf\\_v6\\_second\\_bone};",
 "second-backbone tables from \\texttt{*\\_ngcf\\_v6\\_second\\_bone}; corrected-protocol tables from \\texttt{*\\_lightgcn\\_v7\\_corrected\\_protocol};"),
]
for old, new in EDITS:
    n = src.count(old)
    if n != 1:
        print(f"PATCH FAIL ({n}): {old[:90]!r}")
        sys.exit(1)
    src = src.replace(old, new)

MAIN.write_text(src)
(ROOT / "springer_latex" / "main.tex").write_text(src)
print("v24 applied:", len(src), "chars")
