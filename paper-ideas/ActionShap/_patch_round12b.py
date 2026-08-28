"""State the success convention the release actually implements, and name the file that verifies it.

The audit (audit_success_estimand.py) recomputes all 30 rows of the supplement's decision-quality
block from user_seed_metrics.csv.gz; the values are the seed-averaged per-seed indicator, so the
prose that claimed "indicator of the seed-averaged effect" is corrected to what the data does.
"""
from pathlib import Path

MAIN_OLD = ("For a selected joint action $\\widehat A_{u,g}$, success is the indicator "
            "$\\mathbf{1}[\\bar\\Delta_u^z(\\widehat A_{u,g})>0]$ evaluated on the seed-\\emph{averaged} "
            "realized effect, and this is the quantity every \\emph{headline} success rate in the paper "
            "uses. The pooling order is not cosmetic and must be read with the statistic: the released "
            "matrices additionally store the per-seed indicator "
            "$\\mathbf{1}[\\Delta_{u,r}^z(\\widehat A_{u,g})>0]$ for each seed $r$, and averaging that "
            "indicator over users and seeds gives materially smaller cohort rates than taking the "
            "indicator of the averaged effect (on the primary MovieLens NDCG comparison, $0.461$ versus "
            "$0.149$ for Monte Carlo Shapley). A rate quoted without saying which of the two it is "
            "cannot be reproduced, so the supplement's audit tables state their pooling explicitly and "
            "the two are never compared;")
MAIN_NEW = ("For a selected joint action $\\widehat A_{u,g}$, success is the per-seed indicator "
            "$\\mathbf{1}[\\Delta_{u,r}^z(\\widehat A_{u,g})>0]$ for each seed run $r$, averaged over the "
            "$R_{\\mathrm{seed}}$ seeds within a user and then over the $n$ retained users. A cohort rate "
            "therefore moves on a $1/(nR_{\\mathrm{seed}})$ lattice --- multiples of $0.0002$ for the "
            "$1000$-user primary ItemKNN cohort --- which is why four-decimal values such as $0.2742$ "
            "are expected rather than suspicious. Pooling the other way, the indicator of the "
            "seed-\\emph{averaged} effect $\\mathbf{1}[\\bar\\Delta_u^z(\\widehat A_{u,g})>0]$, is a "
            "different quantity with a different value ($0.2850$ against $0.2742$ for Monte Carlo Shapley "
            "on the primary MovieLens NDCG comparison), so the convention is stated wherever a success "
            "rate is printed, the two are never compared, and every printed rate is recomputed row by "
            "row from the released matrix by \\texttt{audit\\_success\\_estimand.py};")

PROP_OLD = ("Success is the fraction of users with\npositive seed-averaged realized NDCG effect; "
            "abstention is the fraction\nselecting no action.")
PROP_NEW = ("Success is the cohort mean of each user's\nseed-averaged per-seed indicator (positive "
            "realized NDCG effect in that seed run), so\nits values lie on a $1/(n_uR_{\\mathrm{seed}})$ "
            "lattice; abstention is the cohort\nmean of the per-seed abstention indicator. Every row is "
            "recomputed from\n\\texttt{user\\_seed\\_metrics.csv.gz} at the primary ItemKNN slice by\n"
            "\\texttt{audit\\_success\\_estimand.py}.")

MATRIX_OLD = ("The headline success convention in the main text is instead the indicator of the "
              "seed-averaged effect, a different quantity that is systematically larger, and the two are never "
              "compared in this paper.")
MATRIX_NEW = ("The headline success convention in the main text is this same user-level average of the "
              "per-seed indicator; the alternative pooling, the indicator of the seed-averaged effect, is a "
              "different quantity ($0.2850$ against $0.2742$ for Shapley on the primary MovieLens NDCG "
              "comparison) and is never substituted for it.")

targets = [
    ("acmart-primary/acmmanuscript.tex", [(MAIN_OLD, MAIN_NEW)]),
    ("acmart-primary/tables/review3_statistics.tex", [(PROP_OLD, PROP_NEW)]),
    ("actionshap-ipm/tables/review3_statistics.tex", [(PROP_OLD, PROP_NEW)]),
    ("acmart-primary/tables/appendix_intervention_full.tex", [(MATRIX_OLD, MATRIX_NEW)]),
    ("actionshap-ipm/tables/appendix_intervention_full.tex", [(MATRIX_OLD, MATRIX_NEW)]),
]
for rel, edits in targets:
    p = Path(rel)
    if not p.exists():
        print("skip", rel)
        continue
    text = p.read_text(encoding="utf-8")
    for old, new in edits:
        if old in text:
            text = text.replace(old, new, 1)
            print(f"  {rel}: edited")
        elif new[:60] in text:
            print(f"  {rel}: already applied")
        else:
            print(f"  !! {rel}: anchor missing -> {old[:70]!r}")
    p.write_text(text)
