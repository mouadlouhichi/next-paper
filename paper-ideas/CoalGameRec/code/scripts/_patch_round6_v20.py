#!/usr/bin/env python3
"""Round-6 v20 manuscript patch: integrates the user's executed v6 runs
(NGCF second backbone + LOO lambda sweep + validation-tuned lambda)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "code" / "results" / "journal_runs"
MAIN = ROOT / "paper_package" / "main.tex"

src = MAIN.read_text()
EDITS: list[tuple[str, str]] = []

# ============================================================ NGCF table rows
FAM_ORDER = [("unreranked", "unreranked"), ("uniform", "uniform"), ("additive-pref", "additive-pref"),
             ("attention", "attention"), ("heuristic-pop", "heuristic-pop"),
             ("valid-sim", "valid-sim"), ("valid-linear", "valid-linear"),
             ("shapley-mc", "Shapley"), ("loo-marginal", "\\textbf{LOO}")]
ngcf_rows = []
for ds, dsl in [("ml1m", "ML-1M"), ("amazon_books", "Amazon-Book")]:
    df = pd.read_csv(RUNS / f"{ds}_ngcf_v6_second_backbone" / "tables" / "summary_by_seed_family.csv")
    assert sorted(df.seed.unique().tolist()) == [42, 43, 44, 45, 46]
    for fam, lab in FAM_ORDER:
        g = df[df.family == fam]
        hr, hn = g["HitRate@20"].mean(), g["HitRate@20"].std(ddof=1)
        nr, nn = g["NDCG@20"].mean(), g["NDCG@20"].std(ddof=1)
        hr_s = f"{hr:.5f} $\\pm$ {hn:.5f}"
        nr_s = f"{nr:.5f} $\\pm$ {nn:.5f}"
        if fam == "loo-marginal":
            hr_s, nr_s = "\\textbf{" + hr_s + "}", "\\textbf{" + nr_s + "}"
        ngcf_rows.append(f"{dsl} & {lab} & {hr_s} & {nr_s} \\\\")
NGCF_ROWS = "\n".join(ngcf_rows)

# ============================================================ LOO lambda rows (v6 sweep)
loo_rows = []
for ds, dsl in [("ml1m", "ML-1M"), ("amazon_books", "Amazon-Book")]:
    df = pd.read_csv(RUNS / f"{ds}_lightgcn_v6_lambda_sweep" / "tables" / "lambda_sensitivity_all.csv")
    g = df[df.family == "loo-marginal"].groupby("lambda_attr")["NDCG@20"].agg(["mean", "std"])
    cells = " & ".join(f"{g.loc[l, 'mean']:.5f} $\\pm$ {g.loc[l, 'std']:.5f}" for l in [0.0, 0.05, 0.1, 0.2, 0.4])
    cells = cells.replace("0.10", "\\textbf{0.10}", 0)  # placeholder no-op; bold handled below
    loo_rows.append(f"{dsl} & loo-marginal & {cells} \\\\")
LOO_ROWS = "\n".join(loo_rows)

# ============================================================ validation-tuned table rows
tuned_rows = []
for ds, dsl in [("ml1m", "ML-1M"), ("amazon_books", "Amazon-Book")]:
    t = pd.read_csv(RUNS / f"{ds}_lightgcn_v6_lambda_sweep" / "tables" / "validation_tuned_lambda.csv")
    for _, r in t.iterrows():
        fam = "LOO" if r.family == "loo-marginal" else r.family
        lab = f"\\textbf{{{fam}}}" if r.family == "loo-marginal" else fam
        tuned_rows.append(f"{dsl} & {lab} & {r.lambda_selected_on_validation:.2f} & "
                          f"{r.test_ndcg20_mean:.5f} $\\pm$ {r.test_ndcg20_std:.5f} & "
                          f"{r.protocol_ndcg20_mean:.5f} \\\\")
TUNED_ROWS = "\n".join(tuned_rows)

# ============================================================ 1. abstract
EDITS.append((
"A confirmatory re-execution with validation-informed controls preserves the ordering, while single-seed masked-forward tests provide limited evidence that game-derived weights identify influential history.",
"A confirmatory re-execution with validation-informed controls preserves the ordering, and the same family ordering replicates on a structurally different NGCF backbone on both datasets; single-seed masked-forward tests provide limited evidence that game-derived weights identify influential history."))

# ============================================================ 2. contributions
EDITS.append((
"    \\item \\textbf{A boundary result.} Bounded Shapley beats matched non-game controls, but LOO is equivalent or better on NDCG@20 and requires $13.0$--$15.7\\times$ less attribution time.",
"    \\item \\textbf{A boundary result.} Bounded Shapley beats matched non-game controls, but LOO is equivalent or better on NDCG@20 and requires $13.0$--$15.7\\times$ less attribution time; the ordering replicates on an NGCF backbone, under independently validation-tuned reranking strengths, and across the full $\\lambda$ sweep."))

# ============================================================ 3. background backbone sentence
EDITS.append((
"In this paper LightGCN is the sole evaluated backbone; a structurally different nonlinear backbone (NGCF-style aggregation) is released as a second-backbone run script and is required before any cross-architecture claim.",
"LightGCN is the primary backbone; a structurally different nonlinear backbone (NGCF-style aggregation, \\S\\ref{subsec:second_backbone}) is evaluated under the identical frozen protocol as a cross-architecture replication."))

# ============================================================ 4. lambda paragraph rewrite
EDITS.append((
"Two further $\\lambda$ analyses address protocol fairness. First, because the released v3 sweep does not contain the LOO family, a dedicated LOO $\\lambda$-sweep run (identical protocol, five seeds) is released as a script and will be appended to Table~\\ref{tab:ablation_lambda}; until then the LOO family is reported only at the protocol value. Second, Table~\\ref{tab:lambda_oracle} reports a \\emph{test-oracle} upper bound: each family at its best $\\lambda$ on the test split. This is explicitly not a tuning protocol (selection uses the test split), so it only bounds how much per-family tuning could move the comparison; a proper validation-tuned selection run (lambda chosen by validation NDCG@20, test reported once) is released alongside. Even under the oracle, uniform-style controls gain modestly while Shapley gains strongly on ML-1M, confirming that the headline conclusion is a statement about the shared-$\\lambda=0.10$ protocol, not about each family's attainable maximum.",
"Two completed $\\lambda$ analyses address protocol fairness. First, Table~\\ref{tab:ablation_lambda} now includes the LOO family from a dedicated five-seed sweep (v6 re-execution, identical protocol; Fig.~\\ref{fig:lambda_sensitivity}). LOO is the strongest family at every $\\lambda>0$ on both datasets and rises fastest with $\\lambda$ (ML-1M: $0.04959$ at the protocol value to $0.06211$ at $\\lambda=0.40$; Amazon-Book: $0.03250$ to $0.03683$). Second, we ran the reviewers' requested \\emph{independently tuned} comparison: for each family the reranking strength is selected by validation NDCG@20 (full-catalog ranking of the held-out validation item, training items masked), and the test split is evaluated exactly once at the selected value (Table~\\ref{tab:lambda_tuned}). Under proper per-family tuning LOO's lead \\emph{widens}: validation selects $\\lambda=0.40$ for LOO on both datasets, yielding test NDCG@20 $0.06211\\pm0.00040$ on ML-1M ($+30.0\\%$ over tuned uniform, $+28.8\\%$ over tuned additive-pref) and $0.03683\\pm0.00120$ on Amazon-Book ($+23.9\\%$ over tuned uniform). The shared-$\\lambda=0.10$ protocol therefore does not mask a tuned advantage of the competing families; if anything, tuning favors the validation-guided LOO signal. Table~\\ref{tab:lambda_oracle} additionally reports the test-oracle upper bound for the v3 families as a reference."))

# ============================================================ 5. tuned-lambda table insertion (after oracle table)
EDITS.append((
"\\end{tabular}\n\\end{table}\n\n\n\\subsection{Design-factor ablations}\\label{subsec:design_ablations}",
f"""\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[t]
\\caption{{Independently validation-tuned reranking strength (v6 re-execution, five seeds): $\\lambda$ selected per family by validation NDCG@20; test NDCG@20 reported once at the selected value. Protocol column: the same re-execution at the frozen shared $\\lambda=0.10$.}}\\label{{tab:lambda_tuned}}
\\centering
\\small
\\begin{{tabular}}{{llccc}}
\\toprule
Dataset & Family & $\\lambda^*$ (val.) & Test NDCG@20 at $\\lambda^*$ & Protocol ($\\lambda{{=}}0.10$) \\\\
\\midrule
{TUNED_ROWS}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Design-factor ablations}}\\label{{subsec:design_ablations}}"""))

# ============================================================ 6. ablation_lambda table: caption + LOO rows
EDITS.append((
"Bold column: protocol value $\\lambda_{\\text{attr}}=0.10$, fixed a priori and shared. The LOO family is not part of the released $\\lambda$-sweep artifact and is reported only at the protocol value.}",
"Bold column: protocol value $\\lambda_{\\text{attr}}=0.10$, fixed a priori and shared. The LOO rows come from the dedicated five-seed v6 sweep (\\S\\ref{subsec:design_ablations} reports the tuned comparison); other families are as released in the v3 sweep artifact.}"))

EDITS.append((
"Amazon-Book & shapley-mc & 0.02982 $\\pm$ 0.00074 & 0.03097 $\\pm$ 0.00089 & \\textbf{0.03187 $\\pm$ 0.00085} & 0.03376 $\\pm$ 0.00083 & 0.03591 $\\pm$ 0.00101 \\\\",
"Amazon-Book & shapley-mc & 0.02982 $\\pm$ 0.00074 & 0.03097 $\\pm$ 0.00089 & \\textbf{0.03187 $\\pm$ 0.00085} & 0.03376 $\\pm$ 0.00083 & 0.03591 $\\pm$ 0.00101 \\\\\n" + LOO_ROWS))

# bold the LOO protocol cells (0.10 column) in the freshly added rows
EDITS.append((
"ML-1M & loo-marginal & 0.04467 $\\pm$ 0.00030 & 0.04723 $\\pm$ 0.00026 & 0.04959 $\\pm$ 0.00048 & 0.05432 $\\pm$ 0.00064 & 0.06211 $\\pm$ 0.00040 \\\\",
"ML-1M & loo-marginal & 0.04467 $\\pm$ 0.00030 & 0.04723 $\\pm$ 0.00026 & \\textbf{0.04959 $\\pm$ 0.00048} & 0.05432 $\\pm$ 0.00064 & 0.06211 $\\pm$ 0.00040 \\\\"))
EDITS.append((
"Amazon-Book & loo-marginal & 0.02954 $\\pm$ 0.00089 & 0.03117 $\\pm$ 0.00125 & 0.03250 $\\pm$ 0.00100 & 0.03449 $\\pm$ 0.00133 & 0.03683 $\\pm$ 0.00120 \\\\",
"Amazon-Book & loo-marginal & 0.02954 $\\pm$ 0.00089 & 0.03117 $\\pm$ 0.00125 & \\textbf{0.03250 $\\pm$ 0.00100} & 0.03449 $\\pm$ 0.00133 & 0.03683 $\\pm$ 0.00120 \\\\"))

# ============================================================ 7. lambda figure caption
EDITS.append((
"\\caption{Reranking-strength sensitivity, NDCG@20 (illustrative single-seed curves; the five-seed values are in Table~\\ref{tab:ablation_lambda}). The LOO family is not part of this released sweep artifact; its dedicated sweep is a released run script (\\S\\ref{subsec:design_ablations} discusses the pending additions).}",
"\\caption{Reranking-strength sensitivity, NDCG@20, five-seed means: uniform, additive-pref, and LOO from the dedicated v6 sweep; Shapley-MC from the released v3 sweep (Table~\\ref{tab:ablation_lambda}). LOO dominates at every $\\lambda>0$ on both datasets.}"))

# ============================================================ 8. NGCF subsection before Discussion
EDITS.append((
"\\section{Discussion}\\label{sec:discussion}",
f"""\\subsection{{Second backbone: NGCF (cross-architecture replication)}}\\label{{subsec:second_backbone}}

To test whether the boundary result is an artifact of LightGCN's linear propagation, we re-ran the identical frozen matched-controls protocol with an NGCF-style backbone (nonlinear $W_1e_j+W_2(e_j\\odot e_i)$ aggregation with LeakyReLU, mean layer readout; identical hyperparameters, seeds 42--46, $k=24$, 100 validation negatives, native intervention, $\\lambda_{{\\text{{attr}}}}=0.10$; Table~\\ref{{tab:second_backbone}}). The family ordering replicates on both datasets: LOO is the strongest NDCG@20 method, Shapley is second, valid-linear third, and the non-game controls trail, with the unreranked model last. Absolute NDCG@20 is lower than under LightGCN because the shared frozen hyperparameters are not NGCF-tuned; this does not affect the within-run, same-model family comparison, which is the claim under test. Paired per-user permutation inference for this backbone is released with the per-user artifacts.

\\begin{{table*}}[t]
\\caption{{Second-backbone replication under the frozen matched-controls protocol (NGCF, five seeds, mean $\\pm$ SD). Only the propagation scheme differs from the C1b protocol.}}\\label{{tab:second_backbone}}
\\centering
\\small
\\begin{{tabular}}{{llcc}}
\\toprule
Dataset & Method & HitRate@20 & NDCG@20 \\\\
\\midrule
{NGCF_ROWS}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}

\\section{{Discussion}}\\label{{sec:discussion}}"""))

# ============================================================ 9. discussion sentence
EDITS.append((
"The results have three levels. First, validation-guided game signals improve ranking over uniform and similarity-style reweighting.",
"The results have three levels, and each now survives a cross-architecture check. First, validation-guided game signals improve ranking over uniform and similarity-style reweighting --- under LightGCN \\emph{and} NGCF (\\S\\ref{subsec:second_backbone})."))

# ============================================================ 10. limitations update
EDITS.append((
"The evidence is limited to LightGCN, MovieLens-1M, and a custom temporal Amazon-Book split; no external explainable-recommendation method is evaluated. Generalization requires at least a second backbone, another domain, and counterfactual or influence-function baselines. Released scripts for the required extensions accompany this revision: an NGCF-style second backbone under the identical protocol, multi-seed design-factor ablations, multi-seed masked-forward faithfulness curves, the missing LOO $\\lambda$-sweep with validation-tuned $\\lambda$ selection, validation-negative-set-size sensitivity ($|\\mathcal{N}_u^-|\\in\\{50,100,500\\}$), and attribution stability / model-randomization sanity checks; their results will be appended when the runs complete. Until then, the design-factor and masked-forward studies use one training seed per dataset and support no multi-seed claim.",
"The evidence covers two backbones (LightGCN primary, NGCF replication), two datasets, and one reranking intervention; no external explainable-recommendation method is evaluated, and a third domain remains future work. Completed in this revision: the NGCF second-backbone replication (\\S\\ref{subsec:second_backbone}), the LOO $\\lambda$-sweep, and the independently validation-tuned $\\lambda$ comparison (\\S\\ref{subsec:design_ablations}). Remaining released scripts whose results are pending: multi-seed design-factor ablations, multi-seed masked-forward faithfulness curves, validation-negative-set-size sensitivity ($|\\mathcal{N}_u^-|\\in\\{50,100,500\\}$), and attribution stability / model-randomization sanity checks; until they complete, the design-factor and masked-forward studies use one training seed per dataset and support no multi-seed claim. Paired per-user inference for the NGCF backbone follows with the released per-user artifacts."))

# ============================================================ apply
for old, new in EDITS:
    n = src.count(old)
    if n != 1:
        print(f"PATCH FAIL ({n} matches): {old[:100]!r}...")
        sys.exit(1)
    src = src.replace(old, new)

MAIN.write_text(src)
print(f"APPLIED {len(EDITS)} edits -> {len(src)} chars")
