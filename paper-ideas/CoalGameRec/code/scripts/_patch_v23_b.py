#!/usr/bin/env python3
"""v23 patch B: algorithm-1 correctness, specs (negatives, faithfulness,
notation), provenance, conclusion narrowing, C1 reframing."""
from __future__ import annotations

import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parent.parent.parent / "paper_package" / "main.tex"
src = MAIN.read_text()
EDITS: list[tuple[str, str]] = []

# ---------------------------------------------------------------- C1 intro reframe
EDITS.append((
"The primary study compares validation-guided game attribution against non-game controls that have \\emph{no} validation access, leaving open how much of the game-vs-heuristic gap is attributable to validation access itself. Study C1 closes this gap by adding two matched validation-informed non-game controls that consume the same validation item $i_u^+$ as the Shapley/LOO games but use no coalition structure:",
"The primary non-game families use no \\emph{direct} calibration weighting, but they operate on the calibration-selected player set (\\S\\ref{subsec:protocol_timeline}), leaving open how much of the game-vs-heuristic gap is attributable to direct calibration weighting rather than to valuation structure. Study C1 addresses this by adding two matched validation-informed non-game controls that consume the same calibration item $i_u^+$ as the Shapley/LOO games but use no coalition structure:"))

# ---------------------------------------------------------------- main-results asymmetry sentence
EDITS.append((
"they were added in the confirmatory study C1 (\\S\\ref{subsec:c1}, Table~\\ref{tab:c1_main}), which resolves the validation-access asymmetry empirically.",
"they were added in the confirmatory study C1 (\\S\\ref{subsec:c1}, Table~\\ref{tab:c1_main}), which matches direct calibration weighting across game and non-game methods."))

# ---------------------------------------------------------------- Algorithm 1 rewrite
EDITS.append((
"""\\Require User $u$, training history $H_u$, train-only similarity vectors $\\{x_j\\}$ (distinct from native embeddings $\\mathbf{e}$), val positive $i_u^+$, bound $k$; effective budget $m\\gets\\min(k,|H_u|)$
\\Ensure Player set $P_u$, $|P_u|=m$ in deterministic order
\\State Compute $\\text{sim}_{\\text{prof}}(j)=\\cos(x_j,\\bar x_u)$ where $\\bar x_u=\\frac{1}{|H_u|}\\sum_{j\\in H_u}x_j$
\\State Compute $\\text{sim}_{\\text{val}}(j)=\\cos(x_j,x_{i_u^+})$
\\State Let $q=\\lfloor m/3\\rfloor$ and $r=m-3q$ (so short histories scale automatically; for $|H_u|<k$ the same steps run with $m=|H_u|$ and no separate fallback is needed). Select $q$ profile-similar items for $P_1$, then $q$ validation-similar remaining items for $P_2$, then $q+r$ remaining items for $P_3$ by farthest-point sampling with cosine distance $1-\\text{SafeCosine}$ initialized from $P_1\\cup P_2$; ties use item id
\\State $P_u\\gets P_1\\cup P_2\\cup P_3$; if $|P_u|<\\min(k,|H_u|)$, fill the remainder from $H_u\\setminus P_u$ by $\\text{sim}_{\\text{prof}}$ order; sort by item id
\\State Cosines use $\\text{SafeCosine}(a,b)=a^\\top b/(\\|a\\|\\|b\\|+\\epsilon)$ (zero for a zero-norm vector); when $|H_u|<k$, missing slots are filled deterministically from the remaining profile-similarity order. Complexity with maintained minimum distances: $\\mathcal{O}(k|H_u|d)$
\\State \\Return $P_u$""",
"""\\Require User $u$, training history $H_u$, train-only similarity vectors $\\{x_j\\}$ (distinct from native embeddings $\\mathbf{e}$), calibration positive $i_u^+$, bound $k=24$
\\Ensure Player set $P_u$ in deterministic (item-id) order
\\If{$|H_u|\\le k$}
  \\State \\Return $P_u\\gets H_u$ \\Comment{the bounded game covers the full history; selection is inactive, so no stratum is ever initialized from an empty set}
\\EndIf
\\State Compute $\\text{sim}_{\\text{prof}}(j)=\\text{SafeCosine}(x_j,\\bar x_u)$ with $\\bar x_u=\\frac{1}{|H_u|}\\sum_{j\\in H_u}x_j$, and $\\text{sim}_{\\text{val}}(j)=\\text{SafeCosine}(x_j,x_{i_u^+})$
\\State $P_1\\gets$ the $\\lfloor k/3\\rfloor$ highest-$\\text{sim}_{\\text{prof}}$ items (ties by item id); $R\\gets H_u\\setminus P_1$
\\State $P_2\\gets$ the $\\lfloor k/3\\rfloor$ highest-$\\text{sim}_{\\text{val}}$ items in $R$ (ties by item id); $R\\gets R\\setminus P_2$
\\State $P_3\\gets$ greedy farthest-point selection of $k-|P_1|-|P_2|$ items from $R$ under cosine distance $1-\\text{SafeCosine}$, seeded from $P_1\\cup P_2$ (non-empty here because $|H_u|>k\\ge3$); ties by item id
\\State \\Return $P_u\\gets P_1\\cup P_2\\cup P_3$ sorted by item id \\Comment{SafeCosine$(a,b)=a^\\top b/(\\|a\\|\\|b\\|+\\epsilon)$, zero for zero-norm vectors; complexity $\\mathcal{O}(k|H_u|d)$ with maintained minimum distances}"""))

# ---------------------------------------------------------------- negative-sampling spec
EDITS.append((
"Two consequences are stated plainly.",
"Validation negatives for the coalition utility are sampled uniformly without replacement from the catalog excluding the user's training items and the calibration target (the test item is never blocked because it is unknown at calibration time); the draw is deterministic per user (estimation-seed $+10^{5}+u$), fixed across all coalitions and families within a run, and identical across training seeds for a given estimation seed. Two consequences are stated plainly."))

# ---------------------------------------------------------------- faithfulness usage spec
EDITS.append((
"\\subsection{Faithfulness scope}\\label{sec:explainability}\nRanking improvement is not explanation faithfulness. The primary diagnostic is the C1 masked-forward comparison in Table~\\ref{tab:c1_faith}, which removes edges, renormalizes degrees, and re-propagates the frozen model.",
"\\subsection{Faithfulness scope}\\label{sec:explainability}\nRanking improvement is not explanation faithfulness. The primary diagnostic is the C1 masked-forward comparison in Table~\\ref{tab:c1_faith}, which removes edges, renormalizes degrees, and re-propagates the frozen model. Intervention semantics: the masked set is chosen by \\emph{absolute} attribution magnitude over the full training history $H_u$ (players outside $P_u$ have weight zero; ties break by item index); fractions $\\{0.1,0.2,0.3\\}$ are relative to $|H_u|$; in \\emph{deletion} mode the top-weighted edges are removed while the background $B_u$ and all non-player edges are retained; in \\emph{insertion} mode only the top-weighted edges are retained (background removed); the unmasked run with the full history is the reference. Signed attributions are used only through their magnitude here; separating positive and negative weights is part of the released multi-seed extension."))

# ---------------------------------------------------------------- signed-attribution note (reranking section)
EDITS.append((
"Shapley and LOO weights may be signed, but every family uses the same reranker. Candidates exclude training items, and the test item is never used for attribution or tuning.",
"Shapley and LOO weights may be signed, but every family uses the same reranker. Because both the base and attribution score vectors are z-scored, per-user positive rescaling of the weights is inert: the intervention depends on the \\emph{direction and relative signed weighting} of the history representation, not on its magnitude. Candidates exclude training items (and, under the corrected protocol, the calibration item), and the test item is never used for attribution or tuning. The fraction of negative attributions, LOO--Shapley weight correlation, and positive-only/absolute-weight variants are reported with the v7 re-execution artifacts."))

# ---------------------------------------------------------------- friedman explanation
EDITS.append((
"\\subsection{Omnibus ranking test (supplementary)}\\label{subsec:friedman}",
"\\subsection{Omnibus ranking test (supplementary)}\\label{subsec:friedman}\n\nThe omnibus test rejects while no Nemenyi--Holm pair survives because user-block ranks are heavily tied (family differences at the individual level are far smaller than the within-user rank noise) and the Nemenyi post-hoc bound is conservative at nine families; the test confirms that the families are not interchangeable in aggregate, while the paired contrasts of \\S\\ref{subsec:paired}--\\S\\ref{subsec:c1} remain the primary evidence. We retain it here rather than moving it to a supplement for completeness."))

# ---------------------------------------------------------------- conclusion narrowing
EDITS.append((
"This paper argues that cooperative-game attribution should be treated as a precise design language rather than a guaranteed performance enhancer. The LightGCN case study shows that a carefully specified bounded Shapley intervention improves ranking over uniform and similarity-based controls on MovieLens-1M and Amazon-Book. However, leave-one-out marginal attribution captures most or all of the ranking benefit at a fraction of the computational cost. This boundary condition is central: the axiomatic appeal of Shapley does not by itself justify its use unless coalition-context averaging adds measurable ranking, explanation, stability, or interaction-sensitive benefits beyond simpler marginal baselines.\n\nWithin the evaluated protocol, LOO is the computationally preferred validation-guided reranker; Shapley is justified only when follow-up masked-forward faithfulness or interaction-sensitivity results demonstrate benefits that offset its $13.0$--$15.7\\times$ attribution-time cost.",
"This paper argues that cooperative-game attribution should be treated as a precise design language rather than a guaranteed performance enhancer. The contribution is an empirical boundary result for a specific validation-guided reranking design: coalition-context averaging did not improve NDCG@20 over the grand-coalition marginal in the reported runs, while costing substantially more. Both game-derived signals outperform uniform and profile-based reweighting under LightGCN and NGCF; Shapley does not consistently outperform the strongest validation-informed linear control once calibration access is matched, and LOO retains the highest NDCG@20 point estimates in the matched studies. The study does not show that LOO recovers Shapley allocations or that it is generally sufficient for explanation; the axiomatic appeal of Shapley does not by itself justify its cost unless coalition-context averaging adds measurable ranking, explanation, stability, or interaction-sensitive benefits beyond simpler marginal baselines.\n\nWithin the evaluated native-embedding protocol at the shared $\\lambda_{\\mathrm{attr}}=0.10$, LOO is the computationally preferred validation-guided reranker; claims beyond that setting --- other interventions, tuned $\\lambda$, explanation faithfulness --- require the targeted re-runs released with this revision."))

# ---------------------------------------------------------------- provenance subsection
EDITS.append((
"\\bibliography{references}",
"""\\subsection*{Provenance and auditability}
All code, configurations, split manifests, and per-user metrics live in the public repository \\url{https://github.com/mouadlouhichi/next-paper} (branch \\texttt{arena/019fdd75-next-paper}; immutable commit hashes per run are recorded in each run's \\texttt{manifest.json}). Table-level provenance: primary tables derive from runs \\texttt{\\{ml1m,amazon\\_books\\}\\_lightgcn\\_v3\\_prospective}; C1/C1b tables from \\texttt{*\\_v4b\\_matched\\_controls}; second-backbone tables from \\texttt{*\\_ngcf\\_v6\\_second\\_bone}; $\\lambda$-sweep tables from \\texttt{*\\_lightgcn\\_v6\\_lambda\\_sweep}; stability/negset diagnostics from \\texttt{*\\_lightgcn\\_v6\\_randomization\\_sanity} and \\texttt{*\\_lightgcn\\_v6\\_negset\\_sensitivity}. Deviations between executions (device, trainer, candidate exclusion under the corrected protocol) are recorded in the run manifests; the frozen-protocol declaration of 2026-07-15 refers to the in-repository configuration and split artifacts (SHA256 fingerprints in \\texttt{item\\_vectors\\_report.json} and split \\texttt{meta.json}), not to an external preregistration service. The Python environment lock and table/figure regeneration scripts are released under \\texttt{code/}.

\\bibliography{references}"""))

# ---------------------------------------------------------------- notation paragraph
EDITS.append((
"\\subsection{Evaluation protocol}\\label{subsec:evaluation_protocol}\n\nFor each evaluated user, all training items are excluded from the candidate list.",
"\\subsection{Notation}\\label{subsec:notation}\n\n\\begin{table}[t]\n\\caption{Representations used by the protocol (defined once).}\\label{tab:notation}\n\\centering\n\\small\n\\begin{tabular}{lp{0.62\\linewidth}}\n\\toprule\nSymbol & Meaning \\\\\n\\midrule\n$x_j$ & train-only item similarity vector (item--item co-occurrence statistics of the training graph); used for player selection, profile/attention/additive-pref weights, and ILD \\\\\n$\\mathbf{e}_j$ & native final embedding of item $j$ from the frozen backbone (layer-average of the propagated LightGCN/NGCF embeddings); used for the reranking intervention and valid-sim/valid-linear \\\\\n$\\mathbf{e}^{(0)}_j$ & layer-0 (learned, unpropagated) embedding; perturbation experiments act here \\\\\nexternal kernel & the train-only similarity matrix used as an alternative intervention space (\\S\\ref{subsec:design_ablations}) \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\subsection{Evaluation protocol}\\label{subsec:evaluation_protocol}\n\nFor each evaluated user, all training items are excluded from the candidate list (and, under the corrected protocol of \\S\\ref{subsec:protocol_timeline}, the calibration item as well)."""))

# ============================================================ apply
for old, new in EDITS:
    n = src.count(old)
    if n != 1:
        print(f"PATCH FAIL ({n} matches): {old[:110]!r}...")
        sys.exit(1)
    src = src.replace(old, new)
MAIN.write_text(src)
print(f"APPLIED {len(EDITS)} edits -> {len(src)} chars")
