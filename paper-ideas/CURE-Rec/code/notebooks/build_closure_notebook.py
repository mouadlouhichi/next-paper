"""Build the reviewer-closure run notebook (13_reviewer_closure_runs.ipynb)."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "13_reviewer_closure_runs.ipynb"

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# CURE-Rec — reviewer-closure runs (round 2, remaining executions)

This notebook gathers **every experiment that must run outside the review sandbox**, in one place, with manuscript-ready outputs. The sandbox had no MovieLens download access and limited compute; everything else from the revision is already archived under `results/reviewer_phase_assets/`.

**Runs collected here**

| Run | What it produces | Reviewer item | Approx. cost |
|---|---|---|---|
| A | MovieLens-1M per-user paired inference (McNemar / bootstrap / permutation / Holm) | Phase 8 mandatory tests; Issue #7 | ~1–2 h (GPU) / ~3–5 h (CPU) |
| B | Semi-real integration (learned BPR base policy inside CURE-Sim) | Phase 9 #4; Issue #2 | ~1 h |
| C | Integrated scalability with real operator semantics (n=8, n=10) | Phase 9 #5 | n=8 ~1.5 h, n=10 ~6–8 h |
| D | Second independent domain (Amazon-2014 chronological ranking audit) | Phase 9 #6 | ~1 h |
| E | Aggregation into manuscript-ready tables | — | seconds |

**Claim discipline:** Runs A and D are chronological-ranking evidence only. Run B is simulator-conditional. None of these creates external causal policy evidence for CURE-Rec intervention selection; audited real policy logs remain future work (stated in the manuscript conclusion).

Requirements: Python ≥ 3.11, `pip install -e '.[dev,torch]'` (torch optional for CPU-only fallbacks in runs B/C; required for the SASRec part of run A).
"""))

cells.append(code("""from pathlib import Path
import sys, os
CANDIDATES=[Path.cwd(), Path.cwd()/'paper-ideas'/'CURE-Rec'/'code', *Path.cwd().parents]
ROOT=next(p for p in CANDIDATES if (p/'pyproject.toml').exists() and (p/'cure_rec').exists())
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
RESULTS = ROOT/'results'/'reviewer_phase_assets'
print('repo root:', ROOT)
import cure_rec, importlib
print('cure_rec OK')
try:
    import torch; print('torch', torch.__version__)
except ImportError:
    print('torch NOT available — Run A SASRec and torch paths will be skipped')
"""))

cells.append(md("""## Run A — MovieLens-1M per-user paired inference (Issue #7)

The archived MovieLens-1M audit stored seed-level aggregates only. This run re-trains the **frozen final configurations** (BPR hash `9875430c235b`, SASRec hash `7aa511398a7f`) for seeds 42–46, exports one hit/NDCG row per evaluated warm user, and computes the pre-specified user-level analysis:

- paired user bootstrap CIs for mean Recall@10 / NDCG@10 differences;
- exact McNemar tests on per-user hit indicators (sign test on discordant pairs);
- paired sign-flip permutation tests for NDCG;
- Cohen d_z; Holm correction across the model × metric family.

Downloads MovieLens-1M automatically (needs internet)."""))

cells.append(code("""# Run A (blocking). Uses scripts_review/phase_d_ml1m_paired.py
!{sys.executable} scripts_review/phase_d_ml1m_paired.py"""))

cells.append(code("""import pandas as pd
A = RESULTS/'movielens_1m_paired'
ci = pd.read_csv(A/'paired_bootstrap_ci_ml1m.csv'); tests = pd.read_csv(A/'paired_tests_holm_ml1m.csv'); perm = pd.read_csv(A/'paired_permutation_tests_ml1m.csv')
display(ci); display(tests); display(perm)
# Manuscript-ready sentence
r42 = ci[ci.analysis=='seed_42']
print('Manuscript snippet:')
for _,row in r42.iterrows():
    print(f\"{row.model} vs popularity ({row.metric}): diff {row.difference_mean:.4f}, 95% CI [{row.bootstrap_low:.4f}, {row.bootstrap_high:.4f}], dz={row.effect_dz:.2f}\")
"""))

cells.append(md("""## Run B — Semi-real integration (Issue #2, Phase 9 #4)

Deploys a **learned ranker** (BPR-MF trained on simulator-logged feedback) as the base policy inside CURE-Sim and re-runs the exact intervention game on top of it. If the sandbox already archived `semireal_integration/semireal_comparison.csv`, this cell verifies it instead of re-running."""))

cells.append(code("""B = RESULTS/'semireal_integration'/'semireal_comparison.csv'
if B.exists():
    print('semi-real results already archived:')
    display(pd.read_csv(B))
else:
    !{sys.executable} scripts_review/phase_f_semireal.py
    display(pd.read_csv(B))
display(pd.read_csv(RESULTS/'semireal_integration'/'semireal_attribution_comparison.csv'))"""))

cells.append(md("""## Run C — Integrated scalability with real operator semantics (Phase 9 #5)

The four extended operators (`session_length_cap`, `freshness_quota`, `provider_cooldown`, `category_coverage_quota`) are implemented as real slate transformations in `cure_rec/interventions.py`. This runs the **exact integrated game** for n=8 (256 coalitions × 4 scenarios) and n=10 (1024 × 4), reporting wall-clock, peak memory, Shapley efficiency, exact-vs-sampled fidelity, and selection regret.

n=10 is slow (~6–8 h on a laptop); run it unattended or skip it — the n=8 result already upgrades the benchmark from arithmetic-only to integrated."""))

cells.append(code("""C8 = RESULTS/'integrated_scalability'/'players_8'/'integrated_summary.csv'
if C8.exists():
    display(pd.read_csv(C8))
else:
    !{sys.executable} scripts_review/phase_g_integrated_scalability.py 8 42
    display(pd.read_csv(C8))
display(pd.read_csv(RESULTS/'integrated_scalability'/'players_8'/'sampled_fidelity.csv'))"""))

cells.append(code("""# n=10 — heavy; uncomment to run
# !{sys.executable} scripts_review/phase_g_integrated_scalability.py 10 42
# display(pd.read_csv(RESULTS/'integrated_scalability'/'players_10'/'integrated_summary.csv'))"""))

cells.append(md("""## Run D — Second independent domain: Amazon-2014 chronological ranking audit (Phase 9 #6)

MovieLens-25M (second MovieLens release, including an unfavorable BPR result) is already archived. The reviewer also asked for a genuinely different domain. This cell audits **Amazon-2014 ratings (Toys & Games)** under the same chronological leave-one-out, warm-candidate, cold-counting protocol with popularity and BPR-MF (torch SASRec too when available).

If the Stanford mirror is unreachable, download `ratings_Toys_and_Games.csv` manually from https://jmcauley.ucsd.edu/data/amazon/links/ (Amazon-2014) into `data/raw/amazon_toys/` and re-run."""))

cells.append(code("""import urllib.request, shutil
import numpy as np, pandas as pd
from cure_rec.data import _ensure_directory, _safe_extract_zip
from cure_rec.models import chronological_leave_one_out, PopularityRecommender, BPRMFRecommender, build_shared_candidates, evaluate_leave_one_out

AMZ_URL = 'http://snap.stanford.edu/data/amazon/productGraph/ratings_Toys_and_Games.csv'
raw = ROOT/'data'/'raw'/'amazon_toys'
raw.mkdir(parents=True, exist_ok=True)
target = raw/'ratings_Toys_and_Games.csv'
if not target.exists():
    try:
        with urllib.request.urlopen(AMZ_URL, timeout=120) as src, target.open('wb') as out:
            shutil.copyfileobj(src, out)
        print('downloaded', target.stat().st_size, 'bytes')
    except Exception as exc:
        print('DOWNLOAD FAILED:', exc)
        print('Place ratings_Toys_and_Games.csv in', raw, 'and re-run this cell.')

if target.exists():
    df = pd.read_csv(target, header=None, names=['user_id','item_id','rating','timestamp'])
    df['response'] = (df['rating'] >= 4).astype(int)
    df['user_id'] = df['user_id'].astype('int64'); df['item_id'] = df['item_id'].astype('int64')
    print('ratings:', len(df), 'positives:', int(df.response.sum()))
    split = chronological_leave_one_out(df)
    pop = PopularityRecommender().fit(split.train)
    bpr = BPRMFRecommender(factors=64, max_updates=200_000, seed=42).fit(split.train)
    mp = evaluate_leave_one_out(pop, split, k=10, max_users=1000)
    mb = evaluate_leave_one_out(bpr, split, k=10, max_users=1000)
    summary = pd.DataFrame([
        {'dataset':'amazon_toys_2014','model':mp.model,'recall@10':mp.recall_at_k,'ndcg@10':mp.ndcg_at_k,'cold':mp.cold_test_items},
        {'dataset':'amazon_toys_2014','model':mb.model,'recall@10':mb.recall_at_k,'ndcg@10':mb.ndcg_at_k,'cold':mb.cold_test_items},
    ])
    outdir = RESULTS/'second_domain'
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir/'amazon_toys_ranking_audit.csv', index=False)
    display(summary)
else:
    print('skipped: dataset not available')"""))

cells.append(md("""## Run E — aggregation and manuscript insertion checklist

Prints which closure artifacts now exist and which manuscript placeholders they fill."""))

cells.append(code("""checks = {
 'Run A: ML-1M paired CIs': RESULTS/'movielens_1m_paired'/'paired_bootstrap_ci_ml1m.csv',
 'Run A: ML-1M paired tests': RESULTS/'movielens_1m_paired'/'paired_tests_holm_ml1m.csv',
 'Run B: semi-real comparison': RESULTS/'semireal_integration'/'semireal_comparison.csv',
 'Run C: integrated n=8': RESULTS/'integrated_scalability'/'players_8'/'integrated_summary.csv',
 'Run C: integrated n=10': RESULTS/'integrated_scalability'/'players_10'/'integrated_summary.csv',
 'Run D: Amazon toys audit': RESULTS/'second_domain'/'amazon_toys_ranking_audit.csv',
}
for name, path in checks.items():
    print(('DONE  ' if path.exists() else 'MISSING'), name, '->', path.relative_to(ROOT))
print()
print('Insert DONE rows into the manuscript: RQ4/limitations (Run A), Section 5.8 semi-real paragraph (Run B),')
print('scalability paragraph in Section 5.7 (Run C), second-domain paragraph in Section 5.7 (Run D).')"""))

nb = {
 "cells": cells,
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
              "language_info": {"name": "python", "version": "3.11"}},
 "nbformat": 4, "nbformat_minor": 5,
}
NB.write_text(json.dumps(nb, indent=1))
print("wrote", NB)
