# Claim Paper Transformation — Q1 Research Article

**New positioning:** Research article (not systematic review). The systematic-review ambition is removed; the taxonomy is kept as a **conceptual design framework that generates falsifiable claims**.

## New title (Research)
> **Validation-guided interaction attribution for graph-based recommendation: when a simple leave-one-out baseline matches or beats Shapley**
> Short: *Efficient validation-guided attribution for graph recommendation*

## Core claims (falsifiable, supported by assets)

### Claim 1 — Validation-guided attribution beats heuristics (strong positive)
- LightGCN + validation-only pairwise utility + frozen-graph reranking **significantly** beats uniform / additive-pref / attention / popularity on **both** datasets, **both** NDCG@20 and HitRate@20 (Holm-corrected, B=2000, 5 seeds).
- ML-1M: Shapley +7.0% NDCG vs uniform (0.04922 vs 0.04601, Δ=0.00322 [0.00271,0.00372], p<0.0005); LOO +8.1% (0.04976 vs 0.04601, Δ=0.00375). Amazon: +7.0% / +8.7% respectively.
- This is the **Q1-grade positive result vs baselines** the journal expects.

### Claim 2 — LOO Pareto-dominates Shapley for ranking (critical frontier)
- On ranking utility, **LOO ≥ Shapley**: ML-1M NDCG Δ=-0.00053 [-0.00096,-0.00012], p=0.008, Holm reject; Amazon NDCG Δ=-0.00049 [-0.00071,-0.00028], p<0.0005.
- HitRate: ML-1M tie (Δ=0.00037, p=0.575), Amazon LOO wins (p=0.036).
- Coverage: LOO leads on both (ML 0.641 vs 0.634, Amazon 0.236 vs 0.234).
- Cost: ML-1M 2010s vs 31658s (15.7×), Amazon 637s vs 8283s (13×); **NDCG gain per hour 18.3× / 16.1×** for LOO.
- **Deployment rule:** default to LOO; use Shapley only when faithfulness under redundancy/complementarity is demonstrated.

### Taxonomy-derived predictions (framework → claims)
- P1 additive leakage (λ·sim → Shapley = φ(v)+λ·sim) — validated by early pilot vs additive-pref control separation.
- P2 smoothness — pairwise log-sigmoid > hard NDCG as coalition value (pilot).
- P3 cost-dominance — unless contexts expose redundancy, LOO matches Shapley.

## What changed in `springer_latex/main.tex`

- Title, abstract, keywords rewritten to Research claims (no "systematic review").
- Introduction aims 3→2; Contributions rewritten to 5 claim contributions.
- § Systematic review methodology **replaced** by § From taxonomy to testable claims (3 predictions).
- § Implications for systematic review → § How results validate design framework.
- § Recommended final framing → § Deployment recommendation.
- Future work 1 rewritten to backbone/domain coverage (not corpus collection).
- Appendix review protocol renamed to "Supplementary: taxonomy codebook (not a systematic review)"; checklist updated to 8 Research-article items.
- Remaining "systematic review" strings: 2 (both in the new explanatory sentence "Instead of a systematic review...").

## How this is valid for Q1 (Discovery AI Research)

| Q1 expectation | How claim paper meets it |
|---|---|
| **Novel empirical insight** | First cost-aware head-to-head of Shapley vs LOO with matched validation-only utility, full-catalog, 5 seeds, paired bootstrap |
| **Strong baselines** | 5 controls incl. additive-pref (additive-leakage ablation) + attention + popularity; all share backbone/hyperparams |
| **Statistical rigor** | B=2000 within-seed bootstrap, Holm F=8, d_z, CI, % improved/harmed reported |
| **Cost-awareness** | Attribution seconds + gain/hour, Pareto frontier — rare in XAI, valued by Q1 |
| **Honest negative result** | LOO frontier is a discovery, not a failure; defines when Shapley is justified |
| **Reproducibility** | `requirements.lock`, cache-key manifests, SHA256, `run_faithfulness.py`/`run_ablations.py` |
| **Framework** | 5 principles + 5-axis taxonomy = reusable language, not just an algorithm |

## Manuscript structure (Research article, IMRaD-aligned)

1 Introduction (with claim contributions)
2 Background (Top-N, LightGCN, games/Shapley, why Shapley can fail, **Notation Table**)
3 Design framework (5 principles)
4 Conceptual taxonomy (5 axes) + detailed coding dimensions
5 Design rationale (why pairwise, stratified k=24, native intervention)
6 Algorithms & complexity (Algs 1–3, O(Mk|N⁻|) vs O(k|N⁻|))
7 Empirical protocol (leakage, evaluation, **Metrics formulas**, estimand B/Holm, backbone, **Hyperparams Table**)
8 Results (main, paired contrasts, cost)
9 Discussion + Why LOO matters + How to read results + Deployment recommendation
10 Future work, Threats to validity, Limitations, Conclusion
Appendices: taxonomy codebook, reproducibility package, ethics, interpretation scenarios, Research checklist
Declarations: complete (no OSF placeholder for review corpus)

## One-sentence Q1 cover-letter pitch

> "We show that validation-guided interaction attribution improves LightGCN ranking by 7–8% NDCG over heuristic reweighting on two datasets, but a simple leave-one-out marginal matches or beats bounded Shapley at 16–18× lower cost, establishing a cost-effectiveness frontier that redefines when Shapley should be used in recommender XAI."

## Next runs to strengthen the claim (optional but reviewer-pleasing)

```bash
# faithfulness: when does Shapley beat LOO? (true masked-forward)
python -m coalgamerec.pipeline configs/q1_lightgcn_ml1m.yaml
python scripts/run_faithfulness.py --all-seeds  # deletion/insertion AUC curves

# ablations: k, value smoothness, intervention
python scripts/run_ablations.py --ablation all --config configs/q1_lightgcn_ml1m.yaml
```

Add the resulting **deletion/insertion figure** (Fig 4) and **k-sensitivity table** to strengthen Claim 2's boundary condition — the only missing piece for a strong Q1 Research acceptance.
