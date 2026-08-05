# ActionShap paper-v2: submission manuscript and evidence package

This is the enhanced, recommendation-only manuscript package for the ActionShap study. It is written against the canonical plan in `ActionShap_Recommendation_Paper_Structure.md` and uses the Springer Nature `sn-jnl` template.

## Submission source

- `paper.tex`: revised Q1-style manuscript, with a restrained evaluation-framework contribution rather than a universal Shapley claim.
- `paper.bib`: validated bibliography used by the manuscript.
- `sn-jnl.cls`, `sn-basic.bst`: Springer Nature journal template files.

## Evidence package

`final/` contains the generated assets for the real schema-v2 matrix: five seeds, two timestamped datasets, two history-conditioned models, primary 1,000-user cohorts, 250-user robustness/sensitivity cohorts, exact budget-two outcomes, convergence diagnostics, component AIA tables, decision outcomes, and provenance manifests.

The primary ItemKNN results exceed popularity on sampled NDCG and Recall for both datasets. The profile model and one MovieLens profile seed are retained as explicitly reported robustness boundaries. The NDCG-attribution sensitivity is unconverged at `M=1000` and is not used for headline claims.

## Reproduction gate

Run from `code/`:

```bash
python scripts/make_paper_assets.py --raw results/raw --out ../paper-v2
python scripts/validate_manuscript.py --paper ../paper-v2/paper.tex --bib ../paper-v2/paper.bib --require-final
```

The raw archive is distributed separately through the content-addressed release workflow. Verify its checksum against the release manifest before submission. Dataset source files are not included in this paper package.
