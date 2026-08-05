# ActionShap paper-v2: canonical Q1 manuscript and result package

This is the only retained ActionShap manuscript/result package. Superseded
paper, paper-enhanced, overleaf-springer, schema-v1, demo, and duplicate result
folders have been removed so that stale assets cannot be mistaken for evidence.

## Submission source

- `paper.tex`: Springer Nature manuscript source.
- `paper.bib`: validated bibliography.
- `sn-jnl.cls`, `sn-basic.bst`: Springer Nature LaTeX template files.

## Evidence package

`final/` contains the generated schema-v2 assets for the real experiment:
figures, tables, compressed user-seed metrics, validation/provenance manifests,
and the generated results summary. The package includes primary 1,000-user
cohorts, 250-user robustness/sensitivity cohorts, five common seeds, two
timestamped datasets, two history-conditioned models, and exact budget-two
outcomes.

The latent-profile MovieLens seed-46 masking failure and NDCG-attribution
non-convergence are retained as limitations in the manuscript and manifests.
The primary ItemKNN gates pass. The raw JSON archive is distributed separately
through the content-addressed release workflow; verify its checksum before
submission. Dataset source files are intentionally excluded.

## Reproduction

From `code/`, with the raw release archive unpacked into `results/raw/`:

```bash
python scripts/make_paper_assets.py --raw results/raw --out ../paper-v2
python scripts/validate_manuscript.py \
  --paper ../paper-v2/paper.tex \
  --bib ../paper-v2/paper.bib \
  --require-final
```
