# ActionShap paper-v3

Self-contained Springer Nature manuscript package for the recommendation-only
ActionShap study. The canonical source is `actionshap.tex`; generated assets are
under `final/` and use schema-v2 results only.

## Contents

- `actionshap.tex`: manuscript source.
- `paper.bib`, `sn-basic.bst`: complete bibliography and Springer numbered style.
- `sn-jnl.cls`: pinned Springer Nature class used by the manuscript.
- `final/`: compact publication tables, figures, CSV/JSON audit assets, and the
  PASS validation manifest.
- `REPRODUCIBILITY.md`: public source, archive hash, configuration, runtime, and
  environment information.

## Build the manuscript

Run from this directory with a TeX distribution that provides `pdflatex`,
`bibtex`, `natbib`, `tikz`, `algorithm`, `algpseudocode`, `booktabs`, and the
packages listed in the source:

```bash
./build.sh
```

Equivalent manual commands are:

```bash
pdflatex -interaction=nonstopmode actionshap.tex
bibtex actionshap
pdflatex -interaction=nonstopmode actionshap.tex
pdflatex -interaction=nonstopmode actionshap.tex
```

The bibliography must be compiled between the first and second LaTeX passes.
The source contains 47 citation keys, and the pinned `paper.bib` contains the
same 47 entries; `code/scripts/validate_manuscript.py` checks that relationship
before submission.

## Rebuild assets

From `paper-ideas/ActionShap/code/`, after placing the raw schema-v2 archive
contents in `results/raw/`:

```bash
python scripts/make_paper_assets.py --raw results/raw --out ../paper-v3
```

The generator rejects invalidated schema-v1 results, enforces the LOO identity,
removes pointwise AIA rows from budget sensitivities, and writes compact PDF
tables while retaining complete CSV/JSON matrices for supplementary auditing.

## Validation

From `paper-ideas/ActionShap/code/`:

```bash
python scripts/validate_manuscript.py \
  --paper ../paper-v3/actionshap.tex \
  --bib ../paper-v3/paper.bib \
  --require-final
python scripts/validate_review_contract.py --paper-root ../paper-v3
```

The current final manifest is `final/manifests/validation_report.json`. A PASS
means the frozen artifact contract and provenance checks pass; the notes retain
known robustness boundaries, including the weaker profile model and the
unconverged NDCG attribution stress test.
