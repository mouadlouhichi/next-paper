# ActionShap paper-v3

Self-contained full-length Springer Nature manuscript package, moved from
`ActionShap/actionshap.tex` and configured to use the required assets locally.

## Contents

- `actionshap.tex`: full journal manuscript.
- `paper.bib`: numbered bibliography.
- `sn-jnl.cls`, `sn-basic.bst`: Springer Nature template files.
- `final/`: current schema-v2 figures, tables, metrics, manifests, and summary.

## Compile

Run from this directory:

```bash
pdflatex actionshap.tex
bibtex actionshap
pdflatex actionshap.tex
pdflatex actionshap.tex
```

The manuscript uses only `final/` assets and repository-local paths. Raw datasets
and raw JSON result files remain outside the manuscript package.
