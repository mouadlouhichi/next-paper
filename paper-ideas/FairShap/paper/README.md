# FairShap manuscript — Springer Nature Discover AI (sn-jnl)

This folder is a self-contained **Springer Nature LaTeX** paper submission, set up
from the official Springer Nature author template
([latex-author-support](https://www.springernature.com/gp/authors/campaigns/latex-author-support)).

## Contents (all assets)

| File | Purpose |
|---|---|
| `fairshap.tex` | the manuscript (main file) |
| `sn-jnl.cls` | **official Springer Nature journal class** (v3.1, Dec 2024) |
| `sn-mathphys-num.bst` | Math & Physical Sciences **Numbered** reference style |
| `sn-mathphys.bst` | Math & Physical Sciences reference style (source of the numbered flavor) |
| `sn-basic.bst` | Basic Springer reference style (for switching styles) |
| `fairshap-bibliography.bib` | references (7 entries, all cited in the paper) |
| `figs/` | figures: Pareto fronts (ML-1M, Amazon-Book) + efficiency bar chart (PDF + PNG) |
| `user-manual.pdf` | Springer template user manual (kept for reference) |

## How to compile (macOS / TeX Live / MacTeX)

```bash
cd paper-ideas/FairShap/paper
pdflatex fairshap
bibtex   fairshap
pdflatex fairshap
pdflatex fairshap
```

Or with `latexmk`:
```bash
latexmk -pdf fairshap.tex
```

Requires a TeX distribution (TeX Live / MacTeX / MiKTeX). All standard packages
used (`graphicx`, `amsmath`, `amsthm`, `algorithm`, `booktabs`, `appendix`) ship
with any full distribution.

## Switching reference style

The document uses `sn-mathphys-num` (numbered). To switch, edit the
`\documentclass` line, e.g.:
```latex
\documentclass[pdflatex,sn-mathphys-ay]{sn-jnl}   % author-year
\documentclass[pdflatex,sn-basic]{sn-jnl}          % basic numbered
```

## Source of the results

Every number in the tables and figures is taken verbatim from
`paper-ideas/FairShap/results/fairshap_{ml1m,amazon-book}.json`.
No result is fabricated.
