# CoalGameRec Springer Nature paper package

This is the ready-to-use paper folder for the CoalGameRec manuscript draft.

## Contents

```text
paper_package/
├── main.tex
├── references.bib
├── sn-jnl.cls
├── Makefile
├── README.md
├── assets/
│   ├── data/
│   │   ├── lightgcn_cost_effectiveness.csv
│   │   ├── lightgcn_main_results.csv
│   │   └── lightgcn_paired_contrasts.csv
│   ├── figures/
│   │   ├── architecture.svg
│   │   ├── ndcg_results.svg
│   │   └── cost_effectiveness.svg
│   └── tables/
│       ├── lightgcn_cost_effectiveness.md
│       ├── lightgcn_main_results.md
│       └── lightgcn_paired_contrasts.md
└── template/
```

## Springer Nature template note

The manuscript is written for the Springer Nature journal article class:

```latex
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}
```

The official Springer Nature template package should be downloaded from:

https://www.springernature.com/gp/authors/campaigns/latex-author-support/see-where-our-services-will-take-you/18782940

The sandbox could not download the official ZIP because the Springer CMS TLS connection failed. To make this folder usable immediately, I included a small compatibility `sn-jnl.cls` fallback. It is only for local drafting and compilation checks. Before journal submission, replace it with the official Springer Nature files, especially:

```text
sn-jnl.cls
sn-mathphys-num.bst
```

## Compile locally

If you have a LaTeX distribution installed:

```bash
cd paper-ideas/CoalGameRec/paper_package
make pdf
```

or manually:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Manuscript status

This is a Q1-style draft, not a final submission. Before submission, complete:

1. PRISMA systematic-review search and screening.
2. Final bibliography with all reviewed studies.
3. Institutional ethics determination.
4. Data and code availability statements with archive DOI.
5. Declarations and author contributions.
6. Replacement of the fallback `sn-jnl.cls` with the official Springer Nature template.
7. External archive for large raw result files.

## Style note

The manuscript avoids em dash characters by request.

## Figure assets

The manuscript contains TikZ figures directly in `main.tex` for LaTeX compilation. For convenience, viewable SVG versions are also included in:

```text
assets/figures/architecture.svg
assets/figures/ndcg_results.svg
assets/figures/cost_effectiveness.svg
```

These SVG files are paper assets for inspection and conversion. The LaTeX source remains the authoritative version for submission.
