# ActionShap Overleaf submission package

This folder uses the same Springer Nature `sn-jnl` class, numeric
Math--Physical Sciences bibliography style, `latexmkrc`, and directory layout
as the working `signalshap-overleaf.zip` package.

## Main manuscript

- Main file: `actionshap.tex`
- Bibliography: `actionshap-bibliography.bib`
- Template class: `sn-jnl.cls`
- Numeric BibTeX style: `sn-mathphys-num.bst`
- Figures: `figures/`
- Compact manuscript tables: `tables/`

The main manuscript has the six-section SignalShap-style organization,
explicit Appendix A--F lettering, a recommendation-quality Table 2, compact
robustness tables, a computational-cost table, and no internal submission
checklist.

## Supplementary audit

`supplementary.tex` is a separate PDF entry point using the same template. It
contains the expanded alignment, intervention, paired-test, convergence, and
contract tables that are too dense for the main manuscript. The main paper
should be compiled as `actionshap.tex`; compile `supplementary.tex` separately
only when the venue requests the full numerical audit.

## Build

In Overleaf, set the main document to `actionshap.tex`, select pdfLaTeX, and
use BibTeX. Recompile from scratch once. The package contains 42 cited entries
and 42 matching bibliography records; no citation key is left unresolved.

The expanded supplementary PDF is intentionally long and preserves the audit
rows. The main manuscript retains only compact, reader-facing tables.

## Reproducibility

The numerical assets come from the validated schema-v2 release in the parent
ActionShap package. The public source repository is
`https://github.com/mouadlouhichi/next-paper`; its visibility and any DOI-backed
archive deposit must be completed before submission. The exact branch commit and
archive SHA-256 are recorded in the parent `paper-v3/REPRODUCIBILITY.md`.
