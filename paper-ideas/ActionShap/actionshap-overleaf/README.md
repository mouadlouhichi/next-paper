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
explicit Appendices A--E in the main paper, a compact evidence appendix of
approximately 6--7 pages, a recommendation-quality Table 2, a computational-cost
table, and no internal submission checklist. Full robustness and statistical audit
matrices are supplied only in the separate supplementary PDF.

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
rows. The main manuscript retains only compact, reader-facing tables. For final
submission, rename the downloaded outputs to `ActionShap_Manuscript.pdf` and
`ActionShap_Supplementary_Audit.pdf` rather than leaving browser-generated
`(1)` or `__2_` suffixes.

## Reproducibility

The numerical assets come from the validated schema-v2 release in the parent
ActionShap package. The public source repository is
`https://github.com/mouadlouhichi/next-paper`; its visibility and any DOI-backed
archive deposit must be completed before submission. The exact branch commit and
archive SHA-256 are recorded in the parent `paper-v3/REPRODUCIBILITY.md`.

The repository does not track stale compiled PDFs. Compile `actionshap.tex` and `supplementary.tex` from this package in Overleaf so the PDF reflects the current source, table sequence, and release metadata.

## Overleaf troubleshooting

- Upload the **entire folder**, including `tables/`, `figures/`, `sn-jnl.cls`,
  `sn-mathphys-num.bst`, and `latexmkrc`. If `tables/` (or any input asset) is
  missing, the compiler reports a warning and a bold placeholder instead of the
  table; re-upload the complete `tables/` directory and recompile.
- Set the main document to `actionshap.tex` (Menu > Settings > Main file).
  Compile `supplementary.tex` as a separate project entry point only when the
  full numerical audit PDF is required.
- The package was compile-verified end-to-end with pdfLaTeX (40 pages main,
  35 pages supplement, zero errors) using the pinned `sn-jnl` class.

### Table authoring rule (sn-jnl + threeparttable)

The `sn-jnl` class wraps every `table` environment in the real
`threeparttable`, which measures the `tabular` body in its own box. Wrapping a
`tabular` in `\resizebox` (or `\scalebox`) inside these tables breaks group
balance and triggers "Missing \endgroup", "Division by 0", and vanished tables.
All manuscript tables are therefore resize-free and use *plain* `tabular`
only (`tabularx`/`tabular*` inside the threeparttable wrapper are not used);
width is controlled by font size, `tabcolsep`, wrapping `p{...}` columns, and
shortened cell text. Keep it that way when editing tables.

For the same reason, never use the `[H]` placement specifier on `table` (or
`sidewaystable`) environments: `float.sty` implements `[H]` by locally
redefining `\endtable` to its own `\float@endH`, which bypasses the class's
threeparttable/closing code and causes "Extra }, or forgotten \\endgroup",
"\\begin{threeparttable} ... ended by \\end{table}", and a cascade of
"Not in outer par mode" errors. Use `[!htbp]` plus the existing `\\FloatBarrier`
commands instead (figures are unaffected because the class does not wrap them).
