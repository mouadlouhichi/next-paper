# CAVI Negative-Result Manuscript (Discover Artificial Intelligence)

**File:** `cavi-negative-result.tex` — Springer Nature Discover AI template
(`sn-jnl.cls`, `sn-mathphys-num` reference style), matching the house style of
the authors' prior SignalShap/ActionShap drafts.

## How to compile

This repo does **not** bundle the Springer class files (they ship in the journal
template zip and are covered by Springer's license). To compile:

1. **Download the Springer Nature LaTeX template** for Discover AI
   (the `sn-article` template), which provides:
   - `sn-jnl.cls`
   - `sn-mathphys-num.bst`
2. Copy `sn-jnl.cls` and `sn-mathphys-num.bst` into this `manuscript/` directory.
3. Compile from this directory:
   ```bash
   pdflatex cavi-negative-result
   bibtex   cavi-negative-result
   pdflatex cavi-negative-result
   pdflatex cavi-negative-result
   ```
   (or use `latexmk -pdf cavi-negative-result` if you have latexmk).

Requires a TeX distribution (TeX Live / MacTeX / MiKTeX) with the standard
packages: `graphicx`, `amsmath`, `amsthm`, `algorithm`, `booktabs`, `appendix`,
etc.

## Contents

- `cavi-negative-result.tex` — the manuscript.
- `cavi-bibliography.bib` — 12 references.

Every number in the tables is taken verbatim from
`paper-ideas/CAVI/results/*.json`. No result is fabricated.

## Quick check before submission

- Abstract is 150–250 words, unstructured, no citations, no equations. ✅
- 3–6 keywords. ✅ (5 given)
- Declarations block present. ✅
- Bibliography compiles with `bibtex` (all keys match). ✅ (checked)
