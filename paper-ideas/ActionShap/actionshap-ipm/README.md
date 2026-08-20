# ActionShap — Elsevier IPM submission package

Elsevier **cas-sc** (CAS single-column) conversion of the ActionShap
manuscript for **Information Processing & Management** (ISSN 0306-4573),
prepared per the journal's Guide for Authors (double-anonymized review).

## Files

| File | Purpose |
|---|---|
| `actionshap.tex` | **Anonymized manuscript** (upload as the manuscript file) |
| `titlepage.tex` | **Title page** with authors, affiliations, corresponding author, CRediT, funding, competing-interests declaration (separate file, as required) |
| `supplementary.tex` | Supplementary numerical audit (upload as supplementary material) |
| `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst` | Elsevier CAS template (official) |
| `actionshap-bibliography.bib` | Bibliography (author-year, `cas-model2-names`) |
| `tables/`, `figures/` | Editable tables (booktabs, no vertical rules) and vector PDF figures |
| `tables/review5_validation.tex` | Quality-gate and validation tables for the sequential-scorer and estimator stress tests |

## Compliance with the IPM guide

- **Double-anonymized review**: the manuscript contains no author names,
  affiliations, emails, acknowledgements, funding, or identifying URLs
  (the release URL is de-anonymized at acceptance). Identifying information
  lives only in `titlepage.tex`.
- **Abstract**: 200 words (limit 250).
- **Keywords**: 7, no "and"/"of" multi-word phrases.
- **References**: author-year via `natbib` + `cas-model2-names.bst`.
- **Appendices**: lettered A–E with A.1-style table/equation numbering.
- **Tables**: editable text, captions, notes below body, no vertical rules.
- **Figures**: separate vector PDFs, cited in text, captions present.
- **Supplementary material**: cited in the manuscript text.
- **CRediT**: provided on the title page.

## Compile

pdfLaTeX -> BibTeX -> pdfLaTeX -> pdfLaTeX (or `latexmk -pdf`).
Verified with `latexmk -pdf`: manuscript 36 pp and supplement 23 pp, with no
fatal compilation errors.

## Before final submission (checklist)

1. Deposit the complete `actionshap-v1` archive (code, configurations,
   scripts, validation reports, manifests, and machine-readable matrices) in
   a data repository and add its DOI to the availability statement.
2. Verify that the deposited archive reproduces every manuscript asset before
   changing the current "upon acceptance" availability wording.
3. Upload `actionshap.tex`/`supplementary.tex` and the separate title page;
   upload figures as individual files if the system requests it.
