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
| `release/` | Machine-readable matrices + provenance (kept for the data-availability statement) |

## Compliance with the IPM guide

- **Double-anonymized review**: the manuscript contains no author names,
  affiliations, emails, acknowledgements, funding, or identifying URLs
  (the release URL is de-anonymized at acceptance). Identifying information
  lives only in `titlepage.tex`.
- **Abstract**: 225 words (limit 250).
- **Keywords**: 7, no "and"/"of" multi-word phrases.
- **References**: author-year via `natbib` + `cas-model2-names.bst`.
- **Appendices**: lettered A–E with A.1-style table/equation numbering.
- **Tables**: editable text, captions, notes below body, no vertical rules.
- **Figures**: separate vector PDFs, cited in text, captions present.
- **Supplementary material**: cited in the manuscript text.
- **CRediT**: provided on the title page.

## Compile

pdfLaTeX -> BibTeX -> pdfLaTeX -> pdfLaTeX (or `latexmk -pdf`).
Verified in sandbox: manuscript 39 pp, supplement 36 pp, zero errors.

## Before final submission (checklist)

1. Deposit the release archive in a data repository (e.g., Zenodo) and add
   the DOI to the Data availability statement (IPM research-data policy,
   Option C: deposit + cite).
2. If desired, restore the de-anonymized release URL in the
   Computational-cost section (currently anonymized for review).
3. Upload `actionshap.tex`/`supplementary.tex` and the separate title page;
   upload figures as individual files if the system requests it.
