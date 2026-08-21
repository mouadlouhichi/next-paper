# CURE-Rec — ESWA submission package (Elsevier `elsarticle`)

Ready-to-submit package for **Expert Systems with Applications**.

## Contents

| File | Purpose |
|---|---|
| `cure-rec-eswa.tex` | Manuscript converted to the Elsevier `elsarticle` class (`[review]` option: single column, double-spaced, line numbers) |
| `elsarticle.cls` | Elsevier article class (v1.20b) |
| `elsarticle-num.bst` | Numbered (Vancouver-style) bibliography style used by ESWA |
| `cure-rec-bibliography.bib` | Bibliography database |
| `figures/` | Figures 2–5 (PDF + PNG) |
| `highlights.txt` | 3–5 highlights, each ≤ 85 characters (upload as separate Highlights item) |
| `cover_letter.tex` | Cover letter (compile separately or paste into Editorial Manager) |
| `build_clean.sh` | One-command build with strict log checks |

## Build

Requires a full TeX Live / MacTeX installation:

```bash
./build_clean.sh        # produces cure-rec-eswa.pdf
```

Manual equivalent: `pdflatex cure-rec-eswa; bibtex cure-rec-eswa; pdflatex cure-rec-eswa; pdflatex cure-rec-eswa`.

## Editorial Manager checklist

- [ ] Article type: Research Article
- [ ] Manuscript source: `cure-rec-eswa.tex` (+ `elsarticle.cls`, `elsarticle-num.bst`, `.bib`, `figures/`) as a LaTeX archive
- [ ] Highlights: contents of `highlights.txt`
- [ ] Cover letter
- [ ] CRediT authorship contribution statement — included in the manuscript
- [ ] Declaration of competing interest — included in the manuscript
- [ ] Data availability statement — included in the manuscript
- [ ] ORCID iDs for all authors (add at submission)

## Notes

- Reference style is numbered in order of appearance (`elsarticle-num`), matching ESWA.
- The manuscript enforces strict evidence boundaries: CURE-Sim results are
  simulator-conditional; MovieLens-1M results are chronological-ranking evidence
  with user-level paired inference; no real-world causal policy claims are made.
- Companion evidence lives in the repository under
  `code/results/reviewer_phase_assets/` (checksumed) and the closure notebook
  `code/notebooks/13_reviewer_closure_runs.ipynb`.
