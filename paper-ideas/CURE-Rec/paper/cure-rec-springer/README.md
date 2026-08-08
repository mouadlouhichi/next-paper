# CURE-Rec Springer manuscript

This directory is a self-contained Springer Nature (`sn-jnl`) manuscript package
for the empirical CURE-Rec paper.

## Files

- `cure-rec.tex` — the single submission-style manuscript source. It does not use
  `\input{...}` for manuscript sections.
- `cure-rec-bibliography.bib` — references used by the manuscript.
- `sn-jnl.cls` and `bst/sn-mathphys-num.bst` — Springer template assets copied
  from the template supplied in `../sn-article-template 2/`.
- `figures/` — only the generated LHS calibration figures used directly by the
  manuscript. The workflow, controlled-regime recovery, OAT sensitivity, and
  external ranking figures are native black-and-white TikZ/PGFPlots figures and
  therefore have no missing image dependency.

## Compile

From this directory, with a full TeX Live or MacTeX installation:

```bash
pdflatex cure-rec.tex
bibtex cure-rec
pdflatex cure-rec.tex
pdflatex cure-rec.tex
```

The document uses the Springer Nature math/physical-sciences numbered reference
style. The content follows the structure of the supplied Q1 example paper:
Introduction; Literature review; Background and preliminaries; Methodology;
Experimental results; Discussion and broader implications; Conclusion; and
appendices.

## Evidence discipline

All numerical claims in `cure-rec.tex` are drawn from
`../../results/reproducibility_snapshot_latest/`:

- full behavioural 20-seed sweep;
- controlled oracle-regime suite;
- completed OAT and 24-point LHS calibration studies;
- final BPR and SASRec audited five-seed replications.

CURE-Sim claims are controlled causal/behavioural benchmark claims. MovieLens-1M
claims are explicitly limited to audited chronological ranking robustness; the
manuscript does not treat MovieLens ratings as complete causal policy logs.
