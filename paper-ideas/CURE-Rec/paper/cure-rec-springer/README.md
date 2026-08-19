# CURE-Rec Springer manuscript

This directory is a self-contained Springer Nature (`sn-jnl`) manuscript package
for the empirical CURE-Rec paper.

## Files

- `cure-rec.tex` — the single submission-style manuscript source. It does not use
  `\input{...}` for manuscript sections.
- `cure-rec-bibliography.bib` — references used by the manuscript.
- `sn-jnl.cls` and `bst/sn-mathphys-num.bst` — Springer template assets copied
  from the template supplied in `../sn-article-template 2/`.
- Figure 1 is an editable, compact vertical TikZ workflow embedded directly in
  `cure-rec.tex`.
- `scripts/generate_figures.py` — regenerates Figures 2--6 from checksumed result
  tables using a consistent black/grey Matplotlib style.
- `figures/` — generated vector PDF and high-resolution PNG files for the
  controlled-regime, OAT, LHS attribution/decision, and external-ranking figures.

## Regenerate figures

The committed PDF/PNG figures are generated from the archived CSV result tables.
To regenerate them after changing a figure style:

```bash
python -m pip install matplotlib numpy
python scripts/generate_figures.py
```

## Compile

From this directory, with a full TeX Live or MacTeX installation:

```bash
./build_clean.sh
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

Second-round revision evidence (utility-weight sensitivity, constraint frontier,
divergent-configuration selector screening and held-out comparison) is drawn from
`../../code/results/reviewer_phase_assets/` (`objective_constraint_sweeps/` and
`divergent_selector_holdout/`), each with its own manifest; generation scripts live
in `../../code/scripts_review/`.

CURE-Sim claims are controlled causal/behavioural benchmark claims. MovieLens-1M
claims are explicitly limited to audited chronological ranking robustness; the
manuscript does not treat MovieLens ratings as complete causal policy logs.
