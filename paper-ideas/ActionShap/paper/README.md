# ActionShap paper assets

- `paper.tex` is the canonical recommendation-only manuscript.
- `final/` is generated from schema-v2 experiments and is the only numerical
  source the manuscript may cite.
- `legacy_pilot/` preserves invalidated schema-v1 pilot assets for provenance;
  those numbers are not final-paper evidence.

Run `code/scripts/run_final_suite.py` to create final assets. The generator
requires two datasets, two history-conditioned models, five common seeds, the
real-data masking gate, independent convergence, and distinct-user inference.
