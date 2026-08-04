# ActionShap paper assets

- `paper.tex` is the canonical recommendation-only manuscript.
- `final/` is generated from schema-v2 experiments and is the only numerical
  source the manuscript may cite; `final/RESULTS_SUMMARY.md` states the validated
  positive claim and its boundaries.
- `legacy_pilot/` preserves invalidated schema-v1 pilot assets for provenance;
  those numbers are not final-paper evidence.

Use `code/ActionShap_All.ipynb` for the complete download-to-release workflow,
or run `code/scripts/run_final_suite.py` after preparing data. The generator
requires two datasets, two history-conditioned models, five common seeds, the
real-data masking gate, independent convergence, and distinct-user inference.
