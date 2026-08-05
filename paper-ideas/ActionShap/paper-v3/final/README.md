# Final ActionShap assets

Validation status: **PASS** for the tracked schema-v2 release. These assets use
schema-v2 runs and distinct-user hierarchical inference; legacy pilot assets are
not consumed.

## Publication versus audit views

The `.tex` files are compact publication views designed for a readable PDF.
They report the principal ItemKNN conditions and budget-dependent outcomes
without exporting the raw data frame. Complete condition-by-condition values,
confidence intervals, valid-user counts, per-user metrics, paired comparisons,
and provenance remain available in the CSV/JSON assets:

- `tables/aia_components.csv`
- `tables/intervention_outcomes.csv`
- `tables/actionability_gap_robustness.csv`
- `tables/actionability_gap_advantage.csv`
- `tables/method_metrics.csv`
- `tables/paired_tests.csv`
- `data/user_seed_metrics.csv.gz`
- `manifests/asset_manifest.json`

Budget-one and budget-three conditions are decision-only sensitivities. Their
publication table contains joint effect, success, abstention, and regret only;
no singleton AIA or Actionability Gap is reported for those rows. The asset
builder enforces this rule for both publication tables and exported summary
CSVs.

See `RESULTS_SUMMARY.md` for the validated claim boundary and
`../REPRODUCIBILITY.md` for the public source and archive record.
