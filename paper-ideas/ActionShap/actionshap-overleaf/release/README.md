# ActionShap reproducibility release

This directory accompanies the manuscript and supplies the validation report,
asset manifest, frozen configuration, dependency lockfile, preprocessing/run
scripts, and machine-readable matrices behind the compact tables. The raw
schema-v2 result archive is tracked at the repository path recorded in
`archive_sha256.txt` and is identified by its SHA-256 hash.

The release unit is the exact source commit `0673378` named in the parent
`RELEASE_METADATA.md`. The code scripts record independent candidate, user, tie,
model, and attribution seeds. `validation_report.json` must report `PASS` before
any numerical claim is copied into the manuscript.

## Provenance model (three distinct identifiers)

- `numeric_source_commit = 0673378` — the commit at which the schema-v2
  numerical results were generated and validated; the source of every value in
  the manuscript tables and of the result archive
  (SHA-256 `ac4c7fb1993458b6b41054974ebff215710e7a8b5894c7aa6af828e94b2a5b0f`).
- `manifest_generation_commit = d4c55b2…` — the commit at which
  `asset_manifest.json` was (re)generated from the raw archive. It describes
  the archive, not the manuscript.
- `packaging_tag = actionshap-v1` — a moving release tag on the release branch
  that packages the manuscript sources, this release directory, and the
  manifest. The exact packaging commit is identified via the tag history.

These identifiers refer to different artifacts by design and are not expected
to be equal. The Overleaf ZIP SHA-256 (`43228a19…2001df`) identifies the
presentation package (manuscript sources) and is a different artifact from the
result-archive SHA above.

## AIA permutation p-value fields

- `aia_permutation_null.csv` → `permutation_p`: the aggregate one-sided
  upper-tail null p (manuscript Eq. for p_null): seed-averaged observed AIA
  versus seed-averaged matched null draws per dataset–method. These are
  unadjusted descriptive calibration controls (Table 5).
- `method_metrics.csv` → `aia_permutation_p`: the mean over distinct users of
  per-user within-user null p-values; a user-level descriptive summary, not the
  aggregate test. The two fields differ by construction.
