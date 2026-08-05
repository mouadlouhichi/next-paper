# ActionShap reproducibility release

This directory accompanies the manuscript and supplies the validation report,
asset manifest, frozen configuration, dependency lockfile, preprocessing/run
scripts, and machine-readable matrices behind the compact tables. The raw
schema-v2 result archive is tracked at the repository path recorded in
`archive_sha256.txt` and is identified by its SHA-256 hash.

The release unit is the exact source commit `5ca120b` named in the parent
`RELEASE_METADATA.md`. The code scripts record independent candidate, user, tie,
model, and attribution seeds. `validation_report.json` must report `PASS` before
any numerical claim is copied into the manuscript.
