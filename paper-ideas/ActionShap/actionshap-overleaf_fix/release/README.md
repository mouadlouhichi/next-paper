# ActionShap reproducibility release

This compact reviewer bundle accompanies the manuscript and supplies the
validation report, asset manifest, frozen configuration, dependency lockfile,
entry-point scripts, and machine-readable matrices behind the compact tables.
The complete executable `actionshap` Python package and raw schema-v2 result
archive remain in the public source repository. The archive path and SHA-256 are
recorded in `archive_sha256.txt`.

The numerical release unit traces to result-generation commit `0673378`, as
recorded in `asset_manifest.json`. User, candidate, and tie seeds are independent
of stochastic repetition seeds. ItemKNN is deterministic, so its repetitions vary
only stochastic explainers and the random control; profile-model repetitions vary
both model initialization and stochastic attribution. `validation_report.json`
must report `PASS` before a numerical claim is copied into the manuscript.

Run `release/scripts/validate_manuscript.py --require-final` and
`release/scripts/validate_review_contract.py` from the manuscript root before
submission. Experiment entry points must be run from the full repository checkout,
where the `actionshap` package is present.
