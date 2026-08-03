# Raw ActionShap results

Schema-v2 JSON files are generated here by `scripts/run_final_suite.py` and are
ignored by Git because complete per-user attribution records are large. Final
paper assets record each source file's repository-relative path, byte size, and
SHA-256 digest.

Before submission, package all schema-v2 raw JSON and convergence files in a
versioned archival release (for example Zenodo) and add that DOI to the paper.
The tracked schema-v1 `wine_static.json` is legacy clustering material and is
not consumed by the recommendation asset generator.
