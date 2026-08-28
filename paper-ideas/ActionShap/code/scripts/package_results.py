#!/usr/bin/env python3
"""Create a content-addressed raw-result archive for repository release/Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def derived_files(paper_root: Path) -> list[tuple[str, Path]]:
    """The generated half of the deposit, as (arcname, path) pairs.

    ``raw/`` alone is not enough to check a number that appears in a table: the
    manuscript tables are produced by the generators in ``code/scripts`` from the
    release matrices and the review-9 run outputs. Including the generators, their
    tests, the manifest they freeze and the notebook that drives the runs makes the
    deposit the thing the paper was actually built from, which is what issues 16
    and 17 asked for. Missing pieces are skipped rather than fatal, so a partial
    deposit is still buildable.
    """
    code = paper_root / "code"
    groups = {
        "results/review9": sorted((code / "results" / "review9").glob("*.json")),
        "results": [code / "results" / "manifest.json"],
        "tables": sorted((paper_root / "acmart-primary" / "tables")
                         .glob("review9_*.tex")),
        "scripts": [code / "scripts" / name for name in (
            "make_review9_stats.py", "make_review3_stats.py", "make_result_manifest.py",
            "run_review9_experiments.py", "validate_manuscript.py",
            "validate_cross_table.py", "make_review9_notebook.py",
        )],
        "tests": [code / "tests" / "test_review9_publication_integrity.py"],
        "notebooks": sorted((paper_root / "notebooks").glob("*.ipynb")),
        "docs": [paper_root / "docs" / name for name in (
            "REVIEW9_EXPERIMENT_GUIDE.md", "REVIEW9_RESPONSE_PLAN.md")],
    }
    out: list[tuple[str, Path]] = []
    for prefix, paths in groups.items():
        for path in paths:
            if path.exists():
                out.append((f"{prefix}/{path.name}", path))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="results/raw")
    parser.add_argument(
        "--output", default="results/release/actionshap-schema-v2-results.tar.gz"
    )
    parser.add_argument(
        "--no-derived",
        action="store_true",
        help="package only the raw runs; by default the generated half of the "
             "artifact (review-9 run outputs, the tables they produce, the result "
             "manifest, the generators, the tests and the run notebook) is included "
             "too, because a reviewer cannot reproduce a printed number from raw "
             "runs alone",
    )
    parser.add_argument(
        "--allow-no-raw",
        action="store_true",
        help="build an archive of the derived files only (the review-9 addendum)",
    )
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    raw_root = (code_root / args.raw_root).resolve()
    files = []
    for path in sorted(raw_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("schema_version") != 2:
            continue
        is_paper_run = payload.get("status") == "paper_eligible"
        is_convergence = "selected_permutations" in payload and "criterion" in payload
        if is_paper_run or is_convergence:
            files.append(path)
    derived = [] if args.no_derived else derived_files(code_root.parent)
    if not files and not (args.allow_no_raw and derived):
        raise FileNotFoundError(
            "no schema-v2 result files to package (use --allow-no-raw for a "
            "derived-only archive)"
        )
    output = (code_root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum_manifest = {
        "schema_version": 2,
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in files
        ],
        "derived": [
            {"path": arcname, "bytes": path.stat().st_size, "sha256": digest(path)}
            for arcname, path in derived
        ],
    }
    manifest_path = raw_root / "release_checksums.json"
    manifest_path.write_text(json.dumps(checksum_manifest, indent=2))
    try:
        with tarfile.open(output, "w:gz") as archive:
            for path in files:
                archive.add(path, arcname=f"raw/{path.name}")
            for arcname, path in derived:
                archive.add(path, arcname=arcname)
            archive.add(manifest_path, arcname="release_checksums.json")
    finally:
        manifest_path.unlink(missing_ok=True)
    archive_hash = digest(output)
    checksum_path = Path(str(output) + ".sha256")
    checksum_path.write_text(f"{archive_hash}  {output.name}\n")
    print(output)
    print(checksum_path)
    print("sha256", archive_hash)


if __name__ == "__main__":
    main()
