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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="results/raw")
    parser.add_argument(
        "--output", default="results/release/actionshap-schema-v2-results.tar.gz"
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
    if not files:
        raise FileNotFoundError("no schema-v2 result files to package")
    output = (code_root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum_manifest = {
        "schema_version": 2,
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in files
        ],
    }
    manifest_path = raw_root / "release_checksums.json"
    manifest_path.write_text(json.dumps(checksum_manifest, indent=2))
    try:
        with tarfile.open(output, "w:gz") as archive:
            for path in files:
                archive.add(path, arcname=f"raw/{path.name}")
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
