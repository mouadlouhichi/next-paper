"""Checksum the small summary artifacts of the second-round revision evidence.

The heavy per-coalition trajectory artifacts are intentionally excluded: the
summary CSVs, JSON manifests, and run logs are the citable evidence, and each run
directory retains its full configuration manifest for exact re-execution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "reviewer_phase_assets"
INCLUDE_SUFFIXES = {".csv", ".json"}
EXCLUDE_PARTS = {"coalitions"}


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in INCLUDE_SUFFIXES:
            continue
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        yield path


def main() -> None:
    lines = []
    for path in iter_files(ASSETS):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ASSETS)}")
    out = ASSETS / "SHA256SUMS.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} entries)")


if __name__ == "__main__":
    main()
