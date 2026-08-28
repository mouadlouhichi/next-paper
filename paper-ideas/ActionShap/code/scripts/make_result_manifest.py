#!/usr/bin/env python3
"""Freeze a content-addressed manifest of the released result artifacts.

Issue #16 (reproducibility) and Issue #17 (version drift) asked for both
documents to be generated from one versioned result manifest and for each PDF to
state the exact revision and hash. The stamp covers content only, so it survives
being committed and survives being recomputed on another machine; the revision rides
along as a recorded field. This script provides the manifest half of
that; the stamp is a single macro in both documents and a unit test recomputes
it, so a table or matrix edited without updating the stamp fails the suite.

Scope is deliberately the *data* artifacts -- the release matrices and the
generated tables -- and not the .tex sources, so the hash cannot be
self-referential (the sources quote the hash).

Usage:
    python3 scripts/make_result_manifest.py            # write + print
    python3 scripts/make_result_manifest.py --check    # exit 1 on drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../ActionShap/code
PAPER = ROOT.parent                                   # .../ActionShap
MANIFEST = ROOT / "results" / "manifest.json"

SOURCES = [
    PAPER / "actionshap-ipm" / "release" / "matrices",
    PAPER / "acmart-primary" / "tables",
    # The review-9 experiment outputs are inputs to the generated tables, so they
    # are frozen explicitly: a reviewer replaying an ablation must be able to
    # check which per-user records produced the published numbers.
    ROOT / "results" / "review9",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PAPER), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # not a git checkout (e.g. an extracted artifact)
        return "unknown"


def build() -> dict:
    files: dict[str, str] = {}
    for base in SOURCES:
        if not base.is_dir():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            files[str(path.relative_to(PAPER))] = sha256(path)
    payload = {
        "schema": "actionshap-result-manifest-v1",
        "git_revision": git_revision(),
        "file_count": len(files),
        "files": files,
    }
    # The stamp hashes file *contents* only. Folding the checked-out revision in made
    # it impossible for a commit to carry its own stamp -- every commit invalidated the
    # documents that quoted it, so `make check` failed right after a successful
    # regeneration and the check trained people to expect a moving number. The revision
    # is still recorded in the manifest, next to the stamp, as provenance. This also
    # keeps the stamp stable across hosts: it answers "which result content was this
    # typeset from", which is the question the reviewer asked.
    canonical = json.dumps({"files": files}, sort_keys=True)
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    payload["manifest_stamp"] = payload["manifest_sha256"][:12]
    return payload


def tex_stamp(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text()
    marker = "\\newcommand{\\resultmanifeststamp}{"
    if marker not in text:
        return None
    return text.split(marker, 1)[1].split("}", 1)[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify documents, write nothing")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    payload = build()
    errors: list[str] = []

    documents = {
        "acmmanuscript.tex": PAPER / "acmart-primary" / "acmmanuscript.tex",
        "supplementary.tex": PAPER / "acmart-primary" / "supplementary.tex",
    }
    for name, path in documents.items():
        found = tex_stamp(path)
        if found is None:
            errors.append(f"{name}: no \\resultmanifeststamp definition found")
        elif found != payload["manifest_stamp"]:
            errors.append(
                f"{name}: stamp {found} is stale; current manifest is "
                f"{payload['manifest_stamp']} (re-run make_result_manifest.py and update both documents)"
            )

    if args.check:
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(
                f"manifest OK: {payload['file_count']} files, "
                f"stamp {payload['manifest_stamp']}, revision {payload['git_revision'][:12]}"
            )
        return 0

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=1, sort_keys=True))
    if not args.quiet:
        print(f"wrote {MANIFEST}")
        print(
            json.dumps(
                {
                    "manifest_stamp": payload["manifest_stamp"],
                    "git_revision": payload["git_revision"],
                    "file_count": payload["file_count"],
                },
                indent=1,
            )
        )
        for error in errors:
            print(f"NOTE: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
