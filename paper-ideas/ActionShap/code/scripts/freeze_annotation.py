#!/usr/bin/env python
"""Freeze a modifiability annotation: stamp it with a payload hash and a time.

Run once, after all annotators are done and before any result is inspected.
The loader recomputes the hash on every subsequent run, so an edit after
freezing becomes an error rather than a silent revision.

    python scripts/freeze_annotation.py annotations/wine.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionshap.modifiability import RUBRIC, compute_payload_hash  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument(
        "--force", action="store_true",
        help="re-freeze an already-frozen file (disclose this in the paper)",
    )
    args = ap.parse_args()

    spec = yaml.safe_load(args.path.read_text())
    factors = spec.get("factors") or {}
    if not factors:
        return _fail(f"{args.path}: no factors to freeze")

    if spec.get("provisional"):
        return _fail(
            f"{args.path} is marked provisional. Freezing it would misrepresent "
            "an assistant-written placeholder as an elicitation."
        )

    blank = [
        f for f, votes in factors.items()
        if any(v is None for v in votes.values())
    ]
    if blank:
        return _fail(
            f"{args.path}: {len(blank)} factor(s) still unannotated: "
            f"{sorted(blank)}"
        )

    bad = [
        (f, a, v) for f, votes in factors.items()
        for a, v in votes.items() if float(v) not in RUBRIC
    ]
    if bad:
        for f, a, v in bad:
            print(f"  {a} gave {f!r} the value {v}", file=sys.stderr)
        return _fail(f"values outside the rubric {sorted(RUBRIC)}")

    existing = (spec.get("frozen") or {}).get("payload_sha256")
    if existing and existing != "PROVISIONAL" and not args.force:
        return _fail(
            f"{args.path} is already frozen ({existing[:12]}...). Use --force "
            "to re-freeze, and disclose the revision in the paper."
        )

    spec["frozen"] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "payload_sha256": compute_payload_hash(factors),
        "git_commit": _git_head(args.path.parent),
    }
    args.path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

    n_annotators = len({a for votes in factors.values() for a in votes})
    print(f"Froze {args.path}")
    print(f"  factors    {len(factors)}")
    print(f"  annotators {n_annotators}")
    print(f"  sha256     {spec['frozen']['payload_sha256'][:16]}...")
    print("\nCommit it now, so the freeze has an external timestamp:")
    print(f"  git add {args.path} && git commit -m 'Freeze {spec['dataset']} annotation'")
    return 0


def _git_head(cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNRECORDED"


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
