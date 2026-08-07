#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import sys
import urllib.request

DEFAULT_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Books_5.json.gz"


def is_gzip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 2:
        return False
    with path.open("rb") as f:
        return f.read(2) == b"\x1f\x8b"


def download(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        if is_gzip(dest):
            print(f"Already exists and looks like gzip: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
            return
        with dest.open("rb") as f:
            preview = f.read(120).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Existing file is not gzip: {dest}\n"
            f"Preview: {preview!r}\n"
            "It is likely an HTML error page from a failed download. Delete it or rerun with --force."
        )
    print(f"Downloading:\n  {url}\n-> {dest}")
    print("This file is large. If the download is interrupted, rerun with curl -C - as shown in README.")
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        total = r.headers.get("Content-Length")
        total = int(total) if total else None
        seen = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            seen += len(chunk)
            if total:
                print(f"\r{seen / total:6.2%} ({seen / 1e9:.2f}/{total / 1e9:.2f} GB)", end="")
            else:
                print(f"\r{seen / 1e9:.2f} GB", end="")
    print("\nDone.")
    if not is_gzip(dest):
        with dest.open("rb") as f:
            preview = f.read(200).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Downloaded file is not gzip: {dest}\n"
            f"Preview: {preview!r}\n"
            "The server likely returned an HTML error page. Try the cseweb UCSD page manually "
            "and set AMAZON_BOOKS_5 to the valid downloaded file."
        )


def validate(path: Path, n: int = 3) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"Validating gzip and reading {n} JSONL records: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i in range(n):
            line = f.readline()
            if not line:
                raise RuntimeError(f"File ended before {n} lines")
            print(line[:160].rstrip() + ("..." if len(line) > 160 else ""))
    print("Validation OK.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/validate Amazon Reviews 2018 Books_5.json.gz for CoalGameRec.")
    ap.add_argument("--dest", default="data/raw/Books_5.json.gz")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    dest = Path(args.dest).expanduser()
    if not args.validate_only:
        download(args.url, dest, force=args.force)
    validate(dest)
    print(f"\nUse with:\n  export AMAZON_BOOKS_5={dest.resolve()}\n  python scripts/run_q1_pipeline.py --config configs/q1_lightgcn_amazon_template.yaml")


if __name__ == "__main__":
    main()
