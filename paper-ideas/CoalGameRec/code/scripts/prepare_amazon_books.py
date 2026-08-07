#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path
import urllib.error
import urllib.request

DEFAULT_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Books_5.json.gz"


def is_gzip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 2:
        return False
    with path.open("rb") as f:
        return f.read(2) == b"\x1f\x8b"


def remote_size(url: str) -> int | None:
    """Return remote Content-Length if available."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def _preview(path: Path, n: int = 200) -> str:
    with path.open("rb") as f:
        return f.read(n).decode("utf-8", errors="replace")


def _open_response(url: str, start: int = 0, timeout: int = 120):
    headers = {}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def download(url: str, dest: Path, force: bool = False, retries: int = 10, sleep: float = 10.0) -> None:
    """Download with resume support and gzip sanity checks.

    Large UCSD downloads can drop midway. This function resumes a partial gzip
    file using HTTP Range requests when the server supports it. If the server
    ignores the Range header, it safely restarts from byte 0.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)

    if dest.exists() and dest.stat().st_size > 0 and not force:
        if not is_gzip(dest):
            raise RuntimeError(
                f"Existing file is not gzip: {dest}\n"
                f"Preview: {_preview(dest, 120)!r}\n"
                "It is likely an HTML error page from a failed download. Delete it or rerun with --force."
            )
        if expected is not None and dest.stat().st_size >= expected:
            print(f"Already exists and matches/exceeds remote size: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
            return
        if expected is None:
            print(f"Existing gzip found but remote size unknown: {dest} ({dest.stat().st_size / 1e9:.2f} GB). Will validate only; use --force to redownload.")
            return
        print(f"Resuming partial gzip: {dest} ({dest.stat().st_size / 1e9:.2f}/{expected / 1e9:.2f} GB)")
    elif force and dest.exists():
        print(f"--force: removing existing file {dest}")
        dest.unlink()

    print(f"Downloading:\n  {url}\n-> {dest}")
    print("This file is large. The downloader will retry/resume automatically. You can also use curl -L -C -.")

    for attempt in range(1, retries + 1):
        start = dest.stat().st_size if dest.exists() else 0
        mode = "ab" if start > 0 else "wb"
        try:
            with _open_response(url, start=start) as r:
                status = getattr(r, "status", None)
                # If server ignores Range and returns 200, restart rather than append duplicate bytes.
                if start > 0 and status == 200:
                    print("\nServer ignored Range request; restarting download from byte 0.")
                    start = 0
                    mode = "wb"
                total = expected
                seen = start
                with dest.open(mode) as f:
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
            print("\nDownload attempt completed.")
            if expected is None or dest.stat().st_size >= expected:
                break
            print(f"Remote size not reached yet ({dest.stat().st_size}/{expected}); retrying...")
        except (BrokenPipeError, ConnectionError, TimeoutError, urllib.error.URLError, OSError) as e:
            print(f"\nDownload interrupted on attempt {attempt}/{retries}: {type(e).__name__}: {e}")
            if attempt >= retries:
                raise
            print(f"Sleeping {sleep:.1f}s then resuming from {dest.stat().st_size if dest.exists() else 0} bytes...")
            time.sleep(sleep)

    if not is_gzip(dest):
        raise RuntimeError(
            f"Downloaded file is not gzip: {dest}\n"
            f"Preview: {_preview(dest)!r}\n"
            "The server likely returned an HTML error page. Delete it and retry, or download manually from the UCSD page."
        )
    if expected is not None and dest.stat().st_size < expected:
        raise RuntimeError(
            f"Downloaded file is incomplete: {dest.stat().st_size} bytes, expected {expected}.\n"
            "Rerun the same command to resume, or use curl -L -C -."
        )
    print(f"Done: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")


def validate(path: Path, n: int = 3) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not is_gzip(path):
        raise RuntimeError(f"Not gzip: {path}\nPreview: {_preview(path, 120)!r}")
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
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=10.0)
    args = ap.parse_args()
    dest = Path(args.dest).expanduser()
    if not args.validate_only:
        download(args.url, dest, force=args.force, retries=args.retries, sleep=args.sleep)
    validate(dest)
    print(f"\nUse with:\n  export AMAZON_BOOKS_5={dest.resolve()}\n  python scripts/run_q1_pipeline.py --config configs/q1_lightgcn_amazon_template.yaml")


if __name__ == "__main__":
    main()
