#!/usr/bin/env python3
"""Download and prepare every external dataset required by ActionShap.

Downloads are atomic, retried across declared mirrors, and content-addressed.
Users behind restrictive institutional networks can provide local source files
without changing the notebook or scientific configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

MOVIELENS_URLS = (
    "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
)
AMAZON_URLS = (
    (
        "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/"
        "Digital_Music_5.json.gz"
    ),
    (
        "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/"
        "categoryFilesSmall/Digital_Music_5.json.gz"
    ),
    (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
        "categoryFilesSmall/Digital_Music_5.json.gz"
    ),
    "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Digital_Music_5.json.gz",
)
# Verified against GroupLens and the mirror used by the tracked preflight.
MOVIELENS_RATINGS_SHA256 = (
    "506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sources(value: str | Sequence[str]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def _atomic_copy(source: Path, destination: Path) -> Path:
    source = source.resolve()
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(f"local source is missing or empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination.resolve():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _download_once(url: str, destination: Path, timeout: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = Request(url, headers={"User-Agent": "ActionShap-research-artifact/0.2"})
    try:
        with (
            urlopen(request, timeout=timeout) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"download produced an empty file: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _download_with_curl(url: str, destination: Path, timeout: int) -> Path:
    """Fallback for macOS/network stacks where urllib cannot reach UCSD."""
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is not installed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    command = [
        curl,
        "-4",
        "--http1.1",
        "-L",
        "--fail",
        "--retry",
        "1",
        "--retry-all-errors",
        "--connect-timeout",
        str(timeout),
        "--output",
        str(temporary),
        url,
    ]
    try:
        subprocess.run(command, check=True)
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError(f"curl produced an empty file: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def download_from_sources(
    urls: str | Sequence[str],
    destination: Path,
    *,
    force: bool = False,
    attempts: int = 2,
    timeout: int = 45,
) -> tuple[Path, str | None]:
    """Try each mirror with bounded retries, returning the successful URL."""
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination, None
    failures: list[str] = []
    for url in _sources(urls):
        for attempt in range(1, attempts + 1):
            try:
                return _download_once(url, destination, timeout), url
            except (OSError, TimeoutError, URLError, RuntimeError) as error:
                failures.append(f"{url} [attempt {attempt}/{attempts}]: {error}")
                if attempt < attempts:
                    time.sleep(min(2**attempt, 5))
    # urllib on some macOS/institutional networks times out before TLS
    # negotiation. Force IPv4 and HTTP/1.1 through system curl as a final path.
    for url in _sources(urls):
        try:
            return _download_with_curl(url, destination, timeout), url
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            failures.append(f"{url} [curl IPv4 fallback]: {error}")
    destination.unlink(missing_ok=True)
    detail = "\n  - ".join(failures)
    raise RuntimeError(
        f"all download sources failed for {destination.name}:\n  - {detail}\n"
        f"Download the source manually to {destination} and rerun without --force."
    )


def prepare_movielens(
    code_root: Path,
    urls: str | Sequence[str],
    expected_ratings_sha256: str,
    force: bool = False,
    local_archive: Path | None = None,
    attempts: int = 2,
    timeout: int = 45,
) -> dict[str, object]:
    data_root = code_root / "data" / "ml-1m"
    ratings_path = data_root / "ratings.dat"
    source_label: str | None = None
    archive_hash: str | None = None
    if ratings_path.exists() and not force:
        observed = sha256(ratings_path)
        if observed != expected_ratings_sha256:
            raise ValueError(
                f"existing MovieLens ratings hash {observed} != expected "
                f"{expected_ratings_sha256}; use --force only after auditing"
            )
    else:
        with tempfile.TemporaryDirectory(prefix="actionshap-ml1m-") as temporary:
            archive = Path(temporary) / "ml-1m.zip"
            if local_archive is not None:
                _atomic_copy(local_archive, archive)
                source_label = f"file://{local_archive.resolve()}"
            else:
                archive, source_label = download_from_sources(
                    urls,
                    archive,
                    force=True,
                    attempts=attempts,
                    timeout=timeout,
                )
            archive_hash = sha256(archive)
            with zipfile.ZipFile(archive) as bundle:
                member = "ml-1m/ratings.dat"
                if member not in bundle.namelist():
                    raise ValueError(f"MovieLens archive does not contain {member}")
                data_root.mkdir(parents=True, exist_ok=True)
                temporary_ratings = ratings_path.with_suffix(".dat.part")
                with (
                    bundle.open(member) as source,
                    temporary_ratings.open("wb") as output,
                ):
                    shutil.copyfileobj(source, output)
                temporary_ratings.replace(ratings_path)
        observed = sha256(ratings_path)
        if observed != expected_ratings_sha256:
            ratings_path.unlink(missing_ok=True)
            raise ValueError(
                f"downloaded MovieLens ratings hash {observed} != expected "
                f"{expected_ratings_sha256}"
            )
    sidecar = ratings_path.with_suffix(".provenance.json")
    previous = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    provenance = {
        "dataset": "MovieLens-1M",
        "source_url": source_label or previous.get("source_url") or _sources(urls)[0],
        "archive_sha256": archive_hash or previous.get("archive_sha256"),
        "output_file": ratings_path.name,
        "output_sha256": observed,
        "bytes": ratings_path.stat().st_size,
    }
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def prepare_amazon(
    code_root: Path,
    urls: str | Sequence[str],
    force: bool = False,
    local_source: Path | None = None,
    attempts: int = 2,
    timeout: int = 45,
) -> dict[str, object]:
    data_root = code_root / "data" / "amazon-digital-music"
    source = data_root / "Digital_Music_5.json.gz"
    if local_source is not None:
        if force or not source.exists():
            _atomic_copy(local_source, source)
        source_label = f"file://{local_source.resolve()}"
    else:
        source, successful_url = download_from_sources(
            urls,
            source,
            force=force,
            attempts=attempts,
            timeout=timeout,
        )
        source_label = successful_url
    output = data_root / "interactions.csv"
    if force or not output.exists():
        subprocess.run(
            [
                sys.executable,
                str(code_root / "scripts" / "prepare_amazon_digital_music.py"),
                "--input",
                str(source),
                "--output",
                str(output),
            ],
            cwd=code_root,
            check=True,
        )
    provenance_path = output.with_suffix(".provenance.json")
    if not provenance_path.exists():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("source_sha256") != sha256(source):
        raise ValueError("Amazon source hash differs from its preparation sidecar")
    if provenance.get("output_sha256") != sha256(output):
        raise ValueError("Amazon output hash differs from its preparation sidecar")
    provenance["source_url"] = (
        source_label or provenance.get("source_url") or _sources(urls)[0]
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("all", "movielens", "amazon"), default="all"
    )
    parser.add_argument(
        "--accept-dataset-terms",
        action="store_true",
        help="confirm that you reviewed the source dataset terms/citation requirements",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--movielens-url", action="append", default=None)
    parser.add_argument("--amazon-url", action="append", default=None)
    parser.add_argument("--movielens-local-archive", type=Path)
    parser.add_argument("--amazon-local-source", type=Path)
    parser.add_argument("--movielens-ratings-sha256", default=MOVIELENS_RATINGS_SHA256)
    args = parser.parse_args()
    if not args.accept_dataset_terms:
        raise SystemExit(
            "Refusing to download: review both dataset terms and pass "
            "--accept-dataset-terms"
        )
    if args.attempts < 1 or args.timeout < 1:
        raise SystemExit("--attempts and --timeout must be positive")
    code_root = Path(__file__).resolve().parents[1]
    report: dict[str, object] = {}
    if args.dataset in {"all", "movielens"}:
        report["movielens"] = prepare_movielens(
            code_root,
            args.movielens_url or MOVIELENS_URLS,
            args.movielens_ratings_sha256,
            args.force,
            args.movielens_local_archive,
            args.attempts,
            args.timeout,
        )
    if args.dataset in {"all", "amazon"}:
        report["amazon_digital_music"] = prepare_amazon(
            code_root,
            args.amazon_url or AMAZON_URLS,
            args.force,
            args.amazon_local_source,
            args.attempts,
            args.timeout,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
