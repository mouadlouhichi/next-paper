#!/usr/bin/env python3
"""Download and prepare every external dataset required by ActionShap.

The downloader is intentionally explicit about source URLs and terms. It never
commits data, never rewrites a valid payload, uses atomic ``.part`` files, and
records content hashes beside each prepared dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
AMAZON_URL = (
    "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/"
    "Digital_Music_5.json.gz"
)
# Verified against the GroupLens file and the pydata-book mirror used by the
# tracked preflight. Pinning the extracted payload avoids trusting a mutable zip.
MOVIELENS_RATINGS_SHA256 = (
    "506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool = False) -> Path:
    """Stream a URL atomically, preserving an existing non-empty file."""
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = Request(url, headers={"User-Agent": "ActionShap-research-artifact/0.2"})
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"download produced an empty file: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_movielens(
    code_root: Path,
    url: str,
    expected_ratings_sha256: str,
    force: bool = False,
) -> dict[str, object]:
    data_root = code_root / "data" / "ml-1m"
    ratings_path = data_root / "ratings.dat"
    if ratings_path.exists() and not force:
        observed = sha256(ratings_path)
        if observed != expected_ratings_sha256:
            raise ValueError(
                f"existing MovieLens ratings hash {observed} != expected "
                f"{expected_ratings_sha256}; use --force only after auditing"
            )
        archive_hash = None
    else:
        with tempfile.TemporaryDirectory(prefix="actionshap-ml1m-") as temporary:
            archive = download(url, Path(temporary) / "ml-1m.zip", force=True)
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
    provenance = {
        "dataset": "MovieLens-1M",
        "source_url": url,
        "archive_sha256": archive_hash,
        "output_file": ratings_path.name,
        "output_sha256": observed,
        "bytes": ratings_path.stat().st_size,
    }
    ratings_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return provenance


def prepare_amazon(
    code_root: Path,
    url: str,
    force: bool = False,
) -> dict[str, object]:
    data_root = code_root / "data" / "amazon-digital-music"
    source = download(url, data_root / "Digital_Music_5.json.gz", force=force)
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
    provenance["source_url"] = url
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
    parser.add_argument("--movielens-url", default=MOVIELENS_URL)
    parser.add_argument("--amazon-url", default=AMAZON_URL)
    parser.add_argument("--movielens-ratings-sha256", default=MOVIELENS_RATINGS_SHA256)
    args = parser.parse_args()
    if not args.accept_dataset_terms:
        raise SystemExit(
            "Refusing to download: review both dataset terms and pass "
            "--accept-dataset-terms"
        )
    code_root = Path(__file__).resolve().parents[1]
    report: dict[str, object] = {}
    if args.dataset in {"all", "movielens"}:
        report["movielens"] = prepare_movielens(
            code_root,
            args.movielens_url,
            args.movielens_ratings_sha256,
            args.force,
        )
    if args.dataset in {"all", "amazon"}:
        report["amazon_digital_music"] = prepare_amazon(
            code_root, args.amazon_url, args.force
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
