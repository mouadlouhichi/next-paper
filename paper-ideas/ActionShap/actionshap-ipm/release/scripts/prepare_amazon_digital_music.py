#!/usr/bin/env python3
"""Build the timestamped Amazon Digital Music secondary benchmark.

Input is the unmodified `Digital_Music_5.json.gz` review file from the Amazon
Review Data (2018) release. The script retains positive reviews, resolves any
repeated user-item review deterministically, reapplies an iterative k-core after
thresholding, and writes the generic ActionShap CSV schema plus provenance.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iterative_core(frame: pd.DataFrame, minimum: int) -> pd.DataFrame:
    current = frame
    while True:
        user_counts = current["user"].value_counts()
        item_counts = current["item"].value_counts()
        retained = current.loc[
            current["user"].isin(user_counts[user_counts >= minimum].index)
            & current["item"].isin(item_counts[item_counts >= minimum].index)
        ]
        if len(retained) == len(current):
            return retained.copy()
        if retained.empty:
            raise ValueError("k-core filtering removed every interaction")
        current = retained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="unmodified Digital_Music_5.json.gz"
    )
    parser.add_argument(
        "--output", default="data/amazon-digital-music/interactions.csv"
    )
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--core", type=int, default=5)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    records: list[dict[str, object]] = []
    with open_text(source) as stream:
        for row_index, line in enumerate(stream):
            payload = json.loads(line)
            try:
                user = str(payload["reviewerID"])
                item = str(payload["asin"])
                rating = float(payload["overall"])
                timestamp = int(payload["unixReviewTime"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid review at source row {row_index}") from error
            if rating >= args.rating_threshold:
                records.append(
                    {
                        "user": user,
                        "item": item,
                        "timestamp": timestamp,
                        "rating": rating,
                        "original_record_index": row_index,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("no positive reviews remained")
    frame = frame.sort_values(
        ["user", "item", "timestamp", "original_record_index"], kind="mergesort"
    )
    frame = frame.drop_duplicates(["user", "item"], keep="last")
    frame = iterative_core(frame, args.core)
    frame = frame.sort_values(
        ["user", "timestamp", "original_record_index"], kind="mergesort"
    ).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[["user", "item", "timestamp", "rating"]].to_csv(output, index=False)
    metadata = {
        "source_file": source.name,
        "source_sha256": sha256(source),
        "output_file": output.name,
        "output_sha256": sha256(output),
        "rating_threshold": args.rating_threshold,
        "iterative_core": args.core,
        "users": int(frame["user"].nunique()),
        "items": int(frame["item"].nunique()),
        "interactions": len(frame),
        "density": float(
            len(frame) / (frame["user"].nunique() * frame["item"].nunique())
        ),
        "timestamp_min": int(np.min(frame["timestamp"])),
        "timestamp_max": int(np.max(frame["timestamp"])),
        "deduplication": "keep latest by (timestamp, original_record_index)",
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
