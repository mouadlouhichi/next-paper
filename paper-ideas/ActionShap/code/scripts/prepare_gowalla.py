#!/usr/bin/env python3
"""Prepare the Gowalla location check-in dataset (third-domain audit, review 6).

Downloads the LightGCN-format Gowalla split (user adjacency lists) and
converts it to the ActionShap temporal CSV format (user, item, timestamp,
rating). Gowalla has no ratings or absolute timestamps; per-user interaction
order is preserved as given in the source files and assigned consecutive
synthetic timestamps, so the standard ActionShap temporal protocol applies:
the last event is the test target and the penultimate event is the
validation event. All retained events have rating 1.

Usage:
    python scripts/prepare_gowalla.py --out data/gowalla/interactions.csv
"""
from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

TRAIN_URL = "https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla/train.txt"
TEST_URL = "https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla/test.txt"


def read_adjacency(path: Path) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {}
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            u = int(parts[0])
            adj[u] = [int(x) for x in parts[1:]]
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gowalla/interactions.csv")
    ap.add_argument("--cache-dir", default="data/gowalla/raw")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in [("train.txt", TRAIN_URL), ("test.txt", TEST_URL)]:
        dst = cache / name
        if not dst.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dst)

    train = read_adjacency(cache / "train.txt")
    test = read_adjacency(cache / "test.txt")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_users = 0
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["user", "item", "timestamp", "rating"])
        for u, heldout in test.items():
            if not heldout:
                continue
            items = list(train.get(u, [])) + list(heldout)
            if len(items) < 4:
                continue
            for t, item in enumerate(items):
                w.writerow([u, item, t + 1, 1.0])
                n_rows += 1
            n_users += 1
    print(f"wrote {out}: {n_rows} interactions, {n_users} users with test items")
    print("note: synthetic per-user timestamps preserve source interaction order;")
    print("      the last event is the test target, the penultimate is validation.")


if __name__ == "__main__":
    main()
