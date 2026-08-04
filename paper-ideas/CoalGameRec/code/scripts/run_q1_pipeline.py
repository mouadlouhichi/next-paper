#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coalgamerec.pipeline import run_pipeline


def main() -> None:
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description="Run CoalGameRec journal-style empirical pipeline.")
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/q1_mac_ml1m.yaml")
    args = ap.parse_args()
    out = run_pipeline(args.config)
    print(f"Wrote run artifacts to: {out}")


if __name__ == "__main__":
    main()
