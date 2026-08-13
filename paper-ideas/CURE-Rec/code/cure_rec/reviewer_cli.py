"""Reproducible reviewer-revision orchestration helpers.

This module keeps expensive revision experiments explicit and writes a manifest
for every generated table. It intentionally does not label finite CURE-Sim
scenarios as real-data causal evidence.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cure_rec.config import load_settings
from cure_rec.revision import run_selector_holdout_study
from cure_rec.review_assets import aggregate_selector_runs


def run_holdout(config: str | Path, output_root: str | Path, selection_seeds: tuple[int, ...], evaluation_seeds: tuple[int, ...]) -> Path:
    settings = load_settings(config)
    settings.run.output_root = Path(output_root)
    return run_selector_holdout_study(settings, selection_seeds=selection_seeds, evaluation_seeds=evaluation_seeds)


def aggregate(runs: list[str | Path], output_dir: str | Path) -> Path:
    table = aggregate_selector_runs(runs, output_dir)
    out = Path(output_dir)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "input_runs": [str(Path(r)) for r in runs],
        "output": str(out / "reviewer_table_selector_holdout.csv"),
        "rows": int(len(table)),
        "claim_scope": "held-out CURE-Sim seed evaluation; not external causal inference",
    }
    (out / "reviewer_phase_a_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
