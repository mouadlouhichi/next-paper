"""Multi-seed CURE-Sim execution with paired common-random-number summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from cure_rec.config import Settings
from cure_rec.pipeline import run_experiment
from cure_rec.planner import decision_to_dict


@dataclass(frozen=True)
class SeedSweepResult:
    run_dir: Path
    decisions: pd.DataFrame
    attributions: pd.DataFrame


def run_seed_sweep(settings: Settings, seeds: Iterable[int]) -> SeedSweepResult:
    """Run paired environment seeds and aggregate decision/attribution stability.

    Within each seed, CURE-Sim resets every coalition to the same scenario-specific
    random state. This common-random-number design reduces variance for paired
    coalition differences. Seeds are independent across repetitions.
    """
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("At least one seed is required")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep_root = Path(settings.run.output_root) / f"seed-sweep-{stamp}"
    decision_rows: list[dict] = []
    attribution_rows: list[dict] = []

    for seed in seed_list:
        seeded = settings.model_copy(deep=True)
        seeded.run.seed = int(seed)
        seeded.run.name = f"{settings.run.name}-seed-{seed}"
        seeded.run.output_root = sweep_root / "runs"
        logger, game, decision = run_experiment(seeded)
        decision_rows.append({
            "seed": seed,
            "cure_run_dir": str(logger.run_dir),
            **decision_to_dict(decision),
        })
        for row in game.regions.to_dict(orient="records"):
            attribution_rows.append({"seed": seed, **row})

    sweep_root.mkdir(parents=True, exist_ok=True)
    decisions = pd.DataFrame(decision_rows)
    attributions = pd.DataFrame(attribution_rows)
    decisions.to_csv(sweep_root / "seed_sweep_decisions.csv", index=False)
    attributions.to_csv(sweep_root / "seed_sweep_attributions.csv", index=False)
    summary = attributions.groupby("intervention", as_index=False).agg(
        phi_mean_mean=("phi_mean", "mean"),
        phi_mean_std=("phi_mean", "std"),
        phi_lower_mean=("phi_lower", "mean"),
        phi_upper_mean=("phi_upper", "mean"),
        positive_sign_rate=("phi_lower", lambda x: float((x > 0).mean())),
    )
    summary.to_csv(sweep_root / "seed_sweep_attribution_summary.csv", index=False)
    return SeedSweepResult(sweep_root, decisions, attributions)
