"""Focused driver: held-out selector comparison on the first divergent configuration.

Runs the full held-out protocol (selection seeds 42-46, disjoint evaluation seeds
200-219) for the predeclared screening points, in declared order, and stops after
the requested number of divergent configurations have completed. Screening games
that already exist on disk are reused.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase_e_divergence as pe  # noqa: E402
from cure_rec.revision import SelectorChoice  # noqa: E402

MAX_HOLDOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 1
EVAL_SEEDS = tuple(range(int(sys.argv[2]), int(sys.argv[3]))) if len(sys.argv) > 3 else pe.EVALUATION_SEEDS
pe.EVALUATION_SEEDS = EVAL_SEEDS


def screening_choices(point_id: str) -> list[SelectorChoice] | None:
    """Reuse a finished screening game if present; return selector choices."""
    matches = sorted((pe.ASSETS / "screening" / point_id).glob("*/tables/coalition_values.csv"))
    if not matches:
        return None
    run_dir = matches[0].parents[1]
    if not (run_dir / "run.log").read_text().strip().splitlines():
        return None
    # Reconstruct the exact game tables and derive selector choices through the
    # same code path used for live games.
    from cure_rec.calibration import _settings_for_point
    from cure_rec.revision import selector_choices

    points = pe.load_screening_points()
    settings = _settings_for_point(pe.load_settings(pe.BASE_CONFIG), points[point_id], pe.ASSETS / "screening")
    settings.run.seed = 42
    settings.run.name = f"{point_id}-seed-42"

    import numpy as np
    from cure_rec.game import GameResult, ScenarioGame, CoalitionValue, ALL_MASKS

    frame = pd.read_csv(matches[0])
    robust_attr = pd.read_csv(run_dir / "tables" / "robust_game_attribution.csv")
    scenario_games: dict[str, ScenarioGame] = {}
    # Preserve the declared scenario order: the nominal selector reads the first
    # scenario game, which must be the nominal scenario exactly as in live runs.
    for scenario_config in settings.scenarios:
        part = frame[frame["scenario"] == scenario_config.name]
        if part.empty:
            raise ValueError(f"Scenario {scenario_config.name} missing from archived screening table")
        values = {}
        for row in part.itertuples(index=False):
            raw_names = row.active_interventions if isinstance(row.active_interventions, str) else ""
            values[int(row.mask)] = CoalitionValue(
                scenario=scenario_config.name, mask=int(row.mask),
                active_interventions=tuple(n for n in raw_names.split(";") if n),
                cost=float(row.cost), utility=float(row.utility), improvement=float(row.improvement),
                satisfaction=float(row.satisfaction), retention=float(row.retention), fatigue=float(row.fatigue),
                relevance=float(row.relevance), provider_disparity=float(row.provider_disparity),
                catalog_coverage=float(row.catalog_coverage), duration_seconds=float(row.duration_seconds),
                intervention_stats={},
            )
        scenario_games[scenario_config.name] = ScenarioGame(scenario=scenario_config.name, values=values, shapley={}, interactions={}, feasibility_semivalue={})
    game = GameResult(
        scenario_games=scenario_games,
        regions=pd.DataFrame(),
        coalition_table=frame,
        interaction_table=pd.DataFrame(),
        robust_improvements=frame.groupby("mask")["improvement"].min().to_dict(),
        robust_shapley=dict(zip(robust_attr["intervention"], robust_attr["robust_phi"])),
    )
    choices = selector_choices(game, settings, selection_seed=42)
    return choices


def main() -> None:
    pe.ASSETS.mkdir(parents=True, exist_ok=True)
    points = pe.load_screening_points()
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "screening_order": pe.SCREENING_ORDER,
        "selection_seeds": list(pe.SELECTION_SEEDS),
        "evaluation_seeds": list(EVAL_SEEDS),
        "protocol": "selection games are full exact games; evaluation games are lean games over the union of selected masks",
        "claim_scope": "Simulator-conditional held-out seed comparison; not external causal inference.",
    }

    screening_rows = []
    divergent_points: list[str] = []
    for point_id in pe.SCREENING_ORDER:
        choices = screening_choices(point_id)
        if choices is None:
            payload = (point_id, points[point_id].values, 42, str(pe.ASSETS / "screening"))
            result = pe.run_full_game_payload(payload)
            choices = [SelectorChoice(**choice) for choice in result["choices"]]
            print(f"SCREEN {point_id}: fresh game ({result['duration_seconds']:.0f}s)", flush=True)
        else:
            print(f"SCREEN {point_id}: reused archived screening game", flush=True)
        for choice in choices:
            screening_rows.append({"point_id": point_id, **asdict(choice), "selected_interventions": ";".join(choice.selected_interventions)})
        divergent, distinct = pe.divergent_selectors(choices)
        print(f"   divergent={divergent} distinct_masks={distinct}", flush=True)
        if divergent:
            divergent_points.append(point_id)
        if len(divergent_points) >= MAX_HOLDOUT:
            break

    pd.DataFrame(screening_rows).to_csv(pe.ASSETS / "screening_selector_choices.csv", index=False)
    manifest["screened_points"] = sorted({row["point_id"] for row in screening_rows})
    manifest["divergent_points"] = divergent_points
    print(f"DIVERGENT POINTS: {divergent_points}", flush=True)

    for point_id in divergent_points:
        existing = pe.ASSETS / point_id / "heldout_selector_summary.csv"
        if existing.exists():
            print(f"HOLDOUT REUSED {point_id}: {existing}", flush=True)
            print(pd.read_csv(existing).to_string(index=False), flush=True)
            continue
        started = time.time()
        root = pe.holdout_for_point(points[point_id], pe.ASSETS, workers=2)
        summary = pd.read_csv(root / "heldout_selector_summary.csv")
        print(f"HOLDOUT COMPLETE {point_id} ({time.time()-started:.0f}s)", flush=True)
        print(summary.to_string(index=False), flush=True)

    (pe.ASSETS / "revision_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("FOCUSED DIVERGENCE DRIVER DONE", flush=True)


if __name__ == "__main__":
    main()
