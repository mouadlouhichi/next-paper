"""Resumable held-out selector study for one divergent configuration.

Reuses completed selection games and evaluation games already on disk (checked via
`run_finished` in each run log) and only runs what is missing. Selector choices for
reused selection games are reconstructed from the archived coalition tables through
the same selector code path used for live games.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase_e_divergence as pe  # noqa: E402
from cure_rec.calibration import _settings_for_point  # noqa: E402
from cure_rec.game import ALL_MASKS, EMPTY_MASK, GameResult, ScenarioGame, CoalitionValue, coalition_names  # noqa: E402
from cure_rec.revision import SelectorChoice, _paired_summary, selector_choices  # noqa: E402

POINT_ID = sys.argv[1] if len(sys.argv) > 1 else "lhs-009"
SELECTION_SEEDS = pe.SELECTION_SEEDS
EVALUATION_SEEDS = pe.EVALUATION_SEEDS
WORKERS = 2


def completed_run(point_root: Path, stage: str, seed: int) -> Path | None:
    for run_dir in sorted((point_root / stage / POINT_ID).glob(f"{POINT_ID}-seed-{seed}-*")):
        log = run_dir / "run.log"
        if log.exists() and "run_finished" in log.read_text(errors="ignore"):
            return run_dir
    return None


def choices_from_run(run_dir: Path, settings, seed: int) -> list[SelectorChoice]:
    frame = pd.read_csv(run_dir / "tables" / "coalition_values.csv")
    attr = pd.read_csv(run_dir / "tables" / "robust_game_attribution.csv")
    scenario_games: dict[str, ScenarioGame] = {}
    for scenario_config in settings.scenarios:
        part = frame[frame["scenario"] == scenario_config.name]
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
        robust_shapley=dict(zip(attr["intervention"], attr["robust_phi"])),
    )
    return selector_choices(game, settings, selection_seed=seed)


def main() -> None:
    points = pe.load_screening_points()
    point = points[POINT_ID]
    point_root = pe.ASSETS / POINT_ID
    (point_root / "selection" / POINT_ID).mkdir(parents=True, exist_ok=True)
    (point_root / "evaluation" / POINT_ID).mkdir(parents=True, exist_ok=True)

    base_settings = pe.load_settings(pe.BASE_CONFIG)

    def settings_for_seed(seed: int, stage_root: Path):
        s = _settings_for_point(base_settings, point, stage_root)
        s.run.seed = seed
        s.run.name = f"{POINT_ID}-seed-{seed}"
        return s

    # ---------------- selection games ----------------
    all_choices: list[SelectorChoice] = []
    for seed in SELECTION_SEEDS:
        run_dir = completed_run(point_root, "selection", seed)
        if run_dir is not None:
            settings = settings_for_seed(seed, point_root / "selection")
            choices = choices_from_run(run_dir, settings, seed)
            print(f"SELECTION seed {seed}: reused {run_dir.name}", flush=True)
        else:
            payload = (POINT_ID, point.values, seed, str(point_root / "selection"))
            result = pe.run_full_game_payload(payload)
            choices = [SelectorChoice(**choice) for choice in result["choices"]]
            print(f"SELECTION seed {seed}: fresh game ({result['duration_seconds']:.0f}s)", flush=True)
        all_choices.extend(choices)

    choice_frame = pd.DataFrame([{**asdict(choice), "selected_interventions": ";".join(choice.selected_interventions)} for choice in all_choices])
    choice_frame.to_csv(point_root / "selection_choices.csv", index=False)

    masks = tuple(sorted({choice.selected_mask for choice in all_choices}))
    (point_root / "evaluation_masks.json").write_text(
        json.dumps({"masks": list(masks), "names": {str(mask): list(coalition_names(mask)) for mask in masks}}, indent=2)
    )

    # ---------------- evaluation games ----------------
    pending = []
    for seed in EVALUATION_SEEDS:
        if completed_run(point_root, "evaluation", seed) is None:
            pending.append((POINT_ID, point.values, seed, masks, str(point_root / "evaluation")))
    if pending:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for _ in pool.map(pe.run_lean_evaluation_payload, pending):
                pass
    print(f"EVALUATION: {len(EVALUATION_SEEDS) - len(pending)} reused, {len(pending)} fresh", flush=True)

    evaluation_results = []
    for seed in EVALUATION_SEEDS:
        run_dir = completed_run(point_root, "evaluation", seed)
        if run_dir is None:
            raise RuntimeError(f"evaluation seed {seed} did not complete")
        table = pd.read_csv(run_dir / "tables" / "coalition_values.csv")
        robust = table.groupby("mask")["improvement"].min().to_dict()
        rel_base = table[table["mask"] == 0].set_index("scenario")["relevance"]
        settings = settings_for_seed(seed, point_root / "evaluation")
        for choice in all_choices:
            mask = choice.selected_mask
            rows = table[table["mask"] == mask]
            rel_delta = min(rows[rows["scenario"] == sc]["relevance"].iloc[0] - rel_base[sc] for sc in rel_base.index)
            margins = {
                "cost": float(rows["cost"].iloc[0]),
                "relevance_delta_lower": float(rel_delta),
                "provider_disparity_upper": float(rows["provider_disparity"].max()),
                "fatigue_upper": float(rows["fatigue"].max()),
            }
            c = settings.constraints
            feasible = bool(
                margins["cost"] <= c.budget + 1e-12
                and margins["relevance_delta_lower"] >= c.min_relevance_delta
                and margins["provider_disparity_upper"] <= c.max_provider_disparity
                and margins["fatigue_upper"] <= c.max_fatigue
            )
            evaluation_results.append({
                "selection_seed": choice.selection_seed,
                "evaluation_seed": seed,
                "selector": choice.selector,
                "selected_mask": mask,
                "selected_interventions": ";".join(choice.selected_interventions),
                "robust_lower_improvement": robust[mask],
                "feasible": feasible,
                **margins,
            })

    evaluation = pd.DataFrame(evaluation_results)
    evaluation.to_csv(point_root / "heldout_selector_evaluations.csv", index=False)
    summary = _paired_summary(evaluation, "cure_exact_maximin")
    summary.to_csv(point_root / "heldout_selector_summary.csv", index=False)
    (point_root / "revision_manifest.json").write_text(json.dumps({
        "point_id": POINT_ID,
        "point_values": point.values,
        "selection_seeds": list(SELECTION_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_scope": "Simulator-conditional held-out seed comparison; not external causal inference.",
    }, indent=2, default=str), encoding="utf-8")
    print(f"HOLDOUT COMPLETE {POINT_ID}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
