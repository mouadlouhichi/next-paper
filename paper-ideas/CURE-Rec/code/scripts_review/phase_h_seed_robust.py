"""Reviewer item 2 -- calibration-robust planning over scenarios AND seeds.

The shipped planner is scenario-robust on a single initialization seed, so the
archived disagreement studies report frozen portfolios whose held-out
feasibility is 0.70 (lhs-012) and 0.47 (lhs-009).  This driver selects the same
decision under three planners and scores every frozen portfolio on the disjoint
held-out seeds:

1. ``scenario_only``        -- the shipped planner on the seed-42 game (reference);
2. ``seed_scenario_minimax``-- feasibility and value pooled over scenarios and
                               calibration seeds 42-46 (conservative extension);
3. ``seed_scenario_chance`` -- empirical chance constraint
                               (1/K) sum_zeta 1_F(S;zeta) >= 1 - alpha, ranked by
                               the seed-averaged worst-scenario improvement.

Calibration games are reconstructed from the archived exact coalition tables, so
the conservative and chance-constrained planners see exactly the rollouts that
produced the published numbers; no new simulation noise enters selection.
Held-out rows are fresh rollouts on seeds 200-219 and are also cross-checked
against the archived evaluation logs whenever the frozen mask was logged there.

Usage:
    python scripts_review/phase_h_seed_robust.py lhs-012
    python scripts_review/phase_h_seed_robust.py lhs-009 --alpha 0.2 --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cure_rec.config import Settings  # noqa: E402
from cure_rec.game import EMPTY_MASK, CoalitionValue, GameResult, ScenarioGame  # noqa: E402
from cure_rec.planner import decision_to_dict, select_robust_portfolio  # noqa: E402
from cure_rec.revision import _paired_summary  # noqa: E402
from cure_rec.seed_robust import (  # noqa: E402
    _QuietLogger,
    calibration_robust_values,
    heldout_evaluation,
    select_calibration_robust_portfolio,
    seed_scenario_metrics,
)

ASSETS = ROOT / "results" / "reviewer_phase_assets"
DIVERGENT = ASSETS / "divergent_selector_holdout"
CALIBRATION_SEEDS = (42, 43, 44, 45, 46)
EVALUATION_SEEDS = tuple(range(200, 220))


def load_point_settings(point_id: str) -> Settings:
    """Recover the exact configuration of an archived selection run."""
    candidates = sorted((DIVERGENT / point_id / "selection" / point_id).glob(f"{point_id}-seed-42-*/manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"no archived seed-42 manifest for {point_id}")
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    settings = Settings.model_validate(payload["settings"])
    return settings, payload["config_hash"], candidates[0].parent


def table_to_game(frame: pd.DataFrame, settings: Settings) -> GameResult:
    """Rebuild a GameResult value shell from an archived coalition table."""
    scenario_games: dict[str, ScenarioGame] = {}
    for scenario in settings.scenarios:
        part = frame[frame["scenario"] == scenario.name]
        values: dict[int, CoalitionValue] = {}
        for row in part.itertuples(index=False):
            names = row.active_interventions if isinstance(row.active_interventions, str) else ""
            values[int(row.mask)] = CoalitionValue(
                scenario=scenario.name,
                mask=int(row.mask),
                active_interventions=tuple(n for n in names.split(";") if n),
                cost=float(row.cost),
                utility=float(row.utility),
                # The archived base row stores improvement == 0.0 already; keep
                # it explicit so reconstruction cannot silently shift the origin.
                improvement=float(row.improvement),
                satisfaction=float(row.satisfaction),
                retention=float(row.retention),
                fatigue=float(row.fatigue),
                relevance=float(row.relevance),
                provider_disparity=float(row.provider_disparity),
                catalog_coverage=float(row.catalog_coverage),
                duration_seconds=float(row.duration_seconds),
                intervention_stats={},
            )
        if EMPTY_MASK not in values or len(values) != 64:
            raise ValueError(f"incomplete coalition table for scenario {scenario.name}")
        base_utility = values[EMPTY_MASK].utility
        for mask, value in values.items():
            if mask != EMPTY_MASK and not np.isclose(value.improvement, value.utility - base_utility, atol=1e-9):
                raise ValueError(f"improvement origin mismatch at mask {mask} in {scenario.name}")
        scenario_games[scenario.name] = ScenarioGame(
            scenario=scenario.name, values=values, shapley={}, interactions={}, feasibility_semivalue={}
        )
    return GameResult(scenario_games=scenario_games, regions=None, coalition_table=None, interaction_table=None)


def load_calibration_games(point_id: str, settings: Settings) -> dict[int, GameResult]:
    games: dict[int, GameResult] = {}
    for seed in CALIBRATION_SEEDS:
        tables = sorted((DIVERGENT / point_id / "selection" / point_id).glob(f"{point_id}-seed-{seed}-*/tables/coalition_values.csv"))
        if not tables:
            raise FileNotFoundError(f"missing archived coalition table for {point_id} seed {seed}")
        games[seed] = table_to_game(pd.read_csv(tables[0]), settings)
    return games


def archived_holdout_rows(point_id: str, mask: int) -> pd.DataFrame:
    """Held-out feasibility of a mask recovered from archived evaluation logs."""
    rows = []
    for run in sorted((DIVERGENT / point_id / "evaluation" / point_id).glob(f"{point_id}-seed-*/logs/events.jsonl")):
        seed = int(run.parent.parent.name.split("-seed-")[1].split("-")[0])
        per_scenario = {}
        for line in run.open(encoding="utf-8"):
            event = json.loads(line)
            if event.get("event") == "coalition_evaluated" and int(event["mask"]) == mask:
                per_scenario[event["scenario"]] = event
        if per_scenario:
            rows.append({"evaluation_seed": seed, "scenario_count": len(per_scenario), **per_scenario[min(per_scenario)]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("point_id", choices=["lhs-012", "lhs-009"])
    parser.add_argument("--alpha", type=float, default=0.2, help="chance-constraint violation level")
    parser.add_argument("--quick", action="store_true", help="3 held-out seeds instead of 20 (smoke test)")
    args = parser.parse_args()

    settings, config_hash, manifest_dir = load_point_settings(args.point_id)
    out = ASSETS / "seed_robust_planner" / args.point_id
    out.mkdir(parents=True, exist_ok=True)
    evaluation_seeds = tuple(range(200, 203)) if args.quick else EVALUATION_SEEDS

    games = load_calibration_games(args.point_id, settings)
    quiet = _QuietLogger()

    # 1. Published protocol: scenario-only maximin selected INDEPENDENTLY on each
    #    calibration seed.  The archived 0.70 / 0.47 held-out feasibility rates are
    #    averages over these five possibly different frozen portfolios, so this is
    #    the reference the new planners must beat.
    per_seed: list[dict] = []
    for seed in CALIBRATION_SEEDS:
        choice = select_robust_portfolio(games[seed], settings, quiet)
        per_seed.append(
            {
                "selection_seed": seed,
                "selected_mask": choice.selected_mask,
                "selected_interventions": ";".join(choice.selected_interventions),
                "mode": choice.mode.value,
                "status": choice.status.value,
                "lower_improvement": choice.lower_improvement,
            }
        )
    per_seed_frame = pd.DataFrame(per_seed)
    per_seed_frame.to_csv(out / "scenario_only_per_seed_selections.csv", index=False)

    # 2. Calibration-robust planners over scenarios AND seeds 42-46 (one frozen
    #    portfolio each, by construction seed-stable).
    minimax = select_calibration_robust_portfolio(games, settings, quiet, rule="minimax")
    chance = select_calibration_robust_portfolio(games, settings, quiet, rule="chance", alpha=args.alpha)

    selections = pd.DataFrame(
        [
            {
                "planner": "scenario_only_per_seed",
                "rule": "scenario maximin, selected independently per calibration seed (published protocol)",
                "alpha": float("nan"),
                "selection_seed": "42-46 (independent)",
                "selected_mask": ";".join(str(int(m)) for m in per_seed_frame["selected_mask"]),
                "selected_interventions": " | ".join(
                    "none" if not isinstance(v, str) else v for v in per_seed_frame["selected_interventions"]
                ),
                "distinct_portfolios": int(per_seed_frame["selected_mask"].nunique()),
                "calibration_lower_improvement": float(per_seed_frame["lower_improvement"].mean()),
            },
            {
                "planner": "seed_scenario_minimax",
                "rule": "minimax over scenarios x seeds",
                "alpha": float("nan"),
                "selection_seed": "42-46 (pooled)",
                "selected_mask": minimax.decision.selected_mask,
                "selected_interventions": ";".join(minimax.decision.selected_interventions) or "none",
                "distinct_portfolios": 1,
                "calibration_lower_improvement": minimax.calibration_lower_improvement,
            },
            {
                "planner": "seed_scenario_chance",
                "rule": f"empirical chance constraint over scenarios x seeds, alpha={args.alpha}",
                "alpha": args.alpha,
                "selection_seed": "42-46 (pooled)",
                "selected_mask": chance.decision.selected_mask,
                "selected_interventions": ";".join(chance.decision.selected_interventions) or "none",
                "distinct_portfolios": 1,
                "calibration_lower_improvement": chance.calibration_lower_improvement,
            },
        ]
    )
    selections.to_csv(out / "calibration_selections.csv", index=False)

    # 3. Score every frozen portfolio on the disjoint held-out seeds.  The
    #    published protocol contributes one frozen mask per calibration seed.
    frozen: list[tuple[str, int, int]] = [
        ("scenario_only_per_seed", int(row.selection_seed), int(row.selected_mask)) for row in per_seed_frame.itertuples(index=False)
    ]
    frozen.append(("seed_scenario_minimax", 42, int(minimax.decision.selected_mask)))
    frozen.append(("seed_scenario_chance", 42, int(chance.decision.selected_mask)))

    heldout_rows: list[dict] = []
    cache: dict[int, list[dict]] = {}
    for planner, selection_seed, mask in frozen:
        if mask not in cache:
            cache[mask] = heldout_evaluation(settings, mask, evaluation_seeds)
        for row in cache[mask]:
            heldout_rows.append({"selector": planner, "selection_seed": selection_seed, **row})
    heldout = pd.DataFrame(heldout_rows)
    heldout.rename(columns={"selector": "planner"}).to_csv(out / "heldout_planner_evaluations.csv", index=False)

    # Evaluation seed is the independent unit; where the published protocol
    # contributes several frozen portfolios, average them within the seed first so
    # the pairing stays one row per evaluation seed (no pseudoreplication).
    aggregated = (
        heldout.groupby(["selector", "evaluation_seed"], as_index=False)
        .agg(robust_lower_improvement=("robust_lower_improvement", "mean"), feasible=("feasible", "mean"))
        .assign(selection_seed=0)
    )
    summary = _paired_summary(aggregated, "scenario_only_per_seed")
    summary.insert(0, "planner", summary.pop("selector"))
    summary.to_csv(out / "heldout_planner_summary.csv", index=False)

    # 4. Cross-check against archived evaluation logs where the mask exists.
    cross = []
    for planner, selection_seed, mask in frozen:
        label = f"{planner}@seed{selection_seed}"
        archived = archived_holdout_rows(args.point_id, int(mask))
        if archived.empty:
            cross.append({"planner": label, "selected_mask": int(mask), "archived_runs": 0, "archived_note": "mask not in archived evaluation logs"})
            continue
        relevant = [col for col in archived.columns if col.startswith(("relevance", "provider", "fatigue", "cost"))]
        cross.append(
            {
                "planner": label,
                "selected_mask": int(mask),
                "archived_runs": int(len(archived)),
                "archived_provider_disparity_max": float(archived["provider_disparity"].max()),
                "archived_fatigue_max": float(archived["fatigue"].max()),
                "archived_note": "recovered from archived events.jsonl",
            }
        )
    pd.DataFrame(cross).to_csv(out / "archived_crosscheck.csv", index=False)

    calibration_values = calibration_robust_values(games)
    (out / "revision_manifest.json").write_text(
        json.dumps(
            {
                "reviewer_item": "2 -- calibration robustness over scenarios and initialization seeds",
                "point_id": args.point_id,
                "config_hash": config_hash,
                "settings_source": str(manifest_dir.relative_to(ROOT)),
                "calibration_seeds": list(CALIBRATION_SEEDS),
                "evaluation_seeds": list(evaluation_seeds),
                "alpha": args.alpha,
                "planners": list(selections["planner"]),
                "calibration_source": "archived exact coalition tables (no new selection rollouts)",
                "holdout_source": "fresh rollouts via cure_rec.seed_robust.heldout_evaluation",
                "selected": json.loads(selections.to_json(orient="records")),
                "calibration_robust_value_of_selected": {
                    f"{planner}@seed{selection_seed}": calibration_values.get(mask)
                    for planner, selection_seed, mask in frozen
                },
                "per_seed_scenario_only_selections": json.loads(per_seed_frame.to_json(orient="records")),
                "claim": "Simulator-conditional decision robustness only; no external causal claim.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n=== {args.point_id} (config {config_hash}) ===")
    print(selections.to_string(index=False))
    print("\n--- held-out summary over disjoint seeds ---")
    print(summary[["planner", "independent_evaluation_seeds", "robust_lower_improvement_mean", "feasible_rate"]].to_string(index=False))
    print("\n--- archived cross-check ---")
    print(pd.DataFrame(cross).to_string(index=False))
    print(f"\nwrote assets to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
