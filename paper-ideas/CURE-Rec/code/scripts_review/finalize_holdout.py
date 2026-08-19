"""Finalize a held-out selector study from completed lean evaluation runs.

Lean evaluation games store per-coalition JSON artifacts (not the full coalition
table), so this script rebuilds robust values and feasibility margins from
`artifacts/coalitions/<scenario>/mask_*.json` for every completed evaluation run,
joins them with the selection choices, and writes the paired summary.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase_e_divergence as pe  # noqa: E402
from cure_rec.calibration import _settings_for_point  # noqa: E402
from cure_rec.revision import _paired_summary  # noqa: E402

POINT_ID = sys.argv[1] if len(sys.argv) > 1 else "lhs-009"


def completed_run(point_root: Path, stage: str, seed: int) -> Path | None:
    for run_dir in sorted((point_root / stage / POINT_ID).glob(f"{POINT_ID}-seed-{seed}-*")):
        log = run_dir / "run.log"
        if log.exists() and "run_finished" in log.read_text(errors="ignore"):
            return run_dir
    return None


def values_from_run(run_dir: Path) -> dict[int, dict[str, dict]]:
    """mask -> scenario -> raw coalition value record."""
    out: dict[int, dict[str, dict]] = {}
    for path in sorted((run_dir / "artifacts" / "coalitions").glob("*/mask_*.json")):
        record = json.loads(path.read_text())
        mask = int(record["coalition_mask"])
        out.setdefault(mask, {})[record["scenario"]] = record["value"]
    return out


def main() -> None:
    points = pe.load_screening_points()
    point = points[POINT_ID]
    point_root = pe.ASSETS / POINT_ID
    base_settings = pe.load_settings(pe.BASE_CONFIG)
    settings = _settings_for_point(base_settings, point, point_root / "evaluation")

    choices = pd.read_csv(point_root / "selection_choices.csv")
    rows = []
    for seed in pe.EVALUATION_SEEDS:
        run_dir = completed_run(point_root, "evaluation", seed)
        if run_dir is None:
            raise SystemExit(f"evaluation seed {seed} incomplete")
        values = values_from_run(run_dir)
        scenarios = sorted(next(iter(values.values())).keys())
        robust = {mask: min(vals[sc]["improvement"] for sc in scenarios) for mask, vals in values.items()}
        # The empty coalition's improvement is zero by definition; its stored
        # JSON record keeps the raw utility (the zeroing is applied in-memory
        # during live evaluation only).
        robust[0] = 0.0
        base_relevance = {sc: values[0][sc]["relevance"] for sc in scenarios}
        for choice in choices.itertuples(index=False):
            mask = int(choice.selected_mask)
            vals = values[mask]
            rel_delta = min(vals[sc]["relevance"] - base_relevance[sc] for sc in scenarios)
            margins = {
                "cost": float(vals[scenarios[0]]["cost"]),
                "relevance_delta_lower": float(rel_delta),
                "provider_disparity_upper": float(max(vals[sc]["provider_disparity"] for sc in scenarios)),
                "fatigue_upper": float(max(vals[sc]["fatigue"] for sc in scenarios)),
            }
            c = settings.constraints
            feasible = bool(
                margins["cost"] <= c.budget + 1e-12
                and margins["relevance_delta_lower"] >= c.min_relevance_delta
                and margins["provider_disparity_upper"] <= c.max_provider_disparity
                and margins["fatigue_upper"] <= c.max_fatigue
            )
            rows.append({
                "selection_seed": int(choice.selection_seed),
                "evaluation_seed": seed,
                "selector": choice.selector,
                "selected_mask": mask,
                "selected_interventions": choice.selected_interventions,
                "robust_lower_improvement": robust[mask],
                "feasible": feasible,
                **margins,
            })

    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(point_root / "heldout_selector_evaluations.csv", index=False)
    summary = _paired_summary(evaluation, "cure_exact_maximin")
    summary.to_csv(point_root / "heldout_selector_summary.csv", index=False)
    (point_root / "revision_manifest.json").write_text(json.dumps({
        "point_id": POINT_ID,
        "point_values": point.values,
        "selection_seeds": list(pe.SELECTION_SEEDS),
        "evaluation_seeds": list(pe.EVALUATION_SEEDS),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_scope": "Simulator-conditional held-out seed comparison; not external causal inference.",
    }, indent=2, default=str), encoding="utf-8")
    print(f"HOLDOUT FINALIZED {POINT_ID}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
