from __future__ import annotations

from pathlib import Path

import pandas as pd

import cure_rec.calibration as calibration
from cure_rec.calibration import default_oat_points, latin_hypercube_points, run_calibration_sweep
from cure_rec.experiments import SeedSweepResult


def test_calibration_designs_are_reproducible_and_cover_all_assumptions(settings):
    oat = default_oat_points(settings)
    assert oat[0].is_baseline
    assert oat[0].point_id == "baseline"
    assert {point.varied_parameter for point in oat[1:]} == {
        "fatigue_strength", "repeat_threshold", "horizon", "provider_threshold",
        "provider_balance_strength", "novelty_delayed_benefit", "exploration_cost",
    }
    lhs_a = latin_hypercube_points(settings, 4, seed=17)
    lhs_b = latin_hypercube_points(settings, 4, seed=17)
    assert lhs_a == lhs_b
    assert len(lhs_a) == 5  # one disclosed baseline plus four joint configurations
    assert all(1 <= point.values["repeat_threshold"] <= 5 for point in lhs_a[1:])
    assert all(4 <= point.values["horizon"] <= 20 for point in lhs_a[1:])


def test_calibration_aggregates_seed_decisions_without_rerunning_test_models(settings, tmp_path, monkeypatch):
    """Exercise aggregation/asset contracts with a deterministic cheap sweep stub."""
    settings.run.output_root = tmp_path

    def fake_sweep(configured, seeds):
        root = Path(configured.run.output_root) / "fake-sweep"
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        attribution = []
        interactions = []
        for seed in seeds:
            rows.append({
                "seed": seed, "cure_run_dir": str(root), "mode": "repair",
                "status": "repair_selected", "base_feasible": False,
                "selected_interventions": "('repeat_cap',)", "lower_improvement": 0.2,
                "provider_disparity_upper": 0.25, "fatigue_upper": 0.1, "feasible": True,
            })
            attribution.append({"seed": seed, "intervention": "repeat_cap", "phi_mean": 0.1})
            interactions.append({"seed": seed, "intervention_i": "repeat_cap", "intervention_j": "explore_slot", "interaction_mean": 0.0})
        return SeedSweepResult(root, pd.DataFrame(rows), pd.DataFrame(attribution), pd.DataFrame(interactions), pd.DataFrame())

    monkeypatch.setattr(calibration, "run_seed_sweep", fake_sweep)
    result = run_calibration_sweep(settings, seeds=(42, 43), design="oat")
    assert len(result.configurations) == 15
    assert len(result.seed_decisions) == 30
    assert (result.summary["repeat_cap_selection_rate"] == 1.0).all()
    assert (result.summary["repair_rate"] == 1.0).all()
    assert (result.run_dir / "calibration_manifest.json").exists()
    assert (result.run_dir / "figures" / "calibration_figure_oat_lower_improvement.png").exists()


def test_calibration_recovers_completed_child_artifact(settings, tmp_path):
    """A shutdown after a child run must not force its exact game to rerun."""
    import json
    from cure_rec.calibration import _recover_completed_point_runs

    point_root = tmp_path / "baseline"
    run_dir = point_root / "seed-sweep-old" / "runs" / "calibration-baseline-seed-42-old"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "tables").mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"settings": {"run": {"seed": 42}}}))
    decision = {
        "mode": "repair", "status": "repair_selected", "base_feasible": False,
        "selected_mask": 1, "selected_interventions": ["repeat_cap"],
        "lower_improvement": 0.2, "upper_improvement": 0.3, "cost": 0.05,
        "relevance_delta_lower": -0.01, "provider_disparity_upper": 0.2,
        "fatigue_upper": 0.1, "feasible": True, "reason": "fixture", "action": "repair_selected",
    }
    (run_dir / "artifacts" / "run_summary.json").write_text(json.dumps({"decision": decision}))
    pd.DataFrame([{"intervention": "repeat_cap", "phi_mean": 0.1}]).to_csv(run_dir / "tables" / "table_03_attribution_regions.csv", index=False)
    pd.DataFrame([{"intervention_i": "repeat_cap", "intervention_j": "explore_slot", "interaction_mean": 0.0}]).to_csv(run_dir / "tables" / "interaction_regions.csv", index=False)
    pd.DataFrame([{"mask": 0, "provider_disparity": 0.35, "fatigue": 0.2}]).to_csv(run_dir / "tables" / "coalition_values.csv", index=False)

    result, completed = _recover_completed_point_runs(point_root, settings, [42, 43])
    assert completed == {42}
    assert result.decisions["seed"].tolist() == [42]
    assert result.decisions.loc[0, "selected_interventions"] == ("repeat_cap",)
    assert result.attributions["seed"].tolist() == [42]
