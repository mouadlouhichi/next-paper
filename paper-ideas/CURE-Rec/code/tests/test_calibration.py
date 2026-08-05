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
