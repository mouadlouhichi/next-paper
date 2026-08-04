from __future__ import annotations

from cure_rec.experiments import run_seed_sweep


def test_paired_seed_sweep_emits_aggregate_assets(settings, tmp_path):
    settings.run.output_root = tmp_path
    settings.simulator.n_users = 4
    settings.simulator.n_items = 24
    settings.simulator.n_providers = 4
    settings.simulator.n_categories = 4
    settings.simulator.horizon = 2
    settings.simulator.slate_size = 5
    settings.policy.candidate_pool_size = 16
    settings.scenarios = settings.scenarios[:1]

    result = run_seed_sweep(settings, [42, 43])
    assert len(result.decisions) == 2
    assert set(result.attributions["seed"]) == {42, 43}
    assert (result.run_dir / "seed_sweep_decisions.csv").exists()
    assert (result.run_dir / "seed_sweep_attribution_summary.csv").exists()
