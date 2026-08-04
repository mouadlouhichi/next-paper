from __future__ import annotations

from cure_rec.pipeline import run_experiment


def test_end_to_end_small_exact_run(settings, tmp_path):
    settings.run.output_root = tmp_path
    settings.simulator.n_users = 4
    settings.simulator.n_items = 24
    settings.simulator.n_providers = 4
    settings.simulator.n_categories = 4
    settings.simulator.horizon = 2
    settings.simulator.slate_size = 5
    settings.policy.candidate_pool_size = 16
    settings.scenarios = settings.scenarios[:1]

    logger, game, decision = run_experiment(settings)

    assert (logger.run_dir / "manifest.json").exists()
    assert (logger.run_dir / "logs" / "events.jsonl").exists()
    assert len(game.coalition_table) == 64
    assert set(game.regions["intervention"]) == set(settings.interventions.costs)
    assert decision.action in {
        "improve_selected",
        "abstain_keep_base",
        "repair_selected",
        "no_feasible_portfolio",
    }
    assert (logger.run_dir / "artifacts" / "explanation_card.json").exists()
