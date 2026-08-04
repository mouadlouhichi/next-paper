from __future__ import annotations

from cure_rec.workflow import run_full_workflow


def test_full_workflow_loads_analyzes_and_runs_cure(settings, tmp_path):
    data_root = tmp_path / "ml-1m"
    data_root.mkdir()
    lines = []
    for user in range(1, 5):
        for offset, item in enumerate([1, 2, 3, 4]):
            lines.append(f"{user}::{item + user}::5::{1_000 + offset}\n")
    (data_root / "ratings.dat").write_text("".join(lines), encoding="latin-1")

    settings.run.output_root = tmp_path / "runs"
    settings.simulator.n_users = 4
    settings.simulator.n_items = 24
    settings.simulator.n_providers = 4
    settings.simulator.n_categories = 4
    settings.simulator.horizon = 2
    settings.simulator.slate_size = 5
    settings.policy.candidate_pool_size = 16
    settings.scenarios = settings.scenarios[:1]

    result = run_full_workflow(
        settings,
        dataset="movielens_1m",
        source=tmp_path,
        run_bpr=False,
        max_eval_users=4,
    )

    assert result.analysis.run_dir.exists()
    assert (result.cure_run_dir / "artifacts" / "asset_manifest.json").exists()
    assert result.dataset.dataset == "movielens_1m"
