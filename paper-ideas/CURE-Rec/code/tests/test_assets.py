from __future__ import annotations

import json

from cure_rec.pipeline import run_experiment


def test_complete_numbered_asset_contract(settings, tmp_path):
    settings.run.output_root = tmp_path
    settings.simulator.n_users = 4
    settings.simulator.n_items = 24
    settings.simulator.n_providers = 4
    settings.simulator.n_categories = 4
    settings.simulator.horizon = 2
    settings.simulator.slate_size = 5
    settings.policy.candidate_pool_size = 16
    settings.scenarios = settings.scenarios[:1]

    logger, _, _ = run_experiment(settings)
    table_names = {path.name for path in logger.tables_dir.glob("*.csv")}
    figure_names = {path.name for path in logger.figures_dir.glob("*.png")}

    for number in range(1, 9):
        assert any(name.startswith(f"table_{number:02d}_") for name in table_names)
        assert any(name.startswith(f"figure_{number:02d}_") for name in figure_names)
    asset_manifest_path = logger.artifacts_dir / "asset_manifest.json"
    assert asset_manifest_path.exists()
    asset_manifest = json.loads(asset_manifest_path.read_text())
    table_one = next(row for row in asset_manifest if row["id"] == "Table 1")
    assert table_one["exists"] is True
    assert (logger.artifacts_dir / "coalitions" / "nominal" / "mask_00.json").exists()
