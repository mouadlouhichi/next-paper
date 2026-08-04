from __future__ import annotations

from cure_rec.experiments import run_all_variations


def _small(settings, tmp_path, name: str):
    cloned = settings.model_copy(deep=True)
    cloned.run.name = name
    cloned.run.output_root = tmp_path / name
    cloned.simulator.n_users = 4
    cloned.simulator.n_items = 24
    cloned.simulator.n_providers = 4
    cloned.simulator.n_categories = 4
    cloned.simulator.horizon = 2
    cloned.simulator.slate_size = 5
    cloned.policy.candidate_pool_size = 16
    cloned.scenarios = cloned.scenarios[:1]
    return cloned


def test_master_all_variations_orchestrates_every_stage(settings, tmp_path):
    data_root = tmp_path / "ml-1m"
    data_root.mkdir()
    lines = []
    for user in range(1, 5):
        for offset, item in enumerate([1, 2, 3, 4]):
            lines.append(f"{user}::{item + user}::5::{1_000 + offset}\n")
    (data_root / "ratings.dat").write_text("".join(lines), encoding="latin-1")

    result = run_all_variations(
        _small(settings, tmp_path, "quick"),
        _small(settings, tmp_path, "full"),
        dataset="movielens_1m",
        source=tmp_path,
        run_bpr=False,
        bpr_updates=100,
        max_eval_users=4,
        quick_seeds=[42],
        full_seeds=[43],
        final_seeds=[44],
    )

    assert result.data_analysis_dir.exists()
    assert set(result.variation_summary["variation"]) == {
        "quick_single",
        "controlled_regimes",
        "quick_five_seed",
        "full_single",
        "full_five_seed",
        "full_twenty_seed",
    }
    assert (result.run_dir / "all_variations_manifest.json").exists()
    assert (result.run_dir / "all_variations_seed_decisions.csv").exists()
