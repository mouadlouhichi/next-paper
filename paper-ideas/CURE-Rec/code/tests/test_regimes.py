from __future__ import annotations

from cure_rec.observability import RunLogger
from cure_rec.regimes import run_regime_suite


def test_controlled_regime_suite_recovers_expected_structures(settings, tmp_path):
    settings.run.output_root = tmp_path
    logger = RunLogger(settings)
    result = run_regime_suite(settings, logger)
    logger.close()

    summary = result.summary.set_index("regime")
    assert bool(summary.loc["additive", "exact_selection_match"])
    assert bool(summary.loc["complementary", "exact_selection_match"])
    assert bool(summary.loc["redundant", "exact_selection_match"])
    assert bool(summary.loc["antagonistic", "exact_selection_match"])
    assert bool(summary.loc["delayed_fatigue_short", "exact_selection_match"])
    assert bool(summary.loc["delayed_fatigue_long", "exact_selection_match"])
    assert bool(summary.loc["provider_repair_balancing", "exact_selection_match"])
    assert bool(summary.loc["provider_repair_repeat", "exact_selection_match"])
    assert (result.run_dir / "regime_attribution_recovery.csv").exists()
    assert (result.run_dir / "regime_figure_selection_recovery.png").exists()


def test_misspecified_regime_exposes_attribution_error(settings, tmp_path):
    settings.run.output_root = tmp_path
    logger = RunLogger(settings)
    result = run_regime_suite(settings, logger)
    logger.close()
    misspecified = result.attribution_recovery[result.attribution_recovery["regime"] == "misspecified_ambiguity"]
    assert misspecified["absolute_error"].max() > 0
    assert not misspecified["sign_correct"].all()
