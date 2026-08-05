from __future__ import annotations

from cure_rec.observability import RunLogger
from cure_rec.regimes import run_regime_suite


def test_controlled_regime_suite_recovers_expected_structures(settings, tmp_path):
    settings.run.output_root = tmp_path
    logger = RunLogger(settings)
    result = run_regime_suite(settings, logger)
    logger.close()

    summary = result.summary.set_index("regime")
    for regime in [
        "additive",
        "complementary",
        "redundant",
        "antagonistic",
        "delayed_fatigue_short",
        "delayed_fatigue_long",
        "provider_repair_balancing",
        "provider_repair_repeat",
    ]:
        assert bool(summary.loc[regime, "estimated_selection_match"])
        assert bool(summary.loc[regime, "oracle_selection_match"])
    # Under misspecification, optimization can match the estimated game while
    # disagreeing with the oracle game. This is the intended failure mode.
    assert bool(summary.loc["misspecified_ambiguity", "estimated_selection_match"])
    assert not bool(summary.loc["misspecified_ambiguity", "oracle_selection_match"])
    assert summary.loc["misspecified_ambiguity", "oracle_regret"] > 0
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
