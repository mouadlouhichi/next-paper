"""End-to-end experiment runner used by the CLI and notebook."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from cure_rec.config import Settings
from cure_rec.game import GameResult, run_exact_game
from cure_rec.observability import RunLogger
from cure_rec.planner import PortfolioDecision, build_explanation_card, select_robust_portfolio
from cure_rec.reporting import emit_assets


def run_experiment(settings: Settings) -> tuple[RunLogger, GameResult, PortfolioDecision]:
    logger = RunLogger(settings)
    try:
        with logger.span("exact_game"):
            game = run_exact_game(settings, logger)
        with logger.span("robust_planning"):
            decision = select_robust_portfolio(game, settings, logger)
            card = build_explanation_card(game, decision, logger)
        with logger.span("reporting"):
            emit_assets(game, decision, settings, logger)
            logger.write_json("artifacts/run_summary.json", {
                "decision": asdict(decision),
                "selected_attributions": card["selected_attributions"],
                "run_dir": str(logger.run_dir),
            })
        logger.close(status="completed")
        return logger, game, decision
    except Exception:
        logger.close(status="failed")
        raise


def run_from_path(config_path: str | Path):
    from cure_rec.config import load_settings

    return run_experiment(load_settings(config_path))
