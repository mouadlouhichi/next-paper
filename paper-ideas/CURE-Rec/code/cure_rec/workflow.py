"""One-command orchestration: fetch/load -> audit -> model analysis -> CURE-Sim assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cure_rec.analysis import DataAnalysisResult, analyze_dataset
from cure_rec.config import Settings
from cure_rec.data import DatasetLoadResult, load_dataset
from cure_rec.pipeline import run_experiment
from cure_rec.planner import PortfolioDecision


@dataclass(frozen=True)
class FullWorkflowResult:
    dataset: DatasetLoadResult
    analysis: DataAnalysisResult
    cure_run_dir: Path
    decision: PortfolioDecision


def run_full_workflow(
    settings: Settings,
    *,
    dataset: str,
    source: str | Path,
    download: bool = False,
    run_bpr: bool = True,
    bpr_updates: int = 50_000,
    max_eval_users: int = 1_000,
) -> FullWorkflowResult:
    """Run data/model analysis before the CURE-Sim causal benchmark.

    The order is intentional: external data are loaded, standardized, audited, and
    analyzed first. The later CURE-Sim run remains the oracle causal evaluation;
    external baseline metrics are not misrepresented as policy-intervention proof.
    """
    loaded = load_dataset(dataset, source, download=download)
    analysis = analyze_dataset(
        loaded,
        output_root=settings.run.output_root,
        run_bpr=run_bpr,
        bpr_updates=bpr_updates,
        max_eval_users=max_eval_users,
        seed=settings.run.seed,
    )
    logger, _, decision = run_experiment(settings)
    return FullWorkflowResult(loaded, analysis, logger.run_dir, decision)
