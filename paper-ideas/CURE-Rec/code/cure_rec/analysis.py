"""External-data profiling, baseline-model analysis, and generated assets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cure_rec.data import AuditResult, DatasetLoadResult, audit_interactions
from cure_rec.models import BPRMFRecommender, PopularityRecommender, chronological_leave_one_out, evaluate_leave_one_out


@dataclass(frozen=True)
class DataAnalysisResult:
    run_dir: Path
    dataset: str
    audit: AuditResult
    model_metrics: pd.DataFrame
    summary: pd.DataFrame


def _analysis_run_dir(output_root: Path, dataset: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"data-analysis-{dataset}-{stamp}"
    for child in (run_dir / "tables", run_dir / "figures", run_dir / "artifacts"):
        child.mkdir(parents=True, exist_ok=False)
    return run_dir


def _dataset_summary(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    users = frame["user_id"].nunique()
    items = frame["item_id"].nunique()
    interactions = len(frame)
    positive = int((frame["response"] > 0).sum())
    density = interactions / max(users * items, 1)
    return pd.DataFrame([{
        "dataset": dataset,
        "users": users,
        "items": items,
        "interactions": interactions,
        "positive_interactions": positive,
        "positive_rate": positive / max(interactions, 1),
        "density": density,
        "timestamps_available": not frame["timestamp"].isna().all(),
    }])


def _write_profile_assets(frame: pd.DataFrame, output_dir: Path, dataset: str) -> None:
    user_activity = frame.groupby("user_id").size().rename("interactions").reset_index()
    item_activity = frame.groupby("item_id").size().rename("interactions").reset_index()
    user_activity.to_csv(output_dir / "tables" / "data_table_user_activity.csv", index=False)
    item_activity.to_csv(output_dir / "tables" / "data_table_item_activity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(user_activity["interactions"], bins=min(40, max(5, int(np.sqrt(len(user_activity))))), color="#2E86AB")
    axes[0].set_title("User activity")
    axes[0].set_xlabel("Interactions per user")
    axes[0].set_ylabel("Count")
    axes[1].hist(item_activity["interactions"], bins=min(40, max(5, int(np.sqrt(len(item_activity))))), color="#E76F51")
    axes[1].set_title("Item activity / popularity")
    axes[1].set_xlabel("Interactions per item")
    axes[1].set_ylabel("Count")
    fig.suptitle(f"{dataset}: interaction-profile diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "data_figure_activity_distributions.png", dpi=180)
    plt.close(fig)


def analyze_dataset(
    result: DatasetLoadResult,
    *,
    output_root: str | Path,
    run_bpr: bool = True,
    bpr_updates: int = 50_000,
    max_eval_users: int = 1_000,
    seed: int = 42,
) -> DataAnalysisResult:
    """Profile data, audit evidence, and run all registered CPU baselines.

    The function never turns baseline ranking results into causal-policy evidence.
    Its purpose is data/model logic analysis before the CURE-Sim causal workflow.
    """
    frame = result.interactions.copy()
    audit = audit_interactions(frame)
    run_dir = _analysis_run_dir(Path(output_root), result.dataset)
    summary = _dataset_summary(frame, result.dataset)
    summary.to_csv(run_dir / "tables" / "data_table_summary.csv", index=False)
    _write_profile_assets(frame, run_dir, result.dataset)

    metrics_rows: list[dict] = []
    modeling_note = "Chronological model evaluation skipped: timestamps unavailable or insufficient positive history."
    if not frame["timestamp"].isna().all():
        try:
            split = chronological_leave_one_out(frame)
            popularity = PopularityRecommender().fit(split.train)
            metrics_rows.append(asdict(evaluate_leave_one_out(popularity, split, max_users=max_eval_users)))
            if run_bpr:
                bpr = BPRMFRecommender(max_updates=bpr_updates, seed=seed).fit(split.train)
                metrics_rows.append(asdict(evaluate_leave_one_out(bpr, split, max_users=max_eval_users)))
            modeling_note = "Chronological leave-one-out ranking evaluation completed."
        except ValueError as exc:
            modeling_note = f"Chronological model evaluation skipped: {exc}"
    metrics = pd.DataFrame(metrics_rows, columns=["model", "evaluated_users", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"])
    metrics.to_csv(run_dir / "tables" / "data_table_model_metrics.csv", index=False)
    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width / 2, metrics["recall_at_k"], width, label="Recall@10", color="#2E86AB")
        ax.bar(x + width / 2, metrics["ndcg_at_k"], width, label="NDCG@10", color="#2A9D8F")
        ax.set_xticks(x, metrics["model"])
        ax.set_ylim(0, 1)
        ax.set_title(f"{result.dataset}: baseline ranking models")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / "data_figure_model_metrics.png", dpi=180)
        plt.close(fig)

    manifest = {
        "dataset": result.dataset,
        "metadata": result.metadata,
        "audit": asdict(audit),
        "modeling_note": modeling_note,
        "run_bpr": run_bpr,
        "bpr_updates": bpr_updates,
        "max_eval_users": max_eval_users,
        "generated_tables": sorted(path.name for path in (run_dir / "tables").glob("*.csv")),
        "generated_figures": sorted(path.name for path in (run_dir / "figures").glob("*.png")),
    }
    (run_dir / "artifacts" / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return DataAnalysisResult(run_dir, result.dataset, audit, metrics, summary)
