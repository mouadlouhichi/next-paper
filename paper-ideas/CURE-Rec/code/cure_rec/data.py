"""Data loading and auditable schema checks for synthetic and local logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from cure_rec.observability import RunLogger


REQUIRED_INTERACTION_COLUMNS = {
    "user_id",
    "item_id",
    "timestamp",
    "response",
}

CAUSAL_LOG_COLUMNS = {
    "displayed_slate",
    "propensity",
    "position",
    "candidate_set",
}


@dataclass(frozen=True)
class AuditResult:
    rows: int
    columns: list[str]
    missing_required: list[str]
    missing_causal: list[str]
    duplicate_rows: int
    null_counts: dict[str, int]
    permitted_claim: str
    notes: list[str]

    @property
    def supports_causal_ope(self) -> bool:
        return self.permitted_claim in {"short_horizon_ope", "sensitivity_bounded"}


def load_interactions_csv(path: str | Path, logger: RunLogger | None = None) -> pd.DataFrame:
    """Load a local CSV without assuming it supports causal claims."""
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    if logger:
        logger.event("data_loaded", source=str(csv_path), rows=len(frame), columns=list(frame.columns))
    return frame


def audit_interactions(frame: pd.DataFrame, logger: RunLogger | None = None) -> AuditResult:
    """Classify a local log by the strongest claim its fields can support.

    This is intentionally conservative. Passing this check does not prove causal
    identification; it prevents accidental causal language for logs that plainly
    lack treatment timing, slates, or policy-propensity information.
    """
    columns = set(frame.columns)
    missing_required = sorted(REQUIRED_INTERACTION_COLUMNS.difference(columns))
    missing_causal = sorted(CAUSAL_LOG_COLUMNS.difference(columns))
    notes: list[str] = []
    if missing_required:
        permitted_claim = "descriptive_only"
        notes.append("Missing core interaction fields; data cannot be used as a standard recommendation trajectory.")
    elif missing_causal:
        permitted_claim = "descriptive_or_semisynthetic"
        notes.append("Missing logged slate/propensity fields; do not make offline causal-policy claims.")
    else:
        permitted_claim = "short_horizon_ope"
        notes.append("Slate and propensity fields are present; sequential timing, overlap, and estimator diagnostics remain required.")
        if "policy_mixture_assignment" in columns:
            permitted_claim = "sensitivity_bounded"
            notes.append("Policy-mixture assignment is present; Gamma-sensitivity analysis may be possible after treatment-unit audit.")

    null_counts = {column: int(frame[column].isna().sum()) for column in frame.columns if frame[column].isna().any()}
    result = AuditResult(
        rows=int(len(frame)),
        columns=sorted(columns),
        missing_required=missing_required,
        missing_causal=missing_causal,
        duplicate_rows=int(frame.duplicated().sum()),
        null_counts=null_counts,
        permitted_claim=permitted_claim,
        notes=notes,
    )
    if logger:
        logger.event("data_audited", **result.__dict__)
        logger.write_json("artifacts/data_audit.json", result.__dict__)
    return result


def to_temporal_groups(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> Iterable[pd.DataFrame]:
    """Yield stable timestamp groups for lightweight local inspection."""
    if timestamp_column not in frame.columns:
        raise ValueError(f"Missing {timestamp_column!r}")
    sorted_frame = frame.sort_values(timestamp_column, kind="stable")
    for _, group in sorted_frame.groupby(timestamp_column, sort=True):
        yield group


def load_curesim(settings, scenario_name: str | None = None, logger: RunLogger | None = None):
    """Instantiate the disclosed synthetic data-generating environment.

    This mirrors a loader rather than hiding CURE-Sim creation in a notebook.
    It is deliberately separate from real-log loading because simulator truth and
    logged-policy evidence support different causal claims.
    """
    from cure_rec.simulator import CureSim

    scenario = next(
        (candidate for candidate in settings.scenarios if candidate.name == scenario_name),
        settings.scenarios[0],
    )
    simulator = CureSim(settings, scenario)
    if logger:
        logger.event(
            "synthetic_data_loaded",
            source="CURE-Sim",
            scenario=scenario.name,
            n_users=settings.simulator.n_users,
            n_items=settings.simulator.n_items,
            horizon=settings.simulator.horizon,
        )
    return simulator
