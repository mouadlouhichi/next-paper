"""Typed configuration, validation, and reproducibility hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, field_validator, model_validator


INTERVENTION_NAMES = (
    "repeat_cap",
    "explore_slot",
    "tail_slot",
    "diversify",
    "novel_slot",
    "provider_balance",
)


class RunConfig(BaseModel):
    name: str = "cure-rec"
    seed: int = 42
    output_root: Path = Path("runs")
    log_level: str = "INFO"
    # True shares event-indexed shocks across coalitions; False gives every
    # coalition an independent shock stream for the CRN variance ablation.
    common_random_numbers: bool = True


class SimulatorConfig(BaseModel):
    n_users: PositiveInt = 48
    n_items: PositiveInt = 96
    n_providers: PositiveInt = 8
    n_categories: PositiveInt = 6
    embedding_dim: PositiveInt = 6
    horizon: PositiveInt = 6
    slate_size: PositiveInt = 8
    repeat_fatigue_threshold: PositiveInt = 2
    popularity_feedback: float = Field(default=0.03, ge=0.0, le=1.0)
    fatigue_multiplier: PositiveFloat = 1.0
    # Delayed preference drift from novel exposure. Zero preserves the frozen
    # historical CURE-Sim behavior; calibration can vary this assumption.
    novelty_preference_drift: float = Field(default=0.0, ge=0.0, le=1.0)
    satisfaction_shift: float = 0.0
    click_feedback_weight: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def item_count_supports_slate(self) -> "SimulatorConfig":
        if self.n_items <= self.slate_size:
            raise ValueError("n_items must exceed slate_size")
        if self.n_providers > self.n_items or self.n_categories > self.n_items:
            raise ValueError("n_providers and n_categories cannot exceed n_items")
        return self


class PolicyConfig(BaseModel):
    popularity_weight: float = Field(default=0.35, ge=0.0, le=2.0)
    profile_weight: float = Field(default=1.0, ge=0.0, le=3.0)
    candidate_pool_size: PositiveInt = 48


class InterventionConfig(BaseModel):
    repeat_cap: PositiveInt = 2
    injection_capacity: PositiveInt = 2
    proposal_top_n: PositiveInt = 6
    long_tail_quantile: float = Field(default=0.35, gt=0.0, lt=1.0)
    diversity_weight: float = Field(default=0.45, ge=0.0, le=2.0)
    provider_balance_weight: float = Field(default=0.35, ge=0.0, le=2.0)
    exploration_temperature: PositiveFloat = 0.25
    novelty_threshold: float = Field(default=0.15, ge=-1.0, le=1.0)
    costs: dict[str, float] = Field(default_factory=lambda: {name: 0.0 for name in INTERVENTION_NAMES})

    @field_validator("costs")
    @classmethod
    def complete_costs(cls, value: dict[str, float]) -> dict[str, float]:
        missing = set(INTERVENTION_NAMES).difference(value)
        extra = set(value).difference(INTERVENTION_NAMES)
        if missing or extra:
            raise ValueError(f"costs must contain exactly {INTERVENTION_NAMES}; missing={missing}, extra={extra}")
        if any(cost < 0 for cost in value.values()):
            raise ValueError("intervention costs must be non-negative")
        return value


class ConstraintConfig(BaseModel):
    budget: float = Field(default=0.35, ge=0.0)
    min_relevance_delta: float = Field(default=-0.10, le=1.0, ge=-1.0)
    max_provider_disparity: float = Field(default=0.30, ge=0.0, le=1.0)
    max_fatigue: float = Field(default=0.70, ge=0.0, le=1.0)


class UtilityConfig(BaseModel):
    gamma: float = Field(default=0.95, gt=0.0, le=1.0)
    satisfaction_weight: float = Field(default=0.70, ge=0.0)
    retention_weight: float = Field(default=0.30, ge=0.0)
    fatigue_weight: float = Field(default=0.35, ge=0.0)
    cost_weight: float = Field(default=1.0, ge=0.0)


class ScenarioConfig(BaseModel):
    name: str
    fatigue_multiplier: PositiveFloat = 1.0
    popularity_feedback_multiplier: PositiveFloat = 1.0
    satisfaction_shift: float = 0.0
    click_feedback_weight: float = Field(default=0.0, ge=0.0, le=1.0)


class Settings(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    interventions: InterventionConfig = Field(default_factory=InterventionConfig)
    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    utility: UtilityConfig = Field(default_factory=UtilityConfig)
    scenarios: list[ScenarioConfig] = Field(default_factory=lambda: [ScenarioConfig(name="nominal")])

    def canonical_dict(self) -> dict[str, Any]:
        """Stable JSON-compatible representation used in manifests and cache keys."""
        return json.loads(self.model_dump_json())

    def config_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def load_settings(path: str | Path) -> Settings:
    """Load and validate a YAML experiment configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    settings = Settings.model_validate(payload)
    return settings
