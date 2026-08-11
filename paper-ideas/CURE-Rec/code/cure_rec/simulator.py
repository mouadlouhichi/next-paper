"""CURE-Sim: a small, disclosed, cohort-level sequential recommendation SCM.

The simulator is an oracle benchmark, not a claim about any production platform.
It intentionally exposes feedback, fatigue, popularity, and provider outcomes so
CURE-Rec's coalition and attribution logic can be tested against known dynamics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cure_rec.config import ScenarioConfig, Settings


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.allclose(values, 0.0):
        return 0.0
    sorted_values = np.sort(np.maximum(values, 0.0))
    n = sorted_values.size
    return float((2.0 * np.dot(np.arange(1, n + 1), sorted_values) / (n * sorted_values.sum())) - (n + 1) / n)


@dataclass
class Catalog:
    features: np.ndarray
    providers: np.ndarray
    categories: np.ndarray
    base_quality: np.ndarray


@dataclass
class PlatformState:
    """State exposed to a policy.

    `hidden_interest` is retained in the simulator for outcome generation only.
    Policies should use `public_profiles`, exposure history, and catalogue state.
    """

    public_profiles: np.ndarray
    hidden_interest: np.ndarray
    exposure_counts: np.ndarray
    fatigue: np.ndarray
    item_popularity: np.ndarray
    provider_exposure: np.ndarray
    step: int = 0
    traces: list[dict] = field(default_factory=list)


@dataclass
class RolloutSummary:
    utility_before_cost: float
    satisfaction: float
    retention: float
    fatigue: float
    relevance: float
    provider_disparity: float
    catalog_coverage: float
    trajectories: int
    intervention_stats: dict[str, int]
    step_summaries: list[dict]


PolicyFn = Callable[[PlatformState, int, np.random.Generator], tuple[list[int], dict]]


class CureSim:
    """Finite-horizon recommendation environment with cohort-level outcomes."""

    def __init__(self, settings: Settings, scenario: ScenarioConfig):
        self.settings = settings
        self.scenario = scenario
        cfg = settings.simulator
        self.rng_offset = 0
        self.rng = np.random.default_rng(settings.run.seed + self._scenario_offset(scenario.name) + self.rng_offset)
        self.catalog = self._build_catalog()
        self.initial_state = self._build_state()
        self.state = copy.deepcopy(self.initial_state)

    @staticmethod
    def _scenario_offset(name: str) -> int:
        return sum((index + 1) * ord(char) for index, char in enumerate(name)) % 100_000

    def _build_catalog(self) -> Catalog:
        cfg = self.settings.simulator
        features = self.rng.normal(0.0, 1.0, size=(cfg.n_items, cfg.embedding_dim))
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
        providers = self.rng.integers(0, cfg.n_providers, size=cfg.n_items)
        categories = self.rng.integers(0, cfg.n_categories, size=cfg.n_items)
        base_quality = self.rng.normal(0.0, 0.5, size=cfg.n_items)
        return Catalog(features=features, providers=providers, categories=categories, base_quality=base_quality)

    def _build_state(self) -> PlatformState:
        cfg = self.settings.simulator
        hidden_interest = self.rng.normal(0.0, 1.0, size=(cfg.n_users, cfg.embedding_dim))
        hidden_interest /= np.maximum(np.linalg.norm(hidden_interest, axis=1, keepdims=True), 1e-8)
        public_profiles = hidden_interest + self.rng.normal(0.0, 0.35, size=hidden_interest.shape)
        public_profiles /= np.maximum(np.linalg.norm(public_profiles, axis=1, keepdims=True), 1e-8)
        item_popularity = self.rng.lognormal(mean=-1.0, sigma=0.7, size=cfg.n_items)
        return PlatformState(
            public_profiles=public_profiles,
            hidden_interest=hidden_interest,
            exposure_counts=np.zeros((cfg.n_users, cfg.n_items), dtype=np.int16),
            fatigue=np.zeros(cfg.n_users, dtype=float),
            item_popularity=item_popularity,
            provider_exposure=np.zeros(cfg.n_providers, dtype=float),
        )

    def reset(self) -> PlatformState:
        self.state = copy.deepcopy(self.initial_state)
        self.rng = np.random.default_rng(self.settings.run.seed + self._scenario_offset(self.scenario.name) + self.rng_offset)
        return self.state

    def clone(self) -> "CureSim":
        return copy.deepcopy(self)

    def available_items(self, user_id: int) -> np.ndarray:
        # All catalogue items are available in the MVP. The interface leaves room
        # for temporal availability and provider inventories.
        return np.arange(self.settings.simulator.n_items, dtype=int)

    def expected_relevance(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        hidden = self.state.hidden_interest[user_id]
        affinity = self.catalog.features[item_ids] @ hidden
        return sigmoid(1.5 * affinity + self.catalog.base_quality[item_ids])

    def _respond_to_slate(self, user_id: int, slate: list[int]) -> dict:
        cfg = self.settings.simulator
        scenario = self.scenario
        items = np.asarray(slate, dtype=int)
        positions = np.arange(len(items), dtype=float)
        affinity = self.catalog.features[items] @ self.state.hidden_interest[user_id]
        repeat_count = self.state.exposure_counts[user_id, items].astype(float)
        repeated = (repeat_count >= cfg.repeat_fatigue_threshold).astype(float)
        position_bonus = 1.0 / np.sqrt(positions + 1.0)
        popularity_signal = np.log1p(self.state.item_popularity[items])

        response_logit = (
            1.6 * affinity
            + 0.45 * position_bonus
            + 0.15 * popularity_signal
            - 0.85 * cfg.fatigue_multiplier * scenario.fatigue_multiplier * self.state.fatigue[user_id]
            - 0.55 * repeated
            + cfg.satisfaction_shift + scenario.satisfaction_shift
        )
        response_probability = sigmoid(response_logit)
        clicks = self.rng.binomial(1, response_probability)

        satisfaction = sigmoid(1.8 * affinity - 0.7 * repeated - 0.4 * self.state.fatigue[user_id])
        relevance = sigmoid(1.7 * affinity + self.catalog.base_quality[items])
        mean_satisfaction = float(np.mean(satisfaction))
        mean_relevance = float(np.mean(relevance))
        return {
            "items": items,
            "clicks": clicks,
            "satisfaction": mean_satisfaction,
            "relevance": mean_relevance,
            "repeat_rate": float(np.mean(repeated)),
        }

    def step(self, policy: PolicyFn) -> dict:
        cfg = self.settings.simulator
        user_records: list[dict] = []
        intervention_stats: dict[str, int] = {}
        for user_id in range(cfg.n_users):
            slate, transform_info = policy(self.state, user_id, self.rng)
            if len(slate) != cfg.slate_size or len(set(slate)) != len(slate):
                raise ValueError("Policy must return a unique slate with exactly slate_size items")
            response = self._respond_to_slate(user_id, slate)
            items = response["items"]
            self.state.exposure_counts[user_id, items] += 1
            np.add.at(self.state.provider_exposure, self.catalog.providers[items], 1.0)
            self.state.item_popularity[items] += scenario_scale(self.scenario.popularity_feedback_multiplier, cfg.popularity_feedback)
            # Fatigue rises with repeated exposure and slowly recovers.
            self.state.fatigue[user_id] = np.clip(
                0.88 * self.state.fatigue[user_id] + 0.22 * response["repeat_rate"], 0.0, 1.0
            )
            # A small satisfaction-mediated preference drift creates delayed effects.
            direction = self.catalog.features[items].mean(axis=0)
            updated = self.state.hidden_interest[user_id] + 0.03 * response["satisfaction"] * direction
            # This optional term is zero in the archived baseline configuration.
            # When varied in calibration it encodes a disclosed behavioral
            # assumption: exposure to items unlike the public profile can create
            # delayed preference broadening rather than an immediate click gain.
            profile_similarity = self.catalog.features[items] @ self.state.public_profiles[user_id]
            novelty = np.clip(-profile_similarity, 0.0, None)
            if novelty.any() and cfg.novelty_preference_drift > 0:
                novelty_direction = (self.catalog.features[items] * novelty[:, None]).sum(axis=0)
                novelty_direction /= max(float(novelty.sum()), 1e-8)
                updated += cfg.novelty_preference_drift * novelty_direction
            self.state.hidden_interest[user_id] = updated / max(np.linalg.norm(updated), 1e-8)
            for key, value in transform_info.get("stats", {}).items():
                intervention_stats[key] = intervention_stats.get(key, 0) + int(value)
            user_records.append({"user_id": user_id, **response, "transform": transform_info})

        provider_disparity = gini(self.state.provider_exposure)
        step_summary = {
            "step": self.state.step,
            "mean_satisfaction": float(np.mean([record["satisfaction"] for record in user_records])),
            "mean_relevance": float(np.mean([record["relevance"] for record in user_records])),
            "mean_fatigue": float(np.mean(self.state.fatigue)),
            "provider_disparity": provider_disparity,
            "catalog_coverage": float(np.count_nonzero(self.state.item_popularity > self.initial_state.item_popularity) / cfg.n_items),
            "intervention_stats": intervention_stats,
            # A bounded trace makes operator/collision behavior inspectable without
            # writing every user-level slate into the default run artifacts.
            "transform_samples": [record["transform"] for record in user_records[: min(3, len(user_records))]],
        }
        self.state.traces.append(step_summary)
        self.state.step += 1
        return step_summary

    def rollout(self, policy: PolicyFn) -> RolloutSummary:
        self.reset()
        cfg = self.settings.simulator
        utility_cfg = self.settings.utility
        summaries = [self.step(policy) for _ in range(cfg.horizon)]
        discounts = np.asarray([utility_cfg.gamma**step for step in range(cfg.horizon)])
        satisfaction = float(np.average([summary["mean_satisfaction"] for summary in summaries], weights=discounts))
        relevance = float(np.average([summary["mean_relevance"] for summary in summaries], weights=discounts))
        fatigue = float(np.average([summary["mean_fatigue"] for summary in summaries], weights=discounts))
        provider_disparity = float(summaries[-1]["provider_disparity"])
        coverage = float(summaries[-1]["catalog_coverage"])
        retention = float(np.clip(1.0 - fatigue + 0.25 * satisfaction, 0.0, 1.0))
        utility_before_cost = (
            utility_cfg.satisfaction_weight * satisfaction
            + utility_cfg.retention_weight * retention
            - utility_cfg.fatigue_weight * fatigue
        )
        aggregate_stats: dict[str, int] = {}
        for summary in summaries:
            for key, value in summary["intervention_stats"].items():
                aggregate_stats[key] = aggregate_stats.get(key, 0) + int(value)
        return RolloutSummary(
            utility_before_cost=float(utility_before_cost),
            satisfaction=satisfaction,
            retention=retention,
            fatigue=fatigue,
            relevance=relevance,
            provider_disparity=provider_disparity,
            catalog_coverage=coverage,
            trajectories=cfg.n_users * cfg.horizon,
            intervention_stats=aggregate_stats,
            step_summaries=summaries,
        )


def scenario_scale(multiplier: float, base: float) -> float:
    return float(multiplier * base)
