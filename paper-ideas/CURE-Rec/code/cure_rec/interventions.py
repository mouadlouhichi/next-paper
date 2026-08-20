"""Composable recommendation-policy intervention operators.

The operators implement the canonical CURE-Rec order:
repeat cap -> eligibility -> injection allocation -> diversity -> provider balancing.
The Shapley permutation never changes this order; a coalition mask always means
one immutable policy transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

from cure_rec.config import INTERVENTION_NAMES, InterventionConfig
from cure_rec.policies import HistoryAwarePolicy
from cure_rec.simulator import PlatformState


INJECTION_NAMES = ("explore_slot", "tail_slot", "novel_slot")
CANONICAL_ORDER = (
    "repeat_cap",
    "eligibility_filter",
    "injection_allocation",
    "diversify",
    "provider_balance",
)

# Extended player library for the integrated scalability revision. The four
# additional operators are appended after the six core players, so all masks of
# the core library keep their original meaning. Their semantics are disclosed:
#   session_length_cap   demote items already present in the user's previous slate
#                        (windowed, one-step repeat control complementing the
#                        lifetime repeat cap);
#   freshness_quota      guarantee at least `freshness_quota` slate positions for
#                        items never exposed to the user (least-exposed fallback);
#   provider_cooldown    demote items of any provider that occupied at least half
#                        of the user's previous slate (concentration cooldown);
#   category_coverage_quota  greedily enforce at least `category_coverage_quota`
#                        distinct categories in the final slate.
EXTRA_PLAYER_NAMES = (
    "session_length_cap",
    "freshness_quota",
    "provider_cooldown",
    "category_coverage_quota",
)
EXTENDED_CANONICAL_ORDER = CANONICAL_ORDER + (
    "session_length_cap",
    "provider_cooldown",
    "freshness_quota",
    "category_coverage_quota",
)


@dataclass(frozen=True)
class Coalition:
    active: frozenset[str]

    def __post_init__(self) -> None:
        # Core and extended player names are both accepted; mask semantics are
        # determined by the name tuple passed to from_mask/names/mask.
        unknown = self.active.difference(tuple(INTERVENTION_NAMES) + EXTRA_PLAYER_NAMES)
        if unknown:
            raise ValueError(f"Unknown interventions: {sorted(unknown)}")

    @classmethod
    def from_mask(cls, mask: int, names: tuple[str, ...] = INTERVENTION_NAMES) -> "Coalition":
        return cls(frozenset(name for index, name in enumerate(names) if mask & (1 << index)))

    def mask(self, names: tuple[str, ...] = INTERVENTION_NAMES) -> int:
        return sum(1 << index for index, name in enumerate(names) if name in self.active)

    def cost(self, config: InterventionConfig) -> float:
        return float(sum(config.costs[name] for name in self.active))

    def names(self, names: tuple[str, ...] = INTERVENTION_NAMES) -> tuple[str, ...]:
        return tuple(name for name in names if name in self.active)


@dataclass(frozen=True)
class Proposal:
    intervention: str
    item_id: int
    raw_score: float
    normalized_score: float


@dataclass
class TransformResult:
    slate: list[int]
    manifest: dict


def percentile_scores(items: np.ndarray, raw_scores: np.ndarray, intervention: str) -> list[Proposal]:
    """Return proposals with score scales normalized within each intervention."""
    if len(items) == 0:
        return []
    order = np.lexsort((items, raw_scores))
    ranks = np.empty(len(items), dtype=float)
    ranks[order] = np.arange(len(items), dtype=float)
    percentiles = (ranks + 1.0) / len(items)
    return [
        Proposal(intervention=intervention, item_id=int(item), raw_score=float(raw), normalized_score=float(score))
        for item, raw, score in zip(items, raw_scores, percentiles, strict=True)
    ]


def _top_proposals(proposals: list[Proposal], top_n: int) -> list[Proposal]:
    ordered = sorted(proposals, key=lambda p: (-p.normalized_score, p.item_id))
    return ordered[:top_n]


def _proposal_options(proposals: list[Proposal]) -> list[Proposal | None]:
    return [None, *proposals]


def resolve_injections(
    proposal_sets: dict[str, list[Proposal]],
    capacity: int,
) -> tuple[list[Proposal], dict]:
    """Exact small assignment for simultaneous slot interventions.

    One candidate can be assigned to at most one intervention. `None` is a valid
    choice, so a coalition remains defined even when an intervention has no
    eligible candidate or capacity is exhausted.
    """
    names = tuple(sorted(proposal_sets))
    choices = [_proposal_options(proposal_sets[name]) for name in names]
    best: tuple[float, tuple[Proposal | None, ...]] | None = None
    for assignment in product(*choices):
        selected = [proposal for proposal in assignment if proposal is not None]
        item_ids = [proposal.item_id for proposal in selected]
        if len(selected) > capacity or len(set(item_ids)) != len(item_ids):
            continue
        score = float(sum(proposal.normalized_score for proposal in selected))
        tie_key = tuple((proposal.item_id, proposal.intervention) for proposal in selected)
        candidate = (score, assignment)
        if best is None:
            best = candidate
        else:
            best_score, best_assignment = best
            best_tie = tuple((proposal.item_id, proposal.intervention) for proposal in best_assignment if proposal is not None)
            if score > best_score or (np.isclose(score, best_score) and tie_key < best_tie):
                best = candidate
    assert best is not None
    selected = [proposal for proposal in best[1] if proposal is not None]
    no_ops = [name for name in names if not any(p.intervention == name for p in selected)]
    return selected, {
        "selected": [proposal.__dict__ for proposal in selected],
        "no_ops": no_ops,
        "capacity": capacity,
        "proposal_counts": {name: len(values) for name, values in proposal_sets.items()},
    }


def _first_unique(items: Iterable[int], k: int) -> list[int]:
    chosen: list[int] = []
    seen: set[int] = set()
    for item in items:
        item = int(item)
        if item not in seen:
            chosen.append(item)
            seen.add(item)
        if len(chosen) == k:
            break
    return chosen


def _diversify(ranked: list[int], categories: np.ndarray, k: int, weight: float) -> list[int]:
    """Greedy category-aware re-ranking over a fixed ranked candidate pool."""
    if not ranked:
        return []
    selected = [ranked[0]]
    remaining = ranked[1:]
    while remaining and len(selected) < k:
        best_item = None
        best_score = -np.inf
        for rank, item in enumerate(remaining):
            novelty = float(np.mean(categories[item] != categories[np.asarray(selected, dtype=int)]))
            score = -rank + weight * novelty * len(ranked)
            if score > best_score or (np.isclose(score, best_score) and (best_item is None or item < best_item)):
                best_item, best_score = item, score
        selected.append(int(best_item))
        remaining.remove(best_item)
    return selected


def _provider_balance(ranked: list[int], providers: np.ndarray, provider_exposure: np.ndarray, k: int, weight: float) -> list[int]:
    """Greedy re-ranking that favors under-exposed providers within a bounded slate."""
    if not ranked:
        return []
    provider_scale = provider_exposure / max(float(provider_exposure.max()), 1.0)
    selected: list[int] = []
    remaining = list(ranked)
    while remaining and len(selected) < k:
        def score(index_item: tuple[int, int]) -> tuple[float, int]:
            index, item = index_item
            penalty = weight * provider_scale[providers[item]] * len(ranked)
            return (-index - penalty, -item)
        _, item = max(enumerate(remaining), key=score)
        selected.append(int(item))
        remaining.remove(item)
    return selected


def _session_length_cap(ranked: list[int], previous_slate: list[int]) -> list[int]:
    """Demote items already shown in the user's previous slate (one-step turnover)."""
    if not previous_slate:
        return list(ranked)
    previous = set(int(item) for item in previous_slate)
    fresh = [item for item in ranked if int(item) not in previous]
    stale = [item for item in ranked if int(item) in previous]
    return fresh + stale


def _provider_cooldown(ranked: list[int], providers: np.ndarray, previous_slate: list[int], slate_size: int) -> list[int]:
    """Demote items of providers that concentrated the previous slate (>= half)."""
    if not previous_slate:
        return list(ranked)
    counts: dict[int, int] = {}
    for item in previous_slate:
        provider = int(providers[int(item)])
        counts[provider] = counts.get(provider, 0) + 1
    hot = {provider for provider, count in counts.items() if count >= max(1, slate_size // 2)}
    if not hot:
        return list(ranked)
    cool = [item for item in ranked if int(providers[int(item)]) not in hot]
    hot_items = [item for item in ranked if int(providers[int(item)]) in hot]
    return cool + hot_items


def _freshness_quota(slate: list[int], ranked: list[int], exposure_row: np.ndarray, quota: int) -> list[int]:
    """Swap trailing slate items for never-exposed (then least-exposed) candidates."""
    if quota <= 0:
        return slate
    slate = list(slate)
    in_slate = set(int(item) for item in slate)
    never = [int(item) for item in ranked if int(item) not in in_slate and exposure_row[int(item)] == 0]
    if len(never) < quota:
        exposed_sorted = sorted(
            (int(item) for item in ranked if int(item) not in in_slate and exposure_row[int(item)] > 0),
            key=lambda item: (float(exposure_row[item]), item),
        )
        never = never + exposed_sorted
    candidates = never[: max(0, quota)]
    if not candidates:
        return slate
    # Replace exposed items from the tail first, then any remaining positions.
    replaced = 0
    for index in range(len(slate) - 1, -1, -1):
        if replaced >= len(candidates):
            break
        if exposure_row[int(slate[index])] > 0:
            slate[index] = candidates[replaced]
            replaced += 1
    for index in range(len(slate) - 1, -1, -1):
        if replaced >= len(candidates):
            break
        slate[index] = candidates[replaced]
        replaced += 1
    return _first_unique(slate, len(slate))


def _category_coverage_quota(slate: list[int], ranked: list[int], categories: np.ndarray, quota: int) -> list[int]:
    """Greedily enforce a minimum number of distinct categories in the slate."""
    if quota <= 1:
        return slate
    slate = list(slate)

    def distinct(items: list[int]) -> set[int]:
        return {int(categories[int(item)]) for item in items}

    if len(distinct(slate)) >= quota:
        return slate
    present = distinct(slate)
    replacements = [int(item) for item in ranked if int(categories[int(item)]) not in present]
    seen_replacement_categories: set[int] = set()
    for position in range(len(slate) - 1, -1, -1):
        if len(present) >= quota or not replacements:
            break
        category_of_position = int(categories[int(slate[position])])
        duplicated = sum(1 for item in slate if int(categories[int(item)]) == category_of_position) > 1
        if not duplicated:
            continue
        while replacements:
            candidate = replacements.pop(0)
            candidate_category = int(categories[candidate])
            if candidate_category in present or candidate_category in seen_replacement_categories:
                continue
            slate[position] = candidate
            present.add(candidate_category)
            seen_replacement_categories.add(candidate_category)
            break
    return slate


def transform_slate(
    policy: HistoryAwarePolicy,
    state: PlatformState,
    user_id: int,
    coalition: Coalition,
    config: InterventionConfig,
    rng: np.random.Generator,
) -> TransformResult:
    """Apply one coalition with fully deterministic semantics for a given RNG state."""
    ranked = policy.rank_items(state, user_id).tolist()
    slate_size = policy.simulator.settings.simulator.slate_size
    stats: dict[str, int] = {"repeat_cap_removed": 0, "injection_noop": 0}
    manifest: dict = {
        "active_interventions": coalition.names(),
        "canonical_order": CANONICAL_ORDER,
        "user_id": user_id,
        "collision": None,
        "stats": stats,
    }

    # 1. repeat cap
    if "repeat_cap" in coalition.active:
        allowed = [item for item in ranked if state.exposure_counts[user_id, item] < config.repeat_cap]
        stats["repeat_cap_removed"] = len(ranked) - len(allowed)
        ranked = allowed + [item for item in ranked if item not in set(allowed)]

    # 2. top base slate, with unique IDs
    base_slate = _first_unique(ranked, slate_size)

    # 3. injection proposal construction and exact collision allocation
    active_injections = [name for name in INJECTION_NAMES if name in coalition.active]
    proposals: dict[str, list[Proposal]] = {}
    candidate_pool = np.asarray(ranked[: policy.config.candidate_pool_size], dtype=int)
    candidate_pool = candidate_pool[~np.isin(candidate_pool, base_slate)]
    if active_injections:
        if "explore_slot" in active_injections:
            uncertainty = 1.0 / (1.0 + state.exposure_counts[user_id, candidate_pool].astype(float))
            # Small seeded perturbation only breaks exact score ties reproducibly.
            raw = uncertainty + rng.uniform(0.0, 1e-9, size=len(candidate_pool)) * config.exploration_temperature
            proposals["explore_slot"] = _top_proposals(percentile_scores(candidate_pool, raw, "explore_slot"), config.proposal_top_n)
        if "tail_slot" in active_injections:
            threshold = np.quantile(state.item_popularity, config.long_tail_quantile)
            items = candidate_pool[state.item_popularity[candidate_pool] <= threshold]
            raw = -state.item_popularity[items]
            proposals["tail_slot"] = _top_proposals(percentile_scores(items, raw, "tail_slot"), config.proposal_top_n)
        if "novel_slot" in active_injections:
            similarities = policy.simulator.catalog.features[candidate_pool] @ state.public_profiles[user_id]
            items = candidate_pool[similarities <= config.novelty_threshold]
            raw = -similarities[similarities <= config.novelty_threshold]
            proposals["novel_slot"] = _top_proposals(percentile_scores(items, raw, "novel_slot"), config.proposal_top_n)

        selected, collision = resolve_injections(proposals, config.injection_capacity)
        manifest["collision"] = collision
        stats["injection_noop"] = len(collision["no_ops"])
        injected = [proposal.item_id for proposal in selected]
        base_without_injected = [item for item in base_slate if item not in injected]
        keep = max(0, slate_size - len(injected))
        base_slate = _first_unique([*base_without_injected[:keep], *injected, *ranked], slate_size)

    # 4. diversity
    if "diversify" in coalition.active:
        candidate = _first_unique([*base_slate, *ranked], max(slate_size * 3, slate_size))
        base_slate = _diversify(candidate, policy.simulator.catalog.categories, slate_size, config.diversity_weight)

    # 5. provider exposure balancing
    if "provider_balance" in coalition.active:
        candidate = _first_unique([*base_slate, *ranked], max(slate_size * 3, slate_size))
        base_slate = _provider_balance(
            candidate,
            policy.simulator.catalog.providers,
            state.provider_exposure,
            slate_size,
            config.provider_balance_weight,
        )

    # 6-9. extended-library operators; inactive for the core six-player library.
    previous_slate = state.last_slates.get(user_id, [])
    if "session_length_cap" in coalition.active:
        before = list(base_slate)
        base_slate = _session_length_cap(base_slate, previous_slate)
        stats["session_cap_demoted"] = sum(1 for item in before if int(item) in set(map(int, previous_slate)))
    if "provider_cooldown" in coalition.active:
        base_slate = _provider_cooldown(base_slate, policy.simulator.catalog.providers, previous_slate, slate_size)
    if "freshness_quota" in coalition.active:
        base_slate = _freshness_quota(base_slate, ranked, state.exposure_counts[user_id], config.freshness_quota)
        stats["freshness_quota"] = config.freshness_quota
    if "category_coverage_quota" in coalition.active:
        base_slate = _category_coverage_quota(base_slate, ranked, policy.simulator.catalog.categories, config.category_coverage_quota)
        stats["category_coverage_quota"] = config.category_coverage_quota

    slate = _first_unique([*base_slate, *ranked], slate_size)
    if len(slate) != slate_size:
        raise RuntimeError("Could not construct a complete slate; check availability semantics")
    return TransformResult(slate=slate, manifest=manifest)
