from __future__ import annotations

import numpy as np

from cure_rec.interventions import Coalition, Proposal, resolve_injections, transform_slate
from cure_rec.policies import HistoryAwarePolicy
from cure_rec.simulator import CureSim


def test_collision_resolver_respects_capacity_and_unique_items():
    proposals = {
        "explore_slot": [Proposal("explore_slot", 2, 1.0, 1.0)],
        "tail_slot": [Proposal("tail_slot", 2, 3.0, 1.0), Proposal("tail_slot", 3, 2.0, 0.8)],
        "novel_slot": [Proposal("novel_slot", 4, 2.0, 0.9)],
    }
    selected, manifest = resolve_injections(proposals, capacity=2)
    assert len(selected) == 2
    assert len({proposal.item_id for proposal in selected}) == 2
    assert manifest["capacity"] == 2


def test_empty_proposal_set_is_explicit_noop():
    selected, manifest = resolve_injections({"tail_slot": []}, capacity=2)
    assert selected == []
    assert manifest["no_ops"] == ["tail_slot"]


def test_every_injection_coalition_returns_unique_complete_slate(settings):
    simulator = CureSim(settings, settings.scenarios[0])
    policy = HistoryAwarePolicy(simulator, settings.policy)
    state = simulator.reset()
    coalition = Coalition(frozenset({"repeat_cap", "explore_slot", "tail_slot", "novel_slot", "diversify", "provider_balance"}))
    result = transform_slate(policy, state, 0, coalition, settings.interventions, np.random.default_rng(7))
    assert len(result.slate) == settings.simulator.slate_size
    assert len(set(result.slate)) == settings.simulator.slate_size
    assert result.manifest["collision"] is not None
