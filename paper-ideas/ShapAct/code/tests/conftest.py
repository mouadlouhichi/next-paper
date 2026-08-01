"""Shared fixtures for the ShapAct test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results" / "raw"


@pytest.fixture(scope="session")
def real_audit():
    """The ML-1M audit result, if present.

    Integration tests (fidelity invariant, reflexivity identity) are skipped
    when the pipeline has not been run yet, so the unit suite stays green on
    a fresh checkout.
    """
    p = RESULTS / "audit_ml1m_seed42.json"
    if not p.exists():
        pytest.skip("run scripts/run_all.py first to produce the ML-1M audit")
    with open(p) as f:
        return json.load(f)
