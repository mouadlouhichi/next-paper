from __future__ import annotations

import json

import pandas as pd

from cure_rec.data import audit_interactions
from cure_rec.observability import RunLogger


def test_audit_blocks_causal_claim_without_slate_and_propensity():
    frame = pd.DataFrame({
        "user_id": [1],
        "item_id": [3],
        "timestamp": [1],
        "response": [1],
    })
    audit = audit_interactions(frame)
    assert audit.permitted_claim == "descriptive_or_semisynthetic"
    assert not audit.supports_causal_ope


def test_run_logger_writes_manifest_and_jsonl(settings, tmp_path):
    settings.run.output_root = tmp_path
    logger = RunLogger(settings)
    logger.event("unit_test_event", coalition_mask=3)
    logger.close()
    manifest = json.loads((logger.run_dir / "manifest.json").read_text())
    assert manifest["config_hash"] == settings.config_hash()
    events = (logger.run_dir / "logs" / "events.jsonl").read_text().splitlines()
    assert any(json.loads(line)["event"] == "unit_test_event" for line in events)
