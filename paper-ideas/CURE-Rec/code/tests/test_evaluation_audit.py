from __future__ import annotations

import pandas as pd

from cure_rec.evaluation_audit import audit_evaluation
from cure_rec.models import BPRMFRecommender, PopularityRecommender, chronological_leave_one_out


def _temporal_fixture() -> pd.DataFrame:
    rows = []
    for user in range(4):
        for offset, item in enumerate([0, 1, 2, 3]):
            rows.append({
                "user_id": user,
                "item_id": (item + user) % 6,
                "rating": 5,
                "response": 1,
                "timestamp": 100 + offset,
                "split": "observed",
                "source_dataset": "fixture",
            })
    return pd.DataFrame(rows)


def test_shared_candidate_audit_has_zero_critical_violations():
    split = chronological_leave_one_out(_temporal_fixture())
    popularity = PopularityRecommender().fit(split.train)
    bpr = BPRMFRecommender(factors=4, max_updates=100, seed=7).fit(split.train)
    audit, pairwise = audit_evaluation([popularity, bpr], split, max_users=4, pair_samples=20, seed=7)
    assert (audit["seen_item_violations"] == 0).all()
    assert (audit["missing_target_violations"] == 0).all()
    assert (audit["candidate_equality_violations"] == 0).all()
    assert (audit["descending_score_violations"] == 0).all()
    assert (audit["cold_test_targets"] == 1).all()
    assert (pairwise["training_pairs"] > 0).all()
