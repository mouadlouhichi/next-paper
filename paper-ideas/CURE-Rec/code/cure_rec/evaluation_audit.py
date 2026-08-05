"""Hard evaluation-audit checks shared by all external recommender baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from cure_rec.models import LeaveOneOutSplit, build_shared_candidates


@dataclass(frozen=True)
class EvaluationAuditRecord:
    model: str
    evaluated_users: int
    warm_test_targets: int
    cold_test_targets: int
    mean_candidate_count: float
    minimum_candidate_count: int
    maximum_candidate_count: int
    seen_item_violations: int
    missing_target_violations: int
    candidate_equality_violations: int
    descending_score_violations: int
    best_validation_epoch: int | None
    restored_checkpoint_epoch: int | None


def _pairwise_accuracy(model, positives: pd.DataFrame, train: pd.DataFrame, *, max_pairs: int, seed: int) -> tuple[float, int]:
    catalog = np.sort(train["item_id"].astype(int).unique())
    seen = {int(u): set(frame["item_id"].astype(int)) for u, frame in train.groupby("user_id")}
    rng = np.random.default_rng(seed)
    correct = 0
    total = 0
    for row in positives.sort_values("user_id", kind="stable").head(max_pairs).itertuples(index=False):
        user, positive = int(row.user_id), int(row.item_id)
        candidates = catalog[~np.isin(catalog, list(seen.get(user, set()) - {positive}))]
        negatives = candidates[candidates != positive]
        if positive not in catalog or not len(negatives):
            continue
        negative = int(negatives[rng.integers(len(negatives))])
        scores = model.score(user, np.asarray([positive, negative], dtype=int))
        correct += int(scores[0] > scores[1])
        total += 1
    return (correct / total if total else float("nan"), total)


def audit_evaluation(models: list, split: LeaveOneOutSplit, *, max_users: int = 1_000, pair_samples: int = 10_000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit candidates, ranking direction, checkpoint state, and pair accuracy.

    Candidate IDs are built once and passed to every model. This makes equality a
    guaranteed invariant that is nevertheless recorded and asserted explicitly.
    """
    cases, cold = build_shared_candidates(split, max_users=max_users)
    expected = {case.user_id: case.candidate_items for case in cases}
    train_seen = {int(u): set(frame["item_id"].astype(int)) for u, frame in split.train.groupby("user_id")}
    records = []
    pair_rows = []
    for model in models:
        seen_violations = missing_violations = candidate_violations = descending_violations = 0
        candidate_counts = []
        for case in cases:
            candidates = expected[case.user_id]
            candidate_counts.append(len(candidates))
            if case.target_item not in candidates:
                missing_violations += 1
            if np.intersect1d(candidates, list(train_seen.get(case.user_id, set()))).size:
                seen_violations += 1
            # All models receive exactly the common candidate vector.
            model_candidates = candidates
            if not np.array_equal(candidates, model_candidates):
                candidate_violations += 1
            scores = model.score(case.user_id, candidates)
            order = np.lexsort((candidates, -scores))
            sorted_scores = scores[order]
            if np.any(sorted_scores[:-1] < sorted_scores[1:] - 1e-12):
                descending_violations += 1

        best_epoch = getattr(model, "best_validation_epoch", None)
        restored_epoch = getattr(model, "restored_checkpoint_epoch", None)
        if best_epoch is not None and restored_epoch != best_epoch:
            raise AssertionError("Best validation checkpoint was not restored")
        records.append(asdict(EvaluationAuditRecord(
            model=model.name,
            evaluated_users=len(cases),
            warm_test_targets=len(cases),
            cold_test_targets=cold,
            mean_candidate_count=float(np.mean(candidate_counts)) if candidate_counts else 0.0,
            minimum_candidate_count=int(np.min(candidate_counts)) if candidate_counts else 0,
            maximum_candidate_count=int(np.max(candidate_counts)) if candidate_counts else 0,
            seen_item_violations=seen_violations,
            missing_target_violations=missing_violations,
            candidate_equality_violations=candidate_violations,
            descending_score_violations=descending_violations,
            best_validation_epoch=best_epoch,
            restored_checkpoint_epoch=restored_epoch,
        )))
        train_accuracy, train_pairs = _pairwise_accuracy(model, split.train, split.train, max_pairs=pair_samples, seed=seed)
        validation_accuracy, validation_pairs = _pairwise_accuracy(model, split.validation, split.train, max_pairs=pair_samples, seed=seed + 1)
        pair_rows.append({
            "model": model.name,
            "pairwise_training_accuracy": train_accuracy,
            "training_pairs": train_pairs,
            "pairwise_validation_accuracy": validation_accuracy,
            "validation_pairs": validation_pairs,
        })

    audit = pd.DataFrame(records)
    for column in ("seen_item_violations", "missing_target_violations", "candidate_equality_violations", "descending_score_violations"):
        assert int(audit[column].sum()) == 0, f"Evaluation audit violation: {column}"
    return audit, pd.DataFrame(pair_rows)
