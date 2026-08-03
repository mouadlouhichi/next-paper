"""Leakage-safe temporal data preparation for recommendation ActionShap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalDataset:
    """Integer-indexed train/validation/test split.

    Item and user mappings are retained so released per-user examples can be
    translated back to source IDs without using validation or test events in
    model fitting.
    """

    train: dict[int, np.ndarray]
    validation: dict[int, int]
    test: dict[int, int]
    n_items: int
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def users(self) -> np.ndarray:
        return np.array(sorted(self.test), dtype=int)

    def seen_before_test(
        self, user: int, include_validation: bool = True
    ) -> np.ndarray:
        """All items observed before the test event for candidate exclusion."""
        values = np.asarray(self.train[int(user)], dtype=int)
        if include_validation:
            values = np.concatenate((values, [int(self.validation[int(user)])]))
        return np.unique(values)


def _build_temporal_dataset(
    frame: pd.DataFrame, minimum_interactions: int = 4
) -> TemporalDataset:
    required = {"user", "item", "timestamp", "original_record_index"}
    if missing := required - set(frame.columns):
        raise ValueError(f"interaction frame is missing columns: {sorted(missing)}")
    if minimum_interactions < 3:
        raise ValueError("minimum_interactions must be at least three")
    ordered = frame.sort_values(
        ["user", "timestamp", "original_record_index"], kind="mergesort"
    ).copy()
    counts = ordered.groupby("user", sort=True).size()
    eligible_users = counts[counts >= minimum_interactions].index
    ordered = ordered.loc[ordered["user"].isin(eligible_users)].copy()
    if ordered.empty:
        raise ValueError("no users satisfy the temporal-split eligibility rule")

    user_values = np.sort(ordered["user"].unique())
    item_values = np.sort(ordered["item"].unique())
    user_to_idx = {value: i for i, value in enumerate(user_values)}
    item_to_idx = {value: i for i, value in enumerate(item_values)}
    ordered["u"] = ordered["user"].map(user_to_idx).astype(int)
    ordered["i"] = ordered["item"].map(item_to_idx).astype(int)

    train: dict[int, np.ndarray] = {}
    validation: dict[int, int] = {}
    test: dict[int, int] = {}
    for user, rows in ordered.groupby("u", sort=True):
        items = rows["i"].to_numpy(dtype=int)
        train[int(user)] = items[:-2].copy()
        validation[int(user)] = int(items[-2])
        test[int(user)] = int(items[-1])
    return TemporalDataset(
        train=train,
        validation=validation,
        test=test,
        n_items=int(item_values.size),
        user_ids=user_values,
        item_ids=item_values,
    )


def load_movielens_1m(
    path: str | Path,
    rating_threshold: float = 4.0,
    minimum_interactions: int = 4,
) -> TemporalDataset:
    """Load MovieLens-1M with a deterministic last-two-event split.

    Ratings below ``rating_threshold`` are removed before splitting, making the
    task an implicit positive-feedback task.  Ties are resolved by the original
    source-file row index.
    """
    path = Path(path)
    frame = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user", "item", "rating", "timestamp"],
    )
    frame["original_record_index"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.loc[frame["rating"] >= rating_threshold].copy()
    return _build_temporal_dataset(frame, minimum_interactions)


def load_interactions_csv(
    path: str | Path,
    *,
    user_column: str = "user",
    item_column: str = "item",
    timestamp_column: str = "timestamp",
    rating_column: str | None = None,
    rating_threshold: float | None = None,
    minimum_interactions: int = 4,
) -> TemporalDataset:
    """Load a timestamped CSV secondary dataset with explicit column names."""
    frame = pd.read_csv(path)
    required = {user_column, item_column, timestamp_column}
    if missing := required - set(frame.columns):
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if rating_column is not None:
        if rating_column not in frame:
            raise ValueError(f"CSV is missing rating column {rating_column!r}")
        if rating_threshold is not None:
            frame = frame.loc[frame[rating_column] >= rating_threshold].copy()
    frame = frame.rename(
        columns={
            user_column: "user",
            item_column: "item",
            timestamp_column: "timestamp",
        }
    )
    frame["original_record_index"] = np.arange(len(frame), dtype=np.int64)
    return _build_temporal_dataset(frame, minimum_interactions)


def truncate_histories(data: TemporalDataset, n_max: int = 50) -> dict[int, np.ndarray]:
    """Keep the most recent training interactions as attribution players."""
    if n_max < 1:
        raise ValueError("n_max must be positive")
    return {
        u: np.asarray(items[-n_max:], dtype=int).copy()
        for u, items in data.train.items()
    }


def sample_evaluation_users(
    data: TemporalDataset,
    max_users: int = 0,
    seed: int = 42,
    minimum_history: int = 2,
    pool_size: int = 0,
) -> list[int]:
    """Select reproducible random users, optionally nested in a larger cohort."""
    eligible = np.array(
        [u for u in sorted(data.test) if len(data.train[u]) >= minimum_history],
        dtype=int,
    )
    if eligible.size == 0:
        return []
    if max_users <= 0 or max_users >= eligible.size:
        return eligible.tolist()
    rng = np.random.default_rng(seed)
    if pool_size > max_users and pool_size < eligible.size:
        # Draw exactly the same larger cohort as the primary run, then take a
        # deterministic prefix for a matched robustness subset.
        pool = rng.choice(eligible, size=pool_size, replace=False)
        return np.sort(pool[:max_users]).astype(int).tolist()
    return (
        np.sort(rng.choice(eligible, size=max_users, replace=False))
        .astype(int)
        .tolist()
    )
