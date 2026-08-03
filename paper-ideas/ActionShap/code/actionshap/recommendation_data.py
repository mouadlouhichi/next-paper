"""Leakage-safe MovieLens-1M preparation for recommendation ActionShap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalDataset:
    """Integer-indexed temporal split with train histories."""

    train: dict[int, np.ndarray]
    validation: dict[int, int]
    test: dict[int, int]
    n_items: int
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def users(self) -> np.ndarray:
        return np.array(sorted(self.test), dtype=int)


def load_movielens_1m(path: str | Path, rating_threshold: float = 4.0) -> TemporalDataset:
    """Load ``ratings.dat`` and make deterministic last/second-last splits.

    The original row index is retained as the deterministic tie-breaker. Users
    with fewer than three positive interactions are excluded because the split
    needs train, validation, and test observations.
    """
    path = Path(path)
    frame = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user", "item", "rating", "timestamp"],
    )
    frame["row_index"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.loc[frame["rating"] >= rating_threshold].copy()
    frame = frame.sort_values(["user", "timestamp", "row_index"], kind="mergesort")
    grouped = frame.groupby("user", sort=True)
    eligible = grouped.filter(lambda x: len(x) >= 3)
    user_values = np.sort(eligible["user"].unique())
    item_values = np.sort(eligible["item"].unique())
    user_to_idx = {int(v): i for i, v in enumerate(user_values)}
    item_to_idx = {int(v): i for i, v in enumerate(item_values)}
    eligible = eligible.loc[
        eligible["user"].isin(user_to_idx) & eligible["item"].isin(item_to_idx)
    ].copy()
    eligible["u"] = eligible["user"].map(user_to_idx).astype(int)
    eligible["i"] = eligible["item"].map(item_to_idx).astype(int)

    train: dict[int, np.ndarray] = {}
    validation: dict[int, int] = {}
    test: dict[int, int] = {}
    for u, rows in eligible.groupby("u", sort=True):
        items = rows["i"].to_numpy(dtype=int)
        train[int(u)] = items[:-2]
        validation[int(u)] = int(items[-2])
        test[int(u)] = int(items[-1])
    return TemporalDataset(
        train=train,
        validation=validation,
        test=test,
        n_items=len(item_values),
        user_ids=user_values,
        item_ids=item_values,
    )


def truncate_histories(data: TemporalDataset, n_max: int = 50) -> dict[int, np.ndarray]:
    """Keep the most recent training interactions only."""
    if n_max < 1:
        raise ValueError("n_max must be positive")
    return {u: items[-n_max:].copy() for u, items in data.train.items()}
