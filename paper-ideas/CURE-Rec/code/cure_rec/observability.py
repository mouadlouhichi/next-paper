"""Small dependency-free observability layer for reproducible CURE-Rec runs."""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from cure_rec.config import Settings


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


class RunLogger:
    """Writes human-readable and structured logs without an external service.

    Each event is appended to JSONL so notebooks can recover a complete timeline,
    including every coalition evaluation and policy-selection decision.
    """

    def __init__(self, settings: Settings):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{settings.run.name}-{stamp}-{uuid.uuid4().hex[:8]}"
        self.run_dir = Path(settings.run.output_root) / self.run_id
        self.logs_dir = self.run_dir / "logs"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.figures_dir = self.run_dir / "figures"
        self.tables_dir = self.run_dir / "tables"
        for directory in (self.logs_dir, self.artifacts_dir, self.figures_dir, self.tables_dir):
            directory.mkdir(parents=True, exist_ok=False)

        self._events_path = self.logs_dir / "events.jsonl"
        self._metrics_path = self.logs_dir / "metrics.jsonl"
        self._logger = logging.getLogger(f"cure_rec.{self.run_id}")
        self._logger.setLevel(getattr(logging, settings.run.log_level.upper(), logging.INFO))
        self._logger.handlers.clear()
        self._logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self.write_json("manifest.json", {
            "run_id": self.run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config_hash": settings.config_hash(),
            "settings": settings.canonical_dict(),
            "platform": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
        })
        self.event("run_started", run_id=self.run_id, config_hash=settings.config_hash())

    def event(self, name: str, **payload: Any) -> None:
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "event": name,
            **payload,
        }
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")
        # Every event stays in JSONL. Per-coalition events are debug-level in the
        # human console log so a 64-coalition sweep remains readable on a laptop.
        level = logging.DEBUG if name.startswith("coalition_") else logging.INFO
        self._logger.log(level, "%s | %s", name, json.dumps(payload, default=_json_default, sort_keys=True))

    def metric(self, name: str, value: float, **dimensions: Any) -> None:
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "metric": name,
            "value": float(value),
            **dimensions,
        }
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")

    @contextmanager
    def span(self, name: str, **payload: Any) -> Iterator[None]:
        started = time.perf_counter()
        self.event(f"{name}_started", **payload)
        try:
            yield
        except Exception as exc:
            self.event(f"{name}_failed", duration_seconds=time.perf_counter() - started, error=repr(exc), **payload)
            raise
        else:
            self.event(f"{name}_completed", duration_seconds=time.perf_counter() - started, **payload)

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=_json_default, indent=2, sort_keys=True)
        return path

    def write_dataframe(self, relative_path: str, frame: pd.DataFrame) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path

    def close(self, status: str = "completed") -> None:
        self.event("run_finished", status=status, run_dir=str(self.run_dir))
        for handler in self._logger.handlers:
            handler.close()
