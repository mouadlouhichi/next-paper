"""Per-method wall-clock and peak-memory benchmarking (review 3, issue 9)."""
from __future__ import annotations

import resource
import time
from typing import Callable


def bench(fn: Callable[[], object], *, repeat: int = 3) -> dict[str, float]:
    """Run ``fn`` repeatedly; report median wall time and delta peak RSS (MiB)."""
    walls = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        walls.append(time.perf_counter() - start)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # MiB
    return {
        "median_wall_seconds": float(sorted(walls)[len(walls) // 2]),
        "peak_rss_mib": float(peak),
        "repeat": repeat,
    }
