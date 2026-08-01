"""Modifiability elicitation (Definition 1) and its audit trail.

The manuscript makes a pre-registration claim in Section 4.2: annotation was
completed and frozen before any attribution or intervention result was
inspected. That claim is only defensible if the freeze is mechanically
checkable, so this module refuses to hand out modifiability values from a
table whose payload hash does not match the one recorded at freeze time.

Annotation files live in ``annotations/<dataset>.yaml`` and look like:

    dataset: air_quality
    rubric_version: "1.0"
    frozen:
      timestamp: "2026-07-30T09:00:00+01:00"
      payload_sha256: "3f1a..."
      git_commit: "a1b2c3d"
    factors:
      PM2.5:  {annotator_a: 1.0, annotator_b: 1.0, annotator_c: 0.5}
      TEMP:   {annotator_a: 0.0, annotator_b: 0.0, annotator_c: 0.0}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

__all__ = [
    "RUBRIC",
    "ModifiabilityTable",
    "load_modifiability",
    "compute_payload_hash",
    "FreezeViolation",
]

# The three admissible levels. Section 4.2 discretizes Definition 1 into
# exactly these; an annotator supplying anything else is an error, not a
# value to be rounded.
RUBRIC: Mapping[float, str] = {
    1.0: "directly controllable by the decision-maker",
    0.5: "controllable indirectly, or at substantial cost or delay",
    0.0: "observable but immutable",
}

_VALID_LEVELS = frozenset(RUBRIC)


class FreezeViolation(RuntimeError):
    """Raised when an annotation table fails its pre-registration check."""


@dataclass(frozen=True)
class ModifiabilityTable:
    """Elicited modifiability for one dataset, with its annotation provenance."""

    dataset: str
    factors: tuple[str, ...]
    per_annotator: dict[str, np.ndarray]
    consensus: np.ndarray
    rubric_version: str
    frozen_at: str
    git_commit: str

    @property
    def m(self) -> np.ndarray:
        """Consensus modifiability vector, ordered as ``self.factors``."""
        return self.consensus

    @property
    def modifiable_mask(self) -> np.ndarray:
        return self.consensus > 0

    @property
    def n_immutable(self) -> int:
        return int((self.consensus == 0).sum())

    def index_of(self, factor: str) -> int:
        try:
            return self.factors.index(factor)
        except ValueError:
            raise KeyError(
                f"{factor!r} not in {self.dataset} table; have {list(self.factors)}"
            ) from None

    def agreement(self) -> float:
        """Krippendorff's alpha across annotators, on the ordinal scale.

        Reported in the manuscript as INTRA-TEAM agreement. It measures
        whether the rubric is unambiguous, not whether the domain agrees;
        the annotators are the authors. Section 4.2 says so explicitly and
        this docstring exists so the distinction is not lost in the code.
        """
        import krippendorff

        matrix = np.vstack([self.per_annotator[a] for a in sorted(self.per_annotator)])
        return float(
            krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
        )

    def disagreements(self) -> list[tuple[str, dict[str, float]]]:
        """Factors where annotators did not agree, for the appendix table."""
        out = []
        for i, f in enumerate(self.factors):
            votes = {a: float(v[i]) for a, v in self.per_annotator.items()}
            if len(set(votes.values())) > 1:
                out.append((f, votes))
        return out


def compute_payload_hash(factors: Mapping[str, Mapping[str, float]]) -> str:
    """SHA-256 over the annotation payload, insensitive to key ordering.

    Only the factor/annotator/value content is hashed. The ``frozen`` block
    itself is excluded, since it contains the hash.
    """
    canonical = json.dumps(
        {f: {a: float(v) for a, v in sorted(votes.items())}
         for f, votes in sorted(factors.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_modifiability(
    path: str | Path,
    *,
    expected_factors: list[str] | None = None,
    verify_freeze: bool = True,
) -> ModifiabilityTable:
    """Load and validate an annotation file.

    Parameters
    ----------
    expected_factors
        If given, the table must cover exactly these factors, in any order,
        and the returned arrays are ordered to match. This is what prevents a
        silent misalignment between the modifiability vector and the
        attribution vector -- the failure mode that would quietly invalidate
        every downstream metric.
    verify_freeze
        Recompute the payload hash and compare against the recorded one.
        Only disable this while drafting an annotation.
    """
    path = Path(path)
    spec = yaml.safe_load(path.read_text())

    for key in ("dataset", "rubric_version", "factors", "frozen"):
        if key not in spec:
            raise ValueError(f"{path}: missing required key {key!r}")

    raw: dict[str, dict[str, float]] = spec["factors"]
    if not raw:
        raise ValueError(f"{path}: no factors annotated")

    frozen = spec["frozen"]
    if verify_freeze:
        recorded = frozen.get("payload_sha256")
        if not recorded:
            raise FreezeViolation(f"{path}: no payload_sha256 recorded")
        actual = compute_payload_hash(raw)
        if actual != recorded:
            raise FreezeViolation(
                f"{path}: annotation payload has changed since it was frozen "
                f"(recorded {recorded[:12]}..., computed {actual[:12]}...). "
                "The pre-registration claim in Section 4.2 is void unless this "
                "is resolved: either restore the frozen content, or re-freeze "
                "and disclose that the annotation was revised."
            )

    annotators = sorted({a for votes in raw.values() for a in votes})
    if len(annotators) < 2:
        raise ValueError(f"{path}: need >= 2 annotators, found {annotators}")

    factors = sorted(raw)
    if expected_factors is not None:
        missing = set(expected_factors) - set(factors)
        extra = set(factors) - set(expected_factors)
        if missing or extra:
            raise ValueError(
                f"{path}: factor mismatch. Missing {sorted(missing)}, "
                f"unexpected {sorted(extra)}."
            )
        factors = list(expected_factors)

    per_annotator: dict[str, np.ndarray] = {}
    for a in annotators:
        vals = []
        for f in factors:
            if a not in raw[f]:
                raise ValueError(f"{path}: annotator {a!r} did not rate {f!r}")
            v = float(raw[f][a])
            if v not in _VALID_LEVELS:
                raise ValueError(
                    f"{path}: {a!r} gave {f!r} the value {v}, which is not one of "
                    f"{sorted(_VALID_LEVELS)}. The rubric is three-level; a value "
                    "outside it means the rubric was not followed."
                )
            vals.append(v)
        per_annotator[a] = np.array(vals, dtype=float)

    # Median, not mean: the scale is ordinal, and a mean would invent
    # intermediate values the rubric does not define. With three annotators
    # the median is also the majority vote whenever one exists.
    stacked = np.vstack([per_annotator[a] for a in annotators])
    consensus = np.median(stacked, axis=0)

    return ModifiabilityTable(
        dataset=spec["dataset"],
        factors=tuple(factors),
        per_annotator=per_annotator,
        consensus=consensus,
        rubric_version=str(spec["rubric_version"]),
        frozen_at=str(frozen.get("timestamp", "UNRECORDED")),
        git_commit=str(frozen.get("git_commit", "UNRECORDED")),
    )
