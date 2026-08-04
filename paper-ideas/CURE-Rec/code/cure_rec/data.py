"""Dataset loading, standardization, and conservative causal-log audits.

The loaders deliberately distinguish recommendation interaction data from genuine
policy logs. A successful load never upgrades a dataset's causal evidence level.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from cure_rec.observability import RunLogger


COAT_URL = "https://www.cs.cornell.edu/~schnabts/mnar.zip"
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

REQUIRED_INTERACTION_COLUMNS = {
    "user_id",
    "item_id",
    "timestamp",
    "response",
}

CAUSAL_LOG_COLUMNS = {
    "displayed_slate",
    "propensity",
    "position",
    "candidate_set",
}

STANDARD_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
    "response",
    "timestamp",
    "split",
    "source_dataset",
]


@dataclass(frozen=True)
class AuditResult:
    rows: int
    columns: list[str]
    missing_required: list[str]
    missing_causal: list[str]
    duplicate_rows: int
    null_counts: dict[str, int]
    permitted_claim: str
    notes: list[str]

    @property
    def supports_causal_ope(self) -> bool:
        return self.permitted_claim in {"short_horizon_ope", "sensitivity_bounded"}


@dataclass(frozen=True)
class DatasetLoadResult:
    dataset: str
    interactions: pd.DataFrame
    metadata: dict


def _log(logger: RunLogger | None, event: str, **payload) -> None:
    if logger:
        logger.event(event, **payload)


def _ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _download(url: str, destination: Path, logger: RunLogger | None = None) -> Path:
    _ensure_directory(destination.parent)
    if destination.exists():
        _log(logger, "dataset_download_reused", url=url, destination=str(destination))
        return destination
    _log(logger, "dataset_download_started", url=url, destination=str(destination))
    with urllib.request.urlopen(url, timeout=60) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    _log(logger, "dataset_download_completed", url=url, destination=str(destination), bytes=destination.stat().st_size)
    return destination


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(destination)):
                raise ValueError(f"Unsafe zip member: {member.filename}")
        zipped.extractall(destination)


def _find_file(root: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        matches = list(root.rglob(candidate))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any of {candidates} under {root}")


def _standardize(frame: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    result = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["split"] = split
    result["source_dataset"] = dataset
    result["user_id"] = pd.to_numeric(result["user_id"], errors="raise").astype("int64")
    result["item_id"] = pd.to_numeric(result["item_id"], errors="raise").astype("int64")
    result["rating"] = pd.to_numeric(result["rating"], errors="coerce")
    result["response"] = pd.to_numeric(result["response"], errors="coerce")
    result["timestamp"] = pd.to_numeric(result["timestamp"], errors="coerce").astype("Int64")
    return result[STANDARD_COLUMNS + [column for column in result.columns if column not in STANDARD_COLUMNS]]


def load_interactions_csv(path: str | Path, logger: RunLogger | None = None) -> pd.DataFrame:
    """Load a local CSV without assuming it supports causal claims."""
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    _log(logger, "data_loaded", source=str(csv_path), rows=len(frame), columns=list(frame.columns))
    return frame


def load_movielens_1m(
    root: str | Path,
    *,
    download: bool = False,
    logger: RunLogger | None = None,
) -> DatasetLoadResult:
    """Load MovieLens-1M ratings into the common interaction schema.

    MovieLens is useful for reproducibility and semi-synthetic experiments. It
    is not a logged-exposure benchmark; the audit will label it accordingly.
    """
    root_path = _ensure_directory(root)
    ratings_path: Path
    try:
        ratings_path = _find_file(root_path, ("ratings.dat",))
    except FileNotFoundError:
        if not download:
            raise FileNotFoundError("MovieLens ratings.dat not found. Re-run with download=True or place ml-1m under the root.")
        archive = _download(MOVIELENS_1M_URL, root_path / "ml-1m.zip", logger)
        _safe_extract_zip(archive, root_path)
        ratings_path = _find_file(root_path, ("ratings.dat",))
    frame = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    frame["response"] = (frame["rating"] >= 4).astype(int)
    interactions = _standardize(frame, "movielens_1m", "observed")
    metadata = {"ratings_path": str(ratings_path), "rows": len(interactions), "has_exposure_log": False}
    _log(logger, "dataset_loaded", dataset="movielens_1m", **metadata)
    return DatasetLoadResult("movielens_1m", interactions, metadata)


def _matrix_to_interactions(matrix: np.ndarray, dataset: str, split: str) -> pd.DataFrame:
    users, items = np.nonzero(matrix)
    ratings = matrix[users, items]
    frame = pd.DataFrame({
        "user_id": users,
        "item_id": items,
        "rating": ratings,
        "response": (ratings >= 4).astype(int),
        # Coat provides matrix observations, not event ordering. Preserve missing
        # timestamps instead of fabricating a temporal causal claim.
        "timestamp": pd.NA,
    })
    return _standardize(frame, dataset, split)


def load_coat(
    root: str | Path,
    *,
    download: bool = False,
    logger: RunLogger | None = None,
) -> DatasetLoadResult:
    """Load the Coat matrix dataset from a local/extracted archive.

    The loader accepts the standard `train.ascii` / `test.ascii` layout. Coat's
    randomized component is useful for short-horizon bias checks, but it has no
    timestamps or complete slate logs for CURE-Rec's long-horizon policy claims.
    """
    root_path = _ensure_directory(root)
    try:
        train_path = _find_file(root_path, ("train.ascii",))
        test_path = _find_file(root_path, ("test.ascii",))
    except FileNotFoundError:
        if not download:
            raise FileNotFoundError("Coat train.ascii/test.ascii not found. Re-run with download=True or unpack the dataset under the root.")
        archive = _download(COAT_URL, root_path / "coat_mnar.zip", logger)
        _safe_extract_zip(archive, root_path)
        train_path = _find_file(root_path, ("train.ascii",))
        test_path = _find_file(root_path, ("test.ascii",))
    train = np.loadtxt(train_path, dtype=float)
    test = np.loadtxt(test_path, dtype=float)
    interactions = pd.concat(
        [_matrix_to_interactions(train, "coat", "train"), _matrix_to_interactions(test, "coat", "test")],
        ignore_index=True,
    )
    metadata = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": int((train > 0).sum()),
        "test_rows": int((test > 0).sum()),
        "has_exposure_log": False,
        "note": "Matrix observations have no event timestamps or slate propensities.",
    }
    _log(logger, "dataset_loaded", dataset="coat", **metadata)
    return DatasetLoadResult("coat", interactions, metadata)


def _read_yahoo_r3_file(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, names=["user_id", "item_id", "rating"], usecols=[0, 1, 2])
    frame["response"] = (frame["rating"] >= 4).astype(int)
    frame["timestamp"] = pd.NA
    return _standardize(frame, "yahoo_r3", split)


def load_yahoo_r3(root: str | Path, logger: RunLogger | None = None) -> DatasetLoadResult:
    """Load a locally obtained Yahoo! R3 dataset.

    Yahoo Webscope access may require accepting its terms, so this loader never
    downloads it automatically. Place the official train and test text files below
    `root` and call this function.
    """
    root_path = Path(root)
    train_path = _find_file(root_path, ("ydata-ymusic-rating-study-v1_0-train.txt", "train.txt"))
    test_path = _find_file(root_path, ("ydata-ymusic-rating-study-v1_0-test.txt", "test.txt"))
    interactions = pd.concat(
        [_read_yahoo_r3_file(train_path, "train"), _read_yahoo_r3_file(test_path, "test")],
        ignore_index=True,
    )
    metadata = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "rows": len(interactions),
        "has_exposure_log": False,
        "note": "Use as randomized-vs-biased rating evidence; it is not a complete long-horizon slate-policy log.",
    }
    _log(logger, "dataset_loaded", dataset="yahoo_r3", **metadata)
    return DatasetLoadResult("yahoo_r3", interactions, metadata)


def load_dataset(
    dataset: Literal["movielens_1m", "coat", "yahoo_r3", "csv"],
    root: str | Path,
    *,
    download: bool = False,
    logger: RunLogger | None = None,
) -> DatasetLoadResult:
    """Dispatch public/local dataset loaders with explicit download consent."""
    if dataset == "movielens_1m":
        return load_movielens_1m(root, download=download, logger=logger)
    if dataset == "coat":
        return load_coat(root, download=download, logger=logger)
    if dataset == "yahoo_r3":
        return load_yahoo_r3(root, logger=logger)
    if dataset == "csv":
        frame = load_interactions_csv(root, logger=logger)
        return DatasetLoadResult("csv", frame, {"path": str(root), "rows": len(frame), "has_exposure_log": set(CAUSAL_LOG_COLUMNS).issubset(frame.columns)})
    raise ValueError(f"Unknown dataset {dataset!r}")


def write_standardized_dataset(result: DatasetLoadResult, output: str | Path, logger: RunLogger | None = None) -> Path:
    """Write standardized interactions as CSV to avoid optional Parquet engines."""
    output_path = Path(output)
    _ensure_directory(output_path.parent)
    result.interactions.to_csv(output_path, index=False)
    _log(logger, "dataset_standardized_written", dataset=result.dataset, output=str(output_path), rows=len(result.interactions))
    return output_path


def audit_interactions(frame: pd.DataFrame, logger: RunLogger | None = None) -> AuditResult:
    """Classify a local log by the strongest claim its fields can support.

    This is intentionally conservative. Passing this check does not prove causal
    identification; it prevents accidental causal language for logs that plainly
    lack treatment timing, slates, or policy-propensity information.
    """
    columns = set(frame.columns)
    missing_required = sorted(
        column for column in REQUIRED_INTERACTION_COLUMNS
        if column not in columns or frame[column].isna().all()
    )
    missing_causal = sorted(
        column for column in CAUSAL_LOG_COLUMNS
        if column not in columns or frame[column].isna().all()
    )
    notes: list[str] = []
    if missing_required:
        permitted_claim = "descriptive_only"
        notes.append("Missing or entirely null core interaction fields; no temporal or causal-policy claim is permitted.")
    elif missing_causal:
        permitted_claim = "descriptive_or_semisynthetic"
        notes.append("Missing logged slate/propensity fields; do not make offline causal-policy claims.")
    else:
        permitted_claim = "short_horizon_ope"
        notes.append("Slate and propensity fields are present; sequential timing, overlap, and estimator diagnostics remain required.")
        if "policy_mixture_assignment" in columns and not frame["policy_mixture_assignment"].isna().all():
            permitted_claim = "sensitivity_bounded"
            notes.append("Policy-mixture assignment is present; Gamma-sensitivity analysis may be possible after treatment-unit audit.")

    null_counts = {column: int(frame[column].isna().sum()) for column in frame.columns if frame[column].isna().any()}
    result = AuditResult(
        rows=int(len(frame)),
        columns=sorted(columns),
        missing_required=missing_required,
        missing_causal=missing_causal,
        duplicate_rows=int(frame.duplicated().sum()),
        null_counts=null_counts,
        permitted_claim=permitted_claim,
        notes=notes,
    )
    if logger:
        logger.event("data_audited", **result.__dict__)
        logger.write_json("artifacts/data_audit.json", result.__dict__)
    return result


def to_temporal_groups(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> Iterable[pd.DataFrame]:
    """Yield stable timestamp groups for lightweight local inspection."""
    if timestamp_column not in frame.columns or frame[timestamp_column].isna().all():
        raise ValueError(f"Missing usable {timestamp_column!r}; do not fabricate temporal order.")
    sorted_frame = frame.sort_values(timestamp_column, kind="stable")
    for _, group in sorted_frame.groupby(timestamp_column, sort=True):
        yield group


def load_curesim(settings, scenario_name: str | None = None, logger: RunLogger | None = None):
    """Instantiate the disclosed synthetic data-generating environment."""
    from cure_rec.simulator import CureSim

    scenario = next(
        (candidate for candidate in settings.scenarios if candidate.name == scenario_name),
        settings.scenarios[0],
    )
    simulator = CureSim(settings, scenario)
    _log(
        logger,
        "synthetic_data_loaded",
        source="CURE-Sim",
        scenario=scenario.name,
        n_users=settings.simulator.n_users,
        n_items=settings.simulator.n_items,
        horizon=settings.simulator.horizon,
    )
    return simulator
