"""Dataset loaders for the static regime.

Row counts are asserted against the published figures so that a silently
different file (red instead of white wine, an unfiltered air-quality dump)
fails loudly at load time rather than producing plausible wrong numbers.
See ``docs/clustering_spec.md`` for provenance of every constant here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Dataset", "load_wine", "load_air_quality", "DATA_ROOT"]

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

# Canonical UCI column order, quality excluded (the task is unsupervised).
WINE_FEATURES = (
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
)

# RAIN is excluded without comment in the source paper; we match that so the
# feature count stays at the published 11.
AIR_FEATURES = (
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "wd", "WSPM",
)

WINE_N = 4898
AIR_N = 383_585


@dataclass(frozen=True)
class Dataset:
    """A loaded static-regime dataset."""

    name: str
    X: np.ndarray
    feature_names: tuple[str, ...]
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    """Columns held out of the clustering but kept for interpretation.

    Wine ``quality`` lives here. It must not enter ``X`` -- the task is
    unsupervised and the published pipeline excludes it -- but it is what lets
    a cluster be identified as the desirable one, which the intervention target
    needs.
    """

    def __post_init__(self) -> None:
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"{self.name}: {self.X.shape[1]} columns but "
                f"{len(self.feature_names)} names"
            )

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Dataset({self.name}, {self.n_samples}x{self.n_features})"


def load_wine(path: str | Path | None = None, strict: bool = True) -> Dataset:
    """UCI Wine Quality, WHITE variant.

    The variant matters and is not stated in the later paper: 4,898 rows is
    white, 1,599 is red, 6,497 is the two combined. The earlier paper states
    white explicitly and its reported feature means match the white file.
    """
    path = Path(path) if path else DATA_ROOT / "winequality-white.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with:\n"
            "  curl -L -o data/raw/winequality-white.csv "
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "wine-quality/winequality-white.csv"
        )

    df = pd.read_csv(path, sep=";")
    missing = set(WINE_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    if strict and len(df) != WINE_N:
        raise ValueError(
            f"{path} has {len(df)} rows, expected {WINE_N} for the white "
            "variant (red is 1599, combined is 6497). Wrong file?"
        )

    aux = {}
    if "quality" in df.columns:
        aux["quality"] = df["quality"].to_numpy(float)

    return Dataset(
        "wine", df[list(WINE_FEATURES)].to_numpy(float), WINE_FEATURES, aux
    )


def load_air_quality(
    directory: str | Path | None = None,
    strict: bool = True,
    wind_encoding: str = "cyclical",
) -> Dataset:
    """Beijing Multi-Site Air Quality, all 12 stations concatenated.

    Rows with missing values in the retained columns are dropped. The source
    paper contradicts itself here -- Table III says omission, the text says
    imputation -- but only omission reconciles with the published count:
    12 stations x 35,064 hourly rows = 420,768, less 37,183 dropped.

    ``wd`` is a 16-level compass string and the source paper never says how it
    was encoded, though the stated pipeline cannot run without an answer.
    Cyclical sin/cos is the default here because it preserves both the
    published feature count and the circular topology; one-hot would inflate 11
    features to 26 and ordinal would impose a false linear order on a compass.
    """
    directory = Path(directory) if directory else DATA_ROOT / "beijing"
    files = sorted(directory.glob("PRSA_Data_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No PRSA_Data_*.csv under {directory}. Download and unzip:\n"
            "  https://archive.ics.uci.edu/static/public/501/"
            "beijing+multi+site+air+quality+data.zip"
        )
    if strict and len(files) != 12:
        raise ValueError(f"expected 12 station files, found {len(files)}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[list(AIR_FEATURES)].dropna()

    if strict and len(df) != AIR_N:
        raise ValueError(
            f"got {len(df)} complete rows, expected {AIR_N}. The published "
            "count assumes dropna() over exactly these 11 columns."
        )

    names = list(AIR_FEATURES)
    if wind_encoding == "cyclical":
        angle = df["wd"].map(_COMPASS_RADIANS)
        if angle.isna().any():
            bad = sorted(set(df.loc[angle.isna(), "wd"]))
            raise ValueError(f"unrecognized wind directions: {bad}")
        df = df.drop(columns=["wd"])
        df["wd_sin"], df["wd_cos"] = np.sin(angle), np.cos(angle)
        names = [n for n in names if n != "wd"] + ["wd_sin", "wd_cos"]
    elif wind_encoding == "drop":
        df = df.drop(columns=["wd"])
        names = [n for n in names if n != "wd"]
    else:
        raise ValueError(f"unknown wind_encoding {wind_encoding!r}")

    return Dataset("air_quality", df[names].to_numpy(float), tuple(names))


_COMPASS_RADIANS = {
    d: np.deg2rad(i * 22.5)
    for i, d in enumerate(
        ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    )
}
