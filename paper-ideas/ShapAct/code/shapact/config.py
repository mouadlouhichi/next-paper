"""Configuration for the ShapAct audit.

Dataset-specific settings follow the SignalShap Implementation Spec (A.3-A.8)
and the ShapAct Implementation Spec (A.2-A.6). All randomness is seeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "cache"
RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "raw"
TABLE_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"

SEEDS = (42, 43, 44, 45, 46)
MAIN_SEED = 42


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    rating_threshold: int  # rating >= threshold is a positive
    k_core: int
    candidates_n: int          # N, candidate-set size per user
    source_top_n: int          # N_g, top items per source per user
    cf_factors: int
    cf_reg: float
    cf_lr: float
    cf_iters: int
    rec_half_life_days: int
    fusion_pairs: int          # R, pairs per user for the fusion
    fusion_c: float
    fusion_seed: int


def ml1m() -> DatasetConfig:
    return DatasetConfig(
        name="ml1m",
        rating_threshold=4,
        k_core=5,
        candidates_n=500,
        source_top_n=500,
        cf_factors=64,
        cf_reg=0.01,
        cf_lr=0.01,
        cf_iters=100,
        rec_half_life_days=30,
        fusion_pairs=10,
        fusion_c=1.0,
        fusion_seed=MAIN_SEED,
    )


def lastfm() -> DatasetConfig:
    return DatasetConfig(
        name="lastfm",
        rating_threshold=1,      # any listening event is a positive
        k_core=5,
        candidates_n=500,
        source_top_n=500,
        cf_factors=64,
        cf_reg=0.01,
        cf_lr=0.01,
        cf_iters=100,
        rec_half_life_days=30,
        fusion_pairs=10,
        fusion_c=1.0,
        fusion_seed=MAIN_SEED,
    )


SOURCES = ("CF", "CB", "POP", "REC", "SEQ")

CONFIGS = {"ml1m": ml1m, "lastfm": lastfm}
