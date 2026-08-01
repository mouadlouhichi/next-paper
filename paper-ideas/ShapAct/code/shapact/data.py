"""Data loaders, filtering, and temporal split (ShapAct Implementation Spec A.3).

Protocol (SignalShap Implementation Spec A.3):
  * implicit conversion: rating >= threshold counts as positive
  * 5-core filtering applied iteratively to a fixed point
  * temporal leave-one-out per user: last interaction = test,
    second-last = validation, rest = train; timestamp ties broken
    deterministically by original row order
  * the k-core filtered frame (original ids) is cached to disk; id remapping
    and the temporal split are deterministic and recomputed on each load
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CACHE_DIR, RAW_DIR, DatasetConfig


@dataclass
class Dataset:
    name: str
    users: np.ndarray          # contiguous user ids (0..n_users-1)
    items: np.ndarray          # contiguous item ids (0..n_items-1)
    item_meta: dict            # contiguous item_id -> metadata dict
    item_tfidf: np.ndarray     # (n_items, n_terms) TF-IDF vectors
    term_names: list
    train: pd.DataFrame        # columns: user, item, timestamp (contiguous)
    val: pd.DataFrame
    test: pd.DataFrame
    stats: dict


def _cache_key(cfg: DatasetConfig) -> str:
    return hashlib.sha256(
        f"{cfg.name}|{cfg.rating_threshold}|{cfg.k_core}".encode()
    ).hexdigest()[:12]


def _kcore(df: pd.DataFrame, k: int) -> pd.DataFrame:
    while True:
        cnt_u = df["user"].value_counts()
        cnt_i = df["item"].value_counts()
        nxt = df[df["user"].isin(cnt_u[cnt_u >= k].index)
                 & df["item"].isin(cnt_i[cnt_i >= k].index)]
        if len(nxt) == len(df):
            return df.reset_index(drop=True)
        df = nxt


def _load_raw(cfg: DatasetConfig):
    if cfg.name == "ml1m":
        # note: the mirror we use normalizes the canonical '::' format to tabs
        ratings = pd.read_csv(
            RAW_DIR / "ml1m_ratings.dat",
            sep="\t", engine="python", header=None,
            names=["user", "item", "rating", "timestamp"],
        )
        df = ratings[["user", "item", "rating", "timestamp"]]
        items = pd.read_csv(
            RAW_DIR / "ml1m_items.dat",
            sep="\t", engine="python", header=None,
            names=["item", "title", "genres"],
        )
        items["genres"] = items["genres"].fillna("")
        meta = {
            int(i): {"genres": g.split("|") if g else []}
            for i, g in zip(items["item"], items["genres"])
        }
        return df, meta

    if cfg.name == "lastfm":
        ua = pd.read_csv(
            RAW_DIR / "lastfm_user_artists.dat", sep="\t", header=0,
            names=["user", "item", "weight"],
        )
        ut = pd.read_csv(
            RAW_DIR / "lastfm_user_taggedartists-timestamps.dat",
            sep="\t", header=0, names=["user", "item", "tag", "timestamp"],
        )
        first = ut.sort_values("timestamp").groupby(["user", "item"])[
            "timestamp"].min().reset_index()
        ua = ua.merge(first, on=["user", "item"], how="left")
        # interactions without a tag event get the user's median first-tag time;
        # users with no tag events at all fall back to 0 (all their
        # interactions share timestamp 0 and order by original row order)
        ua["timestamp"] = ua["timestamp"].fillna(
            ua.groupby("user")["timestamp"].transform("median")
        ).fillna(0)
        df = ua.rename(columns={"weight": "rating"})[
            ["user", "item", "rating", "timestamp"]]
        tags = pd.read_csv(
            RAW_DIR / "lastfm_user_taggedartists.dat",
            sep="\t", header=0, names=["user", "item", "tag", "day", "month", "year"],
        )
        tag_counts = tags.groupby(["item", "tag"]).size().reset_index(name="c")
        meta = {}
        for it, grp in tag_counts.groupby("item"):
            meta[int(it)] = {
                "tags": [f"t{int(t)}" for t in
                         grp.sort_values("c", ascending=False)["tag"].tolist()[:50]]
            }
        return df, meta

    raise ValueError(cfg.name)


def _tfidf(cfg: DatasetConfig, items: np.ndarray, item_meta: dict):
    """TF-IDF over item metadata terms (genres for ML-1M, tags for LastFM)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = []
    for it in items:
        m = item_meta.get(int(it), {})
        toks = m.get("genres", m.get("tags", []))
        docs.append(" ".join(t.replace(" ", "_") for t in toks))
    vec = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True,
                          min_df=1, max_features=500)
    X = vec.fit_transform(docs).toarray()
    return X, vec.get_feature_names_out().tolist()


def load_dataset(cfg: DatasetConfig) -> Dataset:
    key = _cache_key(cfg)
    cache_path = CACHE_DIR / f"{cfg.name}_{key}.pkl"
    if cache_path.exists():
        df = pd.read_pickle(cache_path)
    else:
        df, meta = _load_raw(cfg)
        df = df[df["rating"] >= cfg.rating_threshold].copy()
        df = df.drop(columns="rating")
        df = _kcore(df, cfg.k_core)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        df = df.copy()

    # contiguous ids
    users = np.sort(df["user"].unique())
    items = np.sort(df["item"].unique())
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {i: j for j, i in enumerate(items)}
    df = df.copy()
    df["user"] = df["user"].map(u_map)
    df["item"] = df["item"].map(i_map)

    # item metadata remapped to contiguous ids
    _, meta = _load_raw(cfg)
    item_meta = {i_map[k]: v for k, v in meta.items() if k in i_map}

    # temporal split, deterministic tie-break by original row order
    df["_order"] = np.arange(len(df))
    df = df.sort_values(["user", "timestamp", "_order"]).reset_index(drop=True)
    last = df.groupby("user").tail(1)
    rest = df.drop(index=last.index)
    second = rest.groupby("user").tail(1)
    train = rest.drop(index=second.index).drop(columns="_order")
    val = second.drop(columns="_order")
    test = last.drop(columns="_order")

    item_tfidf, terms = _tfidf(cfg, np.arange(len(items)), item_meta)

    n = len(df)
    stats = {
        "users": int(len(users)),
        "items": int(len(items)),
        "interactions": int(n),
        "density": float(n / (len(users) * len(items))),
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
    }
    return Dataset(cfg.name, np.arange(len(users)), np.arange(len(items)),
                   item_meta, item_tfidf, terms, train, val, test, stats)
