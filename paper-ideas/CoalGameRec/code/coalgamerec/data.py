from __future__ import annotations

import gzip
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy import sparse

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass
class SplitData:
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    user_col: str = "user"
    item_col: str = "item"
    time_col: str = "timestamp"

    @property
    def train_csr(self) -> sparse.csr_matrix:
        return interactions_to_csr(self.train, self.n_users, self.n_items)


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
    return dest


def load_movielens_1m(root: str | Path = "data/raw") -> pd.DataFrame:
    root = Path(root)
    zip_path = download(ML1M_URL, root / "ml-1m.zip")
    extract_dir = root / "ml-1m"
    ratings_path = extract_dir / "ratings.dat"
    if not ratings_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_raw", "item_raw", "rating", "timestamp"],
        encoding="latin-1",
    )
    df["line_idx"] = np.arange(len(df), dtype=np.int64)
    return df


def _assert_gzip_file(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        magic = f.read(2)
        head = magic + f.read(80)
    if magic != b"\x1f\x8b":
        preview = head[:80].decode("utf-8", errors="replace")
        raise ValueError(
            f"Amazon file is not a gzip stream: {path}\n"
            f"First bytes: {head[:20]!r}\n"
            f"Preview: {preview!r}\n\n"
            "This usually means an HTML error page was saved instead of Books_5.json.gz. "
            "Delete the file and rerun scripts/prepare_amazon_books.py, or set "
            "AMAZON_BOOKS_5 to a valid local Books_5.json.gz downloaded from the UCSD page."
        )


def load_amazon_books_2018(books_5_json_gz: str | Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load Amazon Reviews 2018 Books_5.json.gz.

    The full file is very large. For a Mac laptop, pass a sampled local file or
    max_rows for a feasibility spike. The loader uses only reviewerID, asin,
    overall, unixReviewTime, and an original line index.
    """
    _assert_gzip_file(books_5_json_gz)
    rows = []
    with gzip.open(books_5_json_gz, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows is not None and idx >= max_rows:
                break
            obj = json.loads(line)
            rows.append(
                {
                    "user_raw": obj.get("reviewerID"),
                    "item_raw": obj.get("asin"),
                    "rating": obj.get("overall"),
                    "timestamp": obj.get("unixReviewTime"),
                    "line_idx": idx,
                }
            )
    return pd.DataFrame(rows)


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_raw", "timestamp", "line_idx"], kind="mergesort")
    pos = df.groupby("user_raw").cumcount()
    counts = df.groupby("user_raw")["item_raw"].transform("size")
    test_mask = pos == counts - 1
    val_mask = pos == counts - 2
    train = df.loc[~(test_mask | val_mask)].copy()
    val = df.loc[val_mask].copy()
    test = df.loc[test_mask].copy()
    return train, val, test


def _iterative_train_core(train: pd.DataFrame, min_uc: int = 5, min_ic: int = 5) -> tuple[set, set, list[dict]]:
    cur = train[["user_raw", "item_raw"]].drop_duplicates().copy()
    log: list[dict] = []
    prev_shape = None
    it = 0
    while prev_shape != cur.shape:
        prev_shape = cur.shape
        uc = cur["user_raw"].value_counts()
        ic = cur["item_raw"].value_counts()
        keep_u = set(uc[uc >= min_uc].index)
        keep_i = set(ic[ic >= min_ic].index)
        cur = cur[cur["user_raw"].isin(keep_u) & cur["item_raw"].isin(keep_i)].copy()
        log.append({"iter": it, "users": cur.user_raw.nunique(), "items": cur.item_raw.nunique(), "edges": len(cur)})
        it += 1
    return set(cur.user_raw.unique()), set(cur.item_raw.unique()), log


def preprocess_temporal_loo(
    df: pd.DataFrame,
    name: str,
    rating_threshold: float = 4.0,
    min_uc: int = 5,
    min_ic: int = 5,
    sample_users: Optional[int] = None,
    sample_seed: int = 20260803,
) -> tuple[SplitData, dict]:
    """Convert ratings to positives, use preliminary temporal split, train-period 5-core, rebuild split."""
    df = df.dropna(subset=["user_raw", "item_raw", "rating", "timestamp"]).copy()
    if sample_users is not None and df.user_raw.nunique() > sample_users:
        rng = np.random.default_rng(sample_seed)
        users = np.array(sorted(df.user_raw.unique(), key=str), dtype=object)
        keep = set(rng.choice(users, size=sample_users, replace=False))
        df = df[df.user_raw.isin(keep)].copy()
    pos = df[df["rating"] >= rating_threshold].copy()
    prelim_train, _, _ = _temporal_split(pos)
    keep_u, keep_i, core_log = _iterative_train_core(prelim_train, min_uc=min_uc, min_ic=min_ic)
    filtered = pos[pos.user_raw.isin(keep_u) & pos.item_raw.isin(keep_i)].copy()
    train, val, test = _temporal_split(filtered)
    # enforce final minimum histories and one val/test
    counts = train.groupby("user_raw").size()
    valid_users = set(counts[counts >= min_uc].index) & set(val.user_raw.unique()) & set(test.user_raw.unique())
    valid_items = set(train.item_raw.unique())
    filtered = filtered[filtered.user_raw.isin(valid_users) & filtered.item_raw.isin(valid_items)].copy()
    train, val, test = _temporal_split(filtered)
    # map ids from training graph, keep val/test only if item seen in train
    user_map = {u: i for i, u in enumerate(sorted(train.user_raw.unique(), key=str))}
    item_map = {i: j for j, i in enumerate(sorted(train.item_raw.unique(), key=str))}
    def map_df(x: pd.DataFrame) -> pd.DataFrame:
        y = x[x.user_raw.isin(user_map) & x.item_raw.isin(item_map)].copy()
        y["user"] = y.user_raw.map(user_map).astype(np.int64)
        y["item"] = y.item_raw.map(item_map).astype(np.int64)
        return y[["user", "item", "timestamp", "line_idx", "user_raw", "item_raw"]]
    train_m, val_m, test_m = map_df(train), map_df(val), map_df(test)
    # keep users present in all splits after item filtering
    common = set(train_m.user.unique()) & set(val_m.user.unique()) & set(test_m.user.unique())
    train_m = train_m[train_m.user.isin(common)].copy()
    val_m = val_m[val_m.user.isin(common)].copy()
    test_m = test_m[test_m.user.isin(common)].copy()
    # remap compact users after final filtering
    remap_u = {u: k for k, u in enumerate(sorted(common))}
    for part in (train_m, val_m, test_m):
        part["user"] = part.user.map(remap_u).astype(np.int64)
    split = SplitData(name=name, train=train_m, val=val_m, test=test_m, n_users=len(remap_u), n_items=len(item_map))
    stats = dataset_stats(split) | {"core_log": core_log, "rating_threshold": rating_threshold, "min_uc": min_uc, "min_ic": min_ic}
    return split, stats


def dataset_stats(split: SplitData) -> dict:
    n_train = len(split.train)
    density = n_train / max(1, split.n_users * split.n_items)
    return {
        "name": split.name,
        "users": split.n_users,
        "items": split.n_items,
        "train_interactions": int(len(split.train)),
        "val_interactions": int(len(split.val)),
        "test_interactions": int(len(split.test)),
        "density_train": float(density),
        "mean_train_per_user": float(split.train.groupby("user").size().mean()) if split.n_users else 0.0,
    }


def interactions_to_csr(df: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
    return sparse.csr_matrix((np.ones(len(df), dtype=np.float32), (df.user.values, df.item.values)), shape=(n_users, n_items))


def item_user_vectors(train_csr: sparse.csr_matrix) -> sparse.csr_matrix:
    X = train_csr.T.tocsr().astype(np.float32)
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    inv = sparse.diags(1.0 / norms)
    return inv @ X


def save_split(split: SplitData, out_dir: str | Path) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    split.train.to_parquet(out / "train.parquet")
    split.val.to_parquet(out / "val.parquet")
    split.test.to_parquet(out / "test.parquet")
    (out / "meta.json").write_text(json.dumps({"name": split.name, "n_users": split.n_users, "n_items": split.n_items}, indent=2))
