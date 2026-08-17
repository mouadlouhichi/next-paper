

def load_movielens_25m(root: str | Path, *, download: bool = False, logger: RunLogger | None = None) -> DatasetLoadResult:
    """Load MovieLens-25M for an automatically reproducible second ranking dataset."""
    root_path = _ensure_directory(root)
    ratings_path = root_path / "ratings.csv"
    if not ratings_path.exists():
        if not download:
            raise FileNotFoundError("MovieLens-25M ratings.csv not found; use download=True.")
        archive = _download("https://files.grouplens.org/datasets/movielens/ml-25m.zip", root_path / "ml-25m.zip", logger)
        _safe_extract_zip(archive, root_path)
        ratings_path = _find_file(root_path, ("ratings.csv",))
    frame = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating", "timestamp"])
    frame = frame.rename(columns={"userId":"user_id", "movieId":"item_id"})
    frame["response"] = (frame["rating"] >= 4).astype(int)
    interactions = _standardize(frame, "movielens_25m", "observed")
    metadata = {"ratings_path": str(ratings_path), "rows": len(interactions), "has_exposure_log": False, "note": "chronological ranking only; not causal policy evidence"}
    _log(logger, "dataset_loaded", dataset="movielens_25m", **metadata)
    return DatasetLoadResult("movielens_25m", interactions, metadata)
