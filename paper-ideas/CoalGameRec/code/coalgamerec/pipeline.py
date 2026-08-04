from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .attribution import compute_shapley_for_users
from .data import load_amazon_books_2018, load_movielens_1m, preprocess_temporal_loo, item_user_vectors, save_split
from .metrics import evaluate
from .models import TrainConfig, cache_full_scores, train_bprmf, pick_device
from .rerank import rerank_all
from .utils import sparse_fingerprint, write_json
from .validation import assert_item_vector_isolation, assert_rerank_nonzero, assert_shapley_shapes


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def prepare_split(cfg: dict[str, Any]):
    ds = cfg["dataset"]
    raw_dir = Path(ds.get("raw_dir", "data/raw"))
    if ds["name"] == "ml1m":
        raw = load_movielens_1m(raw_dir)
    elif ds["name"] in {"amazon_books_2018", "amazon_books_2018_custom"}:
        path = ds.get("books_5_json_gz") or raw_dir / "Books_5.json.gz"
        raw = load_amazon_books_2018(path, max_rows=ds.get("max_rows"))
    else:
        raise ValueError(f"Unsupported dataset: {ds['name']}")
    split, stats = preprocess_temporal_loo(
        raw,
        name=ds["name"],
        rating_threshold=float(ds.get("rating_threshold", 4.0)),
        min_uc=int(ds.get("min_user_core", 5)),
        min_ic=int(ds.get("min_item_core", 5)),
        sample_users=ds.get("sample_users"),
        sample_seed=int(ds.get("sample_seed", 20260803)),
    )
    return split, stats


def train_backbone(split, cfg: dict[str, Any], seed: int):
    b = cfg["backbone"]
    if b["name"] != "bprmf":
        raise NotImplementedError(
            f"Backbone {b['name']} is not implemented in this local runner. "
            "Use bprmf for prototype runs, or add the validated HCCF adapter after PORT.md validation."
        )
    train_cfg = TrainConfig(
        dim=int(b.get("dim", 64)),
        lr=float(b.get("lr", 2e-3)),
        weight_decay=float(b.get("weight_decay", 1e-5)),
        epochs=int(b.get("epochs", 20)),
        batch_size=int(b.get("batch_size", 4096)),
        n_neg=int(b.get("n_neg", 4)),
        seed=int(seed),
        device=cfg["run"].get("device", "auto"),
    )
    model = train_bprmf(split.train, split.n_users, split.n_items, train_cfg, verbose=True)
    base_scores = cache_full_scores(
        model,
        split.n_users,
        batch_size=int(b.get("score_batch_size", 512)),
        chunk_items=b.get("chunk_items"),
    )
    return model, base_scores


def run_seed(split, item_vectors, cfg: dict[str, Any], seed: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_dir = out_dir / "raw" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, base_scores = train_backbone(split, cfg, seed)
    base_summary, base_per_user = evaluate(base_scores, split, item_vectors, ks=tuple(cfg["run"].get("ks", [5, 10, 20])))
    write_json(seed_dir / "base_summary.json", base_summary)

    a = cfg["attribution"]
    shapley = compute_shapley_for_users(
        split,
        base_scores,
        item_vectors,
        max_users=a.get("shapley_users"),
        m=int(a.get("m_permutations", 128)),
        exact_threshold=int(a.get("exact_threshold", 8)),
        seed=int(seed),
        max_players_per_user=a.get("max_players_per_user"),
        player_selection=a.get("player_selection", "similarity"),
        checkpoint_path=seed_dir / "shapley_checkpoint.npz",
        save_every=int(a.get("save_every", 25)),
        alpha=float(a.get("alpha", 0.70)),
        beta=float(a.get("beta", 0.30)),
        lambda_pref=float(a.get("lambda_pref", 0.20)),
        lambda_attr_value=float(a.get("lambda_attr_value", 0.10)),
    )
    shape_report = assert_shapley_shapes(split, shapley)
    write_json(seed_dir / "shapley_shape_report.json", shape_report)

    families = list(cfg["reranking"].get("primary_families", [])) + list(cfg["reranking"].get("secondary_families", []))
    rows = []
    per_user_rows = []
    for fam in families:
        scores = rerank_all(
            base_scores,
            split,
            item_vectors,
            fam,
            shapley_by_user=shapley,
            lambda_attr=float(cfg["reranking"].get("lambda_attr", 0.10)),
            tau_att=float(cfg["reranking"].get("tau_att", 0.10)),
        )
        summary, per_user = evaluate(scores, split, item_vectors, ks=tuple(cfg["run"].get("ks", [5, 10, 20])))
        summary.update({"seed": seed, "family": fam, "backbone": cfg["backbone"]["name"], "dataset": split.name})
        rows.append(summary)
        for metric, values in per_user.items():
            per_user_rows.append(pd.DataFrame({"seed": seed, "family": fam, "metric": metric, "user": np.arange(len(values)), "value": values}))
    summary_df = pd.DataFrame(rows)
    per_user_df = pd.concat(per_user_rows, ignore_index=True) if per_user_rows else pd.DataFrame()
    summary_df.to_csv(seed_dir / "summary_by_family.csv", index=False)
    per_user_df.to_csv(seed_dir / "per_user_metrics.csv", index=False)

    # Reranking-strength sensitivity for primary families only.
    sens_rows = []
    for lam in cfg["reranking"].get("lambda_attr_sensitivity", []):
        for fam in cfg["reranking"].get("primary_families", []):
            scores = rerank_all(base_scores, split, item_vectors, fam, shapley_by_user=shapley, lambda_attr=float(lam), tau_att=float(cfg["reranking"].get("tau_att", 0.10)))
            summary, _ = evaluate(scores, split, item_vectors, ks=tuple(cfg["run"].get("ks", [5, 10, 20])))
            summary.update({"seed": seed, "family": fam, "lambda_attr": lam, "backbone": cfg["backbone"]["name"], "dataset": split.name})
            sens_rows.append(summary)
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(seed_dir / "lambda_sensitivity.csv", index=False)

    write_json(seed_dir / "runtime.json", {"seed": seed, "seconds": time.time() - t0})
    return summary_df, per_user_df


def run_pipeline(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    out_dir = Path(cfg["run"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.resolved.json", cfg)

    split, dataset_stats = prepare_split(cfg)
    save_split(split, out_dir / "splits")
    write_json(out_dir / "dataset_stats.json", dataset_stats)

    item_vectors = item_user_vectors(split.train_csr)
    write_json(out_dir / "item_vectors_report.json", assert_item_vector_isolation(split) | {"hash": sparse_fingerprint(item_vectors)})

    all_summary = []
    all_per_user = []
    for seed in cfg["run"].get("seeds", [42]):
        summary_df, per_user_df = run_seed(split, item_vectors, cfg, int(seed), out_dir)
        all_summary.append(summary_df)
        all_per_user.append(per_user_df)
    summary = pd.concat(all_summary, ignore_index=True)
    per_user = pd.concat(all_per_user, ignore_index=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    summary.to_csv(out_dir / "tables" / "summary_by_seed_family.csv", index=False)
    metric_cols = [c for c in summary.columns if c.startswith("HitRate@") or c.startswith("NDCG@") or c.startswith("Coverage@") or c.startswith("ILD@")]
    agg = summary.groupby(["dataset", "backbone", "family"])[metric_cols].agg(["mean", "std"])
    agg.to_csv(out_dir / "tables" / "summary_mean_std.csv")
    per_user.to_csv(out_dir / "raw" / "per_user_metrics_all.csv", index=False)
    write_json(
        out_dir / "manifest.json",
        {
            "note": "Executable journal-style pipeline output. Confirmatory only if config uses validated HCCF and preregistered artifacts exist.",
            "config_path": str(config_path),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(pick_device(cfg["run"].get("device", "auto"))),
            "dataset_stats": dataset_stats,
        },
    )
    return out_dir
