from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .attribution import compute_attribution_for_users, compute_shapley_for_users
from .data import load_amazon_books_2018, load_movielens_1m, preprocess_temporal_loo, item_user_vectors, save_split
from .metrics import evaluate
from .models import TrainConfig, cache_full_scores, get_item_embeddings, train_bprmf, train_lightgcn, pick_device
from .explanation import attribution_concentration, deletion_comprehensiveness, insertion_sufficiency
from .rerank import rerank_all
from .utils import sparse_fingerprint, write_json
from .validation import assert_item_vector_isolation, assert_rerank_nonzero, assert_shapley_shapes


def _hash_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def _hash_dict(obj: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def _resolve_path(value: str | Path) -> Path:
    """Resolve ~ and environment variables in config paths."""
    return Path(os.path.expandvars(str(value))).expanduser()


def _amazon_missing_message(path: Path) -> str:
    return f"""
Amazon Books input file not found: {path}

The Amazon run requires the UCSD Amazon Reviews 2018 5-core Books file:
  Books_5.json.gz

Place it at one of these locations:
  1) {Path('data/raw/Books_5.json.gz').resolve()}
  2) any custom path, then set the environment variable:
       export AMAZON_BOOKS_5=/absolute/path/to/Books_5.json.gz
     and keep the config value as ${{AMAZON_BOOKS_5}}
  3) edit configs/q1_lightgcn_amazon_template.yaml:
       dataset:
         books_5_json_gz: /absolute/path/to/Books_5.json.gz

Download source (UCSD Amazon Reviews 2018):
  https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

Direct file URL commonly used by the UCSD page:
  https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Books_5.json.gz

Example:
  mkdir -p data/raw
  curl -L -C - -o data/raw/Books_5.json.gz \
    https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Books_5.json.gz

The raw file is large. Start with dataset.max_rows or sample_users for a feasibility run.
""".strip()


def prepare_split(cfg: dict[str, Any]):
    ds = cfg["dataset"]
    raw_dir = _resolve_path(ds.get("raw_dir", "data/raw"))
    if ds["name"] == "ml1m":
        raw = load_movielens_1m(raw_dir)
    elif ds["name"] in {"amazon_books_2018", "amazon_books_2018_custom"}:
        path_value = os.environ.get("AMAZON_BOOKS_5") or ds.get("books_5_json_gz") or (raw_dir / "Books_5.json.gz")
        path = _resolve_path(path_value)
        if not path.exists():
            raise FileNotFoundError(_amazon_missing_message(path))
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
    name = b["name"]
    if name == "bprmf":
        model = train_bprmf(split.train, split.n_users, split.n_items, train_cfg, verbose=True)
    elif name == "lightgcn":
        model = train_lightgcn(
            split.train, split.n_users, split.n_items, train_cfg,
            n_layers=int(b.get("n_layers", 2)), verbose=True,
        )
    elif name == "hccf_validated_port":
        raise NotImplementedError(
            "HCCF is intentionally blocked until fork_commit, PORT.md, lockfile, "
            "and official-code validation artifacts exist."
        )
    else:
        raise NotImplementedError(f"Unsupported backbone: {name}")
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
    stage_times = {}
    t_stage = time.time()
    model, base_scores = train_backbone(split, cfg, seed)
    stage_times["train_and_score_seconds"] = time.time() - t_stage
    item_embeddings = get_item_embeddings(model)
    base_summary, base_per_user = evaluate(base_scores, split, item_vectors, ks=tuple(cfg["run"].get("ks", [5, 10, 20])))
    write_json(seed_dir / "base_summary.json", base_summary)

    a = cfg["attribution"]
    base_scores_hash = _hash_array(base_scores)
    attribution_hash = _hash_dict({
        "seed": int(seed),
        "backbone": cfg.get("backbone", {}),
        "attribution": a,
        "dataset": cfg.get("dataset", {}),
        "base_scores_hash": base_scores_hash,
        "cache_version": "v2-base-score-aware",
    })
    cache_tag = attribution_hash[:16]
    write_json(seed_dir / "attribution_cache_manifest.json", {
        "cache_tag": cache_tag,
        "base_scores_hash": base_scores_hash,
        "attribution_hash": attribution_hash,
        "note": "Checkpoint filenames include this tag. Old untagged checkpoints are intentionally ignored to prevent stale attribution reuse after backbone/config changes.",
    })

    t_stage = time.time()
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
        checkpoint_path=seed_dir / f"shapley_checkpoint_{cache_tag}.npz",
        save_every=int(a.get("save_every", 25)),
        alpha=float(a.get("alpha", 1.0)),
        beta=float(a.get("beta", 0.0)),
        lambda_pref=float(a.get("lambda_pref", 0.0)),
        lambda_attr_value=float(a.get("lambda_attr_value", 0.10)),
        value_mode=a.get("value_mode", "pairwise_logsigmoid"),
        n_val_negatives=int(a.get("n_val_negatives", 100)),
        antithetic=bool(a.get("antithetic", True)),
    )
    stage_times["shapley_seconds"] = time.time() - t_stage
    shape_report = assert_shapley_shapes(split, shapley)
    write_json(seed_dir / "shapley_shape_report.json", shape_report)
    explanation_report = {}
    explanation_report.update(attribution_concentration(shapley))
    explanation_report.update(deletion_comprehensiveness(base_scores, split, shapley, fraction=0.2, ks=(20,)))
    explanation_report.update(insertion_sufficiency(base_scores, split, shapley, fraction=0.2, ks=(20,)))
    write_json(seed_dir / "explanation_diagnostics.json", explanation_report)

    loo = None
    families = list(cfg["reranking"].get("primary_families", [])) + list(cfg["reranking"].get("secondary_families", []))
    if "loo-marginal" in families:
        t_stage = time.time()
        loo = compute_attribution_for_users(
            split, base_scores, item_vectors, method="loo-marginal",
            max_users=a.get("shapley_users"),
            exact_threshold=int(a.get("exact_threshold", 8)),
            seed=int(seed),
            max_players_per_user=a.get("max_players_per_user"),
            player_selection=a.get("player_selection", "stratified"),
            checkpoint_path=seed_dir / f"loo_checkpoint_{cache_tag}.npz",
            save_every=int(a.get("save_every", 25)),
            alpha=float(a.get("alpha", 1.0)), beta=float(a.get("beta", 0.0)),
            lambda_pref=float(a.get("lambda_pref", 0.0)),
            lambda_attr_value=float(a.get("lambda_attr_value", 0.10)),
            value_mode=a.get("value_mode", "pairwise_logsigmoid"),
            n_val_negatives=int(a.get("n_val_negatives", 100)),
        )
        stage_times["loo_seconds"] = time.time() - t_stage
        write_json(seed_dir / "loo_shape_report.json", assert_shapley_shapes(split, loo))

    t_stage = time.time()
    rows = []
    per_user_rows = []
    for fam in families:
        scores = rerank_all(
            base_scores,
            split,
            item_vectors,
            fam,
            shapley_by_user=shapley,
            loo_by_user=loo,
            lambda_attr=float(cfg["reranking"].get("lambda_attr", 0.10)),
            tau_att=float(cfg["reranking"].get("tau_att", 0.10)),
            intervention=cfg["reranking"].get("intervention", "native"),
            item_embeddings=item_embeddings,
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
            scores = rerank_all(
                base_scores, split, item_vectors, fam, shapley_by_user=shapley, loo_by_user=loo,
                lambda_attr=float(lam), tau_att=float(cfg["reranking"].get("tau_att", 0.10)),
                intervention=cfg["reranking"].get("intervention", "native"), item_embeddings=item_embeddings,
            )
            summary, _ = evaluate(scores, split, item_vectors, ks=tuple(cfg["run"].get("ks", [5, 10, 20])))
            summary.update({"seed": seed, "family": fam, "lambda_attr": lam, "backbone": cfg["backbone"]["name"], "dataset": split.name})
            sens_rows.append(summary)
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(seed_dir / "lambda_sensitivity.csv", index=False)
    stage_times["rerank_eval_sensitivity_seconds"] = time.time() - t_stage

    write_json(seed_dir / "runtime.json", {"seed": seed, "seconds": time.time() - t0, "stages": stage_times})
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
