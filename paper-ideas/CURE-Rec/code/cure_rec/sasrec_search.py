"""Validation-only SASRec search, frozen audit, and fixed-config replication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from cure_rec.evaluation_audit import audit_evaluation
from cure_rec.models import LeaveOneOutSplit, PopularityRecommender, evaluate_leave_one_out
from cure_rec.sasrec import SASRecConfig, TorchSASRec
from cure_rec.torch_models import torch_available


@dataclass(frozen=True)
class SASRecSearchConfig:
    stage_epochs: int = 30
    final_epochs: int = 120
    max_eval_users: int = 1_000
    top_k_stage_a: int = 2
    seed: int = 42


def _hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def _resume(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict(orient="records") if path.exists() else []


def _checkpoint(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _fit_validate(train, split: LeaveOneOutSplit, config: dict, search: SASRecSearchConfig):
    model = TorchSASRec(SASRecConfig(max_epochs=search.stage_epochs, seed=search.seed, **config))
    model.fit(train, validation_split=split, max_eval_users=search.max_eval_users)
    metric = evaluate_leave_one_out(model, split, use_validation=True, max_users=search.max_eval_users)
    return model, metric


def run_staged_sasrec_search(split: LeaveOneOutSplit, output_dir: str | Path, search: SASRecSearchConfig = SASRecSearchConfig()) -> pd.DataFrame:
    """Select SASRec by validation NDCG only; this function never ranks test data.

    Stage A searches optimizer/dropout/negative sampling. Stage B varies capacity
    and sequence context for the retained validation candidates. Every completed
    row is checkpointed, so an interrupted MPS session can resume safely.
    """
    if not torch_available():
        raise ImportError("PyTorch/MPS unavailable. Install with python -m pip install -e '.[dev,torch]'")
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    stage_a_path = root / "sasrec_search_stage_a.csv"
    stage_a = _resume(stage_a_path); done_a = {row.get("config_hash") for row in stage_a}
    for lr, dropout, strategy in product((0.0005, 0.001, 0.003), (0.10, 0.20), ("uniform", "popularity_mixture")):
        config = {
            "embedding_dim": 64, "max_sequence_length": 50, "num_heads": 2, "num_layers": 2,
            "dropout": dropout, "batch_size": 1024, "learning_rate": lr,
            "weight_decay": 1e-5, "negative_strategy": strategy,
        }
        config_hash = _hash(config)
        if config_hash in done_a:
            continue
        _, metric = _fit_validate(split.train, split, config, search)
        stage_a.append({"stage": "A", "config_hash": config_hash, **config, **asdict(metric)})
        _checkpoint(stage_a_path, stage_a)
    a = pd.DataFrame(stage_a).sort_values("ndcg_at_k", ascending=False)
    a.to_csv(stage_a_path, index=False)
    if a.empty:
        raise RuntimeError("SASRec Stage A produced no validation rows")

    stage_b_path = root / "sasrec_search_stage_b.csv"
    stage_b = _resume(stage_b_path); done_b = {row.get("config_hash") for row in stage_b}
    for parent in a.head(search.top_k_stage_a).to_dict(orient="records"):
        for embedding_dim, num_layers, sequence_length in product((64, 128), (1, 2), (25, 50)):
            # Keep the number of attention heads compatible with every dimension.
            config = {
                "embedding_dim": embedding_dim, "max_sequence_length": sequence_length,
                "num_heads": 2, "num_layers": num_layers, "dropout": float(parent["dropout"]),
                "batch_size": 1024, "learning_rate": float(parent["learning_rate"]),
                "weight_decay": float(parent["weight_decay"]), "negative_strategy": str(parent["negative_strategy"]),
            }
            config_hash = _hash(config)
            if config_hash in done_b:
                continue
            _, metric = _fit_validate(split.train, split, config, search)
            stage_b.append({"stage": "B", "parent_config_hash": parent["config_hash"], "config_hash": config_hash, **config, **asdict(metric)})
            _checkpoint(stage_b_path, stage_b)
    b = pd.DataFrame(stage_b).sort_values("ndcg_at_k", ascending=False)
    b.to_csv(stage_b_path, index=False)
    if b.empty:
        raise RuntimeError("SASRec Stage B produced no validation rows")
    best = b.iloc[0].to_dict()
    keys = ("embedding_dim", "max_sequence_length", "num_heads", "num_layers", "dropout", "batch_size", "learning_rate", "weight_decay", "negative_strategy")
    selected = {key: best[key] for key in keys}
    # Cast CSV-loaded numeric values so the emitted frozen configuration is
    # explicit and accepted by SASRecConfig on every platform.
    for key in ("embedding_dim", "max_sequence_length", "num_heads", "num_layers", "batch_size"):
        selected[key] = int(selected[key])
    for key in ("dropout", "learning_rate", "weight_decay"):
        selected[key] = float(selected[key])
    selection = pd.DataFrame([{
        "model": "torch_sasrec", "selected_config_hash": _hash(selected),
        "selection_metric": "validation_ndcg_at_10", "validation_ndcg_at_10": best["ndcg_at_k"],
        "validation_recall_at_10": best["recall_at_k"], **selected,
    }])
    selection.to_csv(root / "sasrec_search_selection.csv", index=False)
    (root / "sasrec_search_manifest.json").write_text(json.dumps({
        "search": asdict(search), "best_config": selected,
        "selection_rule": "maximum validation NDCG@10; test metrics intentionally not computed during search",
    }, indent=2), encoding="utf-8")
    return selection


def _load_selected_config(search_root: str | Path) -> tuple[dict, int]:
    manifest = json.loads((Path(search_root) / "sasrec_search_manifest.json").read_text())
    return manifest["best_config"], int(manifest["search"]["final_epochs"])


def run_final_sasrec_audit(
    split: LeaveOneOutSplit,
    search_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    max_eval_users: int = 1_000,
) -> pd.DataFrame:
    """Train one frozen selected SASRec configuration and perform its first test audit."""
    if not torch_available():
        raise ImportError("PyTorch/MPS unavailable. Install with python -m pip install -e '.[dev,torch]'")
    config, epochs = _load_selected_config(search_root)
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    model = TorchSASRec(SASRecConfig(max_epochs=epochs, seed=seed, **config))
    model.fit(split.train, validation_split=split, max_eval_users=max_eval_users)
    popularity = PopularityRecommender().fit(split.train)
    sasrec_metric = asdict(evaluate_leave_one_out(model, split, max_users=max_eval_users))
    pop_metric = asdict(evaluate_leave_one_out(popularity, split, max_users=max_eval_users))
    audit, pairwise = audit_evaluation([popularity, model], split, max_users=max_eval_users, seed=seed)
    selected_hash = _hash(config)
    summary = pd.DataFrame([
        {"seed": seed, "model": "popularity", "selected_config_hash": selected_hash, **pop_metric},
        {**sasrec_metric, "seed": seed, "model": "torch_sasrec_final", "selected_config_hash": selected_hash,
         "best_validation_epoch": getattr(model, "best_validation_epoch", None),
         "restored_checkpoint_epoch": getattr(model, "restored_checkpoint_epoch", None)},
    ])
    summary.to_csv(root / "final_sasrec_test_metrics.csv", index=False)
    audit.to_csv(root / "final_sasrec_evaluation_audit.csv", index=False)
    pairwise.to_csv(root / "final_sasrec_pairwise_accuracy.csv", index=False)
    pd.DataFrame(model.loss_history).to_csv(root / "final_sasrec_loss.csv", index=False)
    pd.DataFrame(model.validation_history).to_csv(root / "final_sasrec_validation.csv", index=False)
    (root / "final_sasrec_config.json").write_text(json.dumps({
        "selected_config_hash": selected_hash, "config": config, "seed": seed,
        "best_validation_epoch": getattr(model, "best_validation_epoch", None),
        "restored_checkpoint_epoch": getattr(model, "restored_checkpoint_epoch", None),
    }, indent=2), encoding="utf-8")
    return summary


def run_final_sasrec_seed_replication(
    split: LeaveOneOutSplit,
    search_root: str | Path,
    output_dir: str | Path,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    *,
    max_eval_users: int = 1_000,
) -> pd.DataFrame:
    """Replicate a frozen SASRec configuration without retuning or test selection."""
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    config, _ = _load_selected_config(search_root); selected_hash = _hash(config)
    rows = []
    for seed in seeds:
        seed_root = root / f"seed-{seed}"
        metric_file = seed_root / "final_sasrec_test_metrics.csv"
        summary = None
        if metric_file.exists():
            candidate = pd.read_csv(metric_file)
            labels = set(candidate.get("model", pd.Series(dtype=str)).astype(str))
            hashes = set(candidate.get("selected_config_hash", pd.Series(dtype=str)).dropna().astype(str))
            if not candidate.empty and {"popularity", "torch_sasrec_final"}.issubset(labels) and selected_hash in hashes:
                summary = candidate.copy()
        if summary is None:
            summary = run_final_sasrec_audit(split, search_root, seed_root, seed=seed, max_eval_users=max_eval_users)
        if "seed" not in summary:
            summary = summary.copy(); summary.insert(0, "seed", seed)
        rows.append(summary)
    metrics = pd.concat(rows, ignore_index=True)
    metrics.to_csv(root / "final_sasrec_seed_metrics.csv", index=False)
    sasrec = metrics[metrics["model"] == "torch_sasrec_final"].groupby("seed", as_index=False).first()
    popularity = metrics[metrics["model"] == "popularity"].groupby("seed", as_index=False).first()[["seed", "recall_at_k", "ndcg_at_k"]]
    if len(sasrec) != len(seeds) or len(popularity) != len(seeds):
        raise RuntimeError(f"Incomplete SASRec replication: expected {len(seeds)} paired rows; got {len(sasrec)} SASRec and {len(popularity)} popularity rows.")
    paired = sasrec.merge(popularity.rename(columns={"recall_at_k": "pop_recall_at_k", "ndcg_at_k": "pop_ndcg_at_k"}), on="seed", how="left", validate="one_to_one")
    paired["delta_recall_at_k"] = paired["recall_at_k"] - paired["pop_recall_at_k"]
    paired["delta_ndcg_at_k"] = paired["ndcg_at_k"] - paired["pop_ndcg_at_k"]
    paired.to_csv(root / "final_sasrec_seed_paired_metrics.csv", index=False)
    aggregate = paired[["recall_at_k", "ndcg_at_k", "delta_recall_at_k", "delta_ndcg_at_k"]].agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    aggregate.to_csv(root / "final_sasrec_seed_summary.csv", index=False)
    return paired
