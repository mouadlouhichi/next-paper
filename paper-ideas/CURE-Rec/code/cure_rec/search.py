"""Validation-only staged Torch BPR hyperparameter search."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from cure_rec.models import LeaveOneOutSplit, PopularityHybrid, PopularityRecommender, evaluate_leave_one_out
from cure_rec.torch_models import TorchBPRConfig, TorchBPRMFWithBias, torch_available


@dataclass(frozen=True)
class SearchConfig:
    stage_epochs: int = 40
    final_epochs: int = 200
    max_eval_users: int = 1_000
    top_k_stage_a: int = 3
    seed: int = 42


def _hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def _fit_validate(train, split: LeaveOneOutSplit, config: dict, search: SearchConfig):
    model = TorchBPRMFWithBias(TorchBPRConfig(max_epochs=search.stage_epochs, seed=search.seed, **config))
    model.fit(train, validation_split=split, max_eval_users=search.max_eval_users)
    metric = evaluate_leave_one_out(model, split, use_validation=True, max_users=search.max_eval_users)
    return model, metric


def _resume_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _checkpoint_rows(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def run_staged_bpr_search(split: LeaveOneOutSplit, output_dir: str | Path, search: SearchConfig = SearchConfig()) -> pd.DataFrame:
    """Run Stage A/B/C search without using test metrics for model selection."""
    if not torch_available():
        raise ImportError("PyTorch/MPS unavailable. Install with python -m pip install -e '.[dev,torch]'")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stage_a_path = root / "bpr_search_stage_a.csv"
    stage_a = _resume_rows(stage_a_path)
    completed_a = {row.get("config_hash") for row in stage_a}
    for lr, wd, strategy in product((0.001, 0.003, 0.01), (1e-5, 1e-4, 1e-3), ("uniform", "popularity_mixture", "hard_mixture")):
        cfg = {"embedding_dim": 64, "batch_size": 4096, "learning_rate": lr, "weight_decay": wd, "negative_strategy": strategy}
        config_hash = _hash(cfg)
        if config_hash in completed_a:
            continue
        _, metric = _fit_validate(split.train, split, cfg, search)
        stage_a.append({"stage": "A", "config_hash": config_hash, **cfg, **asdict(metric)})
        _checkpoint_rows(stage_a_path, stage_a)
    a = pd.DataFrame(stage_a).sort_values("ndcg_at_k", ascending=False)
    a.to_csv(stage_a_path, index=False)

    stage_b_path = root / "bpr_search_stage_b.csv"
    stage_b = _resume_rows(stage_b_path)
    completed_b = {row.get("config_hash") for row in stage_b}
    retained = a.head(search.top_k_stage_a).to_dict(orient="records")
    for base in retained:
        for dim, batch in product((32, 64, 128), (2048, 4096, 8192)):
            cfg = {"embedding_dim": dim, "batch_size": batch, "learning_rate": base["learning_rate"], "weight_decay": base["weight_decay"], "negative_strategy": base["negative_strategy"]}
            config_hash = _hash(cfg)
            if config_hash in completed_b:
                continue
            _, metric = _fit_validate(split.train, split, cfg, search)
            stage_b.append({"stage": "B", "parent_config_hash": base["config_hash"], "config_hash": config_hash, **cfg, **asdict(metric)})
            _checkpoint_rows(stage_b_path, stage_b)
    b = pd.DataFrame(stage_b).sort_values("ndcg_at_k", ascending=False)
    b.to_csv(stage_b_path, index=False)

    best = b.iloc[0].to_dict()
    best_cfg = {key: best[key] for key in ("embedding_dim", "batch_size", "learning_rate", "weight_decay", "negative_strategy")}
    final = TorchBPRMFWithBias(TorchBPRConfig(max_epochs=search.final_epochs, seed=search.seed, **best_cfg))
    final.fit(split.train, validation_split=split, max_eval_users=search.max_eval_users)
    popularity = PopularityRecommender().fit(split.train)
    stage_c = []
    for alpha in (0.25, 0.40, 0.55, 0.70, 0.85, 1.0):
        metric = evaluate_leave_one_out(PopularityHybrid(final, popularity, alpha), split, use_validation=True, max_users=search.max_eval_users)
        stage_c.append({"stage": "C", "alpha": alpha, **asdict(metric)})
    c = pd.DataFrame(stage_c).sort_values("ndcg_at_k", ascending=False)
    c.to_csv(root / "bpr_search_stage_c.csv", index=False)
    alpha = float(c.iloc[0]["alpha"])
    final_test = asdict(evaluate_leave_one_out(final, split, max_users=search.max_eval_users))
    hybrid_test = asdict(evaluate_leave_one_out(PopularityHybrid(final, popularity, alpha), split, max_users=search.max_eval_users))
    summary = pd.DataFrame([
        {"model": "torch_bpr_mf_bias", "selected_config_hash": _hash(best_cfg), "best_epoch": getattr(final, "best_validation_epoch", None), **final_test},
        {"model": f"bpr_popularity_hybrid_alpha_{alpha:.2f}", "selected_config_hash": _hash(best_cfg), "best_epoch": getattr(final, "best_validation_epoch", None), **hybrid_test},
    ])
    summary.to_csv(root / "bpr_search_final_test.csv", index=False)
    (root / "bpr_search_manifest.json").write_text(json.dumps({"search": asdict(search), "best_config": best_cfg, "alpha": alpha}, indent=2), encoding="utf-8")
    return summary
