"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cure_rec.analysis import analyze_dataset
from cure_rec.calibration import run_calibration_sweep
from cure_rec.config import load_settings
from cure_rec.data import audit_interactions, load_dataset, load_interactions_csv, write_standardized_dataset
from cure_rec.experiments import postprocess_seed_sweep, run_seed_sweep
from cure_rec.observability import RunLogger
from cure_rec.pipeline import run_experiment
from cure_rec.regimes import run_regime_suite
from cure_rec.search import SearchConfig, run_final_bpr_audit, run_final_bpr_seed_replication, run_staged_bpr_search
from cure_rec.models import chronological_leave_one_out
from cure_rec.workflow import run_full_workflow


PUBLIC_DATASETS = ("movielens_1m", "coat", "yahoo_r3")


def _simulate(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    logger, _, decision = run_experiment(settings)
    print("\nCURE-Rec completed")
    print(f"Run directory: {logger.run_dir}")
    print(json.dumps({"action": decision.action, "portfolio": decision.selected_interventions, "lower_improvement": decision.lower_improvement}, indent=2))
    return 0


def _audit(args: argparse.Namespace) -> int:
    frame = load_interactions_csv(args.csv)
    audit = audit_interactions(frame)
    print(json.dumps(audit.__dict__, indent=2))
    return 0


def _load_data(args: argparse.Namespace) -> int:
    source = Path(args.source)
    result = load_dataset(args.dataset, source, download=args.download)
    audit = audit_interactions(result.interactions)
    payload = {
        "dataset": result.dataset,
        "metadata": result.metadata,
        "audit": audit.__dict__,
    }
    if args.output:
        output = write_standardized_dataset(result, args.output)
        payload["standardized_output"] = str(output)
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _analyze_data(args: argparse.Namespace) -> int:
    result = load_dataset(args.dataset, args.source, download=args.download)
    analysis = analyze_dataset(
        result,
        output_root=args.output_root,
        run_bpr=not args.skip_bpr,
        bpr_updates=args.bpr_updates,
        bpr_epochs=args.bpr_epochs,
        bpr_backend=args.bpr_backend,
        max_eval_users=args.max_eval_users,
        seed=args.seed,
    )
    print(json.dumps({
        "dataset": analysis.dataset,
        "analysis_run": str(analysis.run_dir),
        "permitted_claim": analysis.audit.permitted_claim,
        "model_metrics": analysis.model_metrics.to_dict(orient="records"),
    }, indent=2, default=str))
    return 0


def _search_bpr(args: argparse.Namespace) -> int:
    result = load_dataset(args.dataset, args.source, download=args.download)
    split = chronological_leave_one_out(result.interactions)
    summary = run_staged_bpr_search(
        split,
        args.output_root,
        SearchConfig(stage_epochs=args.stage_epochs, final_epochs=args.final_epochs, max_eval_users=args.max_eval_users, top_k_stage_a=args.top_k, seed=args.seed),
    )
    print(json.dumps({"search_run": str(args.output_root), "final_test": summary.to_dict(orient="records")}, indent=2, default=str))
    return 0


def _final_bpr_audit(args: argparse.Namespace) -> int:
    result = load_dataset(args.dataset, args.source, download=args.download)
    split = chronological_leave_one_out(result.interactions)
    summary = run_final_bpr_audit(split, args.search_root, args.output_root, seed=args.seed, max_eval_users=args.max_eval_users)
    print(json.dumps({"final_audit_run": str(args.output_root), "metrics": summary.to_dict(orient="records")}, indent=2, default=str))
    return 0


def _final_bpr_seeds(args: argparse.Namespace) -> int:
    result = load_dataset(args.dataset, args.source, download=args.download)
    split = chronological_leave_one_out(result.interactions)
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    metrics = run_final_bpr_seed_replication(split, args.search_root, args.output_root, seeds=seeds, max_eval_users=args.max_eval_users)
    print(json.dumps({"final_seed_run": str(args.output_root), "rows": len(metrics)}, indent=2, default=str))
    return 0


def _regimes(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    logger = RunLogger(settings)
    try:
        result = run_regime_suite(settings, logger)
        logger.close(status="completed")
    except Exception:
        logger.close(status="failed")
        raise
    print(json.dumps({
        "regime_run": str(result.run_dir),
        "selection_summary": result.summary.to_dict(orient="records"),
        "mean_attribution_error": float(result.attribution_recovery["absolute_error"].mean()),
    }, indent=2, default=str))
    return 0


def _sweep(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    result = run_seed_sweep(settings, seeds)
    print(json.dumps({
        "seed_sweep_run": str(result.run_dir),
        "seeds": seeds,
        "selection_frequency": result.decisions["selected_interventions"].value_counts().to_dict(),
    }, indent=2, default=str))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    result = run_calibration_sweep(
        settings,
        seeds,
        design=args.design,
        lhs_samples=args.lhs_samples,
        lhs_seed=args.lhs_seed,
    )
    print(json.dumps({
        "calibration_run": str(result.run_dir),
        "design": args.design,
        "configuration_count": len(result.configurations),
        "seed_decision_count": len(result.seed_decisions),
        "summary": result.summary.to_dict(orient="records"),
    }, indent=2, default=str))
    return 0


def _postprocess_sweep(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    result = postprocess_seed_sweep(args.run_dir, settings)
    print(json.dumps({
        "sweep_run": str(result.run_dir),
        "decision_rows": len(result.decisions),
        "base_feasibility_rows": len(result.base_feasibility),
        "generated_figures": str(result.run_dir / "figures"),
    }, indent=2, default=str))
    return 0


def _full_run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    workflow = run_full_workflow(
        settings,
        dataset=args.dataset,
        source=args.source,
        download=args.download,
        run_bpr=not args.skip_bpr,
        bpr_updates=args.bpr_updates,
        bpr_epochs=args.bpr_epochs,
        bpr_backend=args.bpr_backend,
        max_eval_users=args.max_eval_users,
    )
    print(json.dumps({
        "dataset": workflow.dataset.dataset,
        "dataset_analysis_run": str(workflow.analysis.run_dir),
        "cure_run": str(workflow.cure_run_dir),
        "permitted_claim": workflow.analysis.audit.permitted_claim,
        "decision": {
            "action": workflow.decision.action,
            "portfolio": workflow.decision.selected_interventions,
            "lower_improvement": workflow.decision.lower_improvement,
        },
    }, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CURE-Rec implementation scaffold")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Run exact CURE-Sim coalition game")
    simulate.add_argument("--config", type=Path, required=True)
    simulate.set_defaults(handler=_simulate)

    audit = subparsers.add_parser("audit-log", help="Audit a local recommendation interaction CSV")
    audit.add_argument("--csv", type=Path, required=True)
    audit.set_defaults(handler=_audit)

    loader = subparsers.add_parser("load-data", help="Load, standardize, and audit an approved public/local dataset")
    loader.add_argument("--dataset", choices=(*PUBLIC_DATASETS, "csv"), required=True)
    loader.add_argument("--source", type=Path, required=True, help="Dataset root directory, or a CSV path for --dataset csv")
    loader.add_argument("--download", action="store_true", help="Explicitly allow MovieLens-1M or Coat download")
    loader.add_argument("--output", type=Path, help="Optional standardized CSV output path")
    loader.set_defaults(handler=_load_data)

    analyzer = subparsers.add_parser("analyze-data", help="Profile an interaction dataset and run CPU baseline recommenders")
    analyzer.add_argument("--dataset", choices=(*PUBLIC_DATASETS, "csv"), required=True)
    analyzer.add_argument("--source", type=Path, required=True)
    analyzer.add_argument("--download", action="store_true")
    analyzer.add_argument("--output-root", type=Path, default=Path("runs"))
    analyzer.add_argument("--skip-bpr", action="store_true", help="Run popularity baseline only")
    analyzer.add_argument("--bpr-updates", type=int, default=500_000, help="NumPy fallback update budget")
    analyzer.add_argument("--bpr-epochs", type=int, default=200, help="PyTorch Adam maximum epochs")
    analyzer.add_argument("--bpr-backend", choices=("auto", "torch", "numpy"), default="auto")
    analyzer.add_argument("--max-eval-users", type=int, default=1_000)
    analyzer.add_argument("--seed", type=int, default=42)
    analyzer.set_defaults(handler=_analyze_data)

    search = subparsers.add_parser("search-bpr", help="Run staged validation-only Torch BPR and hybrid search")
    search.add_argument("--dataset", choices=(*PUBLIC_DATASETS, "csv"), required=True)
    search.add_argument("--source", type=Path, required=True)
    search.add_argument("--download", action="store_true")
    search.add_argument("--output-root", type=Path, required=True)
    search.add_argument("--stage-epochs", type=int, default=40)
    search.add_argument("--final-epochs", type=int, default=200)
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--max-eval-users", type=int, default=1_000)
    search.add_argument("--seed", type=int, default=42)
    search.set_defaults(handler=_search_bpr)

    final_audit = subparsers.add_parser("final-bpr-audit", help="Retrain and audit the frozen staged-search BPR configuration")
    final_audit.add_argument("--dataset", choices=(*PUBLIC_DATASETS, "csv"), required=True)
    final_audit.add_argument("--source", type=Path, required=True)
    final_audit.add_argument("--search-root", type=Path, required=True)
    final_audit.add_argument("--output-root", type=Path, required=True)
    final_audit.add_argument("--download", action="store_true")
    final_audit.add_argument("--seed", type=int, default=42)
    final_audit.add_argument("--max-eval-users", type=int, default=1_000)
    final_audit.set_defaults(handler=_final_bpr_audit)

    final_seeds = subparsers.add_parser("final-bpr-seeds", help="Replicate the frozen staged-search BPR configuration across seeds")
    final_seeds.add_argument("--dataset", choices=(*PUBLIC_DATASETS, "csv"), required=True)
    final_seeds.add_argument("--source", type=Path, required=True)
    final_seeds.add_argument("--search-root", type=Path, required=True)
    final_seeds.add_argument("--output-root", type=Path, required=True)
    final_seeds.add_argument("--seeds", default="42,43,44,45,46")
    final_seeds.add_argument("--download", action="store_true")
    final_seeds.add_argument("--max-eval-users", type=int, default=1_000)
    final_seeds.set_defaults(handler=_final_bpr_seeds)

    regimes = subparsers.add_parser("regimes", help="Run controlled additive/complementary/redundant/repair benchmark regimes")
    regimes.add_argument("--config", type=Path, required=True)
    regimes.set_defaults(handler=_regimes)

    sweep = subparsers.add_parser("sweep", help="Run paired multi-seed CURE-Sim experiments")
    sweep.add_argument("--config", type=Path, required=True)
    sweep.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated integer seeds")
    sweep.set_defaults(handler=_sweep)

    calibration = subparsers.add_parser("calibrate", help="Run pre-specified CURE-Sim OAT or Latin-hypercube behavioral sensitivity")
    calibration.add_argument("--config", type=Path, required=True)
    calibration.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated independent environment seeds")
    calibration.add_argument("--design", choices=("oat", "lhs"), default="oat")
    calibration.add_argument("--lhs-samples", type=int, default=16, help="Joint configurations for --design lhs")
    calibration.add_argument("--lhs-seed", type=int, default=20260805)
    calibration.set_defaults(handler=_calibrate)

    postprocess = subparsers.add_parser("postprocess-sweep", help="Regenerate aggregate seed assets from an existing completed sweep")
    postprocess.add_argument("--config", type=Path, required=True)
    postprocess.add_argument("--run-dir", type=Path, required=True)
    postprocess.set_defaults(handler=_postprocess_sweep)

    full = subparsers.add_parser("full-run", help="Fetch/load data, audit it, analyze baseline models, then run CURE-Sim")
    full.add_argument("--config", type=Path, required=True)
    full.add_argument("--dataset", choices=(*PUBLIC_DATASETS, "csv"), required=True)
    full.add_argument("--source", type=Path, required=True)
    full.add_argument("--download", action="store_true")
    full.add_argument("--skip-bpr", action="store_true")
    full.add_argument("--bpr-updates", type=int, default=500_000, help="NumPy fallback update budget")
    full.add_argument("--bpr-epochs", type=int, default=200, help="PyTorch Adam maximum epochs")
    full.add_argument("--bpr-backend", choices=("auto", "torch", "numpy"), default="auto")
    full.add_argument("--max-eval-users", type=int, default=1_000)
    full.set_defaults(handler=_full_run)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
