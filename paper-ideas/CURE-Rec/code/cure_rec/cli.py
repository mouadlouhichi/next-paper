"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cure_rec.analysis import analyze_dataset
from cure_rec.config import load_settings
from cure_rec.data import audit_interactions, load_dataset, load_interactions_csv, write_standardized_dataset
from cure_rec.pipeline import run_experiment
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


def _full_run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    workflow = run_full_workflow(
        settings,
        dataset=args.dataset,
        source=args.source,
        download=args.download,
        run_bpr=not args.skip_bpr,
        bpr_updates=args.bpr_updates,
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
    analyzer.add_argument("--bpr-updates", type=int, default=50_000)
    analyzer.add_argument("--max-eval-users", type=int, default=1_000)
    analyzer.add_argument("--seed", type=int, default=42)
    analyzer.set_defaults(handler=_analyze_data)

    full = subparsers.add_parser("full-run", help="Fetch/load data, audit it, analyze baseline models, then run CURE-Sim")
    full.add_argument("--config", type=Path, required=True)
    full.add_argument("--dataset", choices=(*PUBLIC_DATASETS, "csv"), required=True)
    full.add_argument("--source", type=Path, required=True)
    full.add_argument("--download", action="store_true")
    full.add_argument("--skip-bpr", action="store_true")
    full.add_argument("--bpr-updates", type=int, default=50_000)
    full.add_argument("--max-eval-users", type=int, default=1_000)
    full.set_defaults(handler=_full_run)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
