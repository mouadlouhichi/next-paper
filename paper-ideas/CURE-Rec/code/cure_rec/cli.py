"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cure_rec.config import load_settings
from cure_rec.data import audit_interactions, load_dataset, load_interactions_csv, write_standardized_dataset
from cure_rec.pipeline import run_experiment


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

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
