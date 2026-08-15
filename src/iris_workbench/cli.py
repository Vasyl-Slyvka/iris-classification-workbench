"""Command-line interface for validating and evaluating Iris experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import DataValidationError, load_dataset
from .evaluation import compare_models, evaluate_holdout
from .models import MODEL_NAMES
from .reporting import write_csv, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iris-workbench",
        description="Reproducible classical-ML experiments on Iris data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a dataset and print its summary")
    validate.add_argument("--data", type=Path, default=Path("data/iris.csv"))
    validate.add_argument("--json", dest="json_path", type=Path)

    compare = subparsers.add_parser("compare", help="compare registered model pipelines")
    compare.add_argument("--data", type=Path, default=Path("data/iris.csv"))
    compare.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    compare.add_argument("--folds", type=int, default=5)
    compare.add_argument("--repeats", type=int, default=3)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--csv", dest="csv_path", type=Path)
    compare.add_argument("--json", dest="json_path", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="run stratified holdout diagnostics")
    evaluate.add_argument("--data", type=Path, default=Path("data/iris.csv"))
    evaluate.add_argument("--model", choices=MODEL_NAMES, default="knn")
    evaluate.add_argument("--test-size", type=float, default=0.2)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--tune", action="store_true")
    evaluate.add_argument("--tuning-folds", type=int, default=5)
    evaluate.add_argument("--json", dest="json_path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame, summary = load_dataset(args.data)
        if args.command == "validate":
            payload = summary.to_dict()
            if args.json_path:
                write_json(payload, args.json_path)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        if args.command == "compare":
            results = compare_models(
                frame,
                models=args.models,
                folds=args.folds,
                repeats=args.repeats,
                seed=args.seed,
            )
            if args.csv_path:
                write_csv(results, args.csv_path)
            records = results.to_dict(orient="records")
            if args.json_path:
                write_json(records, args.json_path)
            print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
            return 0

        payload = evaluate_holdout(
            frame,
            args.model,
            test_size=args.test_size,
            seed=args.seed,
            tune=args.tune,
            tuning_folds=args.tuning_folds,
        )
        if args.json_path:
            write_json(payload, args.json_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (DataValidationError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
