"""Command-line entry point for selecting evaluated model checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from time_causal_vae.evaluation.model_selection import select_model


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Select a best final_model folder from evaluated candidates.",
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Directory containing candidate training runs with final_model folders.",
    )
    parser.add_argument(
        "--criterion",
        choices=["mmd", "swd", "weighted_sum", "pareto"],
        default="mmd",
        help="Selection criterion. Lower values are better.",
    )
    parser.add_argument(
        "--mmd-weight",
        type=float,
        default=1.0,
        help="MMD weight for weighted_sum and Pareto tie-breaking.",
    )
    parser.add_argument(
        "--swd-weight",
        type=float,
        default=1.0,
        help="SWD weight for weighted_sum and Pareto tie-breaking.",
    )
    parser.add_argument(
        "--compute-missing",
        action="store_true",
        help="Compute missing hyper_metric.pkl files with the target evaluator.",
    )
    parser.add_argument(
        "--base-data-dir",
        default="data",
        help="Base data directory used only when --compute-missing is set.",
    )
    parser.add_argument(
        "--n-sample-test",
        type=int,
        default=1000,
        help="Evaluation sample count used only when --compute-missing is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Evaluation seed used only when --compute-missing is set.",
    )
    parser.add_argument(
        "--output",
        help="Optional metadata path. Defaults to <experiment-dir>/selected_model.json.",
    )
    return parser


def main() -> None:
    """Run the model-selection CLI."""
    args = build_parser().parse_args()
    metadata = select_model(
        experiment_dir=Path(args.experiment_dir),
        criterion=args.criterion,
        mmd_weight=args.mmd_weight,
        swd_weight=args.swd_weight,
        compute_missing=args.compute_missing,
        base_data_dir=args.base_data_dir,
        n_sample_test=args.n_sample_test,
        seed=args.seed,
        output_path=Path(args.output) if args.output else None,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
