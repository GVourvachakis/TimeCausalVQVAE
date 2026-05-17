"""Select registered continuous or discrete model metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from time_causal_vae.experiments.model_registry import (
    load_registry,
    registry_summary,
    select_registered_model,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Select model metadata from trained_models/model_registry.yaml.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Optional registry YAML path. Defaults to trained_models/model_registry.yaml.",
    )
    parser.add_argument(
        "--experiment",
        help="Experiment id, for example sp500_vix or black_scholes.",
    )
    parser.add_argument(
        "--family",
        choices=["continuous", "discrete"],
        help="Model family to select.",
    )
    parser.add_argument(
        "--metric",
        help="Optional lower-is-better metric override, for example mmd or swd.",
    )
    parser.add_argument(
        "--profile",
        choices=["distributional", "tail_risk", "sequential_dependence", "balanced_market"],
        default="balanced_market",
        help="Profile used when the registry has no explicit selected candidate.",
    )
    parser.add_argument(
        "--allow-mmd-only",
        action="store_true",
        help="Allow profile fallback to choose from MMD alone.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print registered experiments and candidates instead of selecting one candidate.",
    )
    return parser


def main() -> None:
    """Run the registry-selection CLI."""
    args = build_parser().parse_args()
    if args.list:
        print(json.dumps(registry_summary(args.registry), indent=2, sort_keys=True))
        return
    if args.experiment is None or args.family is None:
        raise SystemExit("--experiment and --family are required unless --list is used.")

    registry = load_registry(args.registry)
    selected = select_registered_model(
        registry,
        experiment=args.experiment,
        family=args.family,
        metric=args.metric,
        profile=args.profile,
        allow_mmd_only=args.allow_mmd_only,
    )
    print(json.dumps(selected.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
