"""Print experimental multidimensional profile metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from time_causal_vae.experiments.multidim_profiles import (
    list_multidim_profiles,
    select_multidim_profile,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Select metadata from trained_models/multidim_profiles.yaml. "
            "This does not read checkpoint weights or the public model registry."
        ),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=None,
        help=(
            "Optional multidim profile YAML path. Defaults to "
            "trained_models/multidim_profiles.yaml."
        ),
    )
    parser.add_argument(
        "--experiment",
        help="Experiment id, for example multifactor_market or sp500_50_panel.",
    )
    parser.add_argument(
        "--profile",
        help="Profile id, for example portfolio_tail or balanced_empirical.",
    )
    parser.add_argument(
        "--family",
        choices=["continuous", "discrete"],
        help="Optional guard for the selected profile family.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and profiles instead of selecting one profile.",
    )
    return parser


def main() -> None:
    """Run the multidimensional profile selector."""
    args = build_parser().parse_args()
    if args.list:
        payload = list_multidim_profiles(args.profiles)
    else:
        if args.experiment is None or args.profile is None:
            raise SystemExit("--experiment and --profile are required unless --list is used.")
        payload = select_multidim_profile(
            experiment=args.experiment,
            profile=args.profile,
            family=args.family,
            path=args.profiles,
        ).to_dict()
    print(yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
