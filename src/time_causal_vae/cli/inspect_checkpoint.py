"""Command-line entry point for checkpoint inspection."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the checkpoint-inspection parser."""
    parser = argparse.ArgumentParser(description="Inspect a Time-Causal VAE checkpoint directory.")
    parser.add_argument("checkpoint", help="Path to a checkpoint directory.")
    return parser


def main() -> None:
    """Inspect files available in a checkpoint directory."""
    parser = build_parser()
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    for path in sorted(checkpoint.iterdir()):
        print(path.name)
