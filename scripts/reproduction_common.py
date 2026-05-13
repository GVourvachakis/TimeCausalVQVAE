"""Shared helpers for selected paper reproduction scripts."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentSpec:
    """Selected experiment metadata used by reproduction wrappers."""

    name: str
    config_path: str


def main_for_experiment(spec: ExperimentSpec, argv: Sequence[str] | None = None) -> None:
    """Run or print selected train/evaluate commands for one experiment."""
    args = _build_parser(spec).parse_args(argv)
    output_root = Path(args.output_root)
    experiment_dir = output_root / spec.name

    train_command = _train_command(spec.config_path, experiment_dir)
    model_dir = Path(args.model_dir) if args.model_dir is not None else None

    if args.dry_run:
        _print_dry_run(spec, args.mode, train_command, experiment_dir, model_dir)
        return

    if args.mode in {"train", "both"}:
        before = _final_model_dirs(experiment_dir)
        _run_command(train_command)
        after = _final_model_dirs(experiment_dir)
        created = sorted(after - before, key=lambda path: path.stat().st_mtime)
        if args.mode == "both" and model_dir is None:
            if not created:
                raise SystemExit(
                    "Training finished but no new final_model directory was found. "
                    "Pass --model-dir explicitly to evaluate an existing checkpoint."
                )
            model_dir = created[-1]

    if args.mode in {"evaluate", "both"}:
        if model_dir is None:
            raise SystemExit("Evaluation mode requires --model-dir.")
        _run_command(_evaluate_command(spec.config_path, experiment_dir, model_dir))


def _build_parser(spec: ExperimentSpec) -> argparse.ArgumentParser:
    """Build the reproduction wrapper parser."""
    parser = argparse.ArgumentParser(description=f"Reproduce selected {spec.name} workflow.")
    parser.add_argument(
        "--output-root",
        default="outputs/reproduction",
        help="Root output directory; the script writes below <output-root>/<experiment>/.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "both"],
        default="both",
        help="Workflow stage to print or execute.",
    )
    parser.add_argument(
        "--model-dir",
        help="Existing final_model directory for evaluate mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of executing them.",
    )
    return parser


def _print_dry_run(
    spec: ExperimentSpec,
    mode: str,
    train_command: list[str],
    experiment_dir: Path,
    model_dir: Path | None,
) -> None:
    """Print the commands that would be executed."""
    print(f"Selected experiment: {spec.name}")
    print(f"Config: {spec.config_path}")
    if mode in {"train", "both"}:
        print(shlex.join(train_command))
    if mode in {"evaluate", "both"}:
        effective_model_dir = model_dir or experiment_dir / "<training-dir>" / "final_model"
        print(shlex.join(_evaluate_command(spec.config_path, experiment_dir, effective_model_dir)))


def _train_command(config_path: str, experiment_dir: Path) -> list[str]:
    """Return the selected training command."""
    return [
        "poetry",
        "run",
        "tcvae-train",
        "--config",
        config_path,
        "--output-dir",
        str(experiment_dir),
        "--no-wandb",
    ]


def _evaluate_command(config_path: str, experiment_dir: Path, model_dir: Path) -> list[str]:
    """Return the selected evaluation command."""
    return [
        "poetry",
        "run",
        "tcvae-evaluate",
        "--config",
        config_path,
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(experiment_dir / "evaluation"),
        "--base-data-dir",
        "data",
    ]


def _run_command(command: list[str]) -> None:
    """Execute one reproduction command from the repository root."""
    print(shlex.join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def _final_model_dirs(experiment_dir: Path) -> set[Path]:
    """Return final_model directories below one experiment output directory."""
    if not experiment_dir.exists():
        return set()
    return {path for path in experiment_dir.glob("*/final_model") if path.is_dir()}
