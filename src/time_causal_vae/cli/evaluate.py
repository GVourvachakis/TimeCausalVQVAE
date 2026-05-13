"""Command-line entry point for selected evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from time_causal_vae.evaluation.checkpoints import TargetModelEvaluator
from time_causal_vae.experiments.legacy_config_adapter import (
    adapt_selected_config,
    load_selected_config,
)
from time_causal_vae.utils.serialization import save_obj


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a selected Time-Causal VAE checkpoint.",
    )
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the final_model directory containing model.pt.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory under outputs/ for generated evaluation artifacts.",
    )
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory used by market-data datasets.",
    )
    parser.add_argument(
        "--n-sample-test",
        type=int,
        help="Number of test samples for load_data and metric computation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Optional seed for load_data.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate the evaluator and print generated tensor shapes without saving metrics.",
    )
    return parser


def validate_output_dir(output_dir: str) -> Path:
    """Validate that generated evaluation artifacts stay below outputs/."""
    path = Path(output_dir)
    resolved = path.resolve()
    outputs_root = (Path.cwd() / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise SystemExit(
            f"--output-dir must be under ignored outputs/. Received: {output_dir}"
        ) from exc
    return path


def validate_model_dir(model_dir: str) -> Path:
    """Validate the expected legacy final_model checkpoint folder."""
    path = Path(model_dir)
    if not path.exists():
        raise SystemExit(f"--model-dir does not exist: {model_dir}")
    if not path.is_dir():
        raise SystemExit(f"--model-dir must be a directory: {model_dir}")
    if not (path / "model.pt").exists():
        raise SystemExit(f"--model-dir must contain model.pt: {model_dir}")
    return path


def ensure_legacy_exp_config(
    *,
    config_path: str,
    model_dir: Path,
    base_data_dir: str,
) -> Path:
    """Ensure ModelEvaluator can find exp_config.yaml next to final_model."""
    exp_config_path = model_dir.parent / "exp_config.yaml"
    if exp_config_path.exists():
        return exp_config_path

    selected_config = load_selected_config(config_path)
    legacy_config = adapt_selected_config(
        selected_config,
        output_dir=model_dir.parent,
        base_data_dir=base_data_dir,
        wandb=False,
    )
    with exp_config_path.open("w", encoding="utf-8") as handle:
        yaml.dump(legacy_config, handle, default_flow_style=False)
    return exp_config_path


def effective_n_sample_test(n_sample_test: int | None, *, dry_run: bool) -> int:
    """Choose a practical sample count when the CLI caller omits one."""
    if n_sample_test is not None:
        return n_sample_test
    if dry_run:
        return 16
    return 1000


def load_evaluation_batch(
    evaluator: Any,
    *,
    n_sample_test: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load real data, generated data, and reconstructions through ModelEvaluator."""
    real_data, fake_data, recon_data = evaluator.load_data(n_sample_test=n_sample_test, seed=seed)
    return real_data, fake_data, recon_data


def print_summary(
    *,
    backend: str,
    model_dir: Path,
    exp_config_path: Path,
    n_sample_test: int,
    dry_run: bool,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    recon_data: torch.Tensor,
) -> None:
    """Print a compact evaluation summary."""
    print("Evaluator dry run complete." if dry_run else "Evaluator run complete.")
    print(f"backend: {backend}")
    print(f"model_dir: {model_dir}")
    print(f"exp_config: {exp_config_path}")
    print(f"n_sample_test: {n_sample_test}")
    print(f"real_data: {_shape_text(real_data)}")
    print(f"fake_data: {_shape_text(fake_data)}")
    print(f"recon_data: {_shape_text(recon_data)}")


def save_evaluation_outputs(
    *,
    backend: str,
    output_dir: Path,
    hyper_metric: dict[str, Any],
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    recon_data: torch.Tensor,
    model_dir: Path,
    exp_config_path: Path,
    n_sample_test: int,
) -> None:
    """Save minimal evaluation outputs without changing metric formulas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_obj(hyper_metric, str(output_dir / "hyper_metric.pkl"))
    save_obj(
        {
            "real_data": real_data.cpu(),
            "fake_data": fake_data.cpu(),
            "recon_data": recon_data.cpu(),
        },
        str(output_dir / "evaluation_batch.pt"),
    )
    summary = {
        "backend": backend,
        "model_dir": str(model_dir),
        "exp_config": str(exp_config_path),
        "n_sample_test": n_sample_test,
        "real_data_shape": list(real_data.shape),
        "fake_data_shape": list(fake_data.shape),
        "recon_data_shape": list(recon_data.shape),
        "hyper_metric": {key: _json_scalar(value) for key, value in hyper_metric.items()},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _shape_text(value: torch.Tensor) -> str:
    return "x".join(str(dim) for dim in value.shape)


def _json_scalar(value: Any) -> float | int | str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return str(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, float | int | str):
        return value
    return str(value)


def main() -> None:
    """Run the command line interface."""
    parser = build_parser()
    args = parser.parse_args()

    # Load the selected YAML even when an adjacent legacy exp_config.yaml already exists.
    load_selected_config(args.config)
    model_dir = validate_model_dir(args.model_dir)
    output_dir = validate_output_dir(args.output_dir)
    n_sample_test = effective_n_sample_test(args.n_sample_test, dry_run=args.dry_run)
    exp_config_path = ensure_legacy_exp_config(
        config_path=args.config,
        model_dir=model_dir,
        base_data_dir=args.base_data_dir,
    )

    evaluator = TargetModelEvaluator(str(model_dir), base_data_dir=args.base_data_dir)
    real_data, fake_data, recon_data = load_evaluation_batch(
        evaluator,
        n_sample_test=n_sample_test,
        seed=args.seed,
    )
    print_summary(
        backend="target",
        model_dir=model_dir,
        exp_config_path=exp_config_path,
        n_sample_test=n_sample_test,
        dry_run=args.dry_run,
        real_data=real_data,
        fake_data=fake_data,
        recon_data=recon_data,
    )

    if args.dry_run:
        print("No metrics were saved because --dry-run was set.")
        return

    hyper_metric = evaluator.compute_hyper_metric(real_data, fake_data)
    save_evaluation_outputs(
        backend="target",
        output_dir=output_dir,
        hyper_metric=hyper_metric,
        real_data=real_data,
        fake_data=fake_data,
        recon_data=recon_data,
        model_dir=model_dir,
        exp_config_path=exp_config_path,
        n_sample_test=n_sample_test,
    )
    print(f"Saved evaluation outputs to {output_dir}")
