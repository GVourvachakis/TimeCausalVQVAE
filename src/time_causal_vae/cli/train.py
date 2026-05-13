"""Command-line entry point for selected training."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ml_collections

from time_causal_vae.data.pipeline import DataPipeline as TargetDataPipeline
from time_causal_vae.experiments.legacy_config_adapter import load_legacy_config
from time_causal_vae.models.factory import ModelFactory
from time_causal_vae.training.config import BaseTrainerConfig as TargetBaseTrainerConfig
from time_causal_vae.training.pipeline import TrainingPipeline as TargetTrainingPipeline
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train a selected Time-Causal VAE configuration.",
    )
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory under outputs/ for generated checkpoints and logs.",
    )
    parser.add_argument("--device", help="Optional device override passed to the trainer.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory used by market-data datasets.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging for this run.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable the W&B callback for this run. Takes precedence over --wandb.",
    )
    parser.add_argument(
        "--wandb-project",
        default="time-causal-vae",
        help="W&B project name used when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-entity",
        help="Optional W&B entity used when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        help="Optional W&B run name used when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode. Environment variable WANDB_MODE is respected when omitted.",
    )
    parser.add_argument("--epochs", type=int, help="Override the selected config epoch count.")
    parser.add_argument("--seed", type=int, help="Override the selected config random seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build data, model, and trainer config, then print a summary without training.",
    )
    return parser


def build_training_config(
    exp_config: ml_collections.ConfigDict,
) -> TargetBaseTrainerConfig:
    """Build a trainer config from adapted legacy fields."""
    common_kwargs: dict[str, Any] = {
        "output_dir": exp_config.output_dir,
        "learning_rate": exp_config.lr,
        "per_device_train_batch_size": exp_config.train_batch_size,
        "per_device_eval_batch_size": exp_config.eval_batch_size,
        "optimizer_cls": exp_config.optimizer,
        "optimizer_params": None,
        "scheduler_cls": None,
        "scheduler_params": None,
        "steps_saving": exp_config.steps_saving,
        "steps_predict": exp_config.steps_predict,
        "seed": exp_config.seed,
        "num_epochs": exp_config.epochs,
        "wandb_callback": exp_config.wandb,
        "wandb_output_dir": str(Path(exp_config.base_output_dir) / "wandb"),
        "ploter": exp_config.get("ploter", "path"),
    }
    wandb_kwargs: dict[str, Any] = {
        "wandb_project": exp_config.get("wandb_project", "time-causal-vae"),
        "wandb_entity": exp_config.get("wandb_entity"),
        "wandb_run_name": exp_config.get("wandb_run_name"),
        "wandb_mode": exp_config.get("wandb_mode"),
    }
    common_kwargs.update(wandb_kwargs)

    training_config = TargetBaseTrainerConfig(
        **common_kwargs,
    )
    return training_config


def build_target_components(
    exp_config: ml_collections.ConfigDict,
) -> tuple[Any, Any, Any, TargetBaseTrainerConfig]:
    """Build selected target data, model, and trainer config components."""
    set_seed(exp_config.seed)

    data_pipeline = TargetDataPipeline()
    train_dataset, eval_dataset = data_pipeline(exp_config)

    model_factory = ModelFactory()
    model = model_factory(exp_config)

    training_config = build_training_config(exp_config)
    return train_dataset, eval_dataset, model, training_config


def validate_output_dir(output_dir: str) -> Path:
    """Validate that generated training artifacts stay below outputs/."""
    path = Path(output_dir)
    resolved = path.resolve()
    outputs_root = (Path.cwd() / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise SystemExit(
            f"--output-dir must be under ignored outputs/. Received: {output_dir}"
        ) from exc
    if "trained_models" in resolved.parts:
        raise SystemExit("--output-dir must not point into trained_models/")
    return path


def validate_data_requirements(exp_config: ml_collections.ConfigDict) -> None:
    """Fail early with a clear message for selected configs requiring local data."""
    if exp_config.dataset != "SP500VIX":
        return

    required_file = Path(exp_config.base_data_dir) / "sp500vix" / "sp500vix_normalized.npy"
    if required_file.exists():
        return

    raise SystemExit(
        "SP500/VIX training requires the local data file "
        f"{required_file}. Place sp500vix_normalized.npy there before running this config."
    )


def print_dry_run_summary(
    exp_config: ml_collections.ConfigDict,
    train_dataset: Any,
    eval_dataset: Any,
    model: Any,
    training_config: Any,
    backend: str,
) -> None:
    """Print a compact dry-run summary."""
    train_data_shape = _shape_text(train_dataset.data)
    train_label_shape = _shape_text(train_dataset.labels)
    eval_data_shape = _shape_text(eval_dataset.data)
    eval_label_shape = _shape_text(eval_dataset.labels)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())

    print("Dry run complete. No training was started.")
    print(f"backend: {backend}")
    print(f"experiment: {exp_config.experiment_name}")
    print(f"config dataset: {exp_config.dataset}")
    print(f"config model: {exp_config.model}")
    print(f"encoder/decoder: {exp_config.encoder}/{exp_config.decoder}")
    print(f"conditioner/prior: {exp_config.conditioner}/{exp_config.prior}")
    print(f"train data: {len(train_dataset)} samples")
    print(f"train shapes: data {train_data_shape}, labels {train_label_shape}")
    print(
        f"eval data: {len(eval_dataset)} samples, data {eval_data_shape}, labels {eval_label_shape}"
    )
    print(f"model class: {model.__class__.__name__}")
    print(f"parameters: {n_parameters}")
    print(f"output_dir: {training_config.output_dir}")
    print(f"epochs: {training_config.num_epochs}")
    print(f"learning_rate: {training_config.learning_rate}")
    print(f"train_batch_size: {training_config.per_device_train_batch_size}")
    print(f"eval_batch_size: {training_config.per_device_eval_batch_size}")
    print(f"wandb: {training_config.wandb_callback}")
    if training_config.wandb_callback:
        print(f"wandb_project: {training_config.wandb_project}")
        print(f"wandb_entity: {training_config.wandb_entity}")
        print(f"wandb_run_name: {training_config.wandb_run_name}")
        print(f"wandb_mode: {training_config.wandb_mode}")
    print(f"device override: {exp_config.device_name}")


def run_training(
    exp_config: ml_collections.ConfigDict,
    train_dataset: Any,
    eval_dataset: Any,
    model: Any,
    training_config: Any,
) -> None:
    """Run the selected training pipeline."""
    start_time = datetime.now(UTC)
    start_perf = time.perf_counter()
    train_pipeline = TargetTrainingPipeline(
        model=model,
        training_config=training_config,
        exp_config=exp_config,
    )
    trainer = train_pipeline(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        device_name=exp_config.device_name,
    )
    trainer.train(log_output=True)
    end_time = datetime.now(UTC)
    _write_runtime_summary(
        trainer=trainer,
        backend="target",
        config_path=exp_config.get("config_path"),
        start_time=start_time,
        end_time=end_time,
        elapsed_seconds=time.perf_counter() - start_perf,
    )


def _write_runtime_summary(
    *,
    trainer: Any,
    backend: str,
    config_path: str | None,
    start_time: datetime,
    end_time: datetime,
    elapsed_seconds: float,
) -> None:
    """Write runtime metadata beside the legacy-format training artifacts."""
    training_dir = Path(trainer.training_dir)
    final_model_path = training_dir / "final_model"
    summary = {
        "wall_clock_start_time": start_time.isoformat(),
        "wall_clock_end_time": end_time.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "device": str(getattr(trainer, "device", "")),
        "backend": backend,
        "config_path": config_path,
        "training_run_dir": str(training_dir),
        "final_model_path": str(final_model_path),
    }
    summary_path = training_dir / "runtime_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"runtime_summary: {summary_path}")


def _shape_text(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "-"
    return "x".join(str(dim) for dim in shape)


def main() -> None:
    """Run the command line interface."""
    parser = build_parser()
    args = parser.parse_args()

    output_dir = validate_output_dir(args.output_dir)
    wandb_enabled = args.wandb and not args.no_wandb and args.wandb_mode != "disabled"
    exp_config = load_legacy_config(
        args.config,
        output_dir=output_dir,
        device=args.device,
        epochs=args.epochs,
        seed=args.seed,
        wandb=wandb_enabled,
        base_data_dir=args.base_data_dir,
    )
    exp_config.config_path = str(Path(args.config))
    exp_config.wandb_project = args.wandb_project
    exp_config.wandb_entity = args.wandb_entity
    exp_config.wandb_run_name = args.wandb_run_name
    exp_config.wandb_mode = args.wandb_mode
    validate_data_requirements(exp_config)

    train_dataset, eval_dataset, model, training_config = build_target_components(exp_config)

    if args.dry_run:
        print_dry_run_summary(
            exp_config,
            train_dataset,
            eval_dataset,
            model,
            training_config,
            "target",
        )
        return

    run_training(exp_config, train_dataset, eval_dataset, model, training_config)
