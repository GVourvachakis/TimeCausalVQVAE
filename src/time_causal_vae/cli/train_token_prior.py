"""Command-line training entry point for causal token priors."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from time_causal_vae.token_prior import CausalTokenPriorConfig, build_token_prior_model
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the token-prior training CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train a causal autoregressive token prior.",
    )
    parser.add_argument("--config", required=True, help="Path to a token-prior YAML config.")
    parser.add_argument("--output-dir", required=True, help="Base output directory under outputs/.")
    parser.add_argument("--epochs", type=int, help="Override the config epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override train and eval batch sizes.")
    parser.add_argument("--learning-rate", type=float, help="Override the config learning rate.")
    parser.add_argument("--seed", type=int, help="Override the config random seed.")
    parser.add_argument("--device", help="Device override, for example cpu or cuda.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging. Takes precedence over --wandb.",
    )
    parser.add_argument(
        "--wandb-project",
        default="time-causal-vae",
        help="W&B project name used when --wandb is enabled.",
    )
    parser.add_argument("--wandb-entity", help="Optional W&B entity.")
    parser.add_argument("--wandb-run-name", help="Optional W&B display run name.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build token datasets and model, then print a summary without training.",
    )
    return parser


def main() -> None:
    """Run the token-prior training command."""
    parser = build_parser()
    args = parser.parse_args()

    output_dir = validate_output_dir(args.output_dir)
    raw_config = load_token_prior_yaml(args.config)
    run_config = build_run_config(raw_config, args=args, output_dir=output_dir)
    set_seed(cast(int, run_config["seed"]))

    prior_config = build_prior_config(run_config)
    train_tokens, train_conditions, eval_tokens, eval_conditions = load_token_tensors(
        cast(str, run_config["token_data_dir"]),
        prior_config=prior_config,
    )
    validate_token_data(train_tokens, prior_config, split_name="train")
    validate_token_data(eval_tokens, prior_config, split_name="eval")
    validate_condition_data(train_conditions, prior_config, split_name="train")
    validate_condition_data(eval_conditions, prior_config, split_name="eval")
    device = select_device(cast(str | None, run_config["device"]))
    model = build_token_prior_model(prior_config).to(device)

    if args.dry_run:
        print_dry_run_summary(
            run_config=run_config,
            prior_config=prior_config,
            train_tokens=train_tokens,
            train_conditions=train_conditions,
            eval_tokens=eval_tokens,
            eval_conditions=eval_conditions,
            model=model,
            device=device,
        )
        return

    run_training(
        model=model,
        prior_config=prior_config,
        run_config=run_config,
        train_tokens=train_tokens,
        train_conditions=train_conditions,
        eval_tokens=eval_tokens,
        eval_conditions=eval_conditions,
        device=device,
    )


def load_token_prior_yaml(path: str | Path) -> dict[str, Any]:
    """Load a token-prior experiment YAML file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Token-prior config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def build_run_config(
    raw_config: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Flatten YAML sections and CLI overrides into one run config."""
    experiment = require_mapping(raw_config, "experiment")
    data = require_mapping(raw_config, "data")
    model = require_mapping(raw_config, "model")
    training = require_mapping(raw_config, "training")

    seed = int(args.seed if args.seed is not None else experiment.get("seed", 0))
    epochs = int(args.epochs if args.epochs is not None else training.get("epochs", 100))
    batch_size = int(args.batch_size if args.batch_size is not None else training["batch_size"])
    eval_batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else training.get("eval_batch_size", batch_size)
    )
    learning_rate = float(
        args.learning_rate if args.learning_rate is not None else training["learning_rate"]
    )
    wandb_enabled = bool(args.wandb and not args.no_wandb and args.wandb_mode != "disabled")
    experiment_name = str(experiment["name"])
    run_name = f"{experiment_name}_seed{seed}"

    config: dict[str, Any] = {
        "config_path": str(Path(args.config)),
        "experiment_name": experiment_name,
        "dataset": str(experiment["dataset"]),
        "seed": seed,
        "output_dir": str(output_dir),
        "run_name": run_name,
        "run_dir": str(output_dir / run_name),
        "tokenizer_dir": str(data["tokenizer_dir"]),
        "token_data_dir": str(data["token_data_dir"]),
        "device": args.device,
        "wandb": wandb_enabled,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
        "wandb_mode": args.wandb_mode,
        "data": dict(data),
        "model": dict(model),
        "training": dict(training),
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
    }
    validate_positive_int("epochs", epochs)
    validate_positive_int("batch_size", batch_size)
    validate_positive_int("eval_batch_size", eval_batch_size)
    if learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be positive.")
    return config


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required config section as a dictionary."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Token-prior config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def validate_positive_int(name: str, value: int) -> None:
    """Validate a positive integer CLI or config value."""
    if value <= 0:
        raise SystemExit(f"{name} must be positive.")


def validate_output_dir(output_dir: str) -> Path:
    """Validate that token-prior artifacts stay below ignored outputs/."""
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


def load_token_tensors(
    token_data_dir: str | Path,
    *,
    prior_config: CausalTokenPriorConfig,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None]:
    """Load extracted train/eval token-index tensors and optional labels."""
    directory = Path(token_data_dir)
    train_path = directory / "train_tokens.pt"
    eval_path = directory / "eval_tokens.pt"
    if not train_path.exists():
        raise SystemExit(f"Missing train token artifact: {train_path}")
    if not eval_path.exists():
        raise SystemExit(f"Missing eval token artifact: {eval_path}")
    train_payload = torch.load(train_path, map_location="cpu", weights_only=True)
    eval_payload = torch.load(eval_path, map_location="cpu", weights_only=True)
    train_mapping = cast(Mapping[str, Tensor], train_payload)
    eval_mapping = cast(Mapping[str, Tensor], eval_payload)
    return (
        train_mapping["indices"].long(),
        load_conditions_from_payload(train_mapping, prior_config, artifact_path=train_path),
        eval_mapping["indices"].long(),
        load_conditions_from_payload(eval_mapping, prior_config, artifact_path=eval_path),
    )


def load_conditions_from_payload(
    payload: Mapping[str, Tensor],
    prior_config: CausalTokenPriorConfig,
    *,
    artifact_path: Path,
) -> Tensor | None:
    """Return labels as conditions when the token prior is conditional."""
    if prior_config.condition_injection == "none":
        return None
    if "labels" not in payload:
        raise SystemExit(f"Conditional token prior requires a 'labels' tensor in {artifact_path}.")
    labels = payload["labels"].float()
    if labels.ndim == 1:
        labels = labels[:, None]
    return labels


def build_prior_config(run_config: Mapping[str, Any]) -> CausalTokenPriorConfig:
    """Build model config for the causal token prior."""
    model_config = cast(Mapping[str, Any], run_config["model"])
    return CausalTokenPriorConfig(
        codebook_size=int(model_config["codebook_size"]),
        sequence_length=int(model_config["sequence_length"]),
        token_embedding_dim=int(model_config["token_embedding_dim"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        mlp_hidden_dim=int(model_config["mlp_hidden_dim"]),
        dropout=float(model_config.get("dropout", 0.0)),
        bos_token_id=optional_int(model_config.get("bos_token_id")),
        pad_token_id=optional_int(model_config.get("pad_token_id")),
        prediction_convention=str(
            model_config.get("prediction_convention", "bos_shifted_next_token")
        ),
        condition_dim=int(model_config.get("condition_dim", 0)),
        condition_injection=parse_condition_injection(
            model_config.get("condition_injection", "none")
        ),
        condition_hidden_dim=optional_int(model_config.get("condition_hidden_dim")),
        adaln_hidden_dim=optional_int(model_config.get("adaln_hidden_dim")),
        prior_type=parse_prior_type(model_config.get("prior_type", "single_code")),
        index_shape=optional_int_list(model_config.get("index_shape")),
        num_quantizers=int(model_config.get("num_quantizers", 1)),
        groups=int(model_config.get("groups", 1)),
        component_loss_weights=optional_float_list(model_config.get("component_loss_weights")),
    )


def optional_int(value: Any) -> int | None:
    """Return an optional integer config value."""
    if value is None:
        return None
    return int(value)


def optional_int_list(value: Any) -> list[int] | None:
    """Return an optional list of integers."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise SystemExit("index_shape must be a list of integers when provided.")
    return [int(item) for item in value]


def optional_float_list(value: Any) -> list[float] | None:
    """Return an optional list of floats."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise SystemExit("component_loss_weights must be a list of floats when provided.")
    return [float(item) for item in value]


def parse_prior_type(
    value: Any,
) -> Literal["single_code", "factorised_multi_code", "hierarchical_rvq_q2"]:
    """Parse the supported token-prior type."""
    parsed = str(value)
    if parsed not in {"single_code", "factorised_multi_code", "hierarchical_rvq_q2"}:
        raise SystemExit(
            "prior_type must be 'single_code', 'factorised_multi_code', or 'hierarchical_rvq_q2'."
        )
    return cast(Literal["single_code", "factorised_multi_code", "hierarchical_rvq_q2"], parsed)


def parse_condition_injection(value: Any) -> Literal["none", "additive", "adaln_lite"]:
    """Parse the supported condition-injection mode."""
    parsed = str(value)
    if parsed not in {"none", "additive", "adaln_lite"}:
        raise SystemExit("condition_injection must be 'none', 'additive', or 'adaln_lite'.")
    return cast(Literal["none", "additive", "adaln_lite"], parsed)


def validate_token_data(tokens: Tensor, config: CausalTokenPriorConfig, *, split_name: str) -> None:
    """Validate token tensor shape and range."""
    expected_shape: tuple[int, ...]
    if config.prior_type == "single_code":
        expected_shape = (config.sequence_length,)
    else:
        expected_shape = (config.sequence_length, *config.component_shape)
    if tuple(tokens.shape[1:]) != expected_shape:
        raise SystemExit(
            f"{split_name} tokens must have shape after batch {expected_shape}; "
            f"got {tuple(tokens.shape[1:])}."
        )
    if int(tokens.min().item()) < 0 or int(tokens.max().item()) >= config.codebook_size:
        raise SystemExit(f"{split_name} token values must be in [0, {config.codebook_size - 1}].")


def validate_condition_data(
    conditions: Tensor | None,
    config: CausalTokenPriorConfig,
    *,
    split_name: str,
) -> None:
    """Validate optional scalar or temporal condition tensors."""
    if config.condition_injection == "none":
        if conditions is not None:
            raise SystemExit(f"{split_name} conditions were loaded for an unconditional prior.")
        return
    if conditions is None:
        raise SystemExit(f"{split_name} conditions are required for conditional priors.")
    if conditions.ndim not in {2, 3}:
        raise SystemExit(
            f"{split_name} conditions must be [batch, condition_dim] or "
            f"[batch, sequence_length, condition_dim]; got {conditions.shape}."
        )
    if conditions.shape[-1] != config.condition_dim:
        raise SystemExit(
            f"{split_name} condition_dim must be {config.condition_dim}; "
            f"got {conditions.shape[-1]}."
        )
    if conditions.ndim == 3 and conditions.shape[1] != config.sequence_length:
        raise SystemExit(
            f"{split_name} temporal conditions must have length {config.sequence_length}; "
            f"got {conditions.shape[1]}."
        )


def select_device(device_name: str | None) -> torch.device:
    """Select the requested device, or prefer CUDA when available."""
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_dry_run_summary(
    *,
    run_config: Mapping[str, Any],
    prior_config: CausalTokenPriorConfig,
    train_tokens: Tensor,
    train_conditions: Tensor | None,
    eval_tokens: Tensor,
    eval_conditions: Tensor | None,
    model: nn.Module,
    device: torch.device,
) -> None:
    """Print a compact summary without training."""
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print("Dry run complete. No training was started and no artifacts were written.")
    print(f"experiment: {run_config['experiment_name']}")
    print(f"dataset: {run_config['dataset']}")
    print(f"token_data_dir: {run_config['token_data_dir']}")
    print(f"train_tokens_shape: {tuple(train_tokens.shape)}")
    print(
        "train_conditions_shape: "
        f"{None if train_conditions is None else tuple(train_conditions.shape)}"
    )
    print(f"eval_tokens_shape: {tuple(eval_tokens.shape)}")
    print(
        "eval_conditions_shape: "
        f"{None if eval_conditions is None else tuple(eval_conditions.shape)}"
    )
    print(f"token_prior_config: {asdict(prior_config)}")
    print(f"parameters: {n_parameters}")
    print(f"run_dir: {run_config['run_dir']}")
    print(f"epochs: {run_config['epochs']}")
    print(f"batch_size: {run_config['batch_size']}")
    print(f"learning_rate: {run_config['learning_rate']}")
    print(f"device: {device}")
    print(f"wandb: {run_config['wandb']}")


def run_training(
    *,
    model: nn.Module,
    prior_config: CausalTokenPriorConfig,
    run_config: Mapping[str, Any],
    train_tokens: Tensor,
    train_conditions: Tensor | None,
    eval_tokens: Tensor,
    eval_conditions: Tensor | None,
    device: torch.device,
) -> None:
    """Train the token prior and write checkpoint, config, and metric artifacts."""
    run_dir = Path(cast(str, run_config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(UTC)
    start_perf = time.perf_counter()
    wandb_run = maybe_start_wandb(run_config, prior_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(run_config["learning_rate"]))
    train_loader = token_data_loader(
        train_tokens,
        conditions=train_conditions,
        batch_size=int(run_config["batch_size"]),
        shuffle=True,
    )
    eval_loader = token_data_loader(
        eval_tokens,
        conditions=eval_conditions,
        batch_size=int(run_config["eval_batch_size"]),
        shuffle=False,
    )
    log_path = run_dir / "token_prior_training_log.jsonl"
    final_epoch_summary: dict[str, Any] | None = None
    best_epoch_summary: dict[str, Any] | None = None
    best_state_dict: dict[str, Tensor] | None = None

    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(1, int(run_config["epochs"]) + 1):
            train_metrics = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
            )
            eval_metrics = evaluate_epoch(model=model, data_loader=eval_loader, device=device)
            final_epoch_summary = {
                "epoch": epoch,
                **prefixed_metrics("train", train_metrics),
                **prefixed_metrics("eval", eval_metrics),
            }
            if best_epoch_summary is None or (
                final_epoch_summary["eval_cross_entropy"] < best_epoch_summary["eval_cross_entropy"]
            ):
                best_epoch_summary = dict(final_epoch_summary)
                best_state_dict = clone_state_dict_to_cpu(model)
            log_handle.write(json.dumps(final_epoch_summary, sort_keys=True) + "\n")
            log_handle.flush()
            if wandb_run is not None:
                wandb_run.log(final_epoch_summary, step=epoch)
            print(
                f"epoch={epoch} "
                f"train_ce={final_epoch_summary['train_cross_entropy']:.8f} "
                f"train_acc={final_epoch_summary['train_accuracy']:.8f} "
                f"eval_ce={final_epoch_summary['eval_cross_entropy']:.8f} "
                f"eval_acc={final_epoch_summary['eval_accuracy']:.8f}"
            )

    if final_epoch_summary is None:
        raise RuntimeError("Training did not run any epochs.")
    if best_epoch_summary is None or best_state_dict is None:
        raise RuntimeError("Best checkpoint state was not recorded.")

    end_time = datetime.now(UTC)
    elapsed_seconds = time.perf_counter() - start_perf
    runtime_summary = {
        "wall_clock_start_time": start_time.isoformat(),
        "wall_clock_end_time": end_time.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "config_path": run_config["config_path"],
        "run_dir": str(run_dir),
        "wandb_enabled": bool(run_config["wandb"]),
    }
    best_checkpoint_summary = build_best_checkpoint_summary(
        best_epoch_summary=best_epoch_summary,
        final_epoch_summary=final_epoch_summary,
    )
    write_json(run_dir / "token_prior_config.json", asdict(prior_config))
    write_json(run_dir / "training_config.json", serialisable_run_config(run_config))
    write_json(run_dir / "runtime_summary.json", runtime_summary)
    write_json(run_dir / "best_checkpoint_summary.json", best_checkpoint_summary)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "token_prior_config": asdict(prior_config),
            "training_config": serialisable_run_config(run_config),
            "runtime_summary": runtime_summary,
            "final_epoch_summary": final_epoch_summary,
        },
        run_dir / "token_prior.pt",
    )
    write_best_model_artifacts(
        run_dir=run_dir,
        best_state_dict=best_state_dict,
        prior_config=prior_config,
        run_config=run_config,
        runtime_summary=runtime_summary,
        best_checkpoint_summary=best_checkpoint_summary,
    )

    if wandb_run is not None:
        wandb_run.finish()

    print(f"training_complete: {run_dir}")
    print(f"runtime_seconds: {elapsed_seconds:.3f}")
    print(f"final_eval_cross_entropy: {final_epoch_summary['eval_cross_entropy']:.8f}")
    print(f"final_eval_accuracy: {final_epoch_summary['eval_accuracy']:.8f}")
    print(f"final_eval_perplexity: {final_epoch_summary['eval_perplexity']:.8f}")
    print(f"best_epoch: {best_checkpoint_summary['best_epoch']}")
    print(f"best_eval_cross_entropy: {best_checkpoint_summary['best_eval_cross_entropy']:.8f}")
    print(f"best_model_dir: {run_dir / 'best_model'}")


def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, Tensor]:
    """Return a detached CPU copy of model parameters for best-checkpoint saving."""
    return {
        name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()
    }


def build_best_checkpoint_summary(
    *,
    best_epoch_summary: Mapping[str, Any],
    final_epoch_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build metadata comparing the best and final token-prior epochs."""
    best_epoch = int(best_epoch_summary["epoch"])
    final_epoch = int(final_epoch_summary["epoch"])
    return {
        "selection_metric": "eval_cross_entropy",
        "best_epoch": best_epoch,
        "best_eval_cross_entropy": float(best_epoch_summary["eval_cross_entropy"]),
        "best_eval_accuracy": float(best_epoch_summary["eval_accuracy"]),
        "best_eval_perplexity": float(best_epoch_summary["eval_perplexity"]),
        "final_epoch": final_epoch,
        "final_eval_cross_entropy": float(final_epoch_summary["eval_cross_entropy"]),
        "final_eval_accuracy": float(final_epoch_summary["eval_accuracy"]),
        "final_eval_perplexity": float(final_epoch_summary["eval_perplexity"]),
        "best_differs_from_final": best_epoch != final_epoch,
        "best_epoch_summary": serialisable_run_config(best_epoch_summary),
        "final_epoch_summary": serialisable_run_config(final_epoch_summary),
    }


def write_best_model_artifacts(
    *,
    run_dir: Path,
    best_state_dict: Mapping[str, Tensor],
    prior_config: CausalTokenPriorConfig,
    run_config: Mapping[str, Any],
    runtime_summary: Mapping[str, Any],
    best_checkpoint_summary: Mapping[str, Any],
) -> None:
    """Write the selected best token-prior checkpoint under ``best_model/``."""
    best_model_dir = run_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    training_config = serialisable_run_config(run_config)
    write_json(best_model_dir / "token_prior_config.json", asdict(prior_config))
    write_json(best_model_dir / "training_config.json", training_config)
    write_json(best_model_dir / "runtime_summary.json", runtime_summary)
    write_json(best_model_dir / "best_checkpoint_summary.json", best_checkpoint_summary)
    torch.save(
        {
            "model_state_dict": dict(best_state_dict),
            "token_prior_config": asdict(prior_config),
            "training_config": training_config,
            "runtime_summary": dict(runtime_summary),
            "best_checkpoint_summary": dict(best_checkpoint_summary),
        },
        best_model_dir / "token_prior.pt",
    )


def token_data_loader(
    tokens: Tensor,
    *,
    conditions: Tensor | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[Tensor, ...]]:
    """Build a DataLoader over token-index tensors and optional conditions."""
    if conditions is None:
        dataset = cast(Dataset[tuple[Tensor, ...]], TensorDataset(tokens))
    else:
        dataset = cast(Dataset[tuple[Tensor, ...]], TensorDataset(tokens, conditions))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    *,
    model: nn.Module,
    data_loader: DataLoader[tuple[Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train one epoch and return aggregate token-prior metrics."""
    model.train()
    totals = MetricTotals()
    for batch in data_loader:
        tokens = batch[0]
        conditions = batch[1] if len(batch) > 1 else None
        batch_tokens = tokens.to(device)
        batch_conditions = None if conditions is None else conditions.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_tokens, conditions=batch_conditions)
        loss = cast(Tensor, output["loss"])
        loss.backward()
        optimizer.step()
        totals.update(output, batch_size=batch_tokens.shape[0])
    return totals.to_metrics()


def evaluate_epoch(
    *,
    model: nn.Module,
    data_loader: DataLoader[tuple[Tensor, ...]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate one epoch and return aggregate token-prior metrics."""
    model.eval()
    totals = MetricTotals()
    with torch.no_grad():
        for batch in data_loader:
            tokens = batch[0]
            conditions = batch[1] if len(batch) > 1 else None
            batch_tokens = tokens.to(device)
            batch_conditions = None if conditions is None else conditions.to(device)
            output = model(batch_tokens, conditions=batch_conditions)
            totals.update(output, batch_size=batch_tokens.shape[0])
    return totals.to_metrics()


class MetricTotals:
    """Accumulate batch-weighted token-prior metrics."""

    def __init__(self) -> None:
        """Initialise empty totals."""
        self.n_samples = 0
        self.cross_entropy_total = 0.0
        self.accuracy_total = 0.0
        self.perplexity_total = 0.0
        self.extra_totals: dict[str, float] = {}

    def update(self, output: Mapping[str, Any], *, batch_size: int) -> None:
        """Add one batch of metrics."""
        self.n_samples += batch_size
        self.cross_entropy_total += (
            float(cast(Tensor, output["cross_entropy"]).detach().cpu()) * batch_size
        )
        self.accuracy_total += float(cast(Tensor, output["accuracy"]).detach().cpu()) * batch_size
        self.perplexity_total += (
            float(cast(Tensor, output["perplexity"]).detach().cpu()) * batch_size
        )
        for key, value in output.items():
            if not key.startswith("component_"):
                continue
            if not isinstance(value, Tensor) or value.ndim != 0:
                continue
            self.extra_totals[key] = self.extra_totals.get(key, 0.0) + (
                float(value.detach().cpu()) * batch_size
            )

    def to_metrics(self) -> dict[str, float]:
        """Return mean metrics."""
        if self.n_samples == 0:
            raise RuntimeError("Token-prior DataLoader produced no samples.")
        metrics = {
            "cross_entropy": self.cross_entropy_total / self.n_samples,
            "accuracy": self.accuracy_total / self.n_samples,
            "perplexity": self.perplexity_total / self.n_samples,
        }
        metrics.update(
            {
                key: value / self.n_samples
                for key, value in sorted(self.extra_totals.items(), key=lambda item: item[0])
            }
        )
        return metrics


def prefixed_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    """Prefix metric names for train/eval logging."""
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def maybe_start_wandb(
    run_config: Mapping[str, Any],
    prior_config: CausalTokenPriorConfig,
) -> Any | None:
    """Start W&B only when the user explicitly enables it."""
    if not run_config["wandb"]:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "W&B logging was requested but wandb is not installed. "
            "Install the tracking dependency group or rerun with --no-wandb."
        ) from exc
    return wandb.init(
        project=run_config["wandb_project"],
        entity=run_config["wandb_entity"],
        name=run_config["wandb_run_name"] or run_config["run_name"],
        mode=run_config["wandb_mode"],
        config={
            "token_prior_config": asdict(prior_config),
            "training_config": serialisable_run_config(run_config),
        },
    )


def serialisable_run_config(run_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of the flattened run config."""
    return cast(dict[str, Any], json.loads(json.dumps(dict(run_config), default=str)))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a mapping as indented JSON with a trailing newline."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
