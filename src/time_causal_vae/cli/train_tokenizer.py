"""Command-line training entry point for the causal VQ tokenizer."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import ml_collections
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from time_causal_vae.data.base import BaseDataset, DatasetOutput, collate_dataset_output
from time_causal_vae.data.pipeline import DataPipeline
from time_causal_vae.models.discrete.tokenizers import (
    CausalVQTokenizer,
    TokenizerAuxiliaryLossContext,
    VQTokenizerConfig,
)
from time_causal_vae.utils.random import set_seed


@dataclass(frozen=True)
class TokenizerDatasetBundle:
    """Datasets and optional raw train dataset metadata for tokenizer training."""

    train: BaseDataset
    eval: BaseDataset
    raw_train_dataset: Any | None


def build_parser() -> argparse.ArgumentParser:
    """Build the tokenizer training CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train a standalone causal VQ tokenizer.",
    )
    parser.add_argument("--config", required=True, help="Path to a tokenizer experiment YAML file.")
    parser.add_argument("--output-dir", required=True, help="Base output directory under outputs/.")
    parser.add_argument("--epochs", type=int, help="Override the config epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override train and eval batch sizes.")
    parser.add_argument("--learning-rate", type=float, help="Override the config learning rate.")
    parser.add_argument("--seed", type=int, help="Override the config random seed.")
    parser.add_argument("--device", help="Device override, for example cpu or cuda.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory for datasets that require local files.",
    )
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
    parser.add_argument("--wandb-run-name", help="Optional run name and artifact subdirectory.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build data and model, then print a summary without training or writing artifacts.",
    )
    return parser


def main() -> None:
    """Run the tokenizer training command."""
    parser = build_parser()
    args = parser.parse_args()

    output_dir = validate_output_dir(args.output_dir)
    raw_config = load_tokenizer_yaml(args.config)
    run_config = build_run_config(raw_config, args=args, output_dir=output_dir)
    set_seed(cast(int, run_config["seed"]))

    dataset_bundle = build_datasets(run_config)
    tokenizer_config = build_tokenizer_config(run_config)
    tokenizer = CausalVQTokenizer(tokenizer_config)
    device = select_device(cast(str | None, run_config["device"]))
    tokenizer.to(device)
    auxiliary_loss_context = build_auxiliary_loss_context(
        tokenizer_config,
        dataset_bundle.raw_train_dataset,
        device=device,
    )
    validate_requested_auxiliary_context(tokenizer_config, auxiliary_loss_context)

    if args.dry_run:
        print_dry_run_summary(
            run_config=run_config,
            tokenizer_config=tokenizer_config,
            train_dataset=dataset_bundle.train,
            eval_dataset=dataset_bundle.eval,
            tokenizer=tokenizer,
            device=device,
            auxiliary_loss_context=auxiliary_loss_context,
        )
        return

    run_training(
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        run_config=run_config,
        train_dataset=dataset_bundle.train,
        device=device,
        auxiliary_loss_context=auxiliary_loss_context,
    )


def load_tokenizer_yaml(path: str | Path) -> dict[str, Any]:
    """Load a tokenizer experiment YAML file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Tokenizer config must be a mapping: {path}")
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
    epochs = int(args.epochs if args.epochs is not None else training.get("epochs", 20))
    batch_size = int(
        args.batch_size if args.batch_size is not None else training.get("train_batch_size", 256)
    )
    eval_batch_size = int(
        args.batch_size if args.batch_size is not None else training.get("eval_batch_size", 256)
    )
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else training.get("learning_rate", 1e-3)
    )
    wandb_enabled = bool(args.wandb and not args.no_wandb and args.wandb_mode != "disabled")
    experiment_name = str(experiment["name"])
    run_name = args.wandb_run_name or f"{experiment_name}_seed{seed}"

    config: dict[str, Any] = {
        "config_path": str(Path(args.config)),
        "experiment_name": experiment_name,
        "dataset": str(experiment["dataset"]),
        "seed": seed,
        "output_dir": str(output_dir),
        "run_name": run_name,
        "run_dir": str(output_dir / run_name),
        "base_data_dir": args.base_data_dir,
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
        raise SystemExit(f"Tokenizer config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def validate_positive_int(name: str, value: int) -> None:
    """Validate a positive integer CLI or config value."""
    if value <= 0:
        raise SystemExit(f"{name} must be positive.")


def validate_output_dir(output_dir: str) -> Path:
    """Validate that tokenizer artifacts stay below local outputs/."""
    path = Path(output_dir)
    resolved = path.resolve()
    outputs_root = (Path.cwd() / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise SystemExit(
            f"--output-dir must be under local outputs/. Received: {output_dir}"
        ) from exc
    return path


def build_datasets(run_config: Mapping[str, Any]) -> TokenizerDatasetBundle:
    """Build train and eval datasets with the target data pipeline."""
    data_config = cast(Mapping[str, Any], run_config["data"])
    exp_config = ml_collections.ConfigDict()
    exp_config.dataset = run_config["dataset"]
    exp_config.n_sample = int(data_config["n_samples"])
    exp_config.train_n_sample = optional_positive_int(data_config.get("train_n_samples"))
    exp_config.eval_n_sample = optional_positive_int(data_config.get("eval_n_samples"))
    exp_config.n_timestep = int(data_config["n_timesteps"])
    exp_config.base_data_dir = run_config["base_data_dir"]
    exp_config.data_params = dict(cast(Mapping[str, Any], data_config.get("data_params", {})))
    if "rho" in data_config:
        exp_config.rho = data_config["rho"]
    pipeline = DataPipeline()
    train_dataset, eval_dataset = cast(tuple[BaseDataset, BaseDataset], pipeline(exp_config))
    return TokenizerDatasetBundle(
        train=train_dataset,
        eval=eval_dataset,
        raw_train_dataset=pipeline.base_dataset,
    )


def build_tokenizer_config(run_config: Mapping[str, Any]) -> VQTokenizerConfig:
    """Build the model config for the causal VQ tokenizer."""
    model_config = cast(Mapping[str, Any], run_config["model"])
    data_config = cast(Mapping[str, Any], run_config["data"])
    return VQTokenizerConfig(
        data_dim=int(model_config.get("data_dim", data_config["data_dim"])),
        data_length=int(model_config.get("data_length", data_config["n_timesteps"])),
        embedding_dim=int(model_config["embedding_dim"]),
        codebook_size=int(model_config["codebook_size"]),
        commitment_weight=float(model_config["commitment_weight"]),
        encoder_hidden_dim=int(model_config["encoder_hidden_dim"]),
        decoder_hidden_dim=int(model_config["decoder_hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        dilations=tuple(int(value) for value in cast(Iterable[Any], model_config["dilations"])),
        dropout=float(model_config.get("dropout", 0.0)),
        condition_dim=int(model_config.get("condition_dim", data_config.get("condition_dim", 0))),
        kmeans_init=bool(model_config.get("kmeans_init", False)),
        kmeans_iters=int(model_config.get("kmeans_iters", 10)),
        use_cosine_sim=bool(model_config.get("use_cosine_sim", False)),
        codebook_dim=optional_int(model_config.get("codebook_dim")),
        threshold_ema_dead_code=float(model_config.get("threshold_ema_dead_code", 0.0)),
        decay=float(model_config.get("decay", 0.8)),
        usage_regularization_weight=float(model_config.get("usage_regularization_weight", 0.0)),
        usage_regularization_type=cast(
            Literal["none", "entropy"],
            str(model_config.get("usage_regularization_type", "none")),
        ),
        quantizer_type=cast(
            Literal["vector", "standard_vq", "residual_vq", "grouped_residual_vq"],
            str(model_config.get("quantizer_type", "vector")),
        ),
        num_quantizers=int(model_config.get("num_quantizers", 1)),
        groups=int(model_config.get("groups", 1)),
        shared_codebook=bool(model_config.get("shared_codebook", False)),
        stochastic_sample_codes=bool(model_config.get("stochastic_sample_codes", False)),
        sample_codebook_temp=float(model_config.get("sample_codebook_temp", 0.0)),
        factor_reconstruction_loss_weight=float(
            model_config.get("factor_reconstruction_loss_weight", 0.0)
        ),
        factor_covariance_loss_weight=float(model_config.get("factor_covariance_loss_weight", 0.0)),
        factor_correlation_loss_weight=float(
            model_config.get("factor_correlation_loss_weight", 0.0)
        ),
        inverse_projected_covariance_loss_weight=float(
            model_config.get("inverse_projected_covariance_loss_weight", 0.0)
        ),
        inverse_projected_correlation_loss_weight=float(
            model_config.get("inverse_projected_correlation_loss_weight", 0.0)
        ),
        sector_block_loss_weight=float(model_config.get("sector_block_loss_weight", 0.0)),
        equal_weight_portfolio_vol_loss_weight=float(
            model_config.get("equal_weight_portfolio_vol_loss_weight", 0.0)
        ),
    )


def optional_int(value: Any) -> int | None:
    """Return ``None`` or an integer config value."""
    if value is None:
        return None
    return int(value)


def optional_positive_int(value: Any) -> int | None:
    """Return ``None`` or a validated positive integer config value."""
    if value is None:
        return None
    count = int(value)
    if count <= 0:
        raise SystemExit("split-specific sample counts must be positive when provided.")
    return count


def build_auxiliary_loss_context(
    tokenizer_config: VQTokenizerConfig,
    raw_train_dataset: Any | None,
    *,
    device: torch.device,
) -> TokenizerAuxiliaryLossContext | None:
    """Build optional factor-projection auxiliary loss metadata."""
    if not has_requested_auxiliary_losses(tokenizer_config):
        return None
    projection_state = getattr(raw_train_dataset, "projection_state", None)
    if projection_state is None:
        return None

    standardization_stats = getattr(raw_train_dataset, "standardization_stats", None)
    standardization_mean = None
    standardization_std = None
    if isinstance(standardization_stats, Mapping):
        standardization_mean = torch.as_tensor(standardization_stats["mean"]).detach().to(device)
        standardization_std = torch.as_tensor(standardization_stats["std"]).detach().to(device)

    return TokenizerAuxiliaryLossContext(
        projection_basis=torch.as_tensor(projection_state.basis).detach().to(device),
        projection_mean=torch.as_tensor(projection_state.mean).detach().to(device),
        standardization_mean=standardization_mean,
        standardization_std=standardization_std,
        sector_labels=maybe_tensor_to_device(
            getattr(raw_train_dataset, "sector_labels", None), device
        ),
        inverse_project_to_raw=standardization_mean is not None and standardization_std is not None,
    )


def maybe_tensor_to_device(value: Any, device: torch.device) -> Tensor | None:
    """Convert an optional tensor-like value to a detached tensor on ``device``."""
    if value is None:
        return None
    return torch.as_tensor(value).detach().to(device)


def validate_requested_auxiliary_context(
    tokenizer_config: VQTokenizerConfig,
    auxiliary_loss_context: TokenizerAuxiliaryLossContext | None,
) -> None:
    """Fail clearly when requested auxiliary losses cannot be computed in training."""
    if not has_requested_auxiliary_losses(tokenizer_config):
        return
    if needs_inverse_projection_context(tokenizer_config) and auxiliary_loss_context is None:
        raise SystemExit(
            "Tokenizer auxiliary losses requiring inverse projection were requested, "
            "but the dataset did not expose factor-projection metadata."
        )
    if (
        tokenizer_config.sector_block_loss_weight > 0.0
        and auxiliary_loss_context is not None
        and auxiliary_loss_context.sector_labels is None
    ):
        raise SystemExit(
            "sector_block_loss_weight requires sector labels from the factor-projected dataset."
        )


def has_requested_auxiliary_losses(tokenizer_config: VQTokenizerConfig) -> bool:
    """Return whether any tokenizer auxiliary loss has a positive weight."""
    return any(
        weight > 0.0
        for weight in (
            tokenizer_config.factor_reconstruction_loss_weight,
            tokenizer_config.factor_covariance_loss_weight,
            tokenizer_config.factor_correlation_loss_weight,
            tokenizer_config.inverse_projected_covariance_loss_weight,
            tokenizer_config.inverse_projected_correlation_loss_weight,
            tokenizer_config.sector_block_loss_weight,
            tokenizer_config.equal_weight_portfolio_vol_loss_weight,
        )
    )


def needs_inverse_projection_context(tokenizer_config: VQTokenizerConfig) -> bool:
    """Return whether the requested auxiliary losses require projection metadata."""
    return any(
        weight > 0.0
        for weight in (
            tokenizer_config.inverse_projected_covariance_loss_weight,
            tokenizer_config.inverse_projected_correlation_loss_weight,
            tokenizer_config.sector_block_loss_weight,
            tokenizer_config.equal_weight_portfolio_vol_loss_weight,
        )
    )


def select_device(device_name: str | None) -> torch.device:
    """Select the requested device, or prefer CUDA when available."""
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_dry_run_summary(
    *,
    run_config: Mapping[str, Any],
    tokenizer_config: VQTokenizerConfig,
    train_dataset: BaseDataset,
    eval_dataset: BaseDataset,
    tokenizer: CausalVQTokenizer,
    device: torch.device,
    auxiliary_loss_context: TokenizerAuxiliaryLossContext | None,
) -> None:
    """Print a compact summary without training."""
    n_parameters = sum(parameter.numel() for parameter in tokenizer.parameters())
    print("Dry run complete. No training was started and no artifacts were written.")
    print(f"experiment: {run_config['experiment_name']}")
    print(f"dataset: {run_config['dataset']}")
    print(f"train data: {len(train_dataset)} samples, shape {_shape_text(train_dataset.data)}")
    print(f"eval data: {len(eval_dataset)} samples, shape {_shape_text(eval_dataset.data)}")
    print(f"tokenizer_config: {asdict(tokenizer_config)}")
    print(f"parameters: {n_parameters}")
    print(f"run_dir: {run_config['run_dir']}")
    print(f"epochs: {run_config['epochs']}")
    print(f"batch_size: {run_config['batch_size']}")
    print(f"learning_rate: {run_config['learning_rate']}")
    print(f"device: {device}")
    print(f"wandb: {run_config['wandb']}")
    print(f"auxiliary_loss_context: {auxiliary_loss_context is not None}")


def run_training(
    *,
    tokenizer: CausalVQTokenizer,
    tokenizer_config: VQTokenizerConfig,
    run_config: Mapping[str, Any],
    train_dataset: BaseDataset,
    device: torch.device,
    auxiliary_loss_context: TokenizerAuxiliaryLossContext | None,
) -> None:
    """Train the tokenizer and write checkpoint, config, and metric artifacts."""
    run_dir = Path(cast(str, run_config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(UTC)
    start_perf = time.perf_counter()
    wandb_run = maybe_start_wandb(run_config, tokenizer_config)

    data_loader = DataLoader(
        train_dataset,
        batch_size=int(run_config["batch_size"]),
        shuffle=True,
        collate_fn=collate_dataset_output,
    )
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=float(run_config["learning_rate"]))
    log_path = run_dir / "training_log.jsonl"
    final_epoch_summary: dict[str, Any] | None = None

    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(1, int(run_config["epochs"]) + 1):
            final_epoch_summary = train_one_epoch(
                tokenizer=tokenizer,
                data_loader=data_loader,
                optimizer=optimizer,
                device=device,
                codebook_size=tokenizer_config.codebook_size,
                epoch=epoch,
                auxiliary_loss_context=auxiliary_loss_context,
            )
            log_handle.write(json.dumps(final_epoch_summary, sort_keys=True) + "\n")
            log_handle.flush()
            if wandb_run is not None:
                wandb_run.log(final_epoch_summary, step=epoch)
            print(
                f"epoch={epoch} loss={final_epoch_summary['mean_total_loss']:.8f} "
                f"recon={final_epoch_summary['mean_reconstruction_loss']:.8f} "
                f"active_codes={final_epoch_summary['active_code_count']}"
            )

    if final_epoch_summary is None:
        raise RuntimeError("Training did not run any epochs.")

    end_time = datetime.now(UTC)
    elapsed_seconds = time.perf_counter() - start_perf
    codebook_summary = build_codebook_summary(
        final_epoch_summary,
        codebook_size=tokenizer_config.codebook_size,
    )
    runtime_summary = {
        "wall_clock_start_time": start_time.isoformat(),
        "wall_clock_end_time": end_time.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "config_path": run_config["config_path"],
        "run_dir": str(run_dir),
        "wandb_enabled": bool(run_config["wandb"]),
    }

    write_json(run_dir / "tokenizer_config.json", asdict(tokenizer_config))
    write_json(run_dir / "training_config.json", serialisable_run_config(run_config))
    write_json(run_dir / "runtime_summary.json", runtime_summary)
    write_json(run_dir / "codebook_summary.json", codebook_summary)
    torch.save(
        {
            "model_state_dict": tokenizer.state_dict(),
            "tokenizer_config": asdict(tokenizer_config),
            "training_config": serialisable_run_config(run_config),
            "codebook_summary": codebook_summary,
            "runtime_summary": runtime_summary,
        },
        run_dir / "tokenizer.pt",
    )

    if wandb_run is not None:
        wandb_run.finish()

    print(f"training_complete: {run_dir}")
    print(f"runtime_seconds: {elapsed_seconds:.3f}")
    print(f"final_loss: {codebook_summary['mean_total_loss']:.8f}")


def maybe_start_wandb(
    run_config: Mapping[str, Any],
    tokenizer_config: VQTokenizerConfig,
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
        name=run_config["run_name"],
        mode=run_config["wandb_mode"],
        config={
            "tokenizer_config": asdict(tokenizer_config),
            "training_config": serialisable_run_config(run_config),
        },
    )


def train_one_epoch(
    *,
    tokenizer: CausalVQTokenizer,
    data_loader: DataLoader[DatasetOutput],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    codebook_size: int,
    epoch: int,
    auxiliary_loss_context: TokenizerAuxiliaryLossContext | None = None,
) -> dict[str, Any]:
    """Train one epoch and return aggregate loss and codebook statistics."""
    tokenizer.train()
    n_samples = 0
    recon_loss_total = 0.0
    commitment_loss_total = 0.0
    codebook_loss_total = 0.0
    usage_loss_total = 0.0
    auxiliary_loss_total = 0.0
    auxiliary_component_totals: dict[str, float] = {
        "factor_reconstruction_aux_loss": 0.0,
        "factor_covariance_loss": 0.0,
        "factor_correlation_loss": 0.0,
        "inverse_projected_covariance_loss": 0.0,
        "inverse_projected_correlation_loss": 0.0,
        "sector_block_loss": 0.0,
        "equal_weight_portfolio_vol_loss": 0.0,
    }
    total_loss_total = 0.0
    code_counts = torch.zeros(codebook_size, dtype=torch.long)

    for batch in data_loader:
        inputs = cast(Tensor, batch["data"]).to(device)
        conditions = condition_tensor_from_batch(batch, tokenizer.config.condition_dim, device)
        optimizer.zero_grad(set_to_none=True)
        output = tokenizer(inputs, conditions, auxiliary_loss_context)
        loss = cast(Tensor, output.loss)
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        n_samples += batch_size
        recon_loss_total += float(cast(Tensor, output.recon_loss).detach().cpu()) * batch_size
        commitment_loss_total += (
            float(cast(Tensor, output.commitment_loss).detach().cpu()) * batch_size
        )
        codebook_loss_total += float(cast(Tensor, output.codebook_loss).detach().cpu()) * batch_size
        usage_loss_total += float(cast(Tensor, output.usage_loss).detach().cpu()) * batch_size
        auxiliary_loss_total += (
            float(cast(Tensor, output.auxiliary_loss).detach().cpu()) * batch_size
        )
        for key in auxiliary_component_totals:
            auxiliary_component_totals[key] += (
                float(cast(Tensor, output[key]).detach().cpu()) * batch_size
            )
        total_loss_total += float(loss.detach().cpu()) * batch_size
        code_counts += torch.bincount(
            cast(Tensor, output.indices).detach().cpu().reshape(-1),
            minlength=codebook_size,
        )[:codebook_size]

    if n_samples == 0:
        raise RuntimeError("Training dataset produced no samples.")

    code_stats = summarise_code_counts(code_counts)
    return {
        "epoch": epoch,
        "n_samples": n_samples,
        "mean_reconstruction_loss": recon_loss_total / n_samples,
        "mean_commitment_loss": commitment_loss_total / n_samples,
        "mean_codebook_loss": codebook_loss_total / n_samples,
        "mean_usage_loss": usage_loss_total / n_samples,
        "mean_auxiliary_loss": auxiliary_loss_total / n_samples,
        **{f"mean_{key}": value / n_samples for key, value in auxiliary_component_totals.items()},
        "usage_regularization_applied": False,
        "mean_total_loss": total_loss_total / n_samples,
        **code_stats,
    }


def condition_tensor_from_batch(
    batch: DatasetOutput,
    condition_dim: int,
    device: torch.device,
) -> Tensor | None:
    """Return labels as conditions only when the tokenizer config requests them."""
    if condition_dim == 0:
        return None
    labels = cast(Tensor, batch["labels"]).to(device)
    if labels.shape[-1] != condition_dim:
        raise ValueError(
            f"Expected condition_dim={condition_dim}; got labels {tuple(labels.shape)}."
        )
    return labels


def summarise_code_counts(code_counts: Tensor) -> dict[str, Any]:
    """Summarise active code count, ratio, and perplexity from token counts."""
    total = int(code_counts.sum().item())
    active_count = int((code_counts > 0).sum().item())
    codebook_size = int(code_counts.numel())
    if total == 0:
        perplexity = 0.0
    else:
        probabilities = code_counts.float() / float(total)
        active_probabilities = probabilities[probabilities > 0.0]
        entropy = -(active_probabilities * active_probabilities.log()).sum()
        perplexity = float(torch.exp(entropy).item())
    return {
        "codebook_size": codebook_size,
        "active_code_count": active_count,
        "active_code_ratio": active_count / codebook_size,
        "perplexity": perplexity,
        "token_count": total,
    }


def build_codebook_summary(
    epoch_summary: Mapping[str, Any],
    *,
    codebook_size: int,
) -> dict[str, Any]:
    """Build the persisted codebook summary from the final epoch metrics."""
    return {
        "codebook_size": codebook_size,
        "active_code_count": epoch_summary["active_code_count"],
        "active_code_ratio": epoch_summary["active_code_ratio"],
        "perplexity": epoch_summary["perplexity"],
        "mean_reconstruction_loss": epoch_summary["mean_reconstruction_loss"],
        "mean_commitment_loss": epoch_summary["mean_commitment_loss"],
        "mean_codebook_loss": epoch_summary["mean_codebook_loss"],
        "mean_usage_loss": epoch_summary["mean_usage_loss"],
        "mean_auxiliary_loss": epoch_summary.get("mean_auxiliary_loss", 0.0),
        "mean_factor_reconstruction_aux_loss": epoch_summary.get(
            "mean_factor_reconstruction_aux_loss",
            0.0,
        ),
        "mean_factor_covariance_loss": epoch_summary.get("mean_factor_covariance_loss", 0.0),
        "mean_factor_correlation_loss": epoch_summary.get("mean_factor_correlation_loss", 0.0),
        "mean_inverse_projected_covariance_loss": epoch_summary.get(
            "mean_inverse_projected_covariance_loss",
            0.0,
        ),
        "mean_inverse_projected_correlation_loss": epoch_summary.get(
            "mean_inverse_projected_correlation_loss",
            0.0,
        ),
        "mean_sector_block_loss": epoch_summary.get("mean_sector_block_loss", 0.0),
        "mean_equal_weight_portfolio_vol_loss": epoch_summary.get(
            "mean_equal_weight_portfolio_vol_loss",
            0.0,
        ),
        "usage_regularization_applied": epoch_summary["usage_regularization_applied"],
        "mean_total_loss": epoch_summary["mean_total_loss"],
        "source_epoch": epoch_summary["epoch"],
        "token_count": epoch_summary["token_count"],
    }


def serialisable_run_config(run_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of the flattened run config."""
    return cast(dict[str, Any], json.loads(json.dumps(dict(run_config), default=str)))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a mapping as indented JSON with a trailing newline."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _shape_text(value: Any) -> str:
    """Return a compact shape string for dry-run output."""
    shape = getattr(value, "shape", None)
    if shape is None:
        return "-"
    return "x".join(str(dim) for dim in shape)


if __name__ == "__main__":
    main()
