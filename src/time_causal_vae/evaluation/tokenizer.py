"""Evaluation diagnostics for trained causal VQ tokenizers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from time_causal_vae.evaluation.style import apply_clean_style, apply_source_style
from time_causal_vae.models.discrete.tokenizers import CausalVQTokenizer, VQTokenizerConfig
from time_causal_vae.utils.output import ModelOutput


def load_trained_tokenizer(
    tokenizer_dir: str | Path,
    *,
    device: torch.device,
) -> tuple[CausalVQTokenizer, VQTokenizerConfig, dict[str, Any]]:
    """Load a trained tokenizer from a tokenizer training output directory."""
    directory = Path(tokenizer_dir)
    checkpoint_path = directory / "tokenizer.pt"
    config_path = directory / "tokenizer_config.json"
    if not directory.exists():
        raise FileNotFoundError(
            f"Tokenizer directory does not exist: {directory}. Expected a directory such as "
            "outputs/tokenizer_smoke/black_scholes/black_scholes_causal_vq_tokenizer_seed0"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = cast(dict[str, Any], json.load(handle))
    tokenizer_config = VQTokenizerConfig(**config_data)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_mapping = cast(Mapping[str, Any], checkpoint)
    state_dict = cast(Mapping[str, Tensor], checkpoint_mapping["model_state_dict"])

    tokenizer = CausalVQTokenizer(tokenizer_config)
    tokenizer.load_state_dict(state_dict)
    tokenizer.to(device)
    tokenizer.eval()
    return tokenizer, tokenizer_config, dict(checkpoint_mapping)


def evaluate_tokenizer_batch(
    tokenizer: CausalVQTokenizer,
    inputs: Tensor,
    *,
    codebook_size: int,
    conditions: Tensor | None = None,
) -> tuple[ModelOutput, dict[str, Any]]:
    """Run a tokenizer batch and compute reconstruction plus codebook diagnostics."""
    with torch.no_grad():
        output = tokenizer(inputs, conditions)
    metrics = compute_tokenizer_metrics(
        inputs=inputs,
        reconstructions=cast(Tensor, output.recon_x),
        indices=cast(Tensor, output.indices),
        codebook_size=codebook_size,
    )
    if conditions is not None:
        metrics["condition_buckets"] = compute_condition_bucket_metrics(
            inputs=inputs,
            reconstructions=cast(Tensor, output.recon_x),
            indices=cast(Tensor, output.indices),
            conditions=conditions,
            codebook_size=codebook_size,
        )
    metrics.update(
        {
            "model_recon_loss": float(cast(Tensor, output.recon_loss).detach().cpu()),
            "model_commitment_loss": float(cast(Tensor, output.commitment_loss).detach().cpu()),
            "model_codebook_loss": float(cast(Tensor, output.codebook_loss).detach().cpu()),
            "model_usage_loss": float(cast(Tensor, output.usage_loss).detach().cpu()),
            "model_usage_regularization_applied": bool(output.usage_regularization_applied),
            "model_total_loss": float(cast(Tensor, output.loss).detach().cpu()),
        }
    )
    return output, metrics


def compute_tokenizer_metrics(
    *,
    inputs: Tensor,
    reconstructions: Tensor,
    indices: Tensor,
    codebook_size: int,
) -> dict[str, Any]:
    """Compute primary causal VQ tokenizer reconstruction and codebook diagnostics."""
    if inputs.shape != reconstructions.shape:
        raise ValueError(
            f"Input and reconstruction shapes must match; got {inputs.shape} and "
            f"{reconstructions.shape}."
        )
    if indices.shape[:2] != inputs.shape[:2]:
        raise ValueError(
            f"Indices must start with shape [batch, length]; got {indices.shape} for inputs "
            f"{inputs.shape}."
        )

    differences = reconstructions - inputs
    reconstruction_l1 = differences.abs().mean()
    reconstruction_l2 = differences.square().mean().sqrt()
    terminal_return_error = (
        terminal_return(inputs).sub(terminal_return(reconstructions)).abs().mean()
    )
    volatility_error = path_volatility(inputs).sub(path_volatility(reconstructions)).abs().mean()
    code_counts = torch.bincount(indices.detach().cpu().reshape(-1), minlength=codebook_size)[
        :codebook_size
    ]
    code_stats = summarise_code_usage(code_counts)
    component_stats = component_code_usage(indices.detach().cpu(), codebook_size)
    transition_summary = summarise_code_transitions(indices.detach().cpu())
    return {
        "reconstruction_l1": float(reconstruction_l1.detach().cpu()),
        "reconstruction_l2": float(reconstruction_l2.detach().cpu()),
        "terminal_return_error": float(terminal_return_error.detach().cpu()),
        "volatility_reconstruction_error": float(volatility_error.detach().cpu()),
        **code_stats,
        **component_stats,
        **transition_summary,
    }


def terminal_return(paths: Tensor) -> Tensor:
    """Return terminal simple returns for positive-valued path tensors."""
    start = paths[:, 0, :]
    terminal = paths[:, -1, :]
    denominator = start.abs().clamp_min(1e-8)
    return (terminal - start) / denominator


def path_volatility(paths: Tensor) -> Tensor:
    """Return per-path volatility from log returns when valid, otherwise increments."""
    if bool((paths > 0.0).all()):
        returns = paths.clamp_min(1e-8).log().diff(dim=1)
    else:
        returns = paths.diff(dim=1)
    return returns.std(dim=1, unbiased=False)


def summarise_code_usage(code_counts: Tensor) -> dict[str, Any]:
    """Summarise active code count, ratio, entropy, and perplexity."""
    total = int(code_counts.sum().item())
    active_code_count = int((code_counts > 0).sum().item())
    codebook_size = int(code_counts.numel())
    if total == 0:
        index_entropy = 0.0
        perplexity = 0.0
        active_indices: list[int] = []
    else:
        probabilities = code_counts.float() / float(total)
        active_probabilities = probabilities[probabilities > 0.0]
        entropy = -(active_probabilities * active_probabilities.log()).sum()
        index_entropy = float(entropy.item())
        perplexity = float(torch.exp(entropy).item())
        active_indices = [
            int(index) for index in torch.nonzero(code_counts > 0, as_tuple=False).flatten()
        ]
    return {
        "codebook_size": codebook_size,
        "active_code_count": active_code_count,
        "active_code_ratio": active_code_count / codebook_size,
        "codebook_perplexity": perplexity,
        "index_entropy": index_entropy,
        "token_count": total,
        "active_code_indices": active_indices,
        "code_usage_counts": [int(value) for value in code_counts.tolist()],
    }


def component_code_usage(indices: Tensor, codebook_size: int) -> dict[str, Any]:
    """Return per-component code usage for vector, RVQ, and grouped RVQ outputs."""
    if indices.ndim == 2:
        return {"component_usage_note": "single_vector_code_per_time_step"}
    if indices.ndim == 3:
        return {
            "per_quantizer": [
                {
                    "quantizer_index": quantizer_index,
                    **summarise_code_usage(
                        torch.bincount(
                            indices[:, :, quantizer_index].reshape(-1),
                            minlength=codebook_size,
                        )[:codebook_size]
                    ),
                }
                for quantizer_index in range(indices.shape[2])
            ]
        }
    if indices.ndim == 4:
        per_group_quantizer = []
        for group_index in range(indices.shape[2]):
            for quantizer_index in range(indices.shape[3]):
                code_counts = torch.bincount(
                    indices[:, :, group_index, quantizer_index].reshape(-1),
                    minlength=codebook_size,
                )[:codebook_size]
                per_group_quantizer.append(
                    {
                        "group_index": group_index,
                        "quantizer_index": quantizer_index,
                        **summarise_code_usage(code_counts),
                    }
                )
        return {"per_group_quantizer": per_group_quantizer}
    return {
        "component_usage_note": (
            "component code usage omitted for unsupported index rank "
            f"{indices.ndim}; expected 2, 3, or 4."
        )
    }


def summarise_code_transitions(indices: Tensor) -> dict[str, Any]:
    """Return an active-code transition matrix for observed consecutive tokens."""
    if indices.ndim > 2:
        return {
            "transition_active_code_indices": [],
            "code_transition_matrix": [],
            "transition_note": (
                "Transition matrix omitted for multi-index tokenizer output; "
                f"indices shape was {list(indices.shape)}."
            ),
        }
    active_codes = torch.unique(indices).sort().values
    active_code_to_position = {int(code): position for position, code in enumerate(active_codes)}
    transition_matrix = torch.zeros(
        (len(active_codes), len(active_codes)),
        dtype=torch.long,
    )
    if indices.shape[1] > 1:
        previous_tokens = indices[:, :-1].reshape(-1)
        next_tokens = indices[:, 1:].reshape(-1)
        for previous, next_token in zip(
            previous_tokens.tolist(), next_tokens.tolist(), strict=True
        ):
            row = active_code_to_position[int(previous)]
            column = active_code_to_position[int(next_token)]
            transition_matrix[row, column] += 1
    return {
        "transition_active_code_indices": [int(value) for value in active_codes.tolist()],
        "code_transition_matrix": [
            [int(value) for value in row] for row in transition_matrix.tolist()
        ],
    }


def compute_condition_bucket_metrics(
    *,
    inputs: Tensor,
    reconstructions: Tensor,
    indices: Tensor,
    conditions: Tensor,
    codebook_size: int,
    n_buckets: int = 5,
    top_k_codes: int = 10,
) -> list[dict[str, Any]]:
    """Compute tokenizer metrics within condition-quantile buckets.

    Scalar conditions use their single value per path. Temporal conditions, if
    introduced later, are summarised by their per-path mean before bucketing.
    Buckets are equal-count sorted partitions, so duplicate condition values do
    not produce empty quantile intervals.
    """
    if conditions.ndim == 2:
        condition_values = conditions.detach().cpu().reshape(conditions.shape[0], -1).mean(dim=1)
    elif conditions.ndim == 3:
        condition_values = conditions.detach().cpu().mean(dim=(1, 2))
    else:
        raise ValueError(
            "conditions must be [batch, condition_dim] or [batch, length, condition_dim]; "
            f"got {tuple(conditions.shape)}."
        )
    batch_size = inputs.shape[0]
    if condition_values.shape[0] != batch_size:
        raise ValueError(
            f"Expected {batch_size} condition values; got {condition_values.shape[0]}."
        )
    if batch_size == 0:
        return []

    bucket_count = min(n_buckets, batch_size)
    sorted_positions = torch.argsort(condition_values)
    buckets: list[dict[str, Any]] = []
    for bucket_index, bucket_positions in enumerate(
        torch.tensor_split(sorted_positions, bucket_count)
    ):
        if bucket_positions.numel() == 0:
            continue
        positions = bucket_positions.to(device=inputs.device)
        bucket_inputs = inputs.index_select(0, positions)
        bucket_reconstructions = reconstructions.index_select(0, positions)
        bucket_indices = indices.index_select(0, positions)
        bucket_metrics = compute_tokenizer_metrics(
            inputs=bucket_inputs,
            reconstructions=bucket_reconstructions,
            indices=bucket_indices,
            codebook_size=codebook_size,
        )
        code_counts = torch.tensor(
            cast(list[int], bucket_metrics["code_usage_counts"]),
            dtype=torch.long,
        )
        top_count = min(top_k_codes, int((code_counts > 0).sum().item()))
        if top_count > 0:
            top_counts, top_indices = torch.topk(code_counts, k=top_count)
            top_codes = [
                {"code": int(code), "count": int(count)}
                for code, count in zip(top_indices.tolist(), top_counts.tolist(), strict=True)
            ]
        else:
            top_codes = []
        bucket_condition_values = condition_values.index_select(0, bucket_positions)
        buckets.append(
            {
                "bucket_index": bucket_index,
                "bucket_label": condition_bucket_label(bucket_index, bucket_count),
                "n_samples": int(bucket_positions.numel()),
                "condition_min": float(bucket_condition_values.min().item()),
                "condition_max": float(bucket_condition_values.max().item()),
                "condition_mean": float(bucket_condition_values.mean().item()),
                "reconstruction_l1": bucket_metrics["reconstruction_l1"],
                "reconstruction_l2": bucket_metrics["reconstruction_l2"],
                "terminal_return_error": bucket_metrics["terminal_return_error"],
                "volatility_reconstruction_error": bucket_metrics[
                    "volatility_reconstruction_error"
                ],
                "active_code_count": bucket_metrics["active_code_count"],
                "active_code_ratio": bucket_metrics["active_code_ratio"],
                "codebook_perplexity": bucket_metrics["codebook_perplexity"],
                "index_entropy": bucket_metrics["index_entropy"],
                "active_code_indices": bucket_metrics["active_code_indices"],
                "top_codes": top_codes,
            }
        )
    return buckets


def condition_bucket_label(bucket_index: int, bucket_count: int) -> str:
    """Return a human-readable condition bucket label."""
    if bucket_count == 1:
        return "all"
    if bucket_count == 3:
        return ("low", "mid", "high")[bucket_index]
    if bucket_count == 5:
        return ("very_low", "low", "mid", "high", "very_high")[bucket_index]
    return f"bucket_{bucket_index}"


def save_tokenizer_batch(
    path: str | Path,
    *,
    inputs: Tensor,
    output: ModelOutput,
    metrics: Mapping[str, Any],
    conditions: Tensor | None = None,
) -> None:
    """Save tensors from an evaluated tokenizer batch."""
    payload = {
        "x": inputs.detach().cpu(),
        "recon_x": cast(Tensor, output["recon_x"]).detach().cpu(),
        "z_e": cast(Tensor, output["z_e"]).detach().cpu(),
        "z_q": cast(Tensor, output["z_q"]).detach().cpu(),
        "indices": cast(Tensor, output["indices"]).detach().cpu(),
        "metrics": dict(metrics),
    }
    if conditions is not None:
        payload["conditions"] = conditions.detach().cpu()
    torch.save(payload, path)


def save_tokenizer_summary(
    path: str | Path,
    *,
    metrics: Mapping[str, Any],
    tokenizer_config: VQTokenizerConfig,
    tensor_shapes: Mapping[str, list[int]],
    config_path: str,
    tokenizer_dir: str,
    n_sample_test: int,
    device: torch.device,
) -> None:
    """Save a JSON summary for a tokenizer evaluation run."""
    payload = {
        "config_path": config_path,
        "tokenizer_dir": tokenizer_dir,
        "n_sample_test": n_sample_test,
        "device": str(device),
        "tokenizer_config": asdict(tokenizer_config),
        "tensor_shapes": dict(tensor_shapes),
        "metrics": dict(metrics),
        "primary_metrics_note": "Financial time-series diagnostics; PSNR, SSIM, and rFID omitted.",
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def plot_code_usage(
    path: str | Path,
    *,
    code_counts: list[int],
    active_code_count: int,
) -> None:
    """Plot a histogram of observed code usage counts."""
    apply_plot_style()
    counts = torch.tensor(code_counts, dtype=torch.long)
    active_positions = torch.nonzero(counts > 0, as_tuple=False).flatten()
    fig, ax = plt.subplots(figsize=(10, 4))
    if active_positions.numel() == 0:
        ax.text(0.5, 0.5, "No codes observed", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(
            active_positions.numpy(),
            counts[active_positions].numpy(),
            width=1.0,
            color="#4c78a8",
        )
        ax.set_xlabel("Code index")
        ax.set_ylabel("Count")
        ax.set_title(f"Observed code usage ({active_code_count} active codes)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reconstruction_examples(
    path: str | Path,
    *,
    inputs: Tensor,
    reconstructions: Tensor,
    n_examples: int = 6,
) -> None:
    """Plot real versus reconstructed paths for a small batch subset."""
    apply_plot_style()
    inputs_cpu = inputs.detach().cpu()
    recon_cpu = reconstructions.detach().cpu()
    n_plots = min(n_examples, inputs_cpu.shape[0])
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, max(2.0, 1.8 * n_plots)), sharex=True)
    if n_plots == 1:
        axes = [axes]
    for row, ax in enumerate(axes):
        ax.plot(inputs_cpu[row, :, 0].numpy(), label="real", color="#1f77b4", linewidth=1.6)
        ax.plot(
            recon_cpu[row, :, 0].numpy(),
            label="reconstruction",
            color="#d62728",
            linewidth=1.4,
            alpha=0.85,
        )
        ax.set_ylabel(f"path {row}")
        if row == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("Time index")
    fig.suptitle("Causal VQ tokenizer reconstruction examples")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def apply_plot_style() -> None:
    """Apply package plotting style with a clean fallback."""
    try:
        apply_source_style()
    except Exception:
        apply_clean_style()
