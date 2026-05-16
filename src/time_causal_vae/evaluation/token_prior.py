"""Evaluation diagnostics for causal autoregressive token priors."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from time_causal_vae.evaluation.metrics import SWD
from time_causal_vae.evaluation.style import apply_clean_style, apply_source_style
from time_causal_vae.evaluation.token_diagnostics import (
    marginal_code_histogram,
    row_normalize_transition_matrix,
    transition_matrix,
)
from time_causal_vae.evaluation.tokenizer import (
    condition_bucket_label,
    path_volatility,
    summarise_code_usage,
    terminal_return,
)
from time_causal_vae.models.distances import GaussianMMD
from time_causal_vae.token_prior import (
    CausalConvTransformerPrior,
    CausalTokenPriorConfig,
    CausalTokenTransformerPrior,
    FactorisedMultiCodeTokenPrior,
    HierarchicalRVQ2TokenPrior,
    build_token_prior_model,
)
from time_causal_vae.tokenization import CausalVQTokenizer, VQTokenizerConfig


def load_trained_token_prior(
    prior_dir: str | Path,
    *,
    device: torch.device,
) -> tuple[
    CausalTokenTransformerPrior
    | CausalConvTransformerPrior
    | FactorisedMultiCodeTokenPrior
    | HierarchicalRVQ2TokenPrior,
    CausalTokenPriorConfig,
    dict[str, Any],
]:
    """Load a trained causal token prior from an output directory."""
    directory = Path(prior_dir)
    checkpoint_path = directory / "token_prior.pt"
    config_path = directory / "token_prior_config.json"
    if not directory.exists():
        raise FileNotFoundError(f"Token-prior directory does not exist: {directory}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Token-prior checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Token-prior config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = cast(dict[str, Any], json.load(handle))
    prior_config = CausalTokenPriorConfig(**config_data)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_mapping = cast(Mapping[str, Any], checkpoint)
    state_dict = cast(Mapping[str, Tensor], checkpoint_mapping["model_state_dict"])

    prior = build_token_prior_model(prior_config)
    prior.load_state_dict(state_dict)
    prior.to(device)
    prior.eval()
    return prior, prior_config, dict(checkpoint_mapping)


@torch.no_grad()
def decode_token_indices(
    tokenizer: CausalVQTokenizer,
    indices: Tensor,
    *,
    conditions: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Map sampled token indices to quantized embeddings and decoded paths."""
    quantized = tokenizer.quantizer.decode_indices(indices)
    decoded = tokenizer.decoder(quantized, conditions)
    return quantized, cast(Tensor, decoded)


def compute_token_prior_sample_metrics(
    *,
    sampled_tokens: Tensor,
    decoded_paths: Tensor,
    real_paths: Tensor,
    codebook_size: int,
) -> dict[str, Any]:
    """Compute token-usage and financial diagnostics for decoded prior samples."""
    if decoded_paths.shape != real_paths.shape:
        raise ValueError(
            f"decoded_paths and real_paths must have matching shapes; got "
            f"{decoded_paths.shape} and {real_paths.shape}."
        )
    code_counts = torch.bincount(
        sampled_tokens.detach().cpu().reshape(-1),
        minlength=codebook_size,
    )[:codebook_size]
    token_metrics = summarise_code_usage(code_counts)
    component_metrics = component_token_metrics(sampled_tokens.detach().cpu(), codebook_size)
    path_metrics = compute_path_distribution_metrics(
        generated=decoded_paths.detach().cpu(),
        real=real_paths.detach().cpu(),
    )
    return {
        **prefix_keys("sampled_token", token_metrics),
        **prefix_keys("sampled_token", component_metrics),
        **path_metrics,
    }


def compute_condition_bucket_sample_metrics(
    *,
    sampled_tokens: Tensor,
    decoded_paths: Tensor,
    real_paths: Tensor,
    conditions: Tensor,
    codebook_size: int,
    n_buckets: int = 5,
) -> list[dict[str, Any]]:
    """Compute decoded-path and sampled-token metrics by condition quantile."""
    if conditions.ndim == 2:
        condition_values = conditions.detach().cpu().reshape(conditions.shape[0], -1).mean(dim=1)
    elif conditions.ndim == 3:
        condition_values = conditions.detach().cpu().mean(dim=(1, 2))
    else:
        raise ValueError(
            "conditions must be [batch, condition_dim] or [batch, length, condition_dim]; "
            f"got {tuple(conditions.shape)}."
        )
    batch_size = decoded_paths.shape[0]
    if sampled_tokens.shape[0] != batch_size or real_paths.shape[0] != batch_size:
        raise ValueError("sampled tokens, decoded paths, and real paths must share batch size.")
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
        generated_bucket = decoded_paths.detach().cpu().index_select(0, bucket_positions)
        real_bucket = real_paths.detach().cpu().index_select(0, bucket_positions)
        token_bucket = sampled_tokens.detach().cpu().index_select(0, bucket_positions)
        path_metrics = compute_path_distribution_metrics(
            generated=generated_bucket,
            real=real_bucket,
        )
        code_counts = torch.bincount(
            token_bucket.reshape(-1),
            minlength=codebook_size,
        )[:codebook_size]
        code_metrics = summarise_code_usage(code_counts)
        component_metrics = component_token_metrics(token_bucket, codebook_size)
        bucket_condition_values = condition_values.index_select(0, bucket_positions)
        buckets.append(
            {
                "bucket_index": bucket_index,
                "bucket_label": condition_bucket_label(bucket_index, bucket_count),
                "n_samples": int(bucket_positions.numel()),
                "condition_min": float(bucket_condition_values.min().item()),
                "condition_max": float(bucket_condition_values.max().item()),
                "condition_mean": float(bucket_condition_values.mean().item()),
                "mmd": path_metrics["mmd"],
                "swd": path_metrics["swd"],
                "terminal_return_mean_error": path_metrics["terminal_return_mean_error"],
                "terminal_return_wasserstein": path_metrics["terminal_return_wasserstein"],
                "volatility_mean_error": path_metrics["volatility_mean_error"],
                "volatility_wasserstein": path_metrics["volatility_wasserstein"],
                "sampled_active_code_count": code_metrics["active_code_count"],
                "sampled_active_code_ratio": code_metrics["active_code_ratio"],
                "sampled_token_perplexity": code_metrics["codebook_perplexity"],
                "sampled_index_entropy": code_metrics["index_entropy"],
                **prefix_keys("sampled_token", component_metrics),
            }
        )
    return buckets


def component_token_metrics(tokens: Tensor, codebook_size: int) -> dict[str, Any]:
    """Return component-level code usage for multi-code sampled token tensors."""
    if tokens.ndim == 2:
        return {"component_usage_note": "single_vector_code_per_time_step"}
    if tokens.ndim == 3:
        return {
            "per_quantizer": [
                {
                    "quantizer_index": quantizer_index,
                    **summarise_code_usage(
                        torch.bincount(
                            tokens[:, :, quantizer_index].reshape(-1),
                            minlength=codebook_size,
                        )[:codebook_size]
                    ),
                }
                for quantizer_index in range(tokens.shape[2])
            ]
        }
    if tokens.ndim == 4:
        per_group_quantizer = []
        for group_index in range(tokens.shape[2]):
            for quantizer_index in range(tokens.shape[3]):
                code_counts = torch.bincount(
                    tokens[:, :, group_index, quantizer_index].reshape(-1),
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
            "component usage omitted for unsupported sampled token rank "
            f"{tokens.ndim}; expected 2, 3, or 4."
        )
    }


def compute_path_distribution_metrics(*, generated: Tensor, real: Tensor) -> dict[str, Any]:
    """Compute decoded-path distribution diagnostics against real comparison data."""
    generated_float = generated.float()
    real_float = real.float()
    generated_terminal = terminal_return(generated_float)
    real_terminal = terminal_return(real_float)
    generated_volatility = path_volatility(generated_float)
    real_volatility = path_volatility(real_float)

    mmd = GaussianMMD()(generated_float, real_float)
    swd = SWD()(generated_float, real_float)
    terminal_distance = wasserstein_1d(generated_terminal.reshape(-1), real_terminal.reshape(-1))
    volatility_distance = wasserstein_1d(
        generated_volatility.reshape(-1),
        real_volatility.reshape(-1),
    )
    return {
        "decoded_path_shape": list(generated.shape),
        "real_path_shape": list(real.shape),
        "mmd": float(mmd.detach().cpu()),
        "swd": float(swd.detach().cpu()),
        "terminal_return_generated_mean": float(generated_terminal.mean().detach().cpu()),
        "terminal_return_real_mean": float(real_terminal.mean().detach().cpu()),
        "terminal_return_mean_error": float(
            (generated_terminal.mean() - real_terminal.mean()).abs().detach().cpu()
        ),
        "terminal_return_wasserstein": float(terminal_distance.detach().cpu()),
        "volatility_generated_mean": float(generated_volatility.mean().detach().cpu()),
        "volatility_real_mean": float(real_volatility.mean().detach().cpu()),
        "volatility_mean_error": float(
            (generated_volatility.mean() - real_volatility.mean()).abs().detach().cpu()
        ),
        "volatility_wasserstein": float(volatility_distance.detach().cpu()),
    }


def wasserstein_1d(first: Tensor, second: Tensor) -> Tensor:
    """Return equal-weight one-dimensional Wasserstein distance."""
    if first.numel() != second.numel():
        n_values = min(first.numel(), second.numel())
        first = first[:n_values]
        second = second[:n_values]
    return (first.sort().values - second.sort().values).abs().mean()


def prefix_keys(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Prefix metric keys for JSON summaries."""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def save_token_prior_summary(
    path: str | Path,
    *,
    metrics: Mapping[str, Any],
    prior_config: CausalTokenPriorConfig,
    tokenizer_config: VQTokenizerConfig,
    tensor_shapes: Mapping[str, list[int]],
    config_path: str,
    prior_dir: str,
    tokenizer_dir: str,
    n_sample: int,
    seed: int,
    device: torch.device,
    temperature: float,
    top_k: int | None,
) -> None:
    """Save a JSON summary for a token-prior sampling evaluation."""
    payload = {
        "config_path": config_path,
        "prior_dir": prior_dir,
        "tokenizer_dir": tokenizer_dir,
        "n_sample": n_sample,
        "seed": seed,
        "device": str(device),
        "temperature": temperature,
        "top_k": top_k,
        "token_prior_config": asdict(prior_config),
        "tokenizer_config": asdict(tokenizer_config),
        "tensor_shapes": dict(tensor_shapes),
        "metrics": dict(metrics),
        "primary_metrics_note": "Decoded financial time-series diagnostics; no diffusion used.",
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_sampled_tokens(
    path: str | Path,
    *,
    tokens: Tensor,
    metrics: Mapping[str, Any],
) -> None:
    """Save sampled token indices and their metrics."""
    torch.save({"indices": tokens.detach().cpu(), "metrics": dict(metrics)}, path)


def save_decoded_paths(
    path: str | Path,
    *,
    decoded_paths: Tensor,
    real_paths: Tensor,
    quantized: Tensor,
    metrics: Mapping[str, Any],
) -> None:
    """Save decoded paths, comparison paths, embeddings, and metrics."""
    torch.save(
        {
            "decoded_paths": decoded_paths.detach().cpu(),
            "real_paths": real_paths.detach().cpu(),
            "z_q": quantized.detach().cpu(),
            "metrics": dict(metrics),
        },
        path,
    )


def plot_sampled_code_usage(
    path: str | Path,
    *,
    code_counts: list[int],
    active_code_count: int,
) -> None:
    """Plot sampled token usage counts."""
    apply_plot_style()
    counts = torch.tensor(code_counts, dtype=torch.long)
    active_positions = torch.nonzero(counts > 0, as_tuple=False).flatten()
    fig, ax = plt.subplots(figsize=(10, 4))
    if active_positions.numel() == 0:
        ax.text(0.5, 0.5, "No sampled codes", ha="center", va="center")
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
        ax.set_title(f"Sampled code usage ({active_code_count} active codes)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_real_vs_sampled_code_usage(
    path: str | Path,
    *,
    real_tokens: Tensor,
    sampled_tokens: Tensor,
    codebook_size: int,
) -> None:
    """Plot real and sampled marginal code usage side by side."""
    apply_plot_style()
    real_counts = marginal_code_histogram(real_tokens, codebook_size)
    sampled_counts = marginal_code_histogram(sampled_tokens, codebook_size)
    code_indices = torch.arange(codebook_size)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(
        (code_indices - 0.2).numpy(),
        real_counts.numpy(),
        width=0.4,
        label="real tokens",
        color="#4c78a8",
    )
    ax.bar(
        (code_indices + 0.2).numpy(),
        sampled_counts.numpy(),
        width=0.4,
        label="sampled tokens",
        color="#f58518",
    )
    ax.set_xlabel("Code index")
    ax.set_ylabel("Count")
    ax.set_title("Real vs sampled code usage")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_transition_matrix(
    path: str | Path,
    *,
    tokens: Tensor,
    codebook_size: int,
    title: str,
) -> None:
    """Plot a row-normalised token transition matrix."""
    apply_plot_style()
    matrix = row_normalize_transition_matrix(transition_matrix(tokens, codebook_size))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix.numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Next code")
    ax.set_ylabel("Previous code")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_decoded_path_examples(
    path: str | Path,
    *,
    decoded_paths: Tensor,
    real_paths: Tensor,
    n_examples: int = 6,
) -> None:
    """Plot decoded samples against real comparison paths."""
    apply_plot_style()
    generated_cpu = decoded_paths.detach().cpu()
    real_cpu = real_paths.detach().cpu()
    n_plots = min(n_examples, generated_cpu.shape[0], real_cpu.shape[0])
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, max(2.0, 1.8 * n_plots)), sharex=True)
    if n_plots == 1:
        axes = [axes]
    for row, ax in enumerate(axes):
        ax.plot(real_cpu[row, :, 0].numpy(), label="real", color="#1f77b4", linewidth=1.5)
        ax.plot(
            generated_cpu[row, :, 0].numpy(),
            label="decoded sample",
            color="#d62728",
            linewidth=1.3,
            alpha=0.85,
        )
        ax.set_ylabel(f"path {row}")
        if row == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("Time index")
    fig.suptitle("Causal token prior decoded path examples")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def apply_plot_style() -> None:
    """Apply package plotting style with a clean fallback."""
    try:
        apply_source_style()
    except Exception:
        apply_clean_style()
