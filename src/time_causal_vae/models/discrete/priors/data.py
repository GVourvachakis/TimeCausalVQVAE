"""Utilities for extracting frozen tokenizer index datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import ml_collections
import torch
import yaml
from torch import Tensor

from time_causal_vae.data.base import BaseDataset
from time_causal_vae.data.pipeline import DataPipeline
from time_causal_vae.evaluation.tokenizer import load_trained_tokenizer, summarise_code_usage
from time_causal_vae.models.discrete.tokenizers import CausalVQTokenizer, VQTokenizerConfig
from time_causal_vae.utils.output import ModelOutput


def load_tokenizer_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a tokenizer experiment YAML config as a mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Tokenizer experiment config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def build_tokenizer_datasets(
    raw_config: Mapping[str, Any],
    *,
    n_sample: int | None,
    base_data_dir: str,
) -> tuple[BaseDataset, BaseDataset]:
    """Build train and eval datasets with the existing target data pipeline."""
    experiment = require_mapping(raw_config, "experiment")
    data = require_mapping(raw_config, "data")
    sample_count = int(n_sample if n_sample is not None else data["n_samples"])
    if sample_count <= 0:
        raise ValueError("n_sample must be positive.")
    train_sample_count = optional_split_sample_count(data.get("train_n_samples"))
    eval_sample_count = optional_split_sample_count(data.get("eval_n_samples"))

    exp_config = ml_collections.ConfigDict()
    exp_config.dataset = str(experiment["dataset"])
    exp_config.n_sample = sample_count
    exp_config.train_n_sample = sample_count if n_sample is not None else train_sample_count
    exp_config.eval_n_sample = sample_count if n_sample is not None else eval_sample_count
    exp_config.n_timestep = int(data["n_timesteps"])
    exp_config.base_data_dir = base_data_dir
    exp_config.data_params = dict(cast(Mapping[str, Any], data.get("data_params", {})))
    if "rho" in data:
        exp_config.rho = data["rho"]
    return cast(tuple[BaseDataset, BaseDataset], DataPipeline()(exp_config))


def optional_split_sample_count(value: Any) -> int | None:
    """Return ``None`` or a validated split-specific sample count."""
    if value is None:
        return None
    count = int(value)
    if count <= 0:
        raise ValueError("split-specific sample counts must be positive when provided.")
    return count


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required config section as a dictionary."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Tokenizer config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def load_frozen_tokenizer(
    tokenizer_dir: str | Path,
    *,
    device: torch.device,
) -> tuple[CausalVQTokenizer, VQTokenizerConfig]:
    """Load a tokenizer checkpoint in evaluation mode for frozen index extraction."""
    tokenizer, tokenizer_config, _checkpoint = load_trained_tokenizer(tokenizer_dir, device=device)
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    return tokenizer, tokenizer_config


def extract_dataset_tokens(
    tokenizer: CausalVQTokenizer,
    dataset: BaseDataset,
    *,
    device: torch.device,
    include_data: bool = True,
    include_labels: bool = True,
) -> dict[str, Tensor]:
    """Extract tokenizer indices from a dataset and return a serialisable tensor payload."""
    inputs = dataset.data.to(device)
    conditions = condition_tensor(dataset, tokenizer, device)
    with torch.no_grad():
        output = cast(ModelOutput, tokenizer(inputs, conditions))
    indices = cast(Tensor, output["indices"]).detach().cpu()
    payload: dict[str, Tensor] = {"indices": indices}
    if include_data:
        payload["data"] = dataset.data.detach().cpu()
    if include_labels:
        payload["labels"] = dataset.labels.detach().cpu()
    return payload


def condition_tensor(
    dataset: BaseDataset,
    tokenizer: CausalVQTokenizer,
    device: torch.device,
) -> Tensor | None:
    """Return labels as tokenizer conditions only when the tokenizer requests them."""
    if tokenizer.config.condition_dim == 0:
        return None
    labels = dataset.labels.to(device)
    if labels.shape[-1] != tokenizer.config.condition_dim:
        raise ValueError(
            f"Expected condition_dim={tokenizer.config.condition_dim}; got labels "
            f"{tuple(labels.shape)}."
        )
    return labels


def token_dataset_summary(
    *,
    tokenizer_dir: str | Path,
    config_path: str | Path,
    tokenizer_config: VQTokenizerConfig,
    train_payload: Mapping[str, Tensor],
    eval_payload: Mapping[str, Tensor],
    seed: int,
    n_sample: int | None,
) -> dict[str, Any]:
    """Build JSON-safe summary statistics for extracted train/eval token datasets."""
    train_indices = train_payload["indices"]
    eval_indices = eval_payload["indices"]
    combined_indices = torch.cat([train_indices.reshape(-1), eval_indices.reshape(-1)])
    combined_counts = torch.bincount(
        combined_indices,
        minlength=tokenizer_config.codebook_size,
    )[: tokenizer_config.codebook_size]
    combined_token_indices = torch.cat([train_indices, eval_indices], dim=0)
    summary: dict[str, Any] = {
        "tokenizer_dir": str(tokenizer_dir),
        "config_path": str(config_path),
        "seed": seed,
        "n_sample": n_sample,
        "quantizer_type": tokenizer_config.quantizer_type,
        "codebook_size": tokenizer_config.codebook_size,
        "num_quantizers": tokenizer_config.num_quantizers,
        "groups": tokenizer_config.groups,
        "sequence_length": tokenizer_config.data_length,
        "train_token_shape": list(train_indices.shape),
        "eval_token_shape": list(eval_indices.shape),
        "index_shape": list(train_indices.shape[1:]),
        "train": split_code_stats(train_indices, tokenizer_config.codebook_size),
        "eval": split_code_stats(eval_indices, tokenizer_config.codebook_size),
        "combined": {
            **summarise_code_usage(combined_counts),
            **component_code_usage(combined_token_indices, tokenizer_config.codebook_size),
        },
    }
    if "data" in train_payload:
        summary["train_data_shape"] = list(train_payload["data"].shape)
    if "data" in eval_payload:
        summary["eval_data_shape"] = list(eval_payload["data"].shape)
    if "labels" in train_payload:
        train_labels = train_payload["labels"]
        summary["train_label_shape"] = list(train_labels.shape)
        summary["train_condition_stats"] = condition_stats(train_labels)
        summary["train_condition_buckets"] = condition_bucket_code_usage(
            labels=train_labels,
            indices=train_indices,
            codebook_size=tokenizer_config.codebook_size,
        )
    if "labels" in eval_payload:
        eval_labels = eval_payload["labels"]
        summary["eval_label_shape"] = list(eval_labels.shape)
        summary["eval_condition_stats"] = condition_stats(eval_labels)
        summary["eval_condition_buckets"] = condition_bucket_code_usage(
            labels=eval_labels,
            indices=eval_indices,
            codebook_size=tokenizer_config.codebook_size,
        )
    if "labels" in train_payload and "labels" in eval_payload:
        combined_labels = torch.cat([train_payload["labels"], eval_payload["labels"]], dim=0)
        summary["combined_condition_stats"] = condition_stats(combined_labels)
        summary["combined_condition_buckets"] = condition_bucket_code_usage(
            labels=combined_labels,
            indices=combined_token_indices,
            codebook_size=tokenizer_config.codebook_size,
        )
    return summary


def split_code_stats(indices: Tensor, codebook_size: int) -> dict[str, Any]:
    """Return code-usage statistics for one extracted token split."""
    code_counts = torch.bincount(indices.reshape(-1), minlength=codebook_size)[:codebook_size]
    return {
        **summarise_code_usage(code_counts),
        **component_code_usage(indices, codebook_size),
    }


def component_code_usage(indices: Tensor, codebook_size: int) -> dict[str, Any]:
    """Return per-component code usage for vector, RVQ, and GRVQ token indices."""
    if indices.ndim == 2:
        return {"component_usage_note": "single_vector_code_per_time_step"}
    if indices.ndim == 3:
        per_quantizer = []
        for quantizer_index in range(indices.shape[2]):
            component = indices[:, :, quantizer_index]
            code_counts = torch.bincount(
                component.reshape(-1),
                minlength=codebook_size,
            )[:codebook_size]
            per_quantizer.append(
                {
                    "quantizer_index": quantizer_index,
                    **summarise_code_usage(code_counts),
                }
            )
        return {"per_quantizer": per_quantizer}
    if indices.ndim == 4:
        per_group = []
        for group_index in range(indices.shape[2]):
            component = indices[:, :, group_index, :]
            code_counts = torch.bincount(
                component.reshape(-1),
                minlength=codebook_size,
            )[:codebook_size]
            per_group.append(
                {
                    "group_index": group_index,
                    **summarise_code_usage(code_counts),
                }
            )
        per_group_quantizer = []
        for group_index in range(indices.shape[2]):
            for quantizer_index in range(indices.shape[3]):
                component = indices[:, :, group_index, quantizer_index]
                code_counts = torch.bincount(
                    component.reshape(-1),
                    minlength=codebook_size,
                )[:codebook_size]
                per_group_quantizer.append(
                    {
                        "group_index": group_index,
                        "quantizer_index": quantizer_index,
                        **summarise_code_usage(code_counts),
                    }
                )
        return {
            "per_group": per_group,
            "per_group_quantizer": per_group_quantizer,
        }
    return {
        "component_usage_note": (
            "component code usage omitted for unsupported index rank "
            f"{indices.ndim}; expected 2, 3, or 4."
        )
    }


def condition_stats(labels: Tensor) -> dict[str, Any]:
    """Return scalar condition statistics for saved label tensors."""
    values = condition_values(labels)
    if values.numel() == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "quantiles": {},
        }
    quantile_levels = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=values.dtype)
    quantiles = torch.quantile(values, quantile_levels)
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "quantiles": {
            f"{float(level.item()):.1f}": float(value.item())
            for level, value in zip(quantile_levels, quantiles, strict=True)
        },
    }


def condition_bucket_code_usage(
    *,
    labels: Tensor,
    indices: Tensor,
    codebook_size: int,
    n_buckets: int = 5,
) -> list[dict[str, Any]]:
    """Return active-code summaries by equal-count condition buckets."""
    values = condition_values(labels)
    if values.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Expected one condition per token path; got labels {tuple(labels.shape)} "
            f"and indices {tuple(indices.shape)}."
        )
    if values.numel() == 0:
        return []

    bucket_labels = ("very_low", "low", "mid", "high", "very_high")
    bucket_count = min(n_buckets, int(values.shape[0]))
    sorted_positions = torch.argsort(values)
    buckets: list[dict[str, Any]] = []
    for bucket_index, bucket_positions in enumerate(
        torch.tensor_split(sorted_positions, bucket_count)
    ):
        if bucket_positions.numel() == 0:
            continue
        bucket_indices = indices.index_select(0, bucket_positions)
        bucket_values = values.index_select(0, bucket_positions)
        code_counts = torch.bincount(
            bucket_indices.reshape(-1),
            minlength=codebook_size,
        )[:codebook_size]
        usage = summarise_code_usage(code_counts)
        component_usage = component_code_usage(bucket_indices, codebook_size)
        buckets.append(
            {
                "bucket_index": bucket_index,
                "bucket_label": bucket_labels[bucket_index]
                if bucket_index < len(bucket_labels)
                else f"bucket_{bucket_index}",
                "n_samples": int(bucket_positions.numel()),
                "condition_min": float(bucket_values.min().item()),
                "condition_max": float(bucket_values.max().item()),
                "condition_mean": float(bucket_values.mean().item()),
                "active_code_count": usage["active_code_count"],
                "active_code_ratio": usage["active_code_ratio"],
                "codebook_perplexity": usage["codebook_perplexity"],
                "index_entropy": usage["index_entropy"],
                "active_code_indices": usage["active_code_indices"],
                **component_usage,
            }
        )
    return buckets


def condition_values(labels: Tensor) -> Tensor:
    """Collapse scalar or temporal labels to one condition value per path."""
    detached = labels.detach().cpu().float()
    if detached.ndim == 1:
        return detached
    if detached.ndim == 2:
        return detached.reshape(detached.shape[0], -1).mean(dim=1)
    if detached.ndim == 3:
        return detached.mean(dim=(1, 2))
    raise ValueError(
        "labels must be [batch], [batch, condition_dim], or "
        f"[batch, length, condition_dim]; got {tuple(labels.shape)}."
    )


def save_token_dataset(
    output_dir: str | Path,
    *,
    train_payload: Mapping[str, Tensor],
    eval_payload: Mapping[str, Tensor],
    summary: Mapping[str, Any],
) -> None:
    """Write train/eval token tensors and a JSON summary to ``output_dir``."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(dict(train_payload), directory / "train_tokens.pt")
    torch.save(dict(eval_payload), directory / "eval_tokens.pt")
    with (directory / "token_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
