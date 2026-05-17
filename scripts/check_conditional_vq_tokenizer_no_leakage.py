"""Smoke-check conditional causal VQ tokenizer prefix causality on PDV data."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import ml_collections
import torch
import yaml
from torch import Tensor

from time_causal_vae.data.pipeline import DataPipeline
from time_causal_vae.evaluation.tokenizer import load_trained_tokenizer
from time_causal_vae.models.discrete.tokenizers import CausalVQTokenizer, VQTokenizerConfig
from time_causal_vae.models.discrete.tokenizers.causal_vq_tokenizer import (
    assert_tokenizer_no_future_leakage,
)
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the script parser."""
    parser = argparse.ArgumentParser(
        description="Check conditional causal VQ tokenizer no-future-leakage on PDV data.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a conditional tokenizer YAML config.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        help="Optional trained tokenizer directory. If omitted, check a fresh model.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the check.")
    parser.add_argument("--cutoff", type=int, default=29, help="Inclusive zero-indexed cutoff.")
    parser.add_argument("--seed", type=int, help="Override the config seed.")
    parser.add_argument("--device", default="cpu", help="Device for the check.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory for datasets that require local files.",
    )
    return parser


def main() -> int:
    """Run the conditional tokenizer no-leakage smoke check."""
    args = build_parser().parse_args()
    raw_config = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else raw_config["experiment"].get("seed", 0))
    set_seed(seed)

    device = torch.device(args.device)
    train_data, train_labels = build_pdv_batch(
        raw_config,
        batch_size=args.batch_size,
        base_data_dir=args.base_data_dir,
    )
    inputs = train_data.to(device=device, dtype=torch.float32)
    conditions = train_labels.to(device=device, dtype=torch.float32)
    if args.tokenizer_dir:
        tokenizer, tokenizer_config, _checkpoint = load_trained_tokenizer(
            args.tokenizer_dir,
            device=device,
        )
    else:
        tokenizer_config = build_tokenizer_config(raw_config)
        tokenizer = CausalVQTokenizer(tokenizer_config).to(device)
    tokenizer.eval()

    try:
        output = tokenizer(inputs, conditions)
        if not torch.isfinite(cast(Tensor, output.loss)):
            raise AssertionError(f"Expected finite loss; got {output.loss.item()}.")
        if tuple(output.recon_x.shape) != tuple(inputs.shape):
            raise AssertionError(
                f"Expected reconstruction shape {tuple(inputs.shape)}; "
                f"got {tuple(output.recon_x.shape)}."
            )
        expected_index_shape = expected_tokenizer_index_shape(
            tokenizer_config,
            batch_size=inputs.shape[0],
            sequence_length=inputs.shape[1],
        )
        if tuple(output.indices.shape) != expected_index_shape:
            raise AssertionError(
                f"Expected index shape {expected_index_shape}; got {tuple(output.indices.shape)}."
            )

        changed_future_inputs = inputs.clone()
        changed_future_inputs[:, args.cutoff + 1 :] += 5.0 * torch.randn_like(
            changed_future_inputs[:, args.cutoff + 1 :]
        )
        reference_recon, changed_recon = assert_tokenizer_no_future_leakage(
            tokenizer,
            inputs,
            changed_future_inputs,
            args.cutoff,
            conditions=conditions,
            atol=1e-6,
            rtol=1e-5,
        )
        max_prefix_diff = (
            reference_recon[:, : args.cutoff + 1] - changed_recon[:, : args.cutoff + 1]
        ).abs()
        max_prefix_diff_value = float(max_prefix_diff.max().item())
        temporal_condition_status = "skipped_no_temporal_condition_in_current_pdv_dataset"
        temporal_condition_max_prefix_diff_value: float | None = None
        if conditions.ndim == 3:
            changed_future_conditions = conditions.clone()
            changed_future_conditions[:, args.cutoff + 1 :] += 5.0 * torch.randn_like(
                changed_future_conditions[:, args.cutoff + 1 :]
            )
            with torch.no_grad():
                reference_condition_output = tokenizer(inputs, conditions)
                changed_condition_output = tokenizer(inputs, changed_future_conditions)
            condition_prefix_diff = (
                reference_condition_output.recon_x[:, : args.cutoff + 1]
                - changed_condition_output.recon_x[:, : args.cutoff + 1]
            ).abs()
            temporal_condition_max_prefix_diff_value = float(condition_prefix_diff.max().item())
            if not torch.allclose(
                reference_condition_output.recon_x[:, : args.cutoff + 1],
                changed_condition_output.recon_x[:, : args.cutoff + 1],
                atol=1e-6,
                rtol=1e-5,
            ):
                raise AssertionError(
                    "Future temporal condition perturbation changed prefix reconstruction: "
                    f"max_diff={temporal_condition_max_prefix_diff_value:.8e}."
                )
            temporal_condition_status = "passed"
    except Exception as exc:
        print(f"FAIL conditional VQ tokenizer no-leakage check: {exc}")
        return 1

    print("PASS conditional VQ tokenizer no-leakage check")
    print(f"config={args.config}")
    print(f"tokenizer_dir={args.tokenizer_dir or 'fresh_random_initialization'}")
    print(f"dataset={raw_config['experiment']['dataset']}")
    print(f"x={tuple(inputs.shape)}")
    print(f"conditions={tuple(conditions.shape)}")
    print("condition_handling=scalar_repeated_over_time")
    print(f"z_e={tuple(output.z_e.shape)}")
    print(f"z_q={tuple(output.z_q.shape)}")
    print(f"indices={tuple(output.indices.shape)}")
    print(f"recon_x={tuple(output.recon_x.shape)}")
    print(f"cutoff={args.cutoff}")
    print(f"loss={output.loss.item():.8f}")
    print(f"recon_loss={output.recon_loss.item():.8f}")
    print(f"max_prefix_diff={max_prefix_diff_value:.8e}")
    print(f"temporal_condition_check={temporal_condition_status}")
    if temporal_condition_max_prefix_diff_value is not None:
        print(f"temporal_condition_max_prefix_diff={temporal_condition_max_prefix_diff_value:.8e}")
    return 0


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def build_pdv_batch(
    raw_config: Mapping[str, Any],
    *,
    batch_size: int,
    base_data_dir: str,
) -> tuple[Tensor, Tensor]:
    """Build a small PDV batch through the target data pipeline."""
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    experiment = require_mapping(raw_config, "experiment")
    data = require_mapping(raw_config, "data")
    exp_config = ml_collections.ConfigDict()
    exp_config.dataset = experiment["dataset"]
    exp_config.n_sample = max(batch_size, int(data["n_samples"]))
    exp_config.n_timestep = int(data["n_timesteps"])
    exp_config.base_data_dir = base_data_dir
    exp_config.data_params = dict(cast(Mapping[str, Any], data.get("data_params", {})))
    train_dataset, _ = DataPipeline()(exp_config)
    return train_dataset.data[:batch_size], train_dataset.labels[:batch_size]


def build_tokenizer_config(raw_config: Mapping[str, Any]) -> VQTokenizerConfig:
    """Build ``VQTokenizerConfig`` from a tokenizer experiment config."""
    data = require_mapping(raw_config, "data")
    model = require_mapping(raw_config, "model")
    return VQTokenizerConfig(
        data_dim=int(model.get("data_dim", data["data_dim"])),
        data_length=int(model.get("data_length", data["n_timesteps"])),
        embedding_dim=int(model["embedding_dim"]),
        codebook_size=int(model["codebook_size"]),
        commitment_weight=float(model["commitment_weight"]),
        encoder_hidden_dim=int(model["encoder_hidden_dim"]),
        decoder_hidden_dim=int(model["decoder_hidden_dim"]),
        num_layers=int(model["num_layers"]),
        dilations=tuple(int(value) for value in model["dilations"]),
        dropout=float(model.get("dropout", 0.0)),
        condition_dim=int(model.get("condition_dim", data.get("condition_dim", 0))),
        kmeans_init=bool(model.get("kmeans_init", False)),
        kmeans_iters=int(model.get("kmeans_iters", 10)),
        use_cosine_sim=bool(model.get("use_cosine_sim", False)),
        codebook_dim=optional_int(model.get("codebook_dim")),
        threshold_ema_dead_code=float(model.get("threshold_ema_dead_code", 0.0)),
        decay=float(model.get("decay", 0.8)),
        usage_regularization_weight=float(model.get("usage_regularization_weight", 0.0)),
        usage_regularization_type=usage_regularization_type(
            model.get("usage_regularization_type", "none")
        ),
        quantizer_type=quantizer_type(model.get("quantizer_type", "vector")),
        num_quantizers=int(model.get("num_quantizers", 1)),
        groups=int(model.get("groups", 1)),
        shared_codebook=bool(model.get("shared_codebook", False)),
        stochastic_sample_codes=bool(model.get("stochastic_sample_codes", False)),
        sample_codebook_temp=float(model.get("sample_codebook_temp", 0.0)),
    )


def optional_int(value: object) -> int | None:
    """Return an optional integer config value."""
    if value is None:
        return None
    return int(cast(int | str, value))


def usage_regularization_type(value: object) -> Literal["none", "entropy"]:
    """Return a validated usage regularisation type."""
    typed_value = str(value)
    if typed_value not in {"none", "entropy"}:
        raise ValueError("usage_regularization_type must be 'none' or 'entropy'.")
    return cast(Literal["none", "entropy"], typed_value)


def quantizer_type(value: object) -> Literal["vector", "residual_vq", "grouped_residual_vq"]:
    """Return a validated quantizer type."""
    typed_value = str(value)
    if typed_value not in {"vector", "residual_vq", "grouped_residual_vq"}:
        raise ValueError(
            "quantizer_type must be 'vector', 'residual_vq', or 'grouped_residual_vq'."
        )
    return cast(Literal["vector", "residual_vq", "grouped_residual_vq"], typed_value)


def expected_tokenizer_index_shape(
    tokenizer_config: VQTokenizerConfig,
    *,
    batch_size: int,
    sequence_length: int,
) -> tuple[int, ...]:
    """Return the expected tokenizer index shape for the configured quantizer."""
    if tokenizer_config.quantizer_type == "vector":
        return (batch_size, sequence_length)
    if tokenizer_config.quantizer_type == "residual_vq":
        return (batch_size, sequence_length, tokenizer_config.num_quantizers)
    if tokenizer_config.quantizer_type == "grouped_residual_vq":
        return (
            batch_size,
            sequence_length,
            tokenizer_config.groups,
            tokenizer_config.num_quantizers,
        )
    raise ValueError(f"Unsupported quantizer_type={tokenizer_config.quantizer_type!r}.")


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required YAML section as a dictionary."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


if __name__ == "__main__":
    raise SystemExit(main())
