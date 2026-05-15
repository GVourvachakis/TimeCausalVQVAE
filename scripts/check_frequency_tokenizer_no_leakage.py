"""Smoke-check frequency-tokenizer prefix causality on decomposed paths."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import ml_collections
import torch
import yaml
from torch import Tensor

from time_causal_vae.cli.train_tokenizer import build_tokenizer_config
from time_causal_vae.data.frequency import causal_ema_frequency_channels
from time_causal_vae.data.pipeline import DataPipeline
from time_causal_vae.tokenization import CausalVQTokenizer
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the script parser."""
    parser = argparse.ArgumentParser(
        description="Check joint causal EMA frequency-tokenizer no-future-leakage.",
    )
    parser.add_argument("--config", required=True, help="Frequency tokenizer YAML config.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the check.")
    parser.add_argument("--cutoff", type=int, default=29, help="Inclusive zero-indexed cutoff.")
    parser.add_argument("--seed", type=int, default=99, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Device for the check.")
    parser.add_argument("--base-data-dir", default="data/processed")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance.")
    return parser


def main() -> int:
    """Run the frequency-tokenizer no-leakage smoke check."""
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.cutoff < 0:
        raise SystemExit("--cutoff must be non-negative.")

    raw_config = load_yaml(args.config)
    data_config = require_mapping(raw_config, "data")
    if str(data_config.get("frequency_decomposition")).lower() != "ema":
        raise SystemExit("This check requires data.frequency_decomposition: ema.")
    alpha = float(data_config["ema_alpha"])
    set_seed(args.seed)
    device = torch.device(args.device)

    original_paths, conditions = build_batch(
        raw_config,
        batch_size=args.batch_size,
        base_data_dir=args.base_data_dir,
    )
    if args.cutoff >= original_paths.shape[1] - 1:
        raise SystemExit("--cutoff must leave at least one future step to perturb.")

    changed_future_paths = perturb_future(original_paths, cutoff=args.cutoff)
    frequency_paths = causal_ema_frequency_channels(original_paths, alpha).to(device)
    changed_frequency_paths = causal_ema_frequency_channels(changed_future_paths, alpha).to(device)
    conditions = conditions.to(device)

    tokenizer_config = build_tokenizer_config(
        {
            "data": data_config,
            "model": require_mapping(raw_config, "model"),
        }
    )
    tokenizer = CausalVQTokenizer(tokenizer_config).to(device)
    tokenizer.eval()

    try:
        low_prefix_diff = max_abs_difference(
            frequency_paths[:, : args.cutoff + 1, 0:1],
            changed_frequency_paths[:, : args.cutoff + 1, 0:1],
        )
        high_prefix_diff = max_abs_difference(
            frequency_paths[:, : args.cutoff + 1, 1:2],
            changed_frequency_paths[:, : args.cutoff + 1, 1:2],
        )
        if low_prefix_diff > args.atol or high_prefix_diff > args.atol:
            raise AssertionError(
                "Future perturbation changed decomposed prefix: "
                f"low={low_prefix_diff:.8e}, high={high_prefix_diff:.8e}."
            )

        with torch.no_grad():
            reference_z_e = tokenizer.encoder(frequency_paths, conditions)
            changed_z_e = tokenizer.encoder(changed_frequency_paths, conditions)
        encoder_prefix_diff = max_abs_difference(
            reference_z_e[:, : args.cutoff + 1],
            changed_z_e[:, : args.cutoff + 1],
        )
        if encoder_prefix_diff > args.atol + args.rtol:
            raise AssertionError(
                f"Future perturbation changed encoder prefix: {encoder_prefix_diff:.8e}."
            )

        with torch.no_grad():
            _warmup = tokenizer(frequency_paths, conditions)
            reference_output = tokenizer(frequency_paths, conditions)
            changed_output = tokenizer(changed_frequency_paths, conditions)
        reference_indices = cast(Tensor, reference_output.indices)
        changed_indices = cast(Tensor, changed_output.indices)
        token_prefix_mismatch_count = int(
            (reference_indices[:, : args.cutoff + 1] != changed_indices[:, : args.cutoff + 1])
            .sum()
            .detach()
            .cpu()
        )
        reconstruction_prefix_diff = max_abs_difference(
            cast(Tensor, reference_output.recon_x)[:, : args.cutoff + 1],
            cast(Tensor, changed_output.recon_x)[:, : args.cutoff + 1],
        )
        if token_prefix_mismatch_count != 0:
            raise AssertionError(
                f"Token prefix changed after warm-up: {token_prefix_mismatch_count} positions."
            )
        if reconstruction_prefix_diff > args.atol + args.rtol:
            raise AssertionError(
                f"Reconstruction prefix changed after warm-up: {reconstruction_prefix_diff:.8e}."
            )
    except Exception as exc:
        print(f"FAIL frequency tokenizer no-leakage check: {exc}")
        return 1

    print("PASS frequency tokenizer no-leakage check")
    print(f"config={args.config}")
    print(f"batch_size={args.batch_size}")
    print(f"original_shape={tuple(original_paths.shape)}")
    print(f"frequency_shape={tuple(frequency_paths.shape)}")
    print(f"conditions_shape={tuple(conditions.shape)}")
    print(f"alpha={alpha}")
    print(f"cutoff={args.cutoff}")
    print(f"max_low_prefix_diff={low_prefix_diff:.8e}")
    print(f"max_high_prefix_diff={high_prefix_diff:.8e}")
    print(f"max_encoder_prefix_diff={encoder_prefix_diff:.8e}")
    print(f"token_prefix_mismatch_count={token_prefix_mismatch_count}")
    print(f"max_reconstruction_prefix_diff={reconstruction_prefix_diff:.8e}")
    return 0


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required config section as a dictionary."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def build_batch(
    raw_config: Mapping[str, Any],
    *,
    batch_size: int,
    base_data_dir: str,
) -> tuple[Tensor, Tensor]:
    """Build an original one-channel batch through the data pipeline."""
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


def perturb_future(path: Tensor, *, cutoff: int) -> Tensor:
    """Perturb only future target values after an inclusive cutoff."""
    changed = path.clone()
    future = changed[:, cutoff + 1 :]
    changed[:, cutoff + 1 :] = future + 0.25 * torch.randn_like(future)
    return changed


def max_abs_difference(first: Tensor, second: Tensor) -> float:
    """Return the maximum absolute tensor difference as a Python float."""
    return float((first - second).abs().max().item())


if __name__ == "__main__":
    raise SystemExit(main())
