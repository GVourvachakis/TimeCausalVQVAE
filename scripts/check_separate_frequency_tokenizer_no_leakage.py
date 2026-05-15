"""Smoke-check separate low/high frequency-tokenizer prefix causality."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import ml_collections
import torch
import yaml
from torch import Tensor

from time_causal_vae.cli.train_tokenizer import build_tokenizer_config
from time_causal_vae.data.frequency import causal_ema_frequency_component
from time_causal_vae.data.pipeline import DataPipeline
from time_causal_vae.tokenization import CausalVQTokenizer
from time_causal_vae.utils.random import set_seed

DEFAULT_LOW_CONFIG = "configs/experiments/sp500_vix_causal_vq_tokenizer_freq_low_alpha02.yaml"
DEFAULT_HIGH_CONFIG = "configs/experiments/sp500_vix_causal_vq_tokenizer_freq_high_alpha02.yaml"


@dataclass(frozen=True)
class TokenizerCheckResult:
    """Prefix-invariance results for one untrained component tokenizer."""

    component: str
    input_shape: tuple[int, ...]
    max_component_prefix_diff: float
    max_encoder_prefix_diff: float
    deterministic_after_warmup: bool
    token_prefix_mismatch_count: int | None
    max_reconstruction_prefix_diff: float | None


def build_parser() -> argparse.ArgumentParser:
    """Build the script parser."""
    parser = argparse.ArgumentParser(
        description="Check separate causal EMA frequency tokenizers for no-future-leakage.",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="EMA smoothing parameter.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the check.")
    parser.add_argument("--cutoff", type=int, default=29, help="Inclusive zero-indexed cutoff.")
    parser.add_argument("--seed", type=int, default=99, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Device for the check.")
    parser.add_argument("--base-data-dir", default="data/processed")
    parser.add_argument("--low-config", default=DEFAULT_LOW_CONFIG)
    parser.add_argument("--high-config", default=DEFAULT_HIGH_CONFIG)
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance.")
    return parser


def main() -> int:
    """Run the separate-tokenizer no-leakage smoke check."""
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.cutoff < 0:
        raise SystemExit("--cutoff must be non-negative.")

    set_seed(args.seed)
    device = torch.device(args.device)
    low_config = load_yaml(args.low_config)
    high_config = load_yaml(args.high_config)
    original_paths, conditions = build_batch(
        low_config,
        batch_size=args.batch_size,
        base_data_dir=args.base_data_dir,
    )
    if args.cutoff >= original_paths.shape[1] - 1:
        raise SystemExit("--cutoff must leave at least one future step to perturb.")

    changed_future_paths = perturb_future(original_paths, cutoff=args.cutoff)
    conditions = conditions.to(device)

    try:
        low_result = check_component_tokenizer(
            raw_config=low_config,
            component="low",
            original_paths=original_paths,
            changed_future_paths=changed_future_paths,
            conditions=conditions,
            alpha=args.alpha,
            cutoff=args.cutoff,
            device=device,
        )
        high_result = check_component_tokenizer(
            raw_config=high_config,
            component="high",
            original_paths=original_paths,
            changed_future_paths=changed_future_paths,
            conditions=conditions,
            alpha=args.alpha,
            cutoff=args.cutoff,
            device=device,
        )
        results = [low_result, high_result]
        for result in results:
            if result.max_component_prefix_diff > args.atol:
                raise AssertionError(
                    f"Future perturbation changed {result.component} component prefix: "
                    f"{result.max_component_prefix_diff:.8e}."
                )
            if result.max_encoder_prefix_diff > args.atol + args.rtol:
                raise AssertionError(
                    f"Future perturbation changed {result.component} encoder prefix: "
                    f"{result.max_encoder_prefix_diff:.8e}."
                )
            if result.deterministic_after_warmup:
                if result.token_prefix_mismatch_count != 0:
                    raise AssertionError(
                        f"{result.component} token prefix changed after warm-up: "
                        f"{result.token_prefix_mismatch_count} positions."
                    )
                if (
                    result.max_reconstruction_prefix_diff is not None
                    and result.max_reconstruction_prefix_diff > args.atol + args.rtol
                ):
                    raise AssertionError(
                        f"{result.component} reconstruction prefix changed after warm-up: "
                        f"{result.max_reconstruction_prefix_diff:.8e}."
                    )
    except Exception as exc:
        print(f"FAIL separate frequency tokenizer no-leakage check: {exc}")
        return 1

    print("PASS separate frequency tokenizer no-leakage check")
    print(f"low_config={args.low_config}")
    print(f"high_config={args.high_config}")
    print(f"batch_size={args.batch_size}")
    print(f"original_shape={tuple(original_paths.shape)}")
    print(f"conditions_shape={tuple(conditions.shape)}")
    print(f"alpha={args.alpha}")
    print(f"cutoff={args.cutoff}")
    print(f"seed={args.seed}")
    for result in results:
        print(f"{result.component}_shape={result.input_shape}")
        print(
            f"max_{result.component}_component_prefix_diff={result.max_component_prefix_diff:.8e}"
        )
        print(f"max_{result.component}_encoder_prefix_diff={result.max_encoder_prefix_diff:.8e}")
        print(f"{result.component}_deterministic_after_warmup={result.deterministic_after_warmup}")
        print(
            f"{result.component}_token_prefix_mismatch_count={result.token_prefix_mismatch_count}"
        )
        reconstruction_diff = result.max_reconstruction_prefix_diff
        if reconstruction_diff is None:
            print(f"max_{result.component}_reconstruction_prefix_diff=None")
        else:
            print(f"max_{result.component}_reconstruction_prefix_diff={reconstruction_diff:.8e}")
    return 0


def check_component_tokenizer(
    *,
    raw_config: Mapping[str, Any],
    component: str,
    original_paths: Tensor,
    changed_future_paths: Tensor,
    conditions: Tensor,
    alpha: float,
    cutoff: int,
    device: torch.device,
) -> TokenizerCheckResult:
    """Check one component transform and its matching untrained tokenizer."""
    component_paths = causal_ema_frequency_component(
        original_paths,
        alpha,
        cast(Any, component),
    ).to(device)
    changed_component_paths = causal_ema_frequency_component(
        changed_future_paths,
        alpha,
        cast(Any, component),
    ).to(device)
    max_component_prefix_diff = max_abs_difference(
        component_paths[:, : cutoff + 1],
        changed_component_paths[:, : cutoff + 1],
    )

    tokenizer_config = build_tokenizer_config(
        {
            "data": require_mapping(raw_config, "data"),
            "model": require_mapping(raw_config, "model"),
        }
    )
    tokenizer = CausalVQTokenizer(tokenizer_config).to(device)
    tokenizer.eval()

    with torch.no_grad():
        reference_z_e = tokenizer.encoder(component_paths, conditions)
        changed_z_e = tokenizer.encoder(changed_component_paths, conditions)
    max_encoder_prefix_diff = max_abs_difference(
        reference_z_e[:, : cutoff + 1],
        changed_z_e[:, : cutoff + 1],
    )

    with torch.no_grad():
        _warmup = tokenizer(component_paths, conditions)
        reference_output = tokenizer(component_paths, conditions)
        repeated_output = tokenizer(component_paths, conditions)
        changed_output = tokenizer(changed_component_paths, conditions)
    reference_indices = cast(Tensor, reference_output.indices)
    repeated_indices = cast(Tensor, repeated_output.indices)
    deterministic_after_warmup = bool(torch.equal(reference_indices, repeated_indices))

    token_prefix_mismatch_count: int | None = None
    reconstruction_prefix_diff: float | None = None
    if deterministic_after_warmup:
        changed_indices = cast(Tensor, changed_output.indices)
        token_prefix_mismatch_count = int(
            (reference_indices[:, : cutoff + 1] != changed_indices[:, : cutoff + 1])
            .sum()
            .detach()
            .cpu()
        )
        reconstruction_prefix_diff = max_abs_difference(
            cast(Tensor, reference_output.recon_x)[:, : cutoff + 1],
            cast(Tensor, changed_output.recon_x)[:, : cutoff + 1],
        )

    return TokenizerCheckResult(
        component=component,
        input_shape=tuple(component_paths.shape),
        max_component_prefix_diff=max_component_prefix_diff,
        max_encoder_prefix_diff=max_encoder_prefix_diff,
        deterministic_after_warmup=deterministic_after_warmup,
        token_prefix_mismatch_count=token_prefix_mismatch_count,
        max_reconstruction_prefix_diff=reconstruction_prefix_diff,
    )


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
