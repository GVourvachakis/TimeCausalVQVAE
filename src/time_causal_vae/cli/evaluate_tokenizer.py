"""Command-line evaluation entry point for causal VQ tokenizers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import ml_collections
import torch
import yaml

from time_causal_vae.data.base import BaseDataset
from time_causal_vae.data.pipeline import DataPipeline
from time_causal_vae.evaluation.tokenizer import (
    evaluate_tokenizer_batch,
    load_trained_tokenizer,
    plot_code_usage,
    plot_reconstruction_examples,
    save_tokenizer_batch,
    save_tokenizer_summary,
)
from time_causal_vae.models.discrete.tokenizers import CausalVQTokenizer
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the tokenizer evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained causal VQ tokenizer.",
    )
    parser.add_argument("--config", required=True, help="Path to a tokenizer experiment YAML file.")
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        help="Directory containing tokenizer.pt and tokenizer_config.json.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory under outputs/.")
    parser.add_argument(
        "--n-sample-test",
        type=int,
        default=256,
        help="Number of evaluation paths to sample.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the test dataset.")
    parser.add_argument("--device", help="Device override, for example cpu or cuda.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory for datasets that require local files.",
    )
    return parser


def main() -> None:
    """Run the tokenizer evaluation command."""
    parser = build_parser()
    args = parser.parse_args()

    output_dir = validate_output_dir(args.output_dir)
    tokenizer_dir = validate_tokenizer_dir(args.tokenizer_dir)
    if args.n_sample_test <= 0:
        raise SystemExit("--n-sample-test must be positive.")

    set_seed(args.seed)
    device = select_device(args.device)
    raw_config = load_tokenizer_yaml(args.config)
    dataset = build_dataset(
        raw_config,
        n_sample_test=args.n_sample_test,
        base_data_dir=args.base_data_dir,
    )
    tokenizer, tokenizer_config, _checkpoint = load_trained_tokenizer(tokenizer_dir, device=device)
    inputs = dataset.data.to(device)
    conditions = condition_tensor(dataset, tokenizer, device)

    output, metrics = evaluate_tokenizer_batch(
        tokenizer,
        inputs,
        conditions=conditions,
        codebook_size=tokenizer_config.codebook_size,
    )
    tensor_shapes = {
        "x": list(inputs.shape),
        "recon_x": list(output["recon_x"].shape),
        "z_e": list(output["z_e"].shape),
        "z_q": list(output["z_q"].shape),
        "indices": list(output["indices"].shape),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_tokenizer_summary(
        output_dir / "tokenizer_summary.json",
        metrics=metrics,
        tokenizer_config=tokenizer_config,
        tensor_shapes=tensor_shapes,
        config_path=str(Path(args.config)),
        tokenizer_dir=str(tokenizer_dir),
        n_sample_test=args.n_sample_test,
        device=device,
    )
    save_tokenizer_batch(
        output_dir / "tokenizer_batch.pt",
        inputs=inputs,
        output=output,
        metrics=metrics,
        conditions=conditions,
    )
    plot_code_usage(
        output_dir / "code_usage.png",
        code_counts=cast(list[int], metrics["code_usage_counts"]),
        active_code_count=int(metrics["active_code_count"]),
    )
    plot_reconstruction_examples(
        output_dir / "reconstruction_examples.png",
        inputs=inputs,
        reconstructions=cast(torch.Tensor, output["recon_x"]),
    )

    print("Tokenizer evaluation complete.")
    print(f"tokenizer_dir: {tokenizer_dir}")
    print(f"output_dir: {output_dir}")
    print(f"x_shape: {tensor_shapes['x']}")
    print(f"recon_x_shape: {tensor_shapes['recon_x']}")
    print(f"indices_shape: {tensor_shapes['indices']}")
    print(f"reconstruction_l1: {metrics['reconstruction_l1']:.8f}")
    print(f"reconstruction_l2: {metrics['reconstruction_l2']:.8f}")
    print(f"terminal_return_error: {metrics['terminal_return_error']:.8f}")
    print(f"volatility_reconstruction_error: {metrics['volatility_reconstruction_error']:.8f}")
    print(
        "codebook: "
        f"active={metrics['active_code_count']}/{metrics['codebook_size']} "
        f"ratio={metrics['active_code_ratio']:.8f} "
        f"perplexity={metrics['codebook_perplexity']:.8f} "
        f"entropy={metrics['index_entropy']:.8f}"
    )
    if "condition_buckets" in metrics:
        print("condition_buckets:")
        for bucket in cast(list[dict[str, Any]], metrics["condition_buckets"]):
            print(
                "  "
                f"{bucket['bucket_label']} "
                f"n={bucket['n_samples']} "
                f"condition=[{bucket['condition_min']:.8f}, {bucket['condition_max']:.8f}] "
                f"recon_l1={bucket['reconstruction_l1']:.8f} "
                f"vol_err={bucket['volatility_reconstruction_error']:.8f} "
                f"active={bucket['active_code_count']} "
                f"perplexity={bucket['codebook_perplexity']:.8f}"
            )
    print("figures: code_usage.png, reconstruction_examples.png")


def validate_output_dir(output_dir: str) -> Path:
    """Validate that tokenizer evaluation artifacts stay below local outputs/."""
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


def validate_tokenizer_dir(tokenizer_dir: str) -> Path:
    """Validate the tokenizer directory or print the expected smoke path."""
    path = Path(tokenizer_dir)
    if path.exists() and path.is_dir():
        return path
    expected_path = "outputs/tokenizer_smoke/black_scholes/black_scholes_causal_vq_tokenizer_seed0"
    raise SystemExit(
        f"--tokenizer-dir does not exist: {tokenizer_dir}\nExpected smoke path: {expected_path}"
    )


def select_device(device_name: str | None) -> torch.device:
    """Select the requested device, or prefer CUDA when available."""
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer_yaml(path: str | Path) -> dict[str, Any]:
    """Load a tokenizer experiment YAML file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Tokenizer config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def build_dataset(
    raw_config: Mapping[str, Any],
    *,
    n_sample_test: int,
    base_data_dir: str,
) -> BaseDataset:
    """Build an evaluation dataset through the target data pipeline."""
    experiment = require_mapping(raw_config, "experiment")
    data = require_mapping(raw_config, "data")
    exp_config = ml_collections.ConfigDict()
    exp_config.dataset = str(experiment["dataset"])
    exp_config.n_sample = n_sample_test
    exp_config.n_timestep = int(data["n_timesteps"])
    exp_config.base_data_dir = base_data_dir
    exp_config.data_params = dict(cast(Mapping[str, Any], data.get("data_params", {})))
    if "rho" in data:
        exp_config.rho = data["rho"]
    _train_dataset, eval_dataset = DataPipeline()(exp_config)
    return cast(BaseDataset, eval_dataset)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required config section as a dictionary."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Tokenizer config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def condition_tensor(
    dataset: BaseDataset,
    tokenizer: CausalVQTokenizer,
    device: torch.device,
) -> torch.Tensor | None:
    """Return labels as conditions only if the tokenizer was configured to use them."""
    if tokenizer.config.condition_dim == 0:
        return None
    labels = dataset.labels.to(device)
    if labels.shape[-1] != tokenizer.config.condition_dim:
        raise ValueError(
            f"Expected condition_dim={tokenizer.config.condition_dim}; got labels "
            f"{tuple(labels.shape)}."
        )
    return labels


if __name__ == "__main__":
    main()
