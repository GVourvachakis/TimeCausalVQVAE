"""Command-line evaluation entry point for causal token priors."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import ml_collections
import numpy as np
import torch
import yaml

from time_causal_vae.data.base import BaseDataset
from time_causal_vae.data.pipeline import DataPipeline
from time_causal_vae.evaluation.token_diagnostics import (
    compare_token_sequences,
    flatten_token_comparison_metrics,
)
from time_causal_vae.evaluation.token_prior import (
    compute_condition_bucket_sample_metrics,
    compute_token_prior_sample_metrics,
    decode_token_indices,
    load_trained_token_prior,
    plot_decoded_path_examples,
    plot_real_vs_sampled_code_usage,
    plot_sampled_code_usage,
    plot_transition_matrix,
    save_decoded_paths,
    save_sampled_tokens,
    save_token_prior_summary,
)
from time_causal_vae.evaluation.tokenizer import load_trained_tokenizer
from time_causal_vae.tokenization import CausalVQTokenizer
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the token-prior evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Sample and evaluate a causal autoregressive token prior.",
    )
    parser.add_argument("--config", required=True, help="Path to a token-prior YAML config.")
    parser.add_argument(
        "--prior-dir",
        required=True,
        help="Directory containing token_prior.pt and token_prior_config.json.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        help="Directory containing tokenizer.pt and tokenizer_config.json.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory under outputs/.")
    parser.add_argument("--n-sample", type=int, default=256, help="Number of paths to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", help="Device override, for example cpu or cuda.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for token generation.",
    )
    parser.add_argument("--top-k", type=int, help="Optional top-k token sampling.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory for datasets that require local files.",
    )
    return parser


def main() -> None:
    """Run token-prior sampling evaluation."""
    parser = build_parser()
    args = parser.parse_args()

    output_dir = validate_output_dir(args.output_dir)
    prior_dir = validate_prior_dir(args.prior_dir)
    tokenizer_dir = validate_tokenizer_dir(args.tokenizer_dir)
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive.")

    set_seed(args.seed)
    device = select_device(args.device)
    raw_config = load_token_prior_yaml(args.config)
    prior, prior_config, _prior_checkpoint = load_trained_token_prior(prior_dir, device=device)
    tokenizer, tokenizer_config, _tokenizer_checkpoint = load_trained_tokenizer(
        tokenizer_dir,
        device=device,
    )
    freeze_tokenizer(tokenizer)
    conditional_payload = load_conditional_eval_payload(
        raw_config,
        n_sample=args.n_sample,
        prior_config=prior_config,
    )
    condition_sampling_convention = "unconditional_no_conditions"
    sample_conditions = None
    decoder_conditions = None
    if conditional_payload is None:
        real_dataset = build_comparison_dataset(
            raw_config,
            n_sample=args.n_sample,
            sequence_length=prior_config.sequence_length,
            base_data_dir=args.base_data_dir,
        )
        real_paths = real_dataset.data.to(device)
        real_tokens = load_reference_tokens(raw_config, n_sample=args.n_sample)
    else:
        real_paths = conditional_payload["data"].to(device)
        sample_conditions = conditional_payload["labels"].to(device)
        decoder_conditions = select_decoder_conditions(
            conditional_payload,
            tokenizer_condition_dim=tokenizer_config.condition_dim,
            device=device,
        )
        real_tokens = conditional_payload["indices"].detach().cpu().long()
        condition_sampling_convention = "paired_eval_labels_from_token_artifacts"

    sampled_tokens = prior.sample(
        batch_size=args.n_sample,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditions=sample_conditions,
    )
    quantized, decoded_paths = decode_token_indices(
        tokenizer,
        sampled_tokens,
        conditions=decoder_conditions,
    )
    metrics = compute_token_prior_sample_metrics(
        sampled_tokens=sampled_tokens,
        decoded_paths=decoded_paths,
        real_paths=real_paths,
        codebook_size=prior_config.codebook_size,
    )
    real_tokens_for_diagnostics = (
        real_tokens if real_tokens is not None and real_tokens.ndim == 2 else None
    )
    can_run_single_token_diagnostics = real_tokens_for_diagnostics is not None
    if real_tokens is not None and real_tokens.shape == sampled_tokens.detach().cpu().shape:
        metrics["token_comparison_note"] = (
            "single-token transition diagnostics omitted for multi-code token tensors"
            if not can_run_single_token_diagnostics
            else "single-token diagnostics computed"
        )
    if real_tokens_for_diagnostics is not None:
        token_comparison = compare_token_sequences(
            real_tokens=real_tokens_for_diagnostics,
            sampled_tokens=sampled_tokens.detach().cpu(),
            codebook_size=prior_config.codebook_size,
        )
        metrics.update(flatten_token_comparison_metrics(token_comparison))
    tensor_shapes = {
        "sampled_tokens": list(sampled_tokens.shape),
        "z_q": list(quantized.shape),
        "decoded_paths": list(decoded_paths.shape),
        "real_paths": list(real_paths.shape),
    }
    if sample_conditions is not None:
        tensor_shapes["conditions"] = list(sample_conditions.shape)
    if decoder_conditions is not None:
        tensor_shapes["decoder_conditions"] = list(decoder_conditions.shape)
    if sample_conditions is not None:
        metrics["condition_buckets"] = compute_condition_bucket_sample_metrics(
            sampled_tokens=sampled_tokens,
            decoded_paths=decoded_paths,
            real_paths=real_paths,
            conditions=sample_conditions,
            codebook_size=prior_config.codebook_size,
        )
    metrics["condition_sampling_convention"] = condition_sampling_convention
    output_dir.mkdir(parents=True, exist_ok=True)
    save_token_prior_summary(
        output_dir / "token_prior_summary.json",
        metrics=metrics,
        prior_config=prior_config,
        tokenizer_config=tokenizer_config,
        tensor_shapes=tensor_shapes,
        config_path=str(Path(args.config)),
        prior_dir=str(prior_dir),
        tokenizer_dir=str(tokenizer_dir),
        n_sample=args.n_sample,
        seed=args.seed,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    save_sampled_tokens(
        output_dir / "sampled_tokens.pt",
        tokens=sampled_tokens,
        metrics=metrics,
    )
    save_decoded_paths(
        output_dir / "decoded_paths.pt",
        decoded_paths=decoded_paths,
        real_paths=real_paths,
        quantized=quantized,
        metrics=metrics,
    )
    plot_sampled_code_usage(
        output_dir / "sampled_code_usage.png",
        code_counts=cast(list[int], metrics["sampled_token_code_usage_counts"]),
        active_code_count=int(metrics["sampled_token_active_code_count"]),
    )
    plot_decoded_path_examples(
        output_dir / "decoded_path_examples.png",
        decoded_paths=decoded_paths,
        real_paths=real_paths,
    )
    generated_figures = ["sampled_code_usage.png", "decoded_path_examples.png"]
    if real_tokens_for_diagnostics is not None:
        plot_real_vs_sampled_code_usage(
            output_dir / "real_vs_sampled_code_usage.png",
            real_tokens=real_tokens_for_diagnostics,
            sampled_tokens=sampled_tokens.detach().cpu(),
            codebook_size=prior_config.codebook_size,
        )
        plot_transition_matrix(
            output_dir / "transition_matrix_real.png",
            tokens=real_tokens_for_diagnostics,
            codebook_size=prior_config.codebook_size,
            title="Real training-token transition matrix",
        )
        plot_transition_matrix(
            output_dir / "transition_matrix_sampled.png",
            tokens=sampled_tokens.detach().cpu(),
            codebook_size=prior_config.codebook_size,
            title="Sampled token-prior transition matrix",
        )
        generated_figures.extend(
            [
                "real_vs_sampled_code_usage.png",
                "transition_matrix_real.png",
                "transition_matrix_sampled.png",
            ]
        )

    print("Token-prior evaluation complete.")
    print(f"prior_dir: {prior_dir}")
    print(f"tokenizer_dir: {tokenizer_dir}")
    print(f"output_dir: {output_dir}")
    print(f"sampled_tokens_shape: {tensor_shapes['sampled_tokens']}")
    print(f"decoded_paths_shape: {tensor_shapes['decoded_paths']}")
    print(f"condition_sampling_convention: {condition_sampling_convention}")
    if sample_conditions is not None:
        print(f"conditions_shape: {tuple(sample_conditions.shape)}")
    if decoder_conditions is not None:
        print(f"decoder_conditions_shape: {tuple(decoder_conditions.shape)}")
    print(
        "sampled codes: "
        f"active={metrics['sampled_token_active_code_count']}/"
        f"{metrics['sampled_token_codebook_size']} "
        f"ratio={metrics['sampled_token_active_code_ratio']:.8f} "
        f"perplexity={metrics['sampled_token_codebook_perplexity']:.8f} "
        f"entropy={metrics['sampled_token_index_entropy']:.8f}"
    )
    print(f"mmd: {metrics['mmd']:.8f}")
    print(f"swd: {metrics['swd']:.8f}")
    print(f"terminal_return_mean_error: {metrics['terminal_return_mean_error']:.8f}")
    print(f"terminal_return_wasserstein: {metrics['terminal_return_wasserstein']:.8f}")
    print(f"volatility_mean_error: {metrics['volatility_mean_error']:.8f}")
    print(f"volatility_wasserstein: {metrics['volatility_wasserstein']:.8f}")
    if can_run_single_token_diagnostics:
        print(f"marginal_code_l1: {metrics['marginal_code_l1']:.8f}")
        print(f"transition_matrix_l1: {metrics['transition_matrix_l1']:.8f}")
        print(f"run_length_distance: {metrics['run_length_distance']:.8f}")
        print(
            "real/sampled token perplexity: "
            f"{metrics['real_token_perplexity']:.8f}/"
            f"{metrics['sampled_token_perplexity']:.8f}"
        )
    if "condition_buckets" in metrics:
        print("condition_buckets:")
        for bucket in cast(list[dict[str, Any]], metrics["condition_buckets"]):
            print(
                "  "
                f"{bucket['bucket_label']} "
                f"n={bucket['n_samples']} "
                f"condition=[{bucket['condition_min']:.8f}, {bucket['condition_max']:.8f}] "
                f"mmd={bucket['mmd']:.8f} "
                f"swd={bucket['swd']:.8f} "
                f"vol_w1={bucket['volatility_wasserstein']:.8f} "
                f"terminal_w1={bucket['terminal_return_wasserstein']:.8f} "
                f"active={bucket['sampled_active_code_count']} "
                f"perplexity={bucket['sampled_token_perplexity']:.8f}"
            )
    print(f"figures: {', '.join(generated_figures)}")


def validate_output_dir(output_dir: str) -> Path:
    """Validate that token-prior evaluation artifacts stay below ignored outputs/."""
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


def validate_prior_dir(prior_dir: str) -> Path:
    """Validate the token-prior directory or print the expected smoke path."""
    path = Path(prior_dir)
    if path.exists() and path.is_dir():
        return path
    expected_path = (
        "outputs/token_prior/black_scholes_kmeans/prior_smoke/"
        "black_scholes_causal_token_prior_seed0"
    )
    raise SystemExit(
        f"--prior-dir does not exist: {prior_dir}\nExpected smoke path: {expected_path}"
    )


def validate_tokenizer_dir(tokenizer_dir: str) -> Path:
    """Validate the tokenizer directory or print the expected quality-gate path."""
    path = Path(tokenizer_dir)
    if path.exists() and path.is_dir():
        return path
    expected_path = (
        "outputs/tokenizer_quality/kmeans/black_scholes_causal_vq_tokenizer_kmeans_seed0"
    )
    raise SystemExit(
        f"--tokenizer-dir does not exist: {tokenizer_dir}\nExpected path: {expected_path}"
    )


def select_device(device_name: str | None) -> torch.device:
    """Select the requested device, or prefer CUDA when available."""
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_token_prior_yaml(path: str | Path) -> dict[str, Any]:
    """Load a token-prior experiment YAML file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Token-prior config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def build_comparison_dataset(
    raw_config: Mapping[str, Any],
    *,
    n_sample: int,
    sequence_length: int,
    base_data_dir: str,
) -> BaseDataset:
    """Build real comparison data through the target data pipeline."""
    experiment = require_mapping(raw_config, "experiment")
    exp_config = ml_collections.ConfigDict()
    exp_config.dataset = str(experiment["dataset"])
    exp_config.n_sample = n_sample
    exp_config.n_timestep = sequence_length
    exp_config.base_data_dir = base_data_dir
    exp_config.data_params = {}
    _train_dataset, eval_dataset = DataPipeline()(exp_config)
    return cast(BaseDataset, eval_dataset)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required config section as a dictionary."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Token-prior config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def load_reference_tokens(raw_config: Mapping[str, Any], *, n_sample: int) -> torch.Tensor | None:
    """Load extracted training tokens when the config records ``token_data_dir``."""
    data = raw_config.get("data")
    if not isinstance(data, dict):
        return None
    raw_token_data_dir = data.get("token_data_dir")
    if raw_token_data_dir is None:
        return None
    token_data_dir = Path(str(raw_token_data_dir))
    train_path = token_data_dir / "train_tokens.pt"
    if not train_path.exists():
        return None
    loaded = torch.load(train_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or "indices" not in loaded:
        raise SystemExit(f"Token payload must contain an 'indices' tensor: {train_path}")
    indices = cast(torch.Tensor, loaded["indices"]).detach().cpu().long()
    if indices.shape[0] > n_sample:
        return indices[:n_sample]
    return indices


def load_conditional_eval_payload(
    raw_config: Mapping[str, Any],
    *,
    n_sample: int,
    prior_config: Any,
) -> dict[str, torch.Tensor] | None:
    """Load paired eval tokens, paths, and labels for conditional generation."""
    if prior_config.condition_injection == "none":
        return None
    data = raw_config.get("data")
    if not isinstance(data, dict):
        raise SystemExit("Conditional token-prior evaluation requires a data section.")
    raw_token_data_dir = data.get("token_data_dir")
    if raw_token_data_dir is None:
        raise SystemExit("Conditional token-prior evaluation requires data.token_data_dir.")
    eval_path = Path(str(raw_token_data_dir)) / "eval_tokens.pt"
    if not eval_path.exists():
        raise SystemExit(f"Missing conditional eval token artifact: {eval_path}")
    loaded = torch.load(eval_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Eval token artifact must be a mapping: {eval_path}")
    required_keys = {"indices", "data", "labels"}
    missing_keys = sorted(required_keys.difference(loaded))
    if missing_keys:
        raise SystemExit(
            f"Eval token artifact is missing required keys {missing_keys}: {eval_path}"
        )
    indices = cast(torch.Tensor, loaded["indices"]).detach().cpu().long()
    paths = cast(torch.Tensor, loaded["data"]).detach().cpu().float()
    labels = cast(torch.Tensor, loaded["labels"]).detach().cpu().float()
    if labels.ndim == 1:
        labels = labels[:, None]
    decoder_labels = labels
    feature_tensor = load_condition_feature_tensor(
        data,
        split_name="eval",
        expected_n=labels.shape[0],
        base_labels=labels,
    )
    if feature_tensor is not None:
        labels = torch.cat([labels, feature_tensor], dim=1)
    if indices.shape[0] < n_sample or paths.shape[0] < n_sample or labels.shape[0] < n_sample:
        raise SystemExit(
            f"Conditional eval artifact has fewer than {n_sample} samples: {eval_path}"
        )
    if labels.shape[-1] != prior_config.condition_dim:
        raise SystemExit(
            f"Eval labels must have condition_dim {prior_config.condition_dim}; "
            f"got {labels.shape[-1]}."
        )
    return {
        "indices": indices[:n_sample],
        "data": paths[:n_sample],
        "labels": labels[:n_sample],
        "decoder_labels": decoder_labels[:n_sample],
    }


def select_decoder_conditions(
    conditional_payload: Mapping[str, torch.Tensor],
    *,
    tokenizer_condition_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return conditions compatible with the frozen tokenizer decoder."""
    if tokenizer_condition_dim == 0:
        return None
    decoder_labels = conditional_payload.get("decoder_labels")
    if decoder_labels is None:
        decoder_labels = conditional_payload["labels"]
    if decoder_labels.shape[-1] != tokenizer_condition_dim:
        raise SystemExit(
            f"Tokenizer decoder expects condition_dim {tokenizer_condition_dim}; "
            f"got decoder labels with dimension {decoder_labels.shape[-1]}."
        )
    return decoder_labels.to(device)


def load_condition_feature_tensor(
    data_config: Mapping[str, Any],
    *,
    split_name: Literal["eval"],
    expected_n: int,
    base_labels: torch.Tensor,
) -> torch.Tensor | None:
    """Load optional precomputed condition features for conditional evaluation."""
    raw_feature_dir = data_config.get("condition_feature_dir")
    if raw_feature_dir is None:
        return None
    feature_dir = Path(str(raw_feature_dir))
    feature_file = str(
        data_config.get("condition_feature_eval_file", "eval_signature_features.npz")
    )
    feature_key = str(data_config.get("condition_feature_key", "features"))
    feature_path = feature_dir / feature_file
    if not feature_path.exists():
        raise SystemExit(f"Missing {split_name} condition feature artifact: {feature_path}")
    try:
        with np.load(feature_path, allow_pickle=False) as npz_file:
            if feature_key not in npz_file:
                available = sorted(npz_file.files)
                raise SystemExit(
                    f"{split_name} condition feature artifact {feature_path} does not contain "
                    f"key {feature_key!r}. Available keys: {available}"
                )
            features = np.asarray(npz_file[feature_key], dtype=np.float32)
            if "sample_indices" in npz_file:
                sample_indices = np.asarray(npz_file["sample_indices"])
                validate_condition_feature_sample_indices(
                    sample_indices,
                    expected_n=expected_n,
                    split_name=split_name,
                    feature_path=feature_path,
                )
            if "labels" in npz_file:
                feature_labels = np.asarray(npz_file["labels"], dtype=np.float32)
                validate_condition_feature_labels(
                    feature_labels,
                    base_labels=base_labels,
                    split_name=split_name,
                    feature_path=feature_path,
                )
    except OSError as exc:
        raise SystemExit(f"Could not load {split_name} condition features: {feature_path}") from exc
    validate_condition_feature_array(
        features,
        expected_n=expected_n,
        split_name=split_name,
        feature_path=feature_path,
    )
    return torch.from_numpy(features.copy()).float()


def validate_condition_feature_array(
    features: np.ndarray,
    *,
    expected_n: int,
    split_name: str,
    feature_path: Path,
) -> None:
    """Validate the feature matrix loaded from an NPZ artifact."""
    if features.ndim != 2:
        raise SystemExit(
            f"{split_name} condition features in {feature_path} must be a 2-D array; "
            f"got shape {features.shape}."
        )
    if features.shape[0] != expected_n:
        raise SystemExit(
            f"{split_name} condition features in {feature_path} have {features.shape[0]} "
            f"rows, but token labels have {expected_n} rows."
        )
    if features.shape[1] <= 0:
        raise SystemExit(f"{split_name} condition features in {feature_path} are empty.")
    if not np.isfinite(features).all():
        raise SystemExit(f"{split_name} condition features in {feature_path} contain NaN or Inf.")


def validate_condition_feature_sample_indices(
    sample_indices: np.ndarray,
    *,
    expected_n: int,
    split_name: str,
    feature_path: Path,
) -> None:
    """Validate optional sample-index alignment metadata."""
    if sample_indices.ndim != 1 or sample_indices.shape[0] != expected_n:
        raise SystemExit(
            f"{split_name} sample_indices in {feature_path} must have shape ({expected_n},); "
            f"got {sample_indices.shape}."
        )
    if np.unique(sample_indices).shape[0] != expected_n:
        raise SystemExit(f"{split_name} sample_indices in {feature_path} contain duplicates.")


def validate_condition_feature_labels(
    feature_labels: np.ndarray,
    *,
    base_labels: torch.Tensor,
    split_name: str,
    feature_path: Path,
) -> None:
    """Validate label metadata against the token artifact labels."""
    if feature_labels.ndim == 1:
        feature_labels = feature_labels[:, None]
    base_labels_np = base_labels.detach().cpu().numpy().astype(np.float32, copy=False)
    if feature_labels.shape != base_labels_np.shape:
        raise SystemExit(
            f"{split_name} labels in {feature_path} have shape {feature_labels.shape}, "
            f"but token labels have shape {base_labels_np.shape}."
        )
    if not np.allclose(feature_labels, base_labels_np, rtol=1e-5, atol=1e-6):
        raise SystemExit(
            f"{split_name} labels in {feature_path} do not match the token artifact labels."
        )


def freeze_tokenizer(tokenizer: CausalVQTokenizer) -> None:
    """Freeze tokenizer parameters for decode-only evaluation."""
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)


if __name__ == "__main__":
    main()
