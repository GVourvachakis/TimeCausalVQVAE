"""Evaluate Hawkes-jump token priors with log-return-to-price diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import Tensor

from time_causal_vae.cli.evaluate_token_prior import (
    freeze_tokenizer,
    load_conditional_eval_payload,
)
from time_causal_vae.evaluation.jump_diagnostics import (
    detected_jump_sizes,
    jump_count_wasserstein,
    jump_diagnostic_summary,
    log_returns_to_normalized_prices,
)
from time_causal_vae.evaluation.market_diagnostics import (
    compare_market_summaries,
    compute_log_returns,
    wasserstein_1d,
)
from time_causal_vae.evaluation.token_diagnostics import (
    compare_token_sequences,
    flatten_token_comparison_metrics,
)
from time_causal_vae.evaluation.token_prior import (
    component_token_metrics,
    compute_path_distribution_metrics,
    decode_token_indices,
    load_trained_token_prior,
)
from time_causal_vae.evaluation.tokenizer import load_trained_tokenizer, summarise_code_usage
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the Hawkes token-prior evaluation parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Hawkes-jump token priors with jump-aware diagnostics.",
    )
    parser.add_argument("--config", required=True, help="Token-prior experiment YAML config.")
    parser.add_argument("--prior-dir", required=True, help="Directory containing token_prior.pt.")
    parser.add_argument("--tokenizer-dir", required=True, help="Directory containing tokenizer.pt.")
    parser.add_argument("--output-dir", required=True, help="Output directory under outputs/.")
    parser.add_argument("--base-data-dir", default="data/processed")
    parser.add_argument("--n-sample", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="Device override, for example cpu or cuda.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--top-k",
        type=parse_optional_top_k,
        default=None,
        help="Top-k sampling cutoff. Use 'none' for unrestricted sampling.",
    )
    return parser


def parse_optional_top_k(raw_value: str) -> int | None:
    """Parse a top-k CLI value, accepting 'none' for unrestricted sampling."""
    normalised = raw_value.strip().lower()
    if normalised in {"none", "null", "unrestricted"}:
        return None
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("--top-k must be positive or 'none'.")
    return value


def main() -> None:
    """Run Hawkes-jump prior evaluation."""
    args = build_parser().parse_args()
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive.")

    output_dir = validate_output_dir(args.output_dir)
    prior_dir = validate_dir(args.prior_dir, description="prior")
    tokenizer_dir = validate_dir(args.tokenizer_dir, description="tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)

    raw_config = load_yaml(args.config)
    data_output = read_data_output(raw_config)
    prior, prior_config, _prior_checkpoint = load_trained_token_prior(prior_dir, device=device)
    tokenizer, tokenizer_config, _tokenizer_checkpoint = load_trained_tokenizer(
        tokenizer_dir,
        device=device,
    )
    freeze_tokenizer(tokenizer)
    eval_payload = load_conditional_eval_payload(
        raw_config,
        n_sample=args.n_sample,
        prior_config=prior_config,
    )
    if eval_payload is None:
        raise SystemExit("Hawkes-jump prior evaluation expects conditional eval token artifacts.")

    real_tokens = eval_payload["indices"].detach().cpu().long()
    real_decoder_space = eval_payload["data"].to(device)
    conditions = eval_payload["labels"].to(device)
    sampled_tokens = prior.sample(
        batch_size=args.n_sample,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditions=conditions,
    )
    quantized, decoded_decoder_space = decode_token_indices(
        tokenizer,
        sampled_tokens,
        conditions=conditions,
    )

    generated_prices = decoder_output_to_prices(decoded_decoder_space, data_output=data_output)
    real_prices = decoder_output_to_prices(real_decoder_space, data_output=data_output)
    jump_thresholds = fit_common_jump_thresholds(real_prices)
    generated_jumps = detect_jumps_with_threshold(generated_prices, thresholds=jump_thresholds)
    real_jumps = detect_jumps_with_threshold(real_prices, thresholds=jump_thresholds)
    generated_jump_sizes = detected_jump_sizes(generated_prices, generated_jumps)
    real_jump_sizes = detected_jump_sizes(real_prices, real_jumps)

    summary = build_summary(
        args=vars(args),
        raw_config=raw_config,
        data_output=data_output,
        prior_dir=prior_dir,
        tokenizer_dir=tokenizer_dir,
        sampled_tokens=sampled_tokens.detach().cpu(),
        real_tokens=real_tokens,
        decoded_decoder_space=decoded_decoder_space.detach().cpu(),
        real_decoder_space=real_decoder_space.detach().cpu(),
        generated_prices=generated_prices.detach().cpu(),
        real_prices=real_prices.detach().cpu(),
        generated_jumps=generated_jumps.detach().cpu(),
        real_jumps=real_jumps.detach().cpu(),
        generated_jump_sizes=generated_jump_sizes.detach().cpu(),
        real_jump_sizes=real_jump_sizes.detach().cpu(),
        prior_config=prior_config,
        tokenizer_config=tokenizer_config,
        temperature=args.temperature,
        top_k=args.top_k,
        jump_thresholds=jump_thresholds,
    )
    write_json(output_dir / "evaluation_summary.json", summary)
    write_markdown_summary(output_dir / "evaluation_summary.md", summary)
    torch.save(
        {
            "sampled_tokens": sampled_tokens.detach().cpu(),
            "real_tokens": real_tokens,
            "decoded_decoder_space": decoded_decoder_space.detach().cpu(),
            "real_decoder_space": real_decoder_space.detach().cpu(),
            "generated_prices": generated_prices.detach().cpu(),
            "real_prices": real_prices.detach().cpu(),
            "generated_jumps": generated_jumps.detach().cpu(),
            "real_jumps": real_jumps.detach().cpu(),
            "quantized": quantized.detach().cpu(),
            "conditions": conditions.detach().cpu(),
        },
        output_dir / "evaluation_batch.pt",
    )

    print("Hawkes-jump token-prior evaluation complete.")
    print(f"output_dir: {output_dir}")
    print(f"data_output: {data_output}")
    print(f"generated_prices_shape: {list(generated_prices.shape)}")
    print(f"sampled_token_active_codes: {summary['token_metrics']['sampled_active_code_count']}")
    print(f"mmd: {summary['smooth_metrics']['mmd']:.8f}")
    print(f"swd: {summary['smooth_metrics']['swd']:.8f}")
    print(
        "detected jump count W1: "
        f"{summary['jump_comparison']['detected_jump_count_wasserstein']:.8f}"
    )


def validate_output_dir(output_dir: str) -> Path:
    """Validate that evaluation outputs stay below local outputs/."""
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


def validate_dir(raw_path: str, *, description: str) -> Path:
    """Validate an existing directory."""
    path = Path(raw_path)
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"{description} directory does not exist: {path}")
    return path


def select_device(device_name: str | None) -> torch.device:
    """Select the requested device, or prefer CUDA when available."""
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def read_data_output(raw_config: Mapping[str, Any]) -> str:
    """Return decoder output convention from config metadata."""
    data = raw_config.get("data")
    if not isinstance(data, Mapping):
        return "price"
    return str(data.get("data_output", "price"))


def decoder_output_to_prices(decoded: Tensor, *, data_output: str) -> Tensor:
    """Convert decoder-space outputs to price paths for diagnostics."""
    if data_output == "log_return":
        return log_returns_to_normalized_prices(decoded)
    if data_output != "price":
        raise ValueError(f"Unsupported data_output: {data_output}")
    if decoded.ndim != 3 or decoded.shape[-1] != 1:
        raise ValueError(f"Expected price paths with shape [batch, time, 1]; got {decoded.shape}.")
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("Decoded price paths must be finite.")
    if not bool((decoded > 0.0).all()):
        raise ValueError("Decoded price paths must be positive.")
    return decoded.float()


def fit_common_jump_thresholds(
    real_prices: Tensor,
    *,
    threshold_multiplier: float = 4.0,
    min_abs_return: float = 0.0,
) -> dict[str, float]:
    """Fit a robust jump threshold on real Ogata evaluation paths."""
    returns = compute_log_returns(real_prices)
    median = returns.median()
    mad = (returns - median).abs().median()
    robust_scale = (1.4826 * mad).clamp_min(1e-8)
    threshold = max(float((threshold_multiplier * robust_scale).item()), float(min_abs_return))
    return {
        "median": float(median.item()),
        "mad": float(mad.item()),
        "robust_scale": float(robust_scale.item()),
        "threshold_multiplier": float(threshold_multiplier),
        "min_abs_return": float(min_abs_return),
        "absolute_threshold": threshold,
    }


def detect_jumps_with_threshold(paths: Tensor, *, thresholds: Mapping[str, float]) -> Tensor:
    """Detect jumps with a fixed threshold fitted on reference paths."""
    returns = compute_log_returns(paths)
    median = float(thresholds["median"])
    threshold = float(thresholds["absolute_threshold"])
    jump_returns = (returns - median).abs() >= threshold
    leading = torch.zeros(
        (jump_returns.shape[0], 1),
        dtype=torch.bool,
        device=jump_returns.device,
    )
    return torch.cat([leading, jump_returns], dim=1).unsqueeze(-1)


def build_summary(
    *,
    args: Mapping[str, Any],
    raw_config: Mapping[str, Any],
    data_output: str,
    prior_dir: Path,
    tokenizer_dir: Path,
    sampled_tokens: Tensor,
    real_tokens: Tensor,
    decoded_decoder_space: Tensor,
    real_decoder_space: Tensor,
    generated_prices: Tensor,
    real_prices: Tensor,
    generated_jumps: Tensor,
    real_jumps: Tensor,
    generated_jump_sizes: Tensor,
    real_jump_sizes: Tensor,
    prior_config: Any,
    tokenizer_config: Any,
    temperature: float,
    top_k: int | None,
    jump_thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Build a JSON-safe Hawkes prior evaluation summary."""
    sampled_code_counts = torch.bincount(
        sampled_tokens.reshape(-1),
        minlength=prior_config.codebook_size,
    )[: prior_config.codebook_size]
    token_metrics = {
        **rename_usage_keys(summarise_code_usage(sampled_code_counts), prefix="sampled"),
        **component_token_metrics(sampled_tokens, prior_config.codebook_size),
    }
    if real_tokens.ndim == 2 and sampled_tokens.ndim == 2:
        token_metrics.update(
            flatten_token_comparison_metrics(
                compare_token_sequences(
                    real_tokens=real_tokens,
                    sampled_tokens=sampled_tokens,
                    codebook_size=prior_config.codebook_size,
                )
            )
        )
    smooth_metrics = compute_path_distribution_metrics(
        generated=generated_prices,
        real=real_prices,
    )
    market_comparison = compare_market_summaries(
        real_paths=real_prices,
        generated_paths=generated_prices,
    )
    generated_jump_summary = jump_diagnostic_summary(
        generated_prices,
        jump_indicators=generated_jumps,
        jump_sizes=generated_jump_sizes,
        tail_reference_paths=real_prices,
    )
    real_jump_summary = jump_diagnostic_summary(
        real_prices,
        jump_indicators=real_jumps,
        jump_sizes=real_jump_sizes,
        tail_reference_paths=real_prices,
    )
    jump_comparison = {
        "detected_jump_count_wasserstein": jump_count_wasserstein(
            generated_jumps,
            real_jumps,
        ),
        "detected_jump_size_wasserstein": wasserstein_1d(
            generated_jump_sizes[generated_jump_sizes.abs() > 0.0],
            real_jump_sizes[real_jump_sizes.abs() > 0.0],
        ),
    }
    return {
        "manifest": {
            "config_path": str(args["config"]),
            "prior_dir": str(prior_dir),
            "tokenizer_dir": str(tokenizer_dir),
            "n_sample": int(args["n_sample"]),
            "seed": int(args["seed"]),
            "temperature": float(temperature),
            "top_k": top_k,
            "data_output": data_output,
            "condition_convention": "paired_eval_labels_from_token_artifacts",
            "config_data": dict(cast(Mapping[str, Any], raw_config.get("data", {}))),
        },
        "tensor_shapes": {
            "sampled_tokens": list(sampled_tokens.shape),
            "real_tokens": list(real_tokens.shape),
            "decoded_decoder_space": list(decoded_decoder_space.shape),
            "real_decoder_space": list(real_decoder_space.shape),
            "generated_prices": list(generated_prices.shape),
            "real_prices": list(real_prices.shape),
        },
        "token_prior_config": dict(prior_config.__dict__),
        "tokenizer_config": dict(tokenizer_config.__dict__),
        "log_return_to_price_conversion": data_output == "log_return",
        "jump_detection_thresholds_from_real": dict(jump_thresholds),
        "smooth_metrics": smooth_metrics,
        "market_comparison": market_comparison,
        "token_metrics": token_metrics,
        "jump_diagnostics": {
            "generated": generated_jump_summary,
            "real": real_jump_summary,
        },
        "jump_comparison": jump_comparison,
    }


def rename_usage_keys(metrics: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    """Rename core code-usage keys for sampled-token summaries."""
    return {
        f"{prefix}_active_code_count": metrics["active_code_count"],
        f"{prefix}_active_code_ratio": metrics["active_code_ratio"],
        f"{prefix}_codebook_perplexity": metrics["codebook_perplexity"],
        f"{prefix}_index_entropy": metrics["index_entropy"],
        f"{prefix}_code_usage_counts": metrics["code_usage_counts"],
        f"{prefix}_active_code_indices": metrics["active_code_indices"],
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a compact Markdown summary."""
    manifest = cast(Mapping[str, Any], summary["manifest"])
    smooth = cast(Mapping[str, Any], summary["smooth_metrics"])
    token = cast(Mapping[str, Any], summary["token_metrics"])
    jump = cast(Mapping[str, Any], summary["jump_comparison"])
    generated_jump = cast(Mapping[str, Any], summary["jump_diagnostics"])["generated"]
    generated_counts = cast(Mapping[str, Any], generated_jump)["jump_counts"]
    generated_count_dist = cast(Mapping[str, Any], generated_counts)["per_path"]
    lines = [
        "# Hawkes-Jump Token Prior Evaluation",
        "",
        "## Manifest",
        "",
        f"- Config: `{manifest['config_path']}`",
        f"- Prior dir: `{manifest['prior_dir']}`",
        f"- Tokenizer dir: `{manifest['tokenizer_dir']}`",
        f"- Samples: `{manifest['n_sample']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Data output: `{manifest['data_output']}`",
        f"- Log-return conversion: `{summary['log_return_to_price_conversion']}`",
        "",
        "## Smooth Metrics",
        "",
        f"- MMD: `{float(smooth['mmd']):.8f}`",
        f"- SWD: `{float(smooth['swd']):.8f}`",
        f"- Terminal W1: `{float(smooth['terminal_return_wasserstein']):.8f}`",
        f"- Volatility W1: `{float(smooth['volatility_wasserstein']):.8f}`",
        "",
        "## Token Metrics",
        "",
        f"- Sampled active codes: `{token['sampled_active_code_count']}`",
        f"- Sampled perplexity: `{float(token['sampled_codebook_perplexity']):.8f}`",
        f"- Marginal code L1: `{float(token.get('marginal_code_l1', 0.0)):.8f}`",
        f"- Transition matrix L1: `{float(token.get('transition_matrix_l1', 0.0)):.8f}`",
        f"- Run-length distance: `{float(token.get('run_length_distance', 0.0)):.8f}`",
        "",
        "## Jump Metrics",
        "",
        f"- Detected jump count W1: `{float(jump['detected_jump_count_wasserstein']):.8f}`",
        f"- Detected jump-size W1: `{float(jump['detected_jump_size_wasserstein']):.8f}`",
        f"- Generated mean jumps per path: `{float(generated_count_dist['mean']):.8f}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    """Convert tensors and paths into JSON-safe values."""
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
