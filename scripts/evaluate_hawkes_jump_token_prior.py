"""Hawkes/SVMHJD token-prior evaluation with jump and tail diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from time_causal_vae.cli.evaluate_token_prior import (
    freeze_tokenizer,
    load_conditional_eval_payload,
    load_token_prior_yaml,
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
    """Build the script argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Hawkes/SVMHJD causal token prior.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-sample", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=parse_optional_top_k, default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def parse_optional_top_k(raw_value: str) -> int | None:
    """Parse an optional top-k sampling cutoff."""
    normalised = raw_value.strip().lower()
    if normalised in {"none", "null", "unrestricted"}:
        return None
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("--top-k must be positive or 'none'.")
    return value


def main() -> None:
    """Run Hawkes/SVMHJD token-prior evaluation."""
    parser = build_parser()
    args = parser.parse_args()
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive.")

    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_perf = time.perf_counter()
    device = torch.device(args.device)
    set_seed(args.seed)

    raw_config = load_token_prior_yaml(args.config)
    data_config = cast(Mapping[str, Any], raw_config["data"])
    data_output = str(data_config.get("data_output", "price"))
    prior, prior_config, _prior_checkpoint = load_trained_token_prior(
        args.prior_dir,
        device=device,
    )
    tokenizer, tokenizer_config, _tokenizer_checkpoint = load_trained_tokenizer(
        args.tokenizer_dir,
        device=device,
    )
    freeze_tokenizer(tokenizer)
    payload = load_conditional_eval_payload(
        raw_config,
        n_sample=args.n_sample,
        prior_config=prior_config,
    )
    if payload is None:
        raise SystemExit("Hawkes/SVMHJD evaluation requires paired conditional token data.")

    conditions = payload["labels"].to(device)
    real_decoder_space = payload["data"].to(device)
    real_tokens = payload["indices"].detach().cpu().long()
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
    generated_paths, real_paths, converted_log_returns = paths_for_hawkes_diagnostics(
        decoded_decoder_space.detach().cpu(),
        real_decoder_space.detach().cpu(),
        data_output=data_output,
    )
    real_jumps, generated_jumps, threshold_summary = detect_jumps_with_real_threshold(
        real_paths,
        generated_paths,
    )
    real_jump_sizes = detected_jump_sizes(real_paths, real_jumps)
    generated_jump_sizes = detected_jump_sizes(generated_paths, generated_jumps)
    smooth_metrics = compute_path_distribution_metrics(
        generated=generated_paths,
        real=real_paths,
    )
    token_metrics = build_token_metrics(
        real_tokens=real_tokens,
        sampled_tokens=sampled_tokens.detach().cpu(),
        codebook_size=prior_config.codebook_size,
    )
    summary = {
        "manifest": {
            "config_path": args.config,
            "prior_dir": args.prior_dir,
            "tokenizer_dir": args.tokenizer_dir,
            "n_sample": args.n_sample,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "data_output": data_output,
            "condition_convention": "paired_eval_labels_from_token_artifacts",
            "config_data": dict(data_config),
        },
        "log_return_to_price_conversion": converted_log_returns,
        "tensor_shapes": {
            "sampled_tokens": list(sampled_tokens.shape),
            "real_tokens": list(real_tokens.shape),
            "decoded_decoder_space": list(decoded_decoder_space.shape),
            "real_decoder_space": list(real_decoder_space.shape),
            "generated_prices": list(generated_paths.shape),
            "real_prices": list(real_paths.shape),
            "generated_jumps": list(generated_jumps.shape),
            "real_jumps": list(real_jumps.shape),
            "quantized": list(quantized.shape),
            "conditions": list(conditions.shape),
        },
        "smooth_metrics": smooth_metrics,
        "market_comparison": compare_market_summaries(
            real_paths=real_paths,
            generated_paths=generated_paths,
        ),
        "jump_detection_thresholds_from_real": threshold_summary,
        "jump_diagnostics": {
            "real": jump_diagnostic_summary(
                real_paths,
                jump_indicators=real_jumps,
                jump_sizes=real_jump_sizes,
                tail_reference_paths=real_paths,
            ),
            "generated": jump_diagnostic_summary(
                generated_paths,
                jump_indicators=generated_jumps,
                jump_sizes=generated_jump_sizes,
                tail_reference_paths=real_paths,
            ),
        },
        "jump_comparison": {
            "detected_jump_count_wasserstein": jump_count_wasserstein(
                real_jumps,
                generated_jumps,
            ),
            "detected_inter_arrival_wasserstein": inter_arrival_wasserstein(
                real_jumps,
                generated_jumps,
            ),
            "detected_jump_size_wasserstein": detected_jump_size_wasserstein(
                real_jump_sizes,
                generated_jump_sizes,
            ),
        },
        "token_metrics": token_metrics,
        "token_prior_config": asdict(prior_config),
        "tokenizer_config": asdict(tokenizer_config),
        "runtime_summary": {
            "elapsed_seconds": time.perf_counter() - start_perf,
            "device": str(device),
            "output_dir": str(output_dir),
        },
    }
    write_json(output_dir / "evaluation_summary.json", summary)
    write_markdown_summary(output_dir / "evaluation_summary.md", summary)
    torch.save(
        {
            "sampled_tokens": sampled_tokens.detach().cpu(),
            "real_tokens": real_tokens.detach().cpu(),
            "decoded_decoder_space": decoded_decoder_space.detach().cpu(),
            "real_decoder_space": real_decoder_space.detach().cpu(),
            "generated_prices": generated_paths.detach().cpu(),
            "real_prices": real_paths.detach().cpu(),
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
    print(f"generated_prices_shape: {list(generated_paths.shape)}")
    print(f"sampled_token_active_codes: {token_metrics['sampled_active_code_count']}")
    print(f"mmd: {smooth_metrics['mmd']:.8f}")
    print(f"swd: {smooth_metrics['swd']:.8f}")
    print(
        "detected jump count W1: "
        f"{summary['jump_comparison']['detected_jump_count_wasserstein']:.8f}"
    )
    print(f"runtime_seconds: {summary['runtime_summary']['elapsed_seconds']:.3f}")


def validate_output_dir(output_dir: str) -> Path:
    """Validate that generated evaluation artefacts stay below local outputs/."""
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


def paths_for_hawkes_diagnostics(
    decoded_paths: Tensor,
    real_paths: Tensor,
    *,
    data_output: str,
) -> tuple[Tensor, Tensor, bool]:
    """Return positive price paths for market and jump diagnostics."""
    if data_output == "log_return":
        return (
            log_returns_to_normalized_prices(decoded_paths),
            log_returns_to_normalized_prices(real_paths),
            True,
        )
    if data_output == "price":
        return decoded_paths, real_paths, False
    raise SystemExit(f"Unsupported Hawkes data_output={data_output!r}.")


def detect_jumps_with_real_threshold(
    real_paths: Tensor,
    generated_paths: Tensor,
    *,
    threshold_multiplier: float = 4.0,
    min_abs_return: float = 0.0,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Detect real and generated jumps with a threshold fitted on real returns."""
    real_returns = compute_log_returns(real_paths)
    median = real_returns.median()
    mad = (real_returns - median).abs().median()
    robust_scale = (1.4826 * mad).clamp_min(1e-8)
    absolute_threshold = max(
        float(threshold_multiplier * robust_scale.item()),
        float(min_abs_return),
    )
    summary = {
        "median": float(median.item()),
        "mad": float(mad.item()),
        "robust_scale": float(robust_scale.item()),
        "threshold_multiplier": float(threshold_multiplier),
        "min_abs_return": float(min_abs_return),
        "absolute_threshold": absolute_threshold,
    }
    return (
        detect_jumps_from_reference_threshold(real_paths, median, absolute_threshold),
        detect_jumps_from_reference_threshold(generated_paths, median, absolute_threshold),
        summary,
    )


def detect_jumps_from_reference_threshold(
    paths: Tensor,
    median: Tensor,
    absolute_threshold: float,
) -> Tensor:
    """Detect jumps using a fixed real-data return median and threshold."""
    returns = compute_log_returns(paths)
    jump_returns = (returns - median).abs() >= float(absolute_threshold)
    leading = torch.zeros(
        (jump_returns.shape[0], 1),
        dtype=torch.bool,
        device=jump_returns.device,
    )
    return torch.cat([leading, jump_returns], dim=1).unsqueeze(-1).cpu()


def detected_jump_size_wasserstein(real_jump_sizes: Tensor, generated_jump_sizes: Tensor) -> float:
    """Return W1 distance between signed non-zero detected jump sizes."""
    real_nonzero = real_jump_sizes[real_jump_sizes != 0.0]
    generated_nonzero = generated_jump_sizes[generated_jump_sizes != 0.0]
    if real_nonzero.numel() == 0 or generated_nonzero.numel() == 0:
        return 0.0
    return wasserstein_1d(real_nonzero, generated_nonzero)


def inter_arrival_wasserstein(real_jumps: Tensor, generated_jumps: Tensor) -> float:
    """Return W1 distance between within-path inter-arrival gap samples."""
    real_gaps = inter_arrival_gaps(real_jumps)
    generated_gaps = inter_arrival_gaps(generated_jumps)
    if real_gaps.numel() == 0 or generated_gaps.numel() == 0:
        return 0.0
    return wasserstein_1d(real_gaps, generated_gaps)


def inter_arrival_gaps(jump_indicators: Tensor) -> Tensor:
    """Return concatenated within-path gaps between detected jump steps."""
    squeezed = jump_indicators.detach().cpu().bool()
    if squeezed.ndim == 3 and squeezed.shape[-1] == 1:
        squeezed = squeezed[..., 0]
    if squeezed.ndim != 2:
        raise ValueError(f"Expected jump indicators [batch, time, 1]; got {jump_indicators.shape}.")
    gaps: list[float] = []
    for path_indicators in squeezed:
        positions = torch.nonzero(path_indicators, as_tuple=False).flatten().float()
        if positions.numel() > 1:
            gaps.extend(float(value) for value in positions.diff().tolist())
    return torch.tensor(gaps, dtype=torch.float32)


def build_token_metrics(
    *,
    real_tokens: Tensor,
    sampled_tokens: Tensor,
    codebook_size: int,
) -> dict[str, Any]:
    """Return token usage, transition, and run-length diagnostics."""
    code_counts = torch.bincount(sampled_tokens.reshape(-1), minlength=codebook_size)[
        :codebook_size
    ]
    return {
        **prefix_keys("sampled", summarise_code_usage(code_counts)),
        **component_token_metrics(sampled_tokens, codebook_size),
        **flatten_token_comparison_metrics(
            compare_token_sequences(
                real_tokens=real_tokens,
                sampled_tokens=sampled_tokens,
                codebook_size=codebook_size,
            )
        ),
    }


def prefix_keys(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Prefix metric keys for JSON summaries."""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a compact Markdown evaluation summary."""
    manifest = cast(Mapping[str, Any], summary["manifest"])
    smooth = cast(Mapping[str, Any], summary["smooth_metrics"])
    token = cast(Mapping[str, Any], summary["token_metrics"])
    jump_comparison = cast(Mapping[str, Any], summary["jump_comparison"])
    generated_jump = cast(Mapping[str, Any], summary["jump_diagnostics"])["generated"]
    generated_counts = cast(Mapping[str, Any], generated_jump)["jump_counts"]
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
        f"- MMD: `{smooth['mmd']:.8f}`",
        f"- SWD: `{smooth['swd']:.8f}`",
        f"- Terminal W1: `{smooth['terminal_return_wasserstein']:.8f}`",
        f"- Volatility W1: `{smooth['volatility_wasserstein']:.8f}`",
        "",
        "## Token Metrics",
        "",
        f"- Sampled active codes: `{token['sampled_active_code_count']}`",
        f"- Sampled perplexity: `{token['sampled_token_perplexity']:.8f}`",
        f"- Marginal code L1: `{token['marginal_code_l1']:.8f}`",
        f"- Transition matrix L1: `{token['transition_matrix_l1']:.8f}`",
        f"- Run-length distance: `{token['run_length_distance']:.8f}`",
        "",
        "## Jump Metrics",
        "",
        (f"- Detected jump count W1: `{jump_comparison['detected_jump_count_wasserstein']:.8f}`"),
        (f"- Detected jump-size W1: `{jump_comparison['detected_jump_size_wasserstein']:.8f}`"),
        (
            "- Detected inter-arrival W1: "
            f"`{jump_comparison['detected_inter_arrival_wasserstein']:.8f}`"
        ),
        (
            "- Generated mean jumps per path: "
            f"`{cast(Mapping[str, Any], generated_counts)['per_path']['mean']:.8f}`"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
