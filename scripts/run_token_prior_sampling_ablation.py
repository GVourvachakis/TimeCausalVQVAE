"""Run sampling-temperature and top-k ablations for a trained token prior."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from time_causal_vae.cli.evaluate_token_prior import (
    build_comparison_dataset,
    freeze_tokenizer,
    load_conditional_eval_payload,
    load_reference_tokens,
    load_token_prior_yaml,
    select_device,
)
from time_causal_vae.evaluation.token_diagnostics import (
    compare_token_sequences,
    flatten_token_comparison_metrics,
)
from time_causal_vae.evaluation.token_prior import (
    compute_condition_bucket_sample_metrics,
    compute_token_prior_sample_metrics,
    decode_token_indices,
    load_trained_token_prior,
)
from time_causal_vae.evaluation.tokenizer import load_trained_tokenizer
from time_causal_vae.utils.random import set_seed

CSV_FIELDS = [
    "temperature",
    "top_k",
    "mmd",
    "swd",
    "volatility_wasserstein",
    "terminal_return_wasserstein",
    "sampled_code_active_count",
    "sampled_code_active_ratio",
    "sampled_code_perplexity",
    "sampled_code_entropy",
    "marginal_code_l1",
    "transition_matrix_l1",
    "run_length_distance",
    "very_low_mmd",
    "very_low_swd",
    "very_low_volatility_wasserstein",
    "very_low_terminal_return_wasserstein",
    "low_mmd",
    "low_swd",
    "low_volatility_wasserstein",
    "low_terminal_return_wasserstein",
    "score",
    "runtime_seconds",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the sampling-ablation parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate sampling temperature and top-k settings for a trained token prior.",
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
    parser.add_argument("--n-sample", type=int, default=1000, help="Number of paths per run.")
    parser.add_argument("--seed", type=int, default=99, help="Evaluation and sampling seed.")
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.6, 0.8, 1.0, 1.2],
        help="Sampling temperatures to evaluate.",
    )
    parser.add_argument(
        "--top-k-values",
        nargs="+",
        default=["none", "5", "10", "20", "40"],
        help="Top-k values to evaluate. Use 'none' for unrestricted sampling.",
    )
    parser.add_argument("--device", help="Device override, for example cpu or cuda.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory for datasets that require local files.",
    )
    return parser


def main() -> None:
    """Run a sampling ablation grid and write JSON/CSV summaries."""
    args = build_parser().parse_args()
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")
    for temperature in cast(Sequence[float], args.temperatures):
        if temperature <= 0.0:
            raise SystemExit("--temperatures values must be positive.")

    output_dir = validate_output_dir(args.output_dir)
    top_k_values = parse_top_k_values(args.top_k_values)
    device = select_device(args.device)
    raw_config = load_token_prior_yaml(args.config)
    prior, prior_config, _prior_checkpoint = load_trained_token_prior(args.prior_dir, device=device)
    tokenizer, tokenizer_config, _tokenizer_checkpoint = load_trained_tokenizer(
        args.tokenizer_dir,
        device=device,
    )
    freeze_tokenizer(tokenizer)
    set_seed(args.seed)
    conditional_payload = load_conditional_eval_payload(
        raw_config,
        n_sample=args.n_sample,
        prior_config=prior_config,
    )
    sample_conditions = None
    condition_sampling_convention = "unconditional_no_conditions"
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
        real_tokens = conditional_payload["indices"].detach().cpu().long()
        condition_sampling_convention = "paired_eval_labels_from_token_artifacts"

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for temperature in cast(Sequence[float], args.temperatures):
        for top_k in top_k_values:
            result = evaluate_sampling_setting(
                prior=prior,
                tokenizer=tokenizer,
                real_paths=real_paths,
                real_tokens=real_tokens,
                sample_conditions=sample_conditions,
                codebook_size=prior_config.codebook_size,
                n_sample=args.n_sample,
                seed=args.seed,
                device=device,
                temperature=float(temperature),
                top_k=top_k,
            )
            results.append(result)
            print(
                "sampling_ablation "
                f"temperature={result['temperature']} "
                f"top_k={top_k_label(top_k)} "
                f"score={result['score']:.8f} "
                f"mmd={result['mmd']:.8f} "
                f"swd={result['swd']:.8f} "
                f"active={result['sampled_code_active_count']}",
                flush=True,
            )
            write_csv(output_dir / "sampling_ablation_summary.csv", results)

    best_result = min(results, key=lambda row: float(row["score"]))
    default_result = find_default_result(results)
    summary = {
        "config": str(Path(args.config)),
        "prior_dir": args.prior_dir,
        "tokenizer_dir": args.tokenizer_dir,
        "output_dir": str(output_dir),
        "n_sample": args.n_sample,
        "seed": args.seed,
        "device": str(device),
        "temperatures": [float(value) for value in args.temperatures],
        "top_k_values": [top_k_label(value) for value in top_k_values],
        "token_prior_config": prior_config.__dict__,
        "tokenizer_config": tokenizer_config.__dict__,
        "condition_sampling_convention": condition_sampling_convention,
        "score_formula": "MMD + SWD + volatility_wasserstein + terminal_return_wasserstein",
        "best_result": best_result,
        "default_temperature_1_top_k_none": default_result,
        "results": results,
    }
    write_json(output_dir / "sampling_ablation_summary.json", summary)
    write_csv(output_dir / "sampling_ablation_summary.csv", results)
    print(f"best_temperature: {best_result['temperature']}")
    print(f"best_top_k: {best_result['top_k']}")
    print(f"best_score: {best_result['score']:.8f}")
    print(f"wrote: {output_dir / 'sampling_ablation_summary.json'}")
    print(f"wrote: {output_dir / 'sampling_ablation_summary.csv'}", flush=True)


def validate_output_dir(output_dir: str) -> Path:
    """Validate that ablation artifacts stay below ignored outputs/."""
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


def parse_top_k_values(raw_values: Sequence[str]) -> list[int | None]:
    """Parse top-k CLI values, accepting 'none' for unrestricted sampling."""
    parsed: list[int | None] = []
    for raw_value in raw_values:
        normalised = raw_value.strip().lower()
        if normalised in {"none", "null", "unrestricted"}:
            parsed.append(None)
            continue
        value = int(raw_value)
        if value <= 0:
            raise SystemExit("--top-k-values must be positive integers or 'none'.")
        parsed.append(value)
    return parsed


def evaluate_sampling_setting(
    *,
    prior: Any,
    tokenizer: Any,
    real_paths: torch.Tensor,
    real_tokens: torch.Tensor | None,
    sample_conditions: torch.Tensor | None,
    codebook_size: int,
    n_sample: int,
    seed: int,
    device: torch.device,
    temperature: float,
    top_k: int | None,
) -> dict[str, Any]:
    """Evaluate one sampling configuration."""
    set_seed(seed)
    start_time = time.perf_counter()
    sampled_tokens = prior.sample(
        batch_size=n_sample,
        device=device,
        temperature=temperature,
        top_k=top_k,
        conditions=sample_conditions,
    )
    _quantized, decoded_paths = decode_token_indices(
        tokenizer,
        sampled_tokens,
        conditions=sample_conditions,
    )
    metrics = compute_token_prior_sample_metrics(
        sampled_tokens=sampled_tokens,
        decoded_paths=decoded_paths,
        real_paths=real_paths,
        codebook_size=codebook_size,
    )
    if real_tokens is not None:
        token_comparison = compare_token_sequences(
            real_tokens=real_tokens,
            sampled_tokens=sampled_tokens.detach().cpu(),
            codebook_size=codebook_size,
        )
        metrics.update(flatten_token_comparison_metrics(token_comparison))
    if sample_conditions is not None:
        metrics["condition_buckets"] = compute_condition_bucket_sample_metrics(
            sampled_tokens=sampled_tokens,
            decoded_paths=decoded_paths,
            real_paths=real_paths,
            conditions=sample_conditions,
            codebook_size=codebook_size,
        )
    runtime_seconds = time.perf_counter() - start_time
    score = (
        float(metrics["mmd"])
        + float(metrics["swd"])
        + float(metrics["volatility_wasserstein"])
        + float(metrics["terminal_return_wasserstein"])
    )
    result: dict[str, Any] = {
        "temperature": temperature,
        "top_k": top_k_label(top_k),
        "mmd": float(metrics["mmd"]),
        "swd": float(metrics["swd"]),
        "volatility_wasserstein": float(metrics["volatility_wasserstein"]),
        "terminal_return_wasserstein": float(metrics["terminal_return_wasserstein"]),
        "sampled_code_active_count": int(metrics["sampled_token_active_code_count"]),
        "sampled_code_active_ratio": float(metrics["sampled_token_active_code_ratio"]),
        "sampled_code_perplexity": float(metrics["sampled_token_codebook_perplexity"]),
        "sampled_code_entropy": float(metrics["sampled_token_index_entropy"]),
        "marginal_code_l1": optional_float(metrics.get("marginal_code_l1")),
        "transition_matrix_l1": optional_float(metrics.get("transition_matrix_l1")),
        "run_length_distance": optional_float(metrics.get("run_length_distance")),
        "score": score,
        "runtime_seconds": runtime_seconds,
    }
    if "condition_buckets" in metrics:
        result["condition_buckets"] = metrics["condition_buckets"]
    result.update(flatten_selected_condition_buckets(metrics))
    return result


def optional_float(value: object) -> float | None:
    """Convert an optional numeric metric to ``float``."""
    if value is None:
        return None
    return float(value)


def flatten_selected_condition_buckets(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    """Flatten low-condition bucket metrics into stable CSV fields."""
    flattened: dict[str, float | None] = {
        "very_low_mmd": None,
        "very_low_swd": None,
        "very_low_volatility_wasserstein": None,
        "very_low_terminal_return_wasserstein": None,
        "low_mmd": None,
        "low_swd": None,
        "low_volatility_wasserstein": None,
        "low_terminal_return_wasserstein": None,
    }
    buckets = metrics.get("condition_buckets")
    if not isinstance(buckets, list):
        return flattened
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        label = str(bucket.get("bucket_label"))
        if label not in {"very_low", "low"}:
            continue
        flattened[f"{label}_mmd"] = float(bucket["mmd"])
        flattened[f"{label}_swd"] = float(bucket["swd"])
        flattened[f"{label}_volatility_wasserstein"] = float(bucket["volatility_wasserstein"])
        flattened[f"{label}_terminal_return_wasserstein"] = float(
            bucket["terminal_return_wasserstein"]
        )
    return flattened


def top_k_label(top_k: int | None) -> str:
    """Return a stable JSON/CSV top-k label."""
    if top_k is None:
        return "none"
    return str(top_k)


def find_default_result(results: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return the temperature 1.0 / no top-k result when present."""
    for result in results:
        if float(result["temperature"]) == 1.0 and str(result["top_k"]) == "none":
            return dict(result)
    return None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write an indented JSON file."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write ablation rows as CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


if __name__ == "__main__":
    main()
