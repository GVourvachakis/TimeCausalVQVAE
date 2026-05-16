"""Paper-style S&P500/VIX diagnostics for separate-frequency token priors."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import Tensor

from time_causal_vae.cli.evaluate_token_prior import freeze_tokenizer
from time_causal_vae.evaluation.checkpoints import TargetModelEvaluator
from time_causal_vae.evaluation.market_diagnostics import (
    compare_market_summaries,
    market_style_summary,
)
from time_causal_vae.evaluation.token_diagnostics import (
    compare_token_sequences,
    flatten_token_comparison_metrics,
)
from time_causal_vae.evaluation.token_prior import (
    compute_path_distribution_metrics,
    decode_token_indices,
    load_trained_token_prior,
)
from time_causal_vae.evaluation.tokenizer import (
    condition_bucket_label,
    load_trained_tokenizer,
    summarise_code_usage,
)
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the separate-frequency paper-style evaluation parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample a separate-frequency hierarchical prior, decode low/high paths, "
            "and compute S&P500/VIX paper-style diagnostics."
        ),
    )
    parser.add_argument("--config", required=True, help="Separate-frequency prior YAML.")
    parser.add_argument("--prior-dir", required=True, help="Directory containing token_prior.pt.")
    parser.add_argument("--low-tokenizer-dir", help="Low tokenizer directory.")
    parser.add_argument("--high-tokenizer-dir", help="High tokenizer directory.")
    parser.add_argument("--token-data-dir", help="Paired low/high token dataset directory.")
    parser.add_argument("--continuous-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-data-dir", default="data/processed")
    parser.add_argument("--n-sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser


def main() -> None:
    """Run separate-frequency prior evaluation."""
    args = build_parser().parse_args()
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive.")
    if args.top_k is not None and args.top_k <= 0:
        raise SystemExit("--top-k must be positive.")

    raw_config = load_yaml(args.config)
    data_config = require_mapping(raw_config, "data")
    token_data_dir = Path(
        args.token_data_dir or cast(str, data_config["token_data_dir"]),
    )
    low_tokenizer_dir = Path(
        args.low_tokenizer_dir or cast(str, data_config["low_tokenizer_dir"]),
    )
    high_tokenizer_dir = Path(
        args.high_tokenizer_dir or cast(str, data_config["high_tokenizer_dir"]),
    )
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cpu")
    prior, prior_config, _checkpoint = load_trained_token_prior(args.prior_dir, device=device)
    low_tokenizer, _low_config, _low_checkpoint = load_trained_tokenizer(
        low_tokenizer_dir,
        device=device,
    )
    high_tokenizer, _high_config, _high_checkpoint = load_trained_tokenizer(
        high_tokenizer_dir,
        device=device,
    )
    freeze_tokenizer(low_tokenizer)
    freeze_tokenizer(high_tokenizer)

    eval_tensors = load_eval_tensors(token_data_dir, n_sample=args.n_sample)
    conditions = eval_tensors["labels"].to(device)
    real_low_tokens = eval_tensors["low_tokens"]
    real_high_tokens = eval_tensors["high_tokens"]
    real_paths = eval_tensors["data"].to(device)
    real_tokens = torch.stack([real_low_tokens, real_high_tokens], dim=-1).to(device)

    teacher_forced = compute_teacher_forced_metrics(
        prior=prior,
        tokens=real_tokens,
        conditions=conditions,
    )
    set_seed(args.seed)
    sample_output = prior.sample_streams(
        batch_size=args.n_sample,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditions=conditions,
    )
    sampled_low_tokens = cast(Tensor, sample_output.sampled_low_tokens).detach().cpu()
    sampled_high_tokens = cast(Tensor, sample_output.sampled_high_tokens).detach().cpu()
    sampled_tokens = cast(Tensor, sample_output.sampled_tokens).detach().cpu()
    low_quantized, decoded_low = decode_token_indices(
        low_tokenizer,
        sampled_low_tokens.to(device),
        conditions=conditions,
    )
    high_quantized, decoded_high = decode_token_indices(
        high_tokenizer,
        sampled_high_tokens.to(device),
        conditions=conditions,
    )
    decoded_paths = decoded_low + decoded_high

    low_comparison = flatten_token_comparison_metrics(
        compare_token_sequences(
            real_tokens=real_low_tokens,
            sampled_tokens=sampled_low_tokens,
            codebook_size=prior_config.low_codebook_size or prior_config.codebook_size,
        )
    )
    high_comparison = flatten_token_comparison_metrics(
        compare_token_sequences(
            real_tokens=real_high_tokens,
            sampled_tokens=sampled_high_tokens,
            codebook_size=prior_config.high_codebook_size or prior_config.codebook_size,
        )
    )
    pair_comparison = paired_token_metrics(
        real_low=real_low_tokens,
        real_high=real_high_tokens,
        sampled_low=sampled_low_tokens,
        sampled_high=sampled_high_tokens,
        high_codebook_size=prior_config.high_codebook_size or prior_config.codebook_size,
    )
    path_metrics = compute_path_distribution_metrics(
        generated=decoded_paths.detach().cpu(),
        real=real_paths.detach().cpu(),
    )
    market_metrics = compare_market_summaries(
        real_paths=real_paths.detach().cpu(),
        generated_paths=decoded_paths.detach().cpu(),
    )
    continuous = load_continuous_baseline(
        model_dir=args.continuous_model_dir,
        base_data_dir=args.base_data_dir,
        n_sample=args.n_sample,
        seed=args.seed,
    )
    continuous_metrics = (
        {
            **compute_path_distribution_metrics(
                generated=continuous["fake_paths"],
                real=real_paths.detach().cpu(),
            ),
            **compare_market_summaries(
                real_paths=real_paths.detach().cpu(),
                generated_paths=continuous["fake_paths"],
            ),
        }
        if continuous is not None
        else None
    )
    condition_buckets = compute_condition_buckets(
        conditions=conditions.detach().cpu(),
        real_paths=real_paths.detach().cpu(),
        decoded_paths=decoded_paths.detach().cpu(),
        sampled_low=sampled_low_tokens,
        sampled_high=sampled_high_tokens,
        high_codebook_size=prior_config.high_codebook_size or prior_config.codebook_size,
    )
    summary = {
        "manifest": {
            "script": "scripts/evaluate_sp500_vix_separate_frequency_paper_style.py",
            "config": args.config,
            "prior_dir": args.prior_dir,
            "low_tokenizer_dir": str(low_tokenizer_dir),
            "high_tokenizer_dir": str(high_tokenizer_dir),
            "token_data_dir": str(token_data_dir),
            "continuous_model_dir": args.continuous_model_dir,
            "output_dir": str(output_dir),
            "n_sample": args.n_sample,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "condition_convention": "paired_eval_labels_from_token_artifacts",
        },
        "tensor_shapes": {
            "sampled_low_tokens": list(sampled_low_tokens.shape),
            "sampled_high_tokens": list(sampled_high_tokens.shape),
            "sampled_tokens": list(sampled_tokens.shape),
            "decoded_low": list(decoded_low.shape),
            "decoded_high": list(decoded_high.shape),
            "decoded_paths": list(decoded_paths.shape),
            "real_paths": list(real_paths.shape),
            "conditions": list(conditions.shape),
        },
        "teacher_forced_eval": teacher_forced,
        "token_diagnostics": {
            "low": low_comparison,
            "high": high_comparison,
            "same_time_pair": pair_comparison,
        },
        "decoded_composed_path_metrics": {**path_metrics, **market_metrics},
        "source_summaries": {
            "real": market_style_summary(real_paths.detach().cpu()),
            "separate_frequency": market_style_summary(decoded_paths.detach().cpu()),
            **(
                {"continuous_beta_cvae": market_style_summary(continuous["fake_paths"])}
                if continuous is not None
                else {}
            ),
        },
        "continuous_beta_cvae_metrics": continuous_metrics,
        "vix_buckets": condition_buckets,
    }
    write_json(output_dir / "separate_frequency_prior_summary.json", summary)
    write_markdown(output_dir / "separate_frequency_prior_summary.md", summary)
    torch.save(
        {
            "sampled_low_tokens": sampled_low_tokens,
            "sampled_high_tokens": sampled_high_tokens,
            "sampled_tokens": sampled_tokens,
            "decoded_low": decoded_low.detach().cpu(),
            "decoded_high": decoded_high.detach().cpu(),
            "decoded_paths": decoded_paths.detach().cpu(),
            "real_paths": real_paths.detach().cpu(),
            "conditions": conditions.detach().cpu(),
            "low_quantized": low_quantized.detach().cpu(),
            "high_quantized": high_quantized.detach().cpu(),
        },
        output_dir / "separate_frequency_prior_samples.pt",
    )

    print("Separate-frequency paper-style evaluation complete.")
    print(f"output_dir: {output_dir}")
    print(f"sampled_low_tokens_shape: {list(sampled_low_tokens.shape)}")
    print(f"sampled_high_tokens_shape: {list(sampled_high_tokens.shape)}")
    print(f"decoded_paths_shape: {list(decoded_paths.shape)}")
    print(f"eval_low_ce: {teacher_forced['low_cross_entropy']:.8f}")
    print(f"eval_high_ce: {teacher_forced['high_cross_entropy']:.8f}")
    print(f"mmd: {path_metrics['mmd']:.8f}")
    print(f"swd: {path_metrics['swd']:.8f}")
    print(f"volatility_wasserstein: {market_metrics['volatility_wasserstein']:.8f}")
    print(f"terminal_return_wasserstein: {path_metrics['terminal_return_wasserstein']:.8f}")
    print(f"pair_sampled_active_count: {pair_comparison['sampled_active_pair_count']}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Expected YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required mapping from an experiment config."""
    value = raw_config.get(key)
    if not isinstance(value, Mapping):
        raise SystemExit(f"Config requires mapping section {key!r}.")
    return value


def validate_output_dir(output_dir: str) -> Path:
    """Validate that evaluation artifacts stay under ignored outputs/."""
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


def load_eval_tensors(token_data_dir: Path, *, n_sample: int) -> dict[str, Tensor]:
    """Load paired eval tokens, labels, and composed scalar data."""
    required_paths = {
        "low_tokens": token_data_dir / "eval_low_tokens.pt",
        "high_tokens": token_data_dir / "eval_high_tokens.pt",
        "labels": token_data_dir / "eval_labels.pt",
        "data": token_data_dir / "eval_data.pt",
    }
    loaded: dict[str, Tensor] = {}
    for name, path in required_paths.items():
        if not path.exists():
            raise SystemExit(f"Missing paired-token artifact: {path}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, Tensor):
            raise SystemExit(f"Expected tensor artifact at {path}.")
        if tensor.shape[0] < n_sample:
            raise SystemExit(f"{path} has fewer than {n_sample} samples.")
        loaded[name] = tensor[:n_sample].detach().cpu()
    labels = loaded["labels"].float()
    if labels.ndim == 1:
        labels = labels[:, None]
    loaded["labels"] = labels
    loaded["low_tokens"] = loaded["low_tokens"].long()
    loaded["high_tokens"] = loaded["high_tokens"].long()
    loaded["data"] = loaded["data"].float()
    return loaded


def compute_teacher_forced_metrics(
    *,
    prior: Any,
    tokens: Tensor,
    conditions: Tensor,
) -> dict[str, float]:
    """Run teacher-forced eval likelihoods for the paired token streams."""
    with torch.no_grad():
        output = prior(tokens, conditions=conditions)
    return {
        "cross_entropy": float(cast(Tensor, output.cross_entropy).detach().cpu()),
        "accuracy": float(cast(Tensor, output.accuracy).detach().cpu()),
        "perplexity": float(cast(Tensor, output.perplexity).detach().cpu()),
        "low_cross_entropy": float(cast(Tensor, output.component_cross_entropy_low).detach().cpu()),
        "high_cross_entropy": float(
            cast(Tensor, output.component_cross_entropy_high).detach().cpu()
        ),
        "low_accuracy": float(cast(Tensor, output.component_accuracy_low).detach().cpu()),
        "high_accuracy": float(cast(Tensor, output.component_accuracy_high).detach().cpu()),
        "low_perplexity": float(cast(Tensor, output.component_perplexity_low).detach().cpu()),
        "high_perplexity": float(cast(Tensor, output.component_perplexity_high).detach().cpu()),
        "same_time_pair_perplexity": float(
            cast(Tensor, output.same_time_pair_perplexity).detach().cpu()
        ),
    }


def paired_token_metrics(
    *,
    real_low: Tensor,
    real_high: Tensor,
    sampled_low: Tensor,
    sampled_high: Tensor,
    high_codebook_size: int,
) -> dict[str, Any]:
    """Compare same-time low/high token pairs as a flat pair vocabulary."""
    pair_vocab_size = int(real_low.max().item() + 1) * high_codebook_size
    pair_vocab_size = max(pair_vocab_size, int(sampled_low.max().item() + 1) * high_codebook_size)
    real_pairs = real_low.long() * high_codebook_size + real_high.long()
    sampled_pairs = sampled_low.long() * high_codebook_size + sampled_high.long()
    pair_vocab_size = max(pair_vocab_size, int(real_pairs.max().item()) + 1)
    pair_vocab_size = max(pair_vocab_size, int(sampled_pairs.max().item()) + 1)
    comparison = flatten_token_comparison_metrics(
        compare_token_sequences(
            real_tokens=real_pairs,
            sampled_tokens=sampled_pairs,
            codebook_size=pair_vocab_size,
        )
    )
    return {
        "pair_vocab_size": pair_vocab_size,
        "real_active_pair_count": comparison["real_active_code_count"],
        "sampled_active_pair_count": comparison["sampled_active_code_count"],
        "real_pair_perplexity": comparison["real_token_perplexity"],
        "sampled_pair_perplexity": comparison["sampled_token_perplexity"],
        "marginal_pair_l1": comparison["marginal_code_l1"],
        "transition_pair_l1": comparison["transition_matrix_l1"],
        "run_length_distance": comparison["run_length_distance"],
        "real_pair_index_entropy": comparison["real_token_index_entropy"],
        "sampled_pair_index_entropy": comparison["sampled_token_index_entropy"],
    }


def load_continuous_baseline(
    *,
    model_dir: str,
    base_data_dir: str,
    n_sample: int,
    seed: int,
) -> dict[str, Tensor] | None:
    """Load generated continuous baseline paths when the checkpoint is available."""
    model_path = Path(model_dir)
    if not model_path.exists() or not (model_path / "model.pt").exists():
        return None
    evaluator = TargetModelEvaluator(model_path, base_data_dir=base_data_dir)
    _real_paths, fake_paths, recon_paths = evaluator.load_data(n_sample_test=n_sample, seed=seed)
    return {
        "fake_paths": fake_paths.detach().cpu().float(),
        "recon_paths": recon_paths.detach().cpu().float(),
    }


def compute_condition_buckets(
    *,
    conditions: Tensor,
    real_paths: Tensor,
    decoded_paths: Tensor,
    sampled_low: Tensor,
    sampled_high: Tensor,
    high_codebook_size: int,
    n_buckets: int = 5,
) -> list[dict[str, Any]]:
    """Compute composed-path and token usage diagnostics by VIX quantile bucket."""
    condition_values = conditions.reshape(conditions.shape[0], -1).mean(dim=1).detach().cpu()
    sorted_positions = torch.argsort(condition_values)
    buckets: list[dict[str, Any]] = []
    for bucket_index, positions in enumerate(torch.tensor_split(sorted_positions, n_buckets)):
        if positions.numel() == 0:
            continue
        real_bucket = real_paths.index_select(0, positions)
        decoded_bucket = decoded_paths.index_select(0, positions)
        low_bucket = sampled_low.index_select(0, positions)
        high_bucket = sampled_high.index_select(0, positions)
        pair_bucket = low_bucket.long() * high_codebook_size + high_bucket.long()
        values = condition_values.index_select(0, positions)
        low_usage = summarise_code_usage(torch.bincount(low_bucket.reshape(-1), minlength=64)[:64])
        high_usage = summarise_code_usage(
            torch.bincount(high_bucket.reshape(-1), minlength=64)[:64]
        )
        pair_usage = summarise_code_usage(
            torch.bincount(pair_bucket.reshape(-1), minlength=64 * high_codebook_size)[
                : 64 * high_codebook_size
            ]
        )
        buckets.append(
            {
                "bucket_index": bucket_index,
                "bucket_label": condition_bucket_label(bucket_index, n_buckets),
                "n_samples": int(positions.numel()),
                "vix_min": float(values.min().item()),
                "vix_max": float(values.max().item()),
                "vix_mean": float(values.mean().item()),
                "path_metrics": {
                    **compute_path_distribution_metrics(
                        generated=decoded_bucket,
                        real=real_bucket,
                    ),
                    **compare_market_summaries(
                        real_paths=real_bucket,
                        generated_paths=decoded_bucket,
                    ),
                },
                "sampled_low_active_code_count": low_usage["active_code_count"],
                "sampled_low_perplexity": low_usage["codebook_perplexity"],
                "sampled_high_active_code_count": high_usage["active_code_count"],
                "sampled_high_perplexity": high_usage["codebook_perplexity"],
                "sampled_pair_active_count": pair_usage["active_code_count"],
                "sampled_pair_perplexity": pair_usage["codebook_perplexity"],
            }
        )
    return buckets


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a compact Markdown summary for quick inspection."""
    manifest = cast(Mapping[str, Any], summary["manifest"])
    likelihoods = cast(Mapping[str, Any], summary["teacher_forced_eval"])
    tokens = cast(Mapping[str, Any], summary["token_diagnostics"])
    paths = cast(Mapping[str, Any], summary["decoded_composed_path_metrics"])
    pair = cast(Mapping[str, Any], tokens["same_time_pair"])
    continuous = cast(Mapping[str, Any] | None, summary["continuous_beta_cvae_metrics"])
    lines = [
        "# Separate Frequency Prior Evaluation",
        "",
        "## Manifest",
        "",
        f"- Prior: `{manifest['prior_dir']}`",
        f"- Low tokenizer: `{manifest['low_tokenizer_dir']}`",
        f"- High tokenizer: `{manifest['high_tokenizer_dir']}`",
        f"- Token data: `{manifest['token_data_dir']}`",
        f"- Samples: {manifest['n_sample']}, seed {manifest['seed']}, "
        f"temperature {manifest['temperature']}, top_k {manifest['top_k']}",
        "",
        "## Likelihoods",
        "",
        f"- Aggregate CE/perplexity: {likelihoods['cross_entropy']:.8f} / "
        f"{likelihoods['perplexity']:.8f}",
        f"- Low CE/perplexity/accuracy: {likelihoods['low_cross_entropy']:.8f} / "
        f"{likelihoods['low_perplexity']:.8f} / {likelihoods['low_accuracy']:.8f}",
        f"- High CE/perplexity/accuracy: {likelihoods['high_cross_entropy']:.8f} / "
        f"{likelihoods['high_perplexity']:.8f} / {likelihoods['high_accuracy']:.8f}",
        "",
        "## Token Diagnostics",
        "",
        f"- Low sampled active/perplexity: "
        f"{tokens['low']['sampled_active_code_count']} / "
        f"{tokens['low']['sampled_token_perplexity']:.8f}",
        f"- High sampled active/perplexity: "
        f"{tokens['high']['sampled_active_code_count']} / "
        f"{tokens['high']['sampled_token_perplexity']:.8f}",
        f"- Same-time sampled pairs/perplexity: "
        f"{pair['sampled_active_pair_count']} / {pair['sampled_pair_perplexity']:.8f}",
        "",
        "## Composed Paths",
        "",
        f"- MMD/SWD: {paths['mmd']:.8f} / {paths['swd']:.8f}",
        f"- Volatility W1: {paths['volatility_wasserstein']:.8f}",
        f"- Terminal W1: {paths['terminal_return_wasserstein']:.8f}",
        f"- Drawdown W1: {paths['maximum_drawdown_wasserstein']:.8f}",
        f"- Squared-return AC L1: {paths['squared_return_autocorrelation_l1']:.8f}",
        "",
    ]
    if continuous is not None:
        lines.extend(
            [
                "## Continuous Baseline",
                "",
                f"- MMD/SWD: {continuous['mmd']:.8f} / {continuous['swd']:.8f}",
                f"- Volatility W1: {continuous['volatility_wasserstein']:.8f}",
                f"- Terminal W1: {continuous['terminal_return_wasserstein']:.8f}",
                f"- Drawdown W1: {continuous['maximum_drawdown_wasserstein']:.8f}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
