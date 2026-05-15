"""Paper-style S&P500/VIX diagnostics for discrete and continuous models."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from time_causal_vae.cli.evaluate_token_prior import (
    freeze_tokenizer,
    load_conditional_eval_payload,
    load_token_prior_yaml,
)
from time_causal_vae.data.frequency import compose_frequency_channels
from time_causal_vae.evaluation.checkpoints import TargetModelEvaluator
from time_causal_vae.evaluation.market_diagnostics import (
    DEFAULT_AUTOCORRELATION_LAGS,
    compare_market_summaries,
    compute_log_returns,
    market_style_summary,
    max_abs_return_per_path,
    max_rolling_volatility_per_path,
    maximum_drawdown,
    outlier_path_metadata,
    return_tail_thresholds,
    terminal_returns,
    volatility_per_path,
)
from time_causal_vae.evaluation.style import apply_clean_style
from time_causal_vae.evaluation.token_diagnostics import (
    compare_token_sequences,
    flatten_token_comparison_metrics,
)
from time_causal_vae.evaluation.token_prior import (
    component_token_metrics,
    compute_condition_bucket_sample_metrics,
    compute_path_distribution_metrics,
    decode_token_indices,
    load_trained_token_prior,
)
from time_causal_vae.evaluation.tokenizer import load_trained_tokenizer
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the script argument parser."""
    parser = argparse.ArgumentParser(
        description="Run paper-style S&P500/VIX diagnostics for promoted discrete samples.",
    )
    parser.add_argument("--discrete-config", required=True)
    parser.add_argument("--discrete-prior-dir", required=True)
    parser.add_argument("--discrete-tokenizer-dir", required=True)
    parser.add_argument("--continuous-config", required=True)
    parser.add_argument("--continuous-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-data-dir", default="data/processed")
    parser.add_argument("--n-sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--top-k",
        type=parse_optional_top_k,
        default=None,
        help="Top-k sampling cutoff. Use 'none' for unrestricted sampling.",
    )
    return parser


def parse_optional_top_k(raw_value: str) -> int | None:
    """Parse an optional top-k sampling value."""
    normalised = raw_value.strip().lower()
    if normalised in {"none", "null", "unrestricted"}:
        return None
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("--top-k must be positive or 'none'.")
    return value


def main() -> None:
    """Run the S&P500/VIX paper-style smoke or full evaluation."""
    parser = build_parser()
    args = parser.parse_args()
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive.")

    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cpu")

    discrete = generate_discrete_samples(
        config_path=args.discrete_config,
        prior_dir=args.discrete_prior_dir,
        tokenizer_dir=args.discrete_tokenizer_dir,
        n_sample=args.n_sample,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    continuous = load_or_generate_continuous_samples(
        config_path=args.continuous_config,
        model_dir=args.continuous_model_dir,
        base_data_dir=args.base_data_dir,
        n_sample=args.n_sample,
        seed=args.seed,
    )

    real_paths = cast(Tensor, discrete["real_paths"]).detach().cpu()
    discrete_paths = cast(Tensor, discrete["decoded_paths"]).detach().cpu()
    source_paths: dict[str, Tensor] = {
        "real": real_paths,
        "discrete": discrete_paths,
    }
    if continuous is not None and not bool(continuous.get("skipped", False)):
        source_paths["continuous"] = cast(Tensor, continuous["fake_paths"]).detach().cpu()

    summary = build_summary(
        source_paths=source_paths,
        discrete=discrete,
        continuous=continuous,
        args=vars(args),
    )
    write_json(output_dir / "paper_style_summary.json", summary)
    write_markdown_summary(output_dir / "paper_style_summary.md", summary)
    write_markdown_summary(output_dir / "paper_style_tables.md", summary)
    write_json(output_dir / "comparison_manifest.json", summary["manifest"])
    if "token_diagnostics" in summary:
        write_json(output_dir / "token_diagnostics_summary.json", summary["token_diagnostics"])
    if "outlier_diagnostics" in summary:
        write_json(output_dir / "outlier_paths_summary.json", summary["outlier_diagnostics"])
    if "vix_buckets" in summary:
        write_markdown_buckets(output_dir / "vix_bucket_summary.md", summary)

    plot_returns_distribution(output_dir / "returns_distribution.png", source_paths)
    plot_distribution(
        output_dir / "terminal_return_distribution.png",
        source_paths,
        terminal_returns,
        "Terminal return",
    )
    plot_distribution(
        output_dir / "volatility_distribution.png",
        source_paths,
        volatility_per_path,
        "Path volatility",
    )
    plot_distribution(
        output_dir / "maximum_drawdown_distribution.png",
        source_paths,
        maximum_drawdown,
        "Maximum drawdown",
    )
    plot_autocorrelation(
        output_dir / "log_return_autocorrelation.png",
        source_paths,
        squared=False,
    )
    plot_autocorrelation(
        output_dir / "squared_return_autocorrelation.png",
        source_paths,
        squared=True,
    )
    plot_skew_kurtosis(output_dir / "skew_kurtosis.png", summary)
    plot_extreme_discrete_paths(
        output_dir / "extreme_discrete_paths.png",
        discrete_paths,
        max_abs_return_per_path(discrete_paths),
        max_rolling_volatility_per_path(discrete_paths),
    )
    plot_extreme_return_histogram(
        output_dir / "extreme_return_histogram.png",
        source_paths,
        real_paths,
    )
    plot_distribution(
        output_dir / "volatility_tail_comparison.png",
        source_paths,
        max_rolling_volatility_per_path,
        "Maximum rolling volatility",
    )
    conditions = cast(Tensor | None, discrete.get("conditions"))
    if conditions is not None:
        plot_vix_bucket_paths(
            output_dir / "vix_bucket_paths.png",
            source_paths,
            conditions.detach().cpu(),
        )
        plot_vix_bucket_distribution(
            output_dir / "vix_bucket_terminal_returns.png",
            source_paths,
            conditions.detach().cpu(),
            terminal_returns,
            "Terminal return",
        )
        plot_vix_bucket_distribution(
            output_dir / "vix_bucket_volatility.png",
            source_paths,
            conditions.detach().cpu(),
            volatility_per_path,
            "Path volatility",
        )

    save_tensor_artifacts(output_dir, discrete=discrete, continuous=continuous)
    print("S&P500/VIX paper-style evaluation complete.")
    print(f"output_dir: {output_dir}")
    print(f"real_paths: {list(real_paths.shape)}")
    print(f"discrete_paths: {list(discrete_paths.shape)}")
    if continuous is None:
        print("continuous: skipped")
    else:
        print(f"continuous_paths: {list(cast(Tensor, continuous['fake_paths']).shape)}")
    print(f"discrete_mmd: {summary['comparisons']['discrete']['mmd']:.8f}")
    print(f"discrete_swd: {summary['comparisons']['discrete']['swd']:.8f}")


def generate_discrete_samples(
    *,
    config_path: str,
    prior_dir: str,
    tokenizer_dir: str,
    n_sample: int,
    seed: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> dict[str, Any]:
    """Sample the promoted discrete prior and decode through the frozen tokenizer."""
    raw_config = load_token_prior_yaml(config_path)
    prior, prior_config, _prior_checkpoint = load_trained_token_prior(prior_dir, device=device)
    tokenizer, _tokenizer_config, tokenizer_checkpoint = load_trained_tokenizer(
        tokenizer_dir,
        device=device,
    )
    tokenizer_data_config = tokenizer_frequency_data_config(tokenizer_checkpoint)
    freeze_tokenizer(tokenizer)
    payload = load_conditional_eval_payload(
        raw_config,
        n_sample=n_sample,
        prior_config=prior_config,
    )
    if payload is None:
        raise SystemExit("S&P500/VIX paper-style evaluation requires conditional token data.")
    conditions = payload["labels"].to(device)
    real_paths = payload["data"].to(device)
    real_tokens = payload["indices"].detach().cpu().long()
    set_seed(seed)
    sampled_tokens = prior.sample(
        batch_size=n_sample,
        device=device,
        temperature=temperature,
        top_k=top_k,
        conditions=conditions,
    )
    quantized, decoded_paths = decode_token_indices(
        tokenizer,
        sampled_tokens,
        conditions=conditions,
    )
    decoded_frequency_paths = None
    if should_compose_frequency_output(tokenizer_data_config):
        decoded_frequency_paths = decoded_paths
        decoded_paths = compose_if_frequency_channels(decoded_paths)
        real_paths = compose_if_frequency_channels(real_paths)
    sampled_tokens_cpu = sampled_tokens.detach().cpu()
    token_comparison: dict[str, Any]
    if real_tokens.ndim == 2 and sampled_tokens_cpu.ndim == 2:
        token_comparison = flatten_token_comparison_metrics(
            compare_token_sequences(
                real_tokens=real_tokens,
                sampled_tokens=sampled_tokens_cpu,
                codebook_size=prior_config.codebook_size,
            )
        )
    else:
        token_comparison = {
            "token_diagnostics_note": (
                "Single-token marginal, transition, and run-length diagnostics are omitted "
                "for native multi-code token tensors."
            ),
            "real_token_shape": list(real_tokens.shape),
            "sampled_token_shape": list(sampled_tokens_cpu.shape),
            "sampled_component_metrics": component_token_metrics(
                sampled_tokens_cpu,
                prior_config.codebook_size,
            ),
        }
    return {
        "raw_config": raw_config,
        "prior_config": prior_config,
        "real_paths": real_paths.detach().cpu(),
        "decoded_paths": decoded_paths.detach().cpu(),
        "decoded_frequency_paths": (
            decoded_frequency_paths.detach().cpu() if decoded_frequency_paths is not None else None
        ),
        "conditions": conditions.detach().cpu(),
        "sampled_tokens": sampled_tokens_cpu,
        "real_tokens": real_tokens,
        "quantized": quantized.detach().cpu(),
        "token_comparison": token_comparison,
        "condition_buckets": compute_condition_bucket_sample_metrics(
            sampled_tokens=sampled_tokens.detach().cpu(),
            decoded_paths=decoded_paths.detach().cpu(),
            real_paths=real_paths.detach().cpu(),
            conditions=conditions.detach().cpu(),
            codebook_size=prior_config.codebook_size,
        ),
    }


def tokenizer_frequency_data_config(tokenizer_checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the tokenizer training data config if it was persisted."""
    training_config = tokenizer_checkpoint.get("training_config")
    if not isinstance(training_config, Mapping):
        return {}
    data_config = training_config.get("data")
    if not isinstance(data_config, Mapping):
        return {}
    return data_config


def should_compose_frequency_output(data_config: Mapping[str, Any]) -> bool:
    """Return whether decoded tokenizer outputs should be composed."""
    decomposition = data_config.get("frequency_decomposition")
    if decomposition is None or str(decomposition).lower() in {"none", "null"}:
        return False
    if str(decomposition).lower() != "ema":
        raise SystemExit(f"Unsupported tokenizer frequency_decomposition: {decomposition!r}")
    return bool(data_config.get("compose_output", True))


def compose_if_frequency_channels(paths: Tensor) -> Tensor:
    """Compose two-channel frequency paths and leave one-channel paths unchanged."""
    if paths.ndim == 3 and paths.shape[-1] == 2:
        return compose_frequency_channels(paths)
    if paths.ndim == 3 and paths.shape[-1] == 1:
        return paths
    raise ValueError(
        "Expected paths with shape [batch, length, 1] or [batch, length, 2]; "
        f"got {tuple(paths.shape)}."
    )


def load_or_generate_continuous_samples(
    *,
    config_path: str,
    model_dir: str,
    base_data_dir: str,
    n_sample: int,
    seed: int,
) -> dict[str, Any] | None:
    """Generate released continuous samples, or skip if the checkpoint is unavailable."""
    model_path = Path(model_dir)
    if not model_path.exists() or not (model_path / "model.pt").exists():
        return {
            "skipped": True,
            "skip_reason": f"continuous checkpoint not found: {model_dir}",
        }
    exp_config_path = model_path.parent / "exp_config.yaml"
    if not exp_config_path.exists():
        fallback = Path("outputs/released_target_eval_sp500_vix/evaluation_batch.pt")
        if fallback.exists():
            loaded = torch.load(fallback, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict) and {"real_data", "fake_data"}.issubset(loaded):
                return {
                    "skipped": False,
                    "source": str(fallback),
                    "real_paths": cast(Tensor, loaded["real_data"])[:n_sample].detach().cpu(),
                    "fake_paths": cast(Tensor, loaded["fake_data"])[:n_sample].detach().cpu(),
                    "recon_paths": cast(Tensor, loaded.get("recon_data", loaded["real_data"]))[
                        :n_sample
                    ]
                    .detach()
                    .cpu(),
                }
        return {
            "skipped": True,
            "skip_reason": (
                f"continuous exp_config.yaml not found next to checkpoint: {exp_config_path}"
            ),
        }
    evaluator = TargetModelEvaluator(model_path, base_data_dir=base_data_dir)
    real_paths, fake_paths, recon_paths = evaluator.load_data(n_sample_test=n_sample, seed=seed)
    return {
        "skipped": False,
        "source": str(model_path),
        "config_path": config_path,
        "real_paths": real_paths.detach().cpu(),
        "fake_paths": fake_paths.detach().cpu(),
        "recon_paths": recon_paths.detach().cpu(),
    }


def build_summary(
    *,
    source_paths: Mapping[str, Tensor],
    discrete: Mapping[str, Any],
    continuous: Mapping[str, Any] | None,
    args: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the JSON summary payload."""
    real_paths = source_paths["real"]
    source_summaries = {name: market_style_summary(paths) for name, paths in source_paths.items()}
    comparisons: dict[str, Any] = {}
    for name, paths in source_paths.items():
        if name == "real":
            continue
        path_metrics = compute_path_distribution_metrics(
            generated=paths,
            real=real_paths,
        )
        market_metrics = compare_market_summaries(
            real_paths=real_paths,
            generated_paths=paths,
        )
        comparisons[name] = {**path_metrics, **market_metrics}

    conditions = cast(Tensor | None, discrete.get("conditions"))
    buckets = (
        compute_vix_bucket_comparisons(source_paths, conditions) if conditions is not None else []
    )
    real_thresholds = return_tail_thresholds(real_paths)
    outliers = {
        "tail_thresholds_from_real": real_thresholds,
        "sources": {
            name: outlier_path_metadata(paths, top_k=10)
            for name, paths in source_paths.items()
            if name != "real"
        },
        "real_reference": outlier_path_metadata(real_paths, top_k=10),
    }
    continuous_summary = continuous
    if continuous_summary is not None and bool(continuous_summary.get("skipped", False)):
        continuous_summary = {
            "skipped": True,
            "skip_reason": continuous_summary.get("skip_reason", "unknown"),
        }
    elif continuous_summary is not None:
        continuous_summary = {
            "skipped": False,
            "source": continuous_summary.get("source"),
            "config_path": continuous_summary.get("config_path"),
            "fake_path_shape": list(cast(Tensor, continuous_summary["fake_paths"]).shape),
        }
    return {
        "manifest": {
            "script": "scripts/evaluate_sp500_vix_paper_style.py",
            "args": dict(args),
            "condition_convention": "paired_eval_labels_from_token_artifacts",
            "continuous": continuous_summary,
        },
        "source_summaries": source_summaries,
        "comparisons": comparisons,
        "outlier_diagnostics": outliers,
        "token_diagnostics": discrete.get("token_comparison"),
        "vix_buckets": buckets,
        "existing_condition_buckets": discrete.get("condition_buckets"),
        "decision_note": (
            "Promote as empirical discrete baseline only if stylised facts remain "
            "competitive with the continuous reference, not solely because MMD/SWD improve."
        ),
    }


def compute_vix_bucket_comparisons(
    source_paths: Mapping[str, Tensor],
    conditions: Tensor,
    *,
    n_buckets: int = 5,
) -> list[dict[str, Any]]:
    """Compute paper-style diagnostics by VIX quantile bucket."""
    condition_values = conditions.reshape(conditions.shape[0], -1).mean(dim=1).detach().cpu()
    sorted_positions = torch.argsort(condition_values)
    buckets: list[dict[str, Any]] = []
    for bucket_index, positions in enumerate(torch.tensor_split(sorted_positions, n_buckets)):
        if positions.numel() == 0:
            continue
        bucket_paths = {
            name: paths.detach().cpu().index_select(0, positions)
            for name, paths in source_paths.items()
            if paths.shape[0] >= int(positions.max().item()) + 1
        }
        real_bucket = bucket_paths["real"]
        comparisons: dict[str, Any] = {}
        for name, paths in bucket_paths.items():
            if name == "real":
                continue
            comparisons[name] = {
                **compute_path_distribution_metrics(generated=paths, real=real_bucket),
                **compare_market_summaries(real_paths=real_bucket, generated_paths=paths),
            }
        condition_bucket_values = condition_values.index_select(0, positions)
        buckets.append(
            {
                "bucket_index": bucket_index,
                "bucket_label": bucket_label(bucket_index, n_buckets),
                "n_samples": int(positions.numel()),
                "vix_min": float(condition_bucket_values.min().item()),
                "vix_max": float(condition_bucket_values.max().item()),
                "vix_mean": float(condition_bucket_values.mean().item()),
                "comparisons": comparisons,
            }
        )
    return buckets


def bucket_label(index: int, count: int) -> str:
    """Return human-readable quantile bucket names."""
    if count == 5:
        return ["very_low", "low", "mid", "high", "very_high"][index]
    return f"bucket_{index}"


def plot_returns_distribution(path: Path, source_paths: Mapping[str, Tensor]) -> None:
    """Plot flattened one-step return distributions."""
    values = {name: compute_log_returns(paths).flatten() for name, paths in source_paths.items()}
    plot_tensor_histograms(path, values, "One-step return", "Density")


def plot_distribution(
    path: Path,
    source_paths: Mapping[str, Tensor],
    fn: Any,
    xlabel: str,
) -> None:
    """Plot source distributions from a path-to-vector function."""
    values = {name: fn(paths).reshape(-1) for name, paths in source_paths.items()}
    plot_tensor_histograms(path, values, xlabel, "Density")


def plot_tensor_histograms(
    path: Path,
    values: Mapping[str, Tensor],
    xlabel: str,
    ylabel: str,
) -> None:
    """Plot overlapping histograms for named tensors."""
    apply_clean_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, tensor in values.items():
        clean = tensor.detach().cpu().float().reshape(-1)
        clean = clean[torch.isfinite(clean)]
        if clean.numel() == 0:
            continue
        value_min = float(clean.min().item())
        value_max = float(clean.max().item())
        value_scale = max(abs(value_min), abs(value_max), 1.0)
        if abs(value_max - value_min) <= value_scale * 1e-6:
            epsilon = value_scale * 1e-6
            histogram_bins: list[float] = [value_min - epsilon, value_max + epsilon]
        else:
            bin_count = min(50, max(1, int(clean.numel())))
            histogram_bins = torch.linspace(value_min, value_max, bin_count + 1).tolist()
        ax.hist(
            clean.numpy(),
            bins=histogram_bins,
            alpha=0.45,
            density=True,
            label=name,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_autocorrelation(
    path: Path,
    source_paths: Mapping[str, Tensor],
    *,
    squared: bool,
) -> None:
    """Plot within-path return or squared-return autocorrelation curves."""
    from time_causal_vae.evaluation.market_diagnostics import return_autocorrelation

    apply_clean_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    lag_values = list(DEFAULT_AUTOCORRELATION_LAGS)
    for name, paths in source_paths.items():
        values = return_autocorrelation(paths, lags=lag_values, squared=squared)
        ax.plot(lag_values, [values[str(lag)] for lag in lag_values], marker="o", label=name)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(
        "Within-path squared-return autocorrelation"
        if squared
        else "Within-path log-return autocorrelation"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_extreme_discrete_paths(
    path: Path,
    discrete_paths: Tensor,
    max_abs_returns: Tensor,
    max_volatility: Tensor,
    *,
    top_k: int = 5,
) -> None:
    """Plot the most extreme discrete paths by return and volatility."""
    paths_2d = discrete_paths.detach().cpu().float()
    if paths_2d.ndim == 3:
        paths_2d = paths_2d[..., 0]
    k = min(top_k, paths_2d.shape[0])
    return_indices = torch.topk(max_abs_returns.detach().cpu(), k=k).indices
    volatility_indices = torch.topk(max_volatility.detach().cpu(), k=k).indices
    apply_clean_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for index in return_indices:
        axes[0].plot(paths_2d[int(index.item())].numpy(), alpha=0.85)
    axes[0].set_title("Largest absolute one-step returns")
    for index in volatility_indices:
        axes[1].plot(paths_2d[int(index.item())].numpy(), alpha=0.85)
    axes[1].set_title("Largest rolling volatility")
    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Normalised level")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_extreme_return_histogram(
    path: Path,
    source_paths: Mapping[str, Tensor],
    real_paths: Tensor,
) -> None:
    """Plot one-step returns with real q01/q99 and q001/q999 thresholds."""
    thresholds = return_tail_thresholds(real_paths)
    values = {name: compute_log_returns(paths).flatten() for name, paths in source_paths.items()}
    apply_clean_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, tensor in values.items():
        ax.hist(
            tensor.detach().cpu().float().numpy(),
            bins=80,
            alpha=0.4,
            density=True,
            label=name,
        )
    for key, colour in [
        ("q001", "tab:red"),
        ("q01", "tab:orange"),
        ("q99", "tab:orange"),
        ("q999", "tab:red"),
    ]:
        ax.axvline(thresholds[key], color=colour, linestyle="--", linewidth=1.0)
    ax.set_xlabel("One-step return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_skew_kurtosis(path: Path, summary: Mapping[str, Any]) -> None:
    """Plot return skewness and excess kurtosis by source."""
    source_summaries = cast(Mapping[str, Mapping[str, Any]], summary["source_summaries"])
    names = list(source_summaries)
    skew_values = [
        float(cast(Mapping[str, Any], source_summaries[name]["returns"])["skewness"])
        for name in names
    ]
    kurtosis_values = [
        float(cast(Mapping[str, Any], source_summaries[name]["returns"])["excess_kurtosis"])
        for name in names
    ]
    x_positions = torch.arange(len(names)).numpy()
    apply_clean_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x_positions - 0.18, skew_values, width=0.36, label="skewness")
    ax.bar(x_positions + 0.18, kurtosis_values, width=0.36, label="excess kurtosis")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(names)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_vix_bucket_paths(
    path: Path,
    source_paths: Mapping[str, Tensor],
    conditions: Tensor,
) -> None:
    """Plot mean paths by VIX bucket and source."""
    condition_values = conditions.reshape(conditions.shape[0], -1).mean(dim=1).detach().cpu()
    sorted_positions = torch.argsort(condition_values)
    apply_clean_style()
    fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
    for bucket_index, positions in enumerate(torch.tensor_split(sorted_positions, 5)):
        ax = axes[bucket_index]
        for name, paths in source_paths.items():
            if paths.shape[0] < int(positions.max().item()) + 1:
                continue
            bucket_paths = paths.detach().cpu().index_select(0, positions)
            ax.plot(bucket_paths[..., 0].mean(dim=0).numpy(), label=name)
        ax.set_title(bucket_label(bucket_index, 5))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_vix_bucket_distribution(
    path: Path,
    source_paths: Mapping[str, Tensor],
    conditions: Tensor,
    fn: Any,
    ylabel: str,
) -> None:
    """Plot bucket means for a distributional diagnostic."""
    condition_values = conditions.reshape(conditions.shape[0], -1).mean(dim=1).detach().cpu()
    sorted_positions = torch.argsort(condition_values)
    labels = [bucket_label(index, 5) for index in range(5)]
    apply_clean_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    x_positions = torch.arange(5, dtype=torch.float32)
    offsets = torch.linspace(-0.24, 0.24, steps=len(source_paths))
    for offset, (name, paths) in zip(offsets, source_paths.items(), strict=False):
        means = []
        for positions in torch.tensor_split(sorted_positions, 5):
            if paths.shape[0] < int(positions.max().item()) + 1:
                means.append(float("nan"))
            else:
                means.append(float(fn(paths.detach().cpu().index_select(0, positions)).mean()))
        ax.bar((x_positions + offset).numpy(), means, width=0.18, label=name)
    ax.set_xticks(x_positions.numpy())
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a compact Markdown summary."""
    comparisons = cast(Mapping[str, Mapping[str, Any]], summary["comparisons"])
    lines = [
        "# S&P500/VIX Paper-Style Evaluation",
        "",
        "## Global Path Metrics",
        "",
        "| Source | MMD | SWD | Vol. W1 | Terminal W1 | Returns W1 | Drawdown W1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in comparisons.items():
        lines.append(
            "| "
            f"{name} | "
            f"{float(metrics['mmd']):.8f} | "
            f"{float(metrics['swd']):.8f} | "
            f"{float(metrics['volatility_wasserstein']):.8f} | "
            f"{float(metrics['terminal_return_wasserstein']):.8f} | "
            f"{float(metrics['returns_wasserstein']):.8f} | "
            f"{float(metrics['maximum_drawdown_wasserstein']):.8f} |"
        )
    lines.extend(
        [
            "",
            "## Within-Path Autocorrelation",
            "",
            (
                "| Source | Return AC L1 | Squared-return AC L1 | "
                "Flattened return AC L1 | Flattened squared-return AC L1 |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in comparisons.items():
        lines.append(
            "| "
            f"{name} | "
            f"{float(metrics['return_autocorrelation_within_path_l1']):.8f} | "
            f"{float(metrics['squared_return_autocorrelation_within_path_l1']):.8f} | "
            f"{float(metrics['return_autocorrelation_flattened_l1']):.8f} | "
            f"{float(metrics['squared_return_autocorrelation_flattened_l1']):.8f} |"
        )
    lines.extend(
        [
            "",
            "## Tail Exceedance Rates",
            "",
            "| Source | < real q001 | < real q01 | > real q99 | > real q999 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in comparisons.items():
        rates = cast(Mapping[str, Any], metrics["tail_exceedance_rates"])
        lines.append(
            "| "
            f"{name} | "
            f"{float(rates['below_real_q001_fraction']):.8f} | "
            f"{float(rates['below_real_q01_fraction']):.8f} | "
            f"{float(rates['above_real_q99_fraction']):.8f} | "
            f"{float(rates['above_real_q999_fraction']):.8f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Note",
            "",
            str(summary["decision_note"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_buckets(path: Path, summary: Mapping[str, Any]) -> None:
    """Write VIX-bucket comparison tables."""
    buckets = cast(list[Mapping[str, Any]], summary["vix_buckets"])
    lines = [
        "# S&P500/VIX Bucket Diagnostics",
        "",
        "| Bucket | VIX min | VIX max | Source | MMD | SWD | Vol. W1 | Terminal W1 |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for bucket in buckets:
        comparisons = cast(Mapping[str, Mapping[str, Any]], bucket["comparisons"])
        for name, metrics in comparisons.items():
            lines.append(
                "| "
                f"{bucket['bucket_label']} | "
                f"{float(bucket['vix_min']):.8f} | "
                f"{float(bucket['vix_max']):.8f} | "
                f"{name} | "
                f"{float(metrics['mmd']):.8f} | "
                f"{float(metrics['swd']):.8f} | "
                f"{float(metrics['volatility_wasserstein']):.8f} | "
                f"{float(metrics['terminal_return_wasserstein']):.8f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_tensor_artifacts(
    output_dir: Path,
    *,
    discrete: Mapping[str, Any],
    continuous: Mapping[str, Any] | None,
) -> None:
    """Save generated tensors for follow-up plotting."""
    torch.save(
        {
            "decoded_paths": discrete["decoded_paths"],
            "real_paths": discrete["real_paths"],
            "conditions": discrete["conditions"],
            "sampled_tokens": discrete["sampled_tokens"],
            "real_tokens": discrete["real_tokens"],
            "z_q": discrete["quantized"],
        },
        output_dir / "discrete_paper_style_batch.pt",
    )
    if continuous is not None and not bool(continuous.get("skipped", False)):
        torch.save(
            {
                "fake_paths": continuous["fake_paths"],
                "real_paths": continuous["real_paths"],
                "recon_paths": continuous["recon_paths"],
            },
            output_dir / "continuous_paper_style_batch.pt",
        )


def validate_output_dir(output_dir: str) -> Path:
    """Validate output path below ignored outputs."""
    path = Path(output_dir)
    resolved = path.resolve()
    outputs_root = (Path.cwd() / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise SystemExit(f"--output-dir must be under outputs/. Received: {output_dir}") from exc
    return path


def json_safe(value: Any) -> Any:
    """Convert tensors and non-JSON objects into JSON-safe structures."""
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, int | float | str | bool):
            return item
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
