"""Smoke-check the synthetic multifactor market dataset and diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from time_causal_vae.data.multifactor_market import MultifactorMarketDataset
from time_causal_vae.evaluation.cross_sectional_diagnostics import (
    compare_cross_sectional_diagnostics,
    empirical_correlation_matrix,
    empirical_covariance_matrix,
    equal_weight_portfolio_returns,
    factor_beta_loading_diagnostic,
    portfolio_var_es,
    random_portfolio_return_diagnostics,
    sector_block_correlation_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic 50D factor-market dataset and smoke diagnostics.",
    )
    parser.add_argument("--n-samples", type=int, default=256, help="Number of paths to simulate.")
    parser.add_argument("--n-assets", type=int, default=50, help="Number of assets.")
    parser.add_argument("--n-factors", type=int, default=5, help="Number of latent factors.")
    parser.add_argument("--n-timesteps", type=int, default=60, help="Number of return steps.")
    parser.add_argument("--seed", type=int, default=99, help="Deterministic simulation seed.")
    parser.add_argument(
        "--structure-seed",
        type=int,
        help="Seed for persistent market structure. Defaults to --seed.",
    )
    parser.add_argument(
        "--path-seed",
        type=int,
        help="Seed for train path randomness. Defaults to --seed.",
    )
    parser.add_argument(
        "--eval-path-seed",
        type=int,
        help="Seed for eval path randomness. Defaults to train path seed plus one.",
    )
    parser.add_argument(
        "--standardize-returns",
        action="store_true",
        help="Expose train-standardized returns as model-visible data.",
    )
    parser.add_argument(
        "--standardization-epsilon",
        type=float,
        default=1e-6,
        help="Minimum per-asset standard deviation used by return standardization.",
    )
    parser.add_argument(
        "--with-jumps",
        action="store_true",
        help="Enable common and sector self-exciting jump shocks.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/multifactor_market_smoke",
        help="Output directory for JSON and Markdown smoke summaries.",
    )
    return parser


def main() -> int:
    """Run the multifactor market dataset smoke check."""
    args = build_parser().parse_args()
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be positive.")
    if args.n_assets < 2:
        raise SystemExit("--n-assets must be at least two.")
    if args.n_factors <= 0:
        raise SystemExit("--n-factors must be positive.")
    if args.n_factors > args.n_assets:
        raise SystemExit("--n-factors must be no larger than --n-assets.")
    if args.n_timesteps <= 1:
        raise SystemExit("--n-timesteps must be greater than one.")

    structure_seed = args.seed if args.structure_seed is None else args.structure_seed
    path_seed = args.seed if args.path_seed is None else args.path_seed
    eval_path_seed = int(path_seed) + 1 if args.eval_path_seed is None else int(args.eval_path_seed)
    dataset = MultifactorMarketDataset(
        args.n_samples,
        args.n_timesteps,
        n_assets=args.n_assets,
        n_factors=args.n_factors,
        seed=args.seed,
        structure_seed=structure_seed,
        path_seed=path_seed,
        with_jumps=args.with_jumps,
        standardize_returns=args.standardize_returns,
        standardization_epsilon=args.standardization_epsilon,
    )
    eval_dataset = MultifactorMarketDataset(
        args.n_samples,
        args.n_timesteps,
        n_assets=args.n_assets,
        n_factors=args.n_factors,
        seed=args.seed + 1,
        structure_seed=structure_seed,
        path_seed=eval_path_seed,
        with_jumps=args.with_jumps,
        standardize_returns=args.standardize_returns,
        standardization_stats=dataset.standardization_stats,
        standardization_epsilon=args.standardization_epsilon,
    )
    summary = build_summary(dataset, eval_dataset=eval_dataset, args=args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown_summary(summary), encoding="utf-8")

    print("Multifactor market smoke complete.")
    print(f"data_shape={tuple(dataset.data.shape)}")
    print(f"raw_log_returns_shape={tuple(dataset.raw_log_returns.shape)}")
    print(f"label_shape={tuple(dataset.labels.shape)}")
    print(f"loadings_shape={tuple(dataset.loadings.shape)}")
    print(f"structure_seed={structure_seed}")
    print(f"path_seed={path_seed}")
    print(f"eval_path_seed={eval_path_seed}")
    print(f"standardize_returns={bool(args.standardize_returns)}")
    print(f"common_jump_count={int(dataset.common_jump_indicators.sum().item())}")
    print(f"sector_jump_count={int(dataset.sector_jump_indicators.sum().item())}")
    print(f"summary_json={json_path}")
    print(f"summary_md={markdown_path}")
    return 0


def build_summary(
    dataset: MultifactorMarketDataset,
    *,
    eval_dataset: MultifactorMarketDataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a JSON-safe smoke summary for generated train/eval samples."""
    raw_returns = dataset.raw_log_returns
    raw_eval_returns = eval_dataset.raw_log_returns
    covariance = empirical_covariance_matrix(raw_returns)
    correlation = empirical_correlation_matrix(raw_returns)
    equal_weight_returns = equal_weight_portfolio_returns(raw_returns)
    sector_summary = sector_block_correlation_summary(correlation, dataset.sector_labels)
    random_portfolios = random_portfolio_return_diagnostics(
        raw_returns,
        n_portfolios=64,
        seed=args.seed,
    )
    comparison = compare_cross_sectional_diagnostics(
        raw_returns,
        raw_eval_returns,
        sector_labels=dataset.sector_labels,
        true_loadings=dataset.loadings,
        n_random_portfolios=64,
        random_seed=args.seed,
    )
    return {
        "config": {
            "n_samples": int(args.n_samples),
            "n_assets": int(args.n_assets),
            "n_factors": int(args.n_factors),
            "n_timesteps": int(args.n_timesteps),
            "seed": int(args.seed),
            "structure_seed": int(dataset.metadata["structure_seed"]),
            "path_seed": int(dataset.metadata["path_seed"]),
            "eval_path_seed": int(eval_dataset.metadata["path_seed"]),
            "with_jumps": bool(args.with_jumps),
            "standardize_returns": bool(args.standardize_returns),
        },
        "tensor_shapes": {
            "data": list(dataset.data.shape),
            "raw_log_returns": list(dataset.raw_log_returns.shape),
            "labels": list(dataset.labels.shape),
            "loadings": list(dataset.loadings.shape),
            "sector_labels": list(dataset.sector_labels.shape),
            "factor_returns": list(dataset.factor_returns.shape),
            "factor_vol_paths": list(dataset.factor_vol_paths.shape),
            "idiosyncratic_volatilities": list(dataset.idiosyncratic_volatilities.shape),
            "true_covariance": list(dataset.true_covariance.shape),
            "true_correlation": list(dataset.true_correlation.shape),
            "common_jump_indicators": list(dataset.common_jump_indicators.shape),
            "sector_jump_indicators": list(dataset.sector_jump_indicators.shape),
            "jump_sizes": list(dataset.jump_sizes.shape),
        },
        "metadata": metadata_summary(dataset),
        "diagnostics": {
            "covariance_trace": float(torch.trace(covariance).item()),
            "mean_abs_correlation": mean_abs_offdiagonal(correlation),
            "sector_blocks": sector_summary,
            "equal_weight": {
                "one_step_mean": float(equal_weight_returns.mean().item()),
                "one_step_std": float(equal_weight_returns.std(unbiased=False).item()),
                "terminal_log_return_mean": float(equal_weight_returns.sum(dim=-1).mean().item()),
                "var_es": portfolio_var_es(equal_weight_returns),
            },
            "random_portfolios": random_portfolios,
            "factor_loadings": factor_beta_loading_diagnostic(raw_returns, dataset.loadings),
            "seed_split_comparison": comparison,
            "model_visible_standardized": standardized_diagnostics(dataset),
        },
        "smoke_assertions": smoke_assertions(dataset),
    }


def metadata_summary(dataset: MultifactorMarketDataset) -> dict[str, Any]:
    """Return compact metadata fields without serialising large oracle tensors."""
    covariance_summaries = dataset.metadata["true_covariance_summaries"]
    jump_indicators = dataset.metadata["jump_indicators"]
    return {
        "oracle_metadata_model_visible": bool(dataset.metadata["oracle_metadata_model_visible"]),
        "condition_mode": str(dataset.metadata["condition_mode"]),
        "structure_seed": dataset.metadata["structure_seed"],
        "path_seed": dataset.metadata["path_seed"],
        "sector_count": int(dataset.metadata["n_sectors"]),
        "factor_count": int(dataset.metadata["n_factors"]),
        "loadings_shape": list(dataset.loadings.shape),
        "sector_labels": dataset.sector_labels.tolist(),
        "factor_vol_paths_shape": list(dataset.factor_vol_paths.shape),
        "idiosyncratic_volatility_mean": float(dataset.idiosyncratic_volatilities.mean().item()),
        "realised_covariance_trace": float(covariance_summaries["realised_covariance_trace"]),
        "factor_model_covariance_trace": float(
            covariance_summaries["factor_model_covariance_trace"]
        ),
        "top5_realised_covariance_eigenvalues": [
            float(value) for value in covariance_summaries["top5_realised_covariance_eigenvalues"]
        ],
        "mean_abs_offdiagonal_correlation": float(
            covariance_summaries["mean_abs_offdiagonal_correlation"]
        ),
        "jumps": {
            "enabled": bool(jump_indicators["enabled"]),
            "common_jump_count": int(jump_indicators["common_jump_count"]),
            "sector_jump_count": int(jump_indicators["sector_jump_count"]),
            "nonzero_asset_jump_fraction": float(jump_indicators["nonzero_asset_jump_fraction"]),
        },
        "standardization": standardization_summary(dataset),
    }


def smoke_assertions(dataset: MultifactorMarketDataset) -> dict[str, bool]:
    """Return boolean smoke checks without raising on benchmark-quality questions."""
    return {
        "shape_is_batch_time_asset": dataset.data.ndim == 3
        and dataset.data.shape[-1] == dataset.loadings.shape[0],
        "raw_shape_matches_model_visible_shape": dataset.raw_log_returns.shape
        == dataset.data.shape,
        "labels_are_prefix_safe_constant": dataset.labels.ndim == 2
        and dataset.labels.shape[-1] == 1
        and bool(torch.allclose(dataset.labels, torch.ones_like(dataset.labels))),
        "oracle_metadata_not_model_visible": not bool(
            dataset.metadata["oracle_metadata_model_visible"]
        ),
        "returns_finite": bool(torch.isfinite(dataset.data).all().item()),
        "raw_returns_finite": bool(torch.isfinite(dataset.raw_log_returns).all().item()),
        "covariance_finite": bool(torch.isfinite(dataset.true_covariance).all().item()),
        "correlation_finite": bool(torch.isfinite(dataset.true_correlation).all().item()),
        "sector_labels_match_assets": dataset.sector_labels.numel() == dataset.data.shape[-1],
        "factor_vol_paths_match_shape": dataset.factor_vol_paths.shape[:2]
        == dataset.data.shape[:2],
        "standardized_train_mean_near_zero": standardized_train_mean_near_zero(dataset),
        "standardized_train_std_near_one": standardized_train_std_near_one(dataset),
    }


def standardized_diagnostics(dataset: MultifactorMarketDataset) -> dict[str, float | bool]:
    """Return compact diagnostics for model-visible standardised data."""
    data = dataset.data.detach().float()
    asset_mean = data.mean(dim=(0, 1))
    asset_std = data.std(dim=(0, 1), unbiased=False)
    return {
        "enabled": bool(dataset.standardize_returns),
        "mean_abs_asset_mean": float(asset_mean.abs().mean().item()),
        "mean_asset_std": float(asset_std.mean().item()),
        "min_asset_std": float(asset_std.min().item()),
        "max_asset_std": float(asset_std.max().item()),
    }


def standardization_summary(dataset: MultifactorMarketDataset) -> dict[str, Any]:
    """Return JSON-safe standardisation metadata."""
    standardization = dataset.metadata["standardization"]
    summary: dict[str, Any] = {
        "enabled": bool(standardization["enabled"]),
        "stats_source": str(standardization["stats_source"]),
        "raw_log_returns_available_on_dataset": bool(
            standardization["raw_log_returns_available_on_dataset"]
        ),
    }
    if dataset.standardization_stats is not None:
        mean = dataset.standardization_stats["mean"]
        std = dataset.standardization_stats["std"]
        summary.update({
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "mean_abs_mean": float(mean.abs().mean().item()),
            "std_mean": float(std.mean().item()),
            "std_min": float(std.min().item()),
            "std_max": float(std.max().item()),
        })
    return summary


def standardized_train_mean_near_zero(dataset: MultifactorMarketDataset) -> bool:
    """Return true when enabled standardisation centred train assets."""
    if not dataset.standardize_returns:
        return True
    asset_mean = dataset.data.detach().float().mean(dim=(0, 1))
    return bool((asset_mean.abs() < 1e-5).all().item())


def standardized_train_std_near_one(dataset: MultifactorMarketDataset) -> bool:
    """Return true when enabled standardisation scaled train assets."""
    if not dataset.standardize_returns:
        return True
    asset_std = dataset.data.detach().float().std(dim=(0, 1), unbiased=False)
    return bool(((asset_std - 1.0).abs() < 1e-4).all().item())


def mean_abs_offdiagonal(correlation: torch.Tensor) -> float:
    """Return mean absolute off-diagonal correlation."""
    mask = ~torch.eye(correlation.shape[0], dtype=torch.bool)
    return float(correlation[mask].abs().mean().item())


def build_markdown_summary(summary: Mapping[str, Any]) -> str:
    """Render a compact Markdown smoke summary."""
    metadata = summary["metadata"]
    diagnostics = summary["diagnostics"]
    comparison = diagnostics["seed_split_comparison"]
    assertions = summary["smoke_assertions"]
    random_portfolios = diagnostics["random_portfolios"]
    equal_weight = diagnostics["equal_weight"]
    lines = [
        "# Multifactor Market Dataset Smoke",
        "",
        "## Config",
        "",
        f"- Samples: {summary['config']['n_samples']}",
        f"- Assets: {summary['config']['n_assets']}",
        f"- Factors: {summary['config']['n_factors']}",
        f"- Timesteps: {summary['config']['n_timesteps']}",
        f"- Seed: {summary['config']['seed']}",
        f"- Structure seed: {summary['config']['structure_seed']}",
        f"- Train path seed: {summary['config']['path_seed']}",
        f"- Eval path seed: {summary['config']['eval_path_seed']}",
        f"- Jumps enabled: {summary['config']['with_jumps']}",
        f"- Standardize returns: {summary['config']['standardize_returns']}",
        "",
        "## Tensor Shapes",
        "",
        f"- Data: {summary['tensor_shapes']['data']}",
        f"- Raw log returns: {summary['tensor_shapes']['raw_log_returns']}",
        f"- Labels: {summary['tensor_shapes']['labels']}",
        f"- Loadings: {summary['tensor_shapes']['loadings']}",
        f"- Factor volatility paths: {summary['tensor_shapes']['factor_vol_paths']}",
        f"- True covariance: {summary['tensor_shapes']['true_covariance']}",
        f"- Common jump indicators: {summary['tensor_shapes']['common_jump_indicators']}",
        f"- Sector jump indicators: {summary['tensor_shapes']['sector_jump_indicators']}",
        "",
        "## Metadata",
        "",
        f"- Condition mode: {metadata['condition_mode']}",
        f"- Oracle metadata model-visible: {metadata['oracle_metadata_model_visible']}",
        f"- Sector count: {metadata['sector_count']}",
        f"- Factor count: {metadata['factor_count']}",
        f"- Standardization enabled: {metadata['standardization']['enabled']}",
        f"- Standardization stats source: {metadata['standardization']['stats_source']}",
        f"- Common jumps: {metadata['jumps']['common_jump_count']}",
        f"- Sector jumps: {metadata['jumps']['sector_jump_count']}",
        "",
        "## Diagnostics",
        "",
        f"- Covariance trace: {diagnostics['covariance_trace']:.6f}",
        f"- Mean absolute off-diagonal correlation: {diagnostics['mean_abs_correlation']:.6f}",
        f"- Equal-weight one-step std: {equal_weight['one_step_std']:.6f}",
        f"- Equal-weight VaR q01: {equal_weight['var_es']['lower_tail_var_q01']:.6f}",
        f"- Equal-weight ES q01: {equal_weight['var_es']['lower_tail_es_q01']:.6f}",
        f"- Random-portfolio realised volatility mean: "
        f"{random_portfolios['realised_volatility_mean']:.6f}",
        f"- Seed-split covariance Frobenius error: {comparison['covariance_frobenius_error']:.6f}",
        f"- Seed-split correlation Frobenius error: "
        f"{comparison['correlation_frobenius_error']:.6f}",
        f"- Seed-split correlation spectrum distance: "
        f"{comparison['correlation_eigenvalue_spectrum_distance']:.6f}",
        "",
        "## Smoke Assertions",
        "",
        *[f"- {name}: {value}" for name, value in assertions.items()],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
