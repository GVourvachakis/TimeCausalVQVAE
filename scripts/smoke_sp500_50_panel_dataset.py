"""Smoke-check the local S&P500 50-stock panel dataset and diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from time_causal_vae.data.sp500_panel import SP50050PanelDataset
from time_causal_vae.evaluation.cross_sectional_diagnostics import (
    compare_cross_sectional_diagnostics,
    empirical_correlation_matrix,
    empirical_covariance_matrix,
    equal_weight_portfolio_returns,
    portfolio_var_es,
    random_portfolio_return_diagnostics,
    sector_block_correlation_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Load the processed S&P500 50-stock panel and write smoke diagnostics.",
    )
    parser.add_argument("--n-samples", type=int, default=64, help="Windows per split.")
    parser.add_argument(
        "--train-n-samples",
        type=int,
        help="Optional train split sample count. Defaults to --n-samples.",
    )
    parser.add_argument(
        "--eval-n-samples",
        type=int,
        help="Optional eval split sample count. Defaults to --n-samples.",
    )
    parser.add_argument("--n-timesteps", type=int, default=60, help="Expected window length.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base processed data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sp500_50_panel_data_smoke",
        help="Output directory for local smoke summaries.",
    )
    parser.add_argument(
        "--standardize-returns",
        action="store_true",
        help="Expose train-standardised returns while retaining raw returns.",
    )
    return parser


def main() -> int:
    """Run the dataset smoke check."""
    args = build_parser().parse_args()
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be positive.")
    train_n_samples = int(args.train_n_samples or args.n_samples)
    eval_n_samples = int(args.eval_n_samples or args.n_samples)
    if train_n_samples <= 0 or eval_n_samples <= 0:
        raise SystemExit("split sample counts must be positive.")
    if args.n_timesteps <= 1:
        raise SystemExit("--n-timesteps must be greater than one.")

    train_dataset = SP50050PanelDataset(
        train_n_samples,
        args.n_timesteps,
        base_data_dir=args.base_data_dir,
        split="train",
        standardize_returns=args.standardize_returns,
    )
    eval_dataset = SP50050PanelDataset(
        eval_n_samples,
        args.n_timesteps,
        base_data_dir=args.base_data_dir,
        split="eval",
        standardize_returns=args.standardize_returns,
        standardization_stats=train_dataset.standardization_stats,
    )
    summary = build_summary(train_dataset, eval_dataset=eval_dataset, args=args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown_summary(summary), encoding="utf-8")

    print("S&P500 50-stock panel dataset smoke complete.")
    print(f"train_data_shape={tuple(train_dataset.data.shape)}")
    print(f"train_label_shape={tuple(train_dataset.labels.shape)}")
    print(f"eval_data_shape={tuple(eval_dataset.data.shape)}")
    print(f"eval_label_shape={tuple(eval_dataset.labels.shape)}")
    print(f"summary_json={json_path}")
    print(f"summary_md={markdown_path}")
    return 0


def build_summary(
    train_dataset: SP50050PanelDataset,
    *,
    eval_dataset: SP50050PanelDataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a JSON-safe smoke summary."""
    covariance = empirical_covariance_matrix(train_dataset.data)
    correlation = empirical_correlation_matrix(train_dataset.data)
    equal_weight_returns = equal_weight_portfolio_returns(train_dataset.data)
    sector_blocks = sector_block_correlation_summary(correlation, train_dataset.sector_labels)
    random_portfolios = random_portfolio_return_diagnostics(
        train_dataset.data,
        n_portfolios=64,
        seed=0,
    )
    split_comparison = compare_cross_sectional_diagnostics(
        train_dataset.data,
        eval_dataset.data,
        sector_labels=train_dataset.sector_labels,
        n_random_portfolios=64,
        random_seed=0,
    )
    return {
        "config": {
            "n_samples": int(args.n_samples),
            "train_n_samples": int(train_dataset.data.shape[0]),
            "eval_n_samples": int(eval_dataset.data.shape[0]),
            "n_timesteps": int(args.n_timesteps),
            "base_data_dir": str(args.base_data_dir),
            "standardize_returns": bool(args.standardize_returns),
        },
        "tensor_shapes": {
            "train_data": list(train_dataset.data.shape),
            "train_labels": list(train_dataset.labels.shape),
            "eval_data": list(eval_dataset.data.shape),
            "eval_labels": list(eval_dataset.labels.shape),
        },
        "metadata": metadata_summary(train_dataset),
        "diagnostics": {
            "covariance_trace": float(torch.trace(covariance).item()),
            "mean_abs_correlation": mean_abs_offdiagonal(correlation),
            "sector_blocks": sector_blocks,
            "equal_weight": {
                "one_step_mean": float(equal_weight_returns.mean().item()),
                "one_step_std": float(equal_weight_returns.std(unbiased=False).item()),
                "var_es": portfolio_var_es(equal_weight_returns),
            },
            "random_portfolios": random_portfolios,
            "train_eval_comparison": split_comparison,
        },
        "smoke_assertions": smoke_assertions(train_dataset, eval_dataset),
    }


def metadata_summary(dataset: SP50050PanelDataset) -> dict[str, Any]:
    """Return compact metadata fields."""
    metadata = dataset.metadata
    missing_data = metadata["missing_data"]
    date_range = metadata["date_range"]
    split = metadata["split"]
    return {
        "universe_id": str(metadata["universe_id"]),
        "tickers": list(dataset.tickers),
        "sectors": list(dataset.sectors),
        "condition_names": list(dataset.condition_names),
        "date_range": {
            "first_window_start_date": str(date_range["first_window_start_date"]),
            "last_window_end_date": str(date_range["last_window_end_date"]),
        },
        "missing_data": {
            "handling": str(missing_data["handling"]),
            "forward_fill_used": bool(missing_data["forward_fill_used"]),
            "raw_price_rows": int(missing_data["raw_price_rows"]),
            "aligned_price_rows": int(missing_data["aligned_price_rows"]),
            "dropped_price_rows": int(missing_data["dropped_price_rows"]),
        },
        "split": {
            "train_window_count": int(split["train_window_count"]),
            "eval_window_count": int(split["eval_window_count"]),
        },
        "standardization": standardization_summary(metadata),
        "yfinance_version": str(metadata["yfinance"]["version"]),
    }


def standardization_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact return-standardisation metadata."""
    standardization = metadata.get("standardization", {})
    if not isinstance(standardization, Mapping):
        return {"enabled": False}
    return {
        "enabled": bool(standardization.get("enabled", False)),
        "fit_split": str(standardization.get("fit_split", "")),
        "stats_source": str(standardization.get("stats_source", "")),
        "raw_log_returns_available_on_dataset": bool(
            standardization.get("raw_log_returns_available_on_dataset", False)
        ),
        "mean_abs_standardized_asset_mean": optional_float(
            standardization.get("mean_abs_standardized_asset_mean")
        ),
        "mean_standardized_asset_std": optional_float(
            standardization.get("mean_standardized_asset_std")
        ),
    }


def optional_float(value: Any) -> float | None:
    """Return a float when ``value`` is present."""
    if value is None:
        return None
    return float(value)


def smoke_assertions(
    train_dataset: SP50050PanelDataset,
    eval_dataset: SP50050PanelDataset,
) -> dict[str, bool]:
    """Return boolean smoke checks without raising on benchmark-quality questions."""
    return {
        "train_shape_is_batch_time_asset": train_dataset.data.ndim == 3
        and train_dataset.data.shape[-1] == 50,
        "eval_shape_is_batch_time_asset": eval_dataset.data.ndim == 3
        and eval_dataset.data.shape[-1] == 50,
        "condition_dim_is_two": train_dataset.labels.ndim == 2
        and train_dataset.labels.shape[-1] == 2,
        "returns_finite": bool(torch.isfinite(train_dataset.data).all().item())
        and bool(torch.isfinite(eval_dataset.data).all().item()),
        "labels_finite": bool(torch.isfinite(train_dataset.labels).all().item())
        and bool(torch.isfinite(eval_dataset.labels).all().item()),
        "sector_labels_match_assets": train_dataset.sector_labels.numel()
        == train_dataset.data.shape[-1],
        "raw_returns_available": train_dataset.raw_log_returns.shape == train_dataset.data.shape
        and eval_dataset.raw_log_returns.shape == eval_dataset.data.shape,
        "eval_uses_train_standardization_when_enabled": (
            train_dataset.standardization_stats is None
            and eval_dataset.standardization_stats is None
        )
        or (
            train_dataset.standardization_stats is not None
            and eval_dataset.standardization_stats is not None
            and torch.allclose(
                train_dataset.standardization_stats["mean"],
                eval_dataset.standardization_stats["mean"],
            )
            and torch.allclose(
                train_dataset.standardization_stats["std"],
                eval_dataset.standardization_stats["std"],
            )
        ),
    }


def mean_abs_offdiagonal(correlation: torch.Tensor) -> float:
    """Return mean absolute off-diagonal correlation."""
    mask = ~torch.eye(correlation.shape[0], dtype=torch.bool)
    return float(correlation[mask].abs().mean().item())


def build_markdown_summary(summary: Mapping[str, Any]) -> str:
    """Render a compact Markdown smoke summary."""
    metadata = summary["metadata"]
    diagnostics = summary["diagnostics"]
    equal_weight = diagnostics["equal_weight"]
    comparison = diagnostics["train_eval_comparison"]
    assertions = summary["smoke_assertions"]
    lines = [
        "# S&P500 50-Stock Panel Dataset Smoke",
        "",
        "## Shapes",
        "",
        f"- Train data: {summary['tensor_shapes']['train_data']}",
        f"- Train labels: {summary['tensor_shapes']['train_labels']}",
        f"- Eval data: {summary['tensor_shapes']['eval_data']}",
        f"- Eval labels: {summary['tensor_shapes']['eval_labels']}",
        "",
        "## Metadata",
        "",
        f"- Universe: {metadata['universe_id']}",
        f"- Date range: {metadata['date_range']['first_window_start_date']} to "
        f"{metadata['date_range']['last_window_end_date']}",
        f"- Missing-data handling: {metadata['missing_data']['handling']}",
        f"- Dropped price rows: {metadata['missing_data']['dropped_price_rows']}",
        f"- Conditions: {metadata['condition_names']}",
        f"- yfinance version: {metadata['yfinance_version']}",
        f"- Standardize returns: {summary['config']['standardize_returns']}",
        f"- Standardization stats source: {metadata['standardization']['stats_source']}",
        "",
        "## Diagnostics",
        "",
        f"- Covariance trace: {diagnostics['covariance_trace']:.8f}",
        f"- Mean absolute off-diagonal correlation: {diagnostics['mean_abs_correlation']:.8f}",
        f"- Equal-weight one-step std: {equal_weight['one_step_std']:.8f}",
        f"- Equal-weight VaR q01: {equal_weight['var_es']['lower_tail_var_q01']:.8f}",
        f"- Equal-weight ES q01: {equal_weight['var_es']['lower_tail_es_q01']:.8f}",
        f"- Train/eval covariance Frobenius error: {comparison['covariance_frobenius_error']:.8f}",
        f"- Train/eval correlation Frobenius error: "
        f"{comparison['correlation_frobenius_error']:.8f}",
        "",
        "## Smoke Assertions",
        "",
        *[f"- {name}: {value}" for name, value in assertions.items()],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
