"""Smoke-check the synthetic Hawkes-jump dataset and jump diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from time_causal_vae.data.hawkes_jump import HawkesJumpDataset
from time_causal_vae.evaluation.jump_diagnostics import jump_diagnostic_summary
from time_causal_vae.evaluation.market_diagnostics import market_style_summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate a small Hawkes-jump dataset and write smoke diagnostics.",
    )
    parser.add_argument("--n-samples", type=int, default=256, help="Number of paths to simulate.")
    parser.add_argument("--n-timesteps", type=int, default=60, help="Path length including start.")
    parser.add_argument("--seed", type=int, default=99, help="Deterministic simulation seed.")
    parser.add_argument(
        "--output-dir",
        default="outputs/hawkes_jump_smoke",
        help="Output directory for JSON and Markdown smoke summaries.",
    )
    parser.add_argument(
        "--no-volatility-excitation",
        action="store_true",
        help="Disable jump-excited volatility for this smoke run.",
    )
    return parser


def main() -> int:
    """Run the Hawkes-jump dataset smoke check."""
    args = build_parser().parse_args()
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be positive.")
    if args.n_timesteps <= 1:
        raise SystemExit("--n-timesteps must be greater than one.")

    dataset = HawkesJumpDataset(
        args.n_samples,
        args.n_timesteps,
        seed=args.seed,
        volatility_excitation=not args.no_volatility_excitation,
    )
    summary = build_summary(dataset, args=args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown_summary(summary), encoding="utf-8")

    print("Hawkes-jump smoke complete.")
    print(f"data_shape={tuple(dataset.data.shape)}")
    print(f"label_shape={tuple(dataset.labels.shape)}")
    print(f"total_jumps={int(dataset.jump_counts.sum().item())}")
    print(f"summary_json={json_path}")
    print(f"summary_md={markdown_path}")
    return 0


def build_summary(dataset: HawkesJumpDataset, *, args: argparse.Namespace) -> dict[str, Any]:
    """Build a JSON-safe smoke summary for one generated dataset."""
    market_summary = market_style_summary(dataset.data)
    oracle_jump_summary = jump_diagnostic_summary(
        dataset.data,
        jump_indicators=dataset.jump_indicators,
        jump_counts=dataset.jump_counts,
        jump_sizes=dataset.jump_sizes,
    )
    detected_jump_summary = jump_diagnostic_summary(dataset.data)
    return {
        "config": {
            "n_samples": int(args.n_samples),
            "n_timesteps": int(args.n_timesteps),
            "seed": int(args.seed),
            "volatility_excitation": not bool(args.no_volatility_excitation),
        },
        "tensor_shapes": {
            "data": list(dataset.data.shape),
            "labels": list(dataset.labels.shape),
            "log_returns": list(dataset.log_returns.shape),
            "jump_indicators": list(dataset.jump_indicators.shape),
            "jump_counts": list(dataset.jump_counts.shape),
            "jump_sizes": list(dataset.jump_sizes.shape),
            "intensities": list(dataset.intensities.shape),
            "volatilities": list(dataset.volatilities.shape),
        },
        "dataset_metadata": dict(dataset.metadata),
        "market_summary": market_summary,
        "oracle_jump_summary": oracle_jump_summary,
        "detected_jump_summary": detected_jump_summary,
        "smoke_assertions": smoke_assertions(dataset),
    }


def smoke_assertions(dataset: HawkesJumpDataset) -> dict[str, bool]:
    """Return boolean smoke checks without raising on benchmark-quality questions."""
    paths_positive = bool((dataset.data > 0.0).all().item())
    has_expected_rank = dataset.data.ndim == 3 and dataset.data.shape[-1] == 1
    finite_paths = bool(torch.isfinite(dataset.data).all().item())
    finite_returns = bool(torch.isfinite(dataset.log_returns).all().item())
    has_jumps = bool((dataset.jump_counts.sum() > 0).item())
    negative_jumps = bool(((dataset.jump_sizes < 0.0) & dataset.jump_indicators).any().item())
    intensity_finite = bool(torch.isfinite(dataset.intensities).all().item())
    return {
        "paths_positive": paths_positive,
        "shape_is_batch_time_channel": has_expected_rank,
        "paths_finite": finite_paths,
        "log_returns_finite": finite_returns,
        "has_at_least_one_jump": has_jumps,
        "has_negative_jump_step": negative_jumps,
        "intensities_finite": intensity_finite,
    }


def build_markdown_summary(summary: Mapping[str, Any]) -> str:
    """Render a compact Markdown smoke summary."""
    metadata = summary["dataset_metadata"]
    oracle_jumps = summary["oracle_jump_summary"]
    clustering = oracle_jumps["clustering"]
    var_es = oracle_jumps["var_es"]
    assertions = summary["smoke_assertions"]
    lines = [
        "# Hawkes-Jump Dataset Smoke",
        "",
        "## Config",
        "",
        f"- Samples: {summary['config']['n_samples']}",
        f"- Timesteps: {summary['config']['n_timesteps']}",
        f"- Seed: {summary['config']['seed']}",
        f"- Volatility excitation: {summary['config']['volatility_excitation']}",
        "",
        "## Dataset",
        "",
        f"- Data shape: {summary['tensor_shapes']['data']}",
        f"- Total jumps: {metadata['total_jumps']:.0f}",
        f"- Mean jumps per path: {metadata['mean_jump_count_per_path']:.4f}",
        f"- Paths with jumps: {metadata['paths_with_jump_fraction']:.4f}",
        f"- Negative jump fraction: {metadata['negative_jump_fraction']:.4f}",
        f"- Max intensity observed: {metadata['max_intensity_observed']:.4f}",
        f"- Max volatility observed: {metadata['max_volatility_observed']:.4f}",
        "",
        "## Jump Diagnostics",
        "",
        f"- Adjacent jump pairs: {clustering['adjacent_jump_pair_count']}",
        f"- Paths with adjacent jumps: {clustering['paths_with_adjacent_jump_fraction']:.4f}",
        f"- Count over-dispersion: {clustering['count_overdispersion']:.4f}",
        f"- Lower-tail VaR q01: {var_es['lower_tail_var_q01']:.6f}",
        f"- Lower-tail ES q01: {var_es['lower_tail_es_q01']:.6f}",
        "",
        "## Smoke Assertions",
        "",
        *[f"- {name}: {value}" for name, value in assertions.items()],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
