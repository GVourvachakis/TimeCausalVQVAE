"""Generate visual diagnostics for the Hawkes-jump benchmark dataset."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import matplotlib
import torch
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from time_causal_vae.data.hawkes_jump import HawkesJumpDataset
from time_causal_vae.evaluation.jump_diagnostics import jump_diagnostic_summary
from time_causal_vae.evaluation.market_diagnostics import (
    compute_log_returns,
    distribution_summary,
    market_style_summary,
    maximum_drawdown,
)

SimulationScheme = Literal["fixed_grid", "ogata"]
FIGURE_NAMES = (
    "sample_price_paths.png",
    "sample_log_return_paths.png",
    "jump_indicator_raster.png",
    "intensity_trajectories.png",
    "volatility_trajectories.png",
    "jump_count_histogram.png",
    "inter_arrival_histogram.png",
    "jump_size_distribution.png",
    "return_tail_histogram.png",
    "var_es_summary.png",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plot and summarise a synthetic Hawkes-jump dataset.",
    )
    parser.add_argument(
        "--simulation-scheme",
        choices=("fixed_grid", "ogata"),
        default="ogata",
        help="Hawkes-jump simulator backend to inspect.",
    )
    parser.add_argument(
        "--compare-schemes",
        action="store_true",
        help="Run fixed-grid and Ogata backends and write comparison tables.",
    )
    parser.add_argument("--n-samples", type=int, default=512, help="Number of paths to simulate.")
    parser.add_argument("--n-timesteps", type=int, default=60, help="Path length including start.")
    parser.add_argument("--seed", type=int, default=99, help="Deterministic simulation seed.")
    parser.add_argument(
        "--output-dir",
        default="outputs/hawkes_jump_plots",
        help="Output directory for plots and summaries.",
    )
    parser.add_argument(
        "--no-volatility-excitation",
        action="store_true",
        help="Disable jump-excited volatility for this diagnostic run.",
    )
    return parser


def main() -> int:
    """Run Hawkes-jump plotting diagnostics."""
    args = build_parser().parse_args()
    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_schemes:
        comparison = run_comparison(args=args)
        write_json(output_dir / "comparison_summary.json", comparison)
        (output_dir / "comparison_summary.md").write_text(
            build_comparison_markdown(comparison),
            encoding="utf-8",
        )
        print("Hawkes-jump comparison diagnostics complete.")
        print(f"output_dir={output_dir}")
        return 0

    dataset, summary, runtime_seconds = simulate_and_summarise(
        scheme=args.simulation_scheme,
        args=args,
    )
    summary["runtime_seconds"] = runtime_seconds
    figure_paths = plot_dataset(dataset, summary=summary, output_dir=output_dir)
    summary["figures"] = [path.name for path in figure_paths]
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(build_markdown_summary(summary), encoding="utf-8")

    print("Hawkes-jump visual diagnostics complete.")
    print(f"simulation_scheme={args.simulation_scheme}")
    print(f"data_shape={tuple(dataset.data.shape)}")
    print(f"total_jumps={int(dataset.jump_counts.sum().item())}")
    print(f"output_dir={output_dir}")
    return 0


def simulate_and_summarise(
    *,
    scheme: SimulationScheme,
    args: argparse.Namespace,
) -> tuple[HawkesJumpDataset, dict[str, Any], float]:
    """Simulate one dataset and return the summary plus elapsed wall time."""
    start = time.perf_counter()
    dataset = HawkesJumpDataset(
        args.n_samples,
        args.n_timesteps,
        seed=args.seed,
        simulation_scheme=scheme,
        volatility_excitation=not args.no_volatility_excitation,
    )
    runtime_seconds = time.perf_counter() - start
    summary = build_summary(dataset, scheme=scheme, args=args)
    return dataset, summary, runtime_seconds


def build_summary(
    dataset: HawkesJumpDataset,
    *,
    scheme: SimulationScheme,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a JSON-safe dataset summary."""
    market_summary = market_style_summary(dataset.data)
    jump_summary = jump_diagnostic_summary(
        dataset.data,
        jump_indicators=dataset.jump_indicators,
        jump_counts=dataset.jump_counts,
        jump_sizes=dataset.jump_sizes,
    )
    returns = compute_log_returns(dataset.data)
    severe_tail = distribution_summary(returns.flatten())
    drawdown_summary = distribution_summary(maximum_drawdown(dataset.data))
    return {
        "config": {
            "n_samples": int(args.n_samples),
            "n_timesteps": int(args.n_timesteps),
            "seed": int(args.seed),
            "simulation_scheme": scheme,
            "volatility_excitation": not bool(args.no_volatility_excitation),
        },
        "tensor_shapes": {
            "prices": list(dataset.prices.shape),
            "data": list(dataset.data.shape),
            "log_returns": list(dataset.log_returns.shape),
            "jump_indicators": list(dataset.jump_indicators.shape),
            "jump_counts": list(dataset.jump_counts.shape),
            "jump_sizes": list(dataset.jump_sizes.shape),
            "intensities": list(dataset.intensities.shape),
            "volatilities": list(dataset.volatilities.shape),
        },
        "checks": build_checks(dataset),
        "dataset_metadata": dict(dataset.metadata),
        "market_summary": market_summary,
        "jump_summary": jump_summary,
        "severe_tail_statistics": {
            "one_step_log_returns": severe_tail,
            "max_abs_return_per_path": market_summary["max_abs_return_per_path"],
            "maximum_drawdown": drawdown_summary,
        },
    }


def build_checks(dataset: HawkesJumpDataset) -> dict[str, bool]:
    """Return basic dataset health checks."""
    return {
        "prices_positive": bool((dataset.prices > 0.0).all().item()),
        "data_finite": bool(torch.isfinite(dataset.data).all().item()),
        "log_returns_finite": bool(torch.isfinite(dataset.log_returns).all().item()),
        "expected_price_shape": dataset.prices.ndim == 3 and dataset.prices.shape[-1] == 1,
        "has_jumps": bool((dataset.jump_counts.sum() > 0).item()),
        "has_negative_jump_step": bool(
            ((dataset.jump_sizes < 0.0) & dataset.jump_indicators).any().item()
        ),
        "intensities_finite": bool(torch.isfinite(dataset.intensities).all().item()),
        "volatilities_finite": bool(torch.isfinite(dataset.volatilities).all().item()),
    }


def plot_dataset(
    dataset: HawkesJumpDataset,
    *,
    summary: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Write the full figure inventory for one Hawkes-jump dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = [
        plot_sample_paths(
            dataset.prices,
            output_dir / "sample_price_paths.png",
            title="Sample Price Paths",
            ylabel="Price",
        ),
        plot_sample_paths(
            dataset.log_returns,
            output_dir / "sample_log_return_paths.png",
            title="Sample Log-Return Paths",
            ylabel="Log return",
        ),
        plot_jump_raster(dataset.jump_indicators, output_dir / "jump_indicator_raster.png"),
        plot_sample_paths(
            dataset.intensities,
            output_dir / "intensity_trajectories.png",
            title="Hawkes Intensity Trajectories",
            ylabel="Intensity",
        ),
        plot_sample_paths(
            dataset.volatilities,
            output_dir / "volatility_trajectories.png",
            title="Volatility Trajectories",
            ylabel="Volatility",
        ),
        plot_jump_count_histogram(dataset.jump_counts, output_dir / "jump_count_histogram.png"),
        plot_inter_arrival_histogram(
            dataset.jump_indicators,
            output_dir / "inter_arrival_histogram.png",
        ),
        plot_jump_size_distribution(dataset.jump_sizes, output_dir / "jump_size_distribution.png"),
        plot_return_tail_histogram(dataset.data, output_dir / "return_tail_histogram.png"),
        plot_var_es_summary(summary, output_dir / "var_es_summary.png"),
    ]
    return figure_paths


def plot_sample_paths(values: Tensor, path: Path, *, title: str, ylabel: str) -> Path:
    """Plot a small deterministic subset of path trajectories."""
    paths = _to_2d_numpy(values)
    n_paths = min(paths.shape[0], 24)
    fig, ax = plt.subplots(figsize=(9, 5))
    for index in range(n_paths):
        ax.plot(paths[index], linewidth=1.0, alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_jump_raster(jump_indicators: Tensor, path: Path) -> Path:
    """Plot jump indicators as a path-by-time raster."""
    indicators = _to_2d_numpy(jump_indicators).astype(float)
    max_paths = min(indicators.shape[0], 256)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(
        indicators[:max_paths],
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
    )
    ax.set_title("Jump Indicator Raster")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Path index")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_jump_count_histogram(jump_counts: Tensor, path: Path) -> Path:
    """Plot per-path jump-count histogram."""
    counts = _to_2d_numpy(jump_counts).sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(0, int(counts.max()) + 2)
    ax.hist(counts, bins=bins, edgecolor="black", alpha=0.8)
    ax.set_title("Jump Count Distribution")
    ax.set_xlabel("Jumps per path")
    ax.set_ylabel("Path count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_inter_arrival_histogram(jump_indicators: Tensor, path: Path) -> Path:
    """Plot within-path inter-arrival gaps."""
    gaps = inter_arrival_gaps(jump_indicators)
    fig, ax = plt.subplots(figsize=(8, 5))
    if gaps.numel() > 0:
        ax.hist(gaps.numpy(), bins=range(1, int(gaps.max().item()) + 2), edgecolor="black")
    ax.set_title("Inter-Arrival Time Distribution")
    ax.set_xlabel("Grid steps between jumps")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_jump_size_distribution(jump_sizes: Tensor, path: Path) -> Path:
    """Plot non-zero aggregate jump sizes."""
    sizes = jump_sizes.detach().float().reshape(-1)
    sizes = sizes[sizes.abs() > 0.0]
    fig, ax = plt.subplots(figsize=(8, 5))
    if sizes.numel() > 0:
        ax.hist(sizes.numpy(), bins=60, edgecolor="black", alpha=0.8)
        ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_title("Jump Size Distribution")
    ax.set_xlabel("Aggregate jump log-return")
    ax.set_ylabel("Jump-step count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_return_tail_histogram(paths: Tensor, path: Path) -> Path:
    """Plot one-step return histogram with tail quantiles."""
    returns = compute_log_returns(paths).detach().float().reshape(-1)
    quantiles = torch.quantile(
        returns,
        torch.tensor([0.001, 0.01, 0.99, 0.999], dtype=returns.dtype),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(returns.numpy(), bins=90, edgecolor="black", alpha=0.75)
    for value, label in zip(quantiles.tolist(), ("q001", "q01", "q99", "q999"), strict=True):
        ax.axvline(value, linestyle="--", linewidth=1.2, label=label)
    ax.set_title("One-Step Return Tail Histogram")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_var_es_summary(summary: Mapping[str, Any], path: Path) -> Path:
    """Plot lower-tail VaR and ES values as bars."""
    var_es = summary["jump_summary"]["var_es"]
    labels = ("VaR 1%", "ES 1%", "VaR 5%", "ES 5%")
    values = [
        var_es["lower_tail_var_q01"],
        var_es["lower_tail_es_q01"],
        var_es["lower_tail_var_q05"],
        var_es["lower_tail_es_q05"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Lower-Tail VaR and ES")
    ax.set_ylabel("Log return")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_comparison(*, args: argparse.Namespace) -> dict[str, Any]:
    """Run both simulator schemes and return a compact comparison summary."""
    rows: dict[str, dict[str, Any]] = {}
    for scheme in ("fixed_grid", "ogata"):
        _, summary, runtime_seconds = simulate_and_summarise(scheme=scheme, args=args)
        rows[scheme] = comparison_row(summary, runtime_seconds=runtime_seconds)
    return {
        "config": {
            "n_samples": int(args.n_samples),
            "n_timesteps": int(args.n_timesteps),
            "seed": int(args.seed),
            "volatility_excitation": not bool(args.no_volatility_excitation),
        },
        "schemes": rows,
    }


def comparison_row(summary: Mapping[str, Any], *, runtime_seconds: float) -> dict[str, Any]:
    """Extract comparison metrics for one scheme."""
    metadata = summary["dataset_metadata"]
    clustering = summary["jump_summary"]["clustering"]
    var_es = summary["jump_summary"]["var_es"]
    return {
        "runtime_seconds": runtime_seconds,
        "mean_jump_count_per_path": metadata["mean_jump_count_per_path"],
        "paths_with_jump_fraction": metadata["paths_with_jump_fraction"],
        "count_overdispersion": clustering["count_overdispersion"],
        "adjacent_jump_pair_count": clustering["adjacent_jump_pair_count"],
        "lower_tail_var_q01": var_es["lower_tail_var_q01"],
        "lower_tail_es_q01": var_es["lower_tail_es_q01"],
        "lower_tail_var_q05": var_es["lower_tail_var_q05"],
        "lower_tail_es_q05": var_es["lower_tail_es_q05"],
        "maximum_drawdown": summary["market_summary"]["maximum_drawdown"],
    }


def build_markdown_summary(summary: Mapping[str, Any]) -> str:
    """Render a compact Markdown report for one dataset."""
    metadata = summary["dataset_metadata"]
    jump_summary = summary["jump_summary"]
    clustering = jump_summary["clustering"]
    var_es = jump_summary["var_es"]
    severe_tail = summary["severe_tail_statistics"]["one_step_log_returns"]
    figures = summary.get("figures", FIGURE_NAMES)
    checks = summary["checks"]
    lines = [
        "# Hawkes-Jump Dataset Visual Diagnostics",
        "",
        "## Config",
        "",
        f"- Simulation scheme: {summary['config']['simulation_scheme']}",
        f"- Samples: {summary['config']['n_samples']}",
        f"- Timesteps: {summary['config']['n_timesteps']}",
        f"- Seed: {summary['config']['seed']}",
        f"- Volatility excitation: {summary['config']['volatility_excitation']}",
        f"- Runtime seconds: {summary['runtime_seconds']:.4f}",
        "",
        "## Figure Inventory",
        "",
        *[f"- `{figure}`" for figure in figures],
        "",
        "## Key Statistics",
        "",
        f"- Price shape: {summary['tensor_shapes']['prices']}",
        f"- Log-return shape: {summary['tensor_shapes']['log_returns']}",
        f"- Total jumps: {metadata['total_jumps']:.0f}",
        f"- Mean jumps per path: {metadata['mean_jump_count_per_path']:.4f}",
        f"- Paths with jumps fraction: {metadata['paths_with_jump_fraction']:.4f}",
        f"- Negative jump fraction: {metadata['negative_jump_fraction']:.4f}",
        f"- Branching ratio proxy: {metadata['branching_ratio_proxy']:.4f}",
        f"- Max intensity: {metadata['max_intensity_observed']:.4f}",
        f"- Max volatility: {metadata['max_volatility_observed']:.4f}",
        f"- Count over-dispersion: {clustering['count_overdispersion']:.4f}",
        f"- Adjacent jump pairs: {clustering['adjacent_jump_pair_count']}",
        f"- Paths with adjacent jumps fraction: "
        f"{clustering['paths_with_adjacent_jump_fraction']:.4f}",
        f"- Return q001: {severe_tail['q001']:.6f}",
        f"- Return q999: {severe_tail['q999']:.6f}",
        f"- Lower-tail VaR q01: {var_es['lower_tail_var_q01']:.6f}",
        f"- Lower-tail ES q01: {var_es['lower_tail_es_q01']:.6f}",
        "",
        "## Checks",
        "",
        *[f"- {name}: {value}" for name, value in checks.items()],
        "",
    ]
    return "\n".join(lines)


def build_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """Render the fixed-grid versus Ogata comparison table."""
    rows = comparison["schemes"]
    lines = [
        "# Hawkes-Jump Simulator Comparison",
        "",
        "## Config",
        "",
        f"- Samples: {comparison['config']['n_samples']}",
        f"- Timesteps: {comparison['config']['n_timesteps']}",
        f"- Seed: {comparison['config']['seed']}",
        f"- Volatility excitation: {comparison['config']['volatility_excitation']}",
        "",
        "## Comparison",
        "",
        "| Scheme | Runtime | Mean jumps/path | Paths with jumps | "
        "Count over-dispersion | VaR q01 | ES q01 | Max drawdown mean | "
        "Max drawdown q99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scheme in ("fixed_grid", "ogata"):
        row = rows[scheme]
        drawdown = row["maximum_drawdown"]
        lines.append(
            f"| `{scheme}` | {row['runtime_seconds']:.4f} | "
            f"{row['mean_jump_count_per_path']:.4f} | "
            f"{row['paths_with_jump_fraction']:.4f} | "
            f"{row['count_overdispersion']:.4f} | "
            f"{row['lower_tail_var_q01']:.6f} | "
            f"{row['lower_tail_es_q01']:.6f} | "
            f"{drawdown['mean']:.6f} | "
            f"{drawdown['q99']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The comparison mode writes summary tables only; detailed visual figures are "
            "generated by single-scheme runs.",
            "",
        ]
    )
    return "\n".join(lines)


def inter_arrival_gaps(jump_indicators: Tensor) -> Tensor:
    """Return within-path gaps between jump steps."""
    indicators = _to_2d_bool(jump_indicators)
    gaps: list[float] = []
    for path in indicators:
        positions = torch.nonzero(path, as_tuple=False).flatten().float()
        if positions.numel() > 1:
            gaps.extend(positions.diff().tolist())
    return torch.tensor(gaps, dtype=torch.float32)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable, indented JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_2d_numpy(values: Tensor) -> Any:
    squeezed = values.detach().cpu()
    if squeezed.ndim == 3 and squeezed.shape[-1] == 1:
        squeezed = squeezed[..., 0]
    if squeezed.ndim != 2:
        raise ValueError(f"Expected [batch, length] or [batch, length, 1]; got {values.shape}.")
    return squeezed.numpy()


def _to_2d_bool(values: Tensor) -> Tensor:
    squeezed = values.detach().cpu()
    if squeezed.ndim == 3 and squeezed.shape[-1] == 1:
        squeezed = squeezed[..., 0]
    if squeezed.ndim != 2:
        raise ValueError(f"Expected [batch, length] or [batch, length, 1]; got {values.shape}.")
    return squeezed.bool()


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be positive.")
    if args.n_timesteps <= 1:
        raise SystemExit("--n-timesteps must be greater than one.")


if __name__ == "__main__":
    raise SystemExit(main())
