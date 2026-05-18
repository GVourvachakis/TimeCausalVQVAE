"""Aggregate Hawkes continuous-ablation and discrete log-return evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

MetricRow = dict[str, str | int | float | None]


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "mmd": ("smooth_metrics", "mmd"),
    "swd": ("smooth_metrics", "swd"),
    "terminal_w1": ("smooth_metrics", "terminal_return_wasserstein"),
    "volatility_w1": ("smooth_metrics", "volatility_wasserstein"),
    "drawdown_w1": ("market_comparison", "maximum_drawdown_wasserstein"),
    "returns_w1": ("market_comparison", "returns_wasserstein"),
    "jump_count_w1": ("jump_comparison", "detected_jump_count_wasserstein"),
    "jump_size_w1": ("jump_comparison", "detected_jump_size_wasserstein"),
    "generated_jump_count_mean": (
        "jump_diagnostics",
        "generated",
        "jump_counts",
        "per_path",
        "mean",
    ),
    "real_jump_count_mean": (
        "jump_diagnostics",
        "real",
        "jump_counts",
        "per_path",
        "mean",
    ),
    "paths_with_jump_fraction": (
        "jump_diagnostics",
        "generated",
        "jump_counts",
        "paths_with_jump_fraction",
    ),
    "count_overdispersion": (
        "jump_diagnostics",
        "generated",
        "clustering",
        "count_overdispersion",
    ),
    "adjacent_pair_per_jump_step": (
        "jump_diagnostics",
        "generated",
        "clustering",
        "adjacent_pair_per_jump_step",
    ),
    "negative_jump_fraction": (
        "jump_diagnostics",
        "generated",
        "jump_sizes",
        "negative_jump_fraction",
    ),
    "var_01": ("jump_diagnostics", "generated", "var_es", "lower_tail_var_q01"),
    "es_01": ("jump_diagnostics", "generated", "var_es", "lower_tail_es_q01"),
    "var_05": ("jump_diagnostics", "generated", "var_es", "lower_tail_var_q05"),
    "es_05": ("jump_diagnostics", "generated", "var_es", "lower_tail_es_q05"),
    "below_real_q01_fraction": (
        "market_comparison",
        "tail_exceedance_rates",
        "below_real_q01_fraction",
    ),
    "below_real_q001_fraction": (
        "market_comparison",
        "tail_exceedance_rates",
        "below_real_q001_fraction",
    ),
    "sampled_active_codes": ("token_metrics", "sampled_active_code_count"),
    "sampled_code_perplexity": ("token_metrics", "sampled_codebook_perplexity"),
    "transition_matrix_l1": ("token_metrics", "transition_matrix_l1"),
    "run_length_w1": ("token_metrics", "run_length_wasserstein"),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the aggregation parser."""
    parser = argparse.ArgumentParser(
        description="Aggregate Hawkes log-return continuous ablations and discrete evaluations.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--continuous-root",
        default="outputs/hawkes_jump_continuous_logreturn_identity",
        help="Root containing repaired BetaCVAE identity evaluations by seed.",
    )
    parser.add_argument(
        "--info-continuous-root",
        default="outputs/hawkes_jump_continuous_logreturn_info_identity",
        help="Root containing InfoCVAE identity evaluations by seed.",
    )
    parser.add_argument(
        "--discrete-eval-root",
        default="outputs/hawkes_jump_logreturn_robustness/evaluations",
    )
    parser.add_argument(
        "--discrete-prior-root",
        default="outputs/hawkes_jump_logreturn_robustness/priors",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hawkes_jump_matched_continuous_comparison",
    )
    return parser


def main() -> None:
    """Aggregate all requested Hawkes comparison rows."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[MetricRow] = []
    for seed in args.seeds:
        additive_run = (
            f"hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive_seed{seed}"
        )
        conv_run = (
            f"hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_seed{seed}"
        )
        rows.append(
            load_model_row(
                model="continuous_logreturn_beta_cvae",
                seed=seed,
                summary_path=Path(args.continuous_root)
                / f"seed{seed}"
                / "evaluation"
                / "evaluation_summary.json",
                batch_path=Path(args.continuous_root)
                / f"seed{seed}"
                / "evaluation"
                / "evaluation_batch.pt",
                runtime_path=find_continuous_runtime(Path(args.continuous_root), seed),
            ),
        )
        rows.append(
            load_model_row(
                model="continuous_logreturn_info_cvae",
                seed=seed,
                summary_path=Path(args.info_continuous_root)
                / f"seed{seed}"
                / "evaluation"
                / "evaluation_summary.json",
                batch_path=Path(args.info_continuous_root)
                / f"seed{seed}"
                / "evaluation"
                / "evaluation_batch.pt",
                runtime_path=find_continuous_runtime(Path(args.info_continuous_root), seed),
            ),
        )
        rows.append(
            load_model_row(
                model="cb64_additive_ar",
                seed=seed,
                summary_path=Path(args.discrete_eval_root)
                / f"additive_seed{seed}"
                / "evaluation_summary.json",
                batch_path=Path(args.discrete_eval_root)
                / f"additive_seed{seed}"
                / "evaluation_batch.pt",
                runtime_path=Path(args.discrete_prior_root)
                / "additive"
                / additive_run
                / "runtime_summary.json",
            ),
        )
        rows.append(
            load_model_row(
                model="cb64_conv_transformer_k3",
                seed=seed,
                summary_path=Path(args.discrete_eval_root)
                / f"conv_transformer_seed{seed}"
                / "evaluation_summary.json",
                batch_path=Path(args.discrete_eval_root)
                / f"conv_transformer_seed{seed}"
                / "evaluation_batch.pt",
                runtime_path=Path(args.discrete_prior_root)
                / "conv_transformer"
                / conv_run
                / "runtime_summary.json",
            ),
        )

    aggregate = {
        "manifest": {
            "seeds": args.seeds,
            "continuous_root": args.continuous_root,
            "info_continuous_root": args.info_continuous_root,
            "discrete_eval_root": args.discrete_eval_root,
            "discrete_prior_root": args.discrete_prior_root,
        },
        "rows": rows,
        "by_model": summarise_rows(rows),
    }
    write_json(output_dir / "aggregate_summary.json", aggregate)
    write_csv(output_dir / "aggregate_summary.csv", rows)
    write_markdown(output_dir / "aggregate_summary.md", aggregate)

    print("Hawkes continuous ablation aggregation complete.")
    print(f"output_dir: {output_dir}")
    for model, stats in aggregate["by_model"].items():
        print(
            f"{model}: mmd={format_mean_std(stats['mmd'])}, "
            f"jump_count_w1={format_mean_std(stats['jump_count_w1'])}"
        )


def load_model_row(
    *,
    model: str,
    seed: int,
    summary_path: Path,
    batch_path: Path,
    runtime_path: Path | None,
) -> MetricRow:
    """Load one seed/model row from evaluation outputs."""
    summary = load_json(summary_path)
    row: MetricRow = {
        "model": model,
        "seed": seed,
        "summary_path": str(summary_path),
        "batch_path": str(batch_path),
    }
    for name, path in METRIC_PATHS.items():
        row[name] = as_float(get_nested(summary, path))
    row["inter_arrival_w1"] = inter_arrival_w1_from_batch(batch_path)
    row["training_runtime_seconds"] = load_runtime_seconds(runtime_path)
    return row


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return data


def find_continuous_runtime(root: Path, seed: int) -> Path | None:
    """Find the continuous training runtime summary for a seed."""
    matches = sorted((root / f"seed{seed}").glob("*/runtime_summary.json"))
    if not matches:
        return None
    return matches[-1]


def load_runtime_seconds(path: Path | None) -> float | None:
    """Load runtime seconds if a runtime summary is available."""
    if path is None or not path.exists():
        return None
    return as_float(load_json(path).get("elapsed_seconds"))


def get_nested(data: Mapping[str, Any], path: Iterable[str]) -> Any:
    """Read a nested metric path, returning ``None`` when missing."""
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def as_float(value: Any) -> float | None:
    """Convert numeric values to float and preserve missing values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def inter_arrival_w1_from_batch(batch_path: Path) -> float | None:
    """Compute jump inter-arrival Wasserstein distance from saved jump masks."""
    if not batch_path.exists():
        return None
    batch = torch.load(batch_path, map_location="cpu")
    generated = batch.get("generated_jumps")
    real = batch.get("real_jumps")
    if generated is None or real is None:
        return None
    generated_gaps = inter_arrival_gaps(generated)
    real_gaps = inter_arrival_gaps(real)
    return wasserstein_1d(generated_gaps, real_gaps)


def inter_arrival_gaps(jumps: torch.Tensor) -> torch.Tensor:
    """Return within-path gaps between detected jump steps."""
    if jumps.ndim == 3:
        jumps = jumps.squeeze(-1)
    gaps: list[torch.Tensor] = []
    for path_jumps in jumps.bool():
        indices = torch.nonzero(path_jumps, as_tuple=False).flatten().float()
        if indices.numel() > 1:
            gaps.append(indices[1:] - indices[:-1])
    if not gaps:
        return torch.empty(0)
    return torch.cat(gaps)


def wasserstein_1d(left: torch.Tensor, right: torch.Tensor) -> float | None:
    """Approximate equal-weight 1D Wasserstein distance by quantile matching."""
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    if left.numel() == 0 or right.numel() == 0:
        populated = left if left.numel() > 0 else right
        return float(populated.abs().mean().item())
    n_quantiles = int(max(left.numel(), right.numel()))
    grid = torch.linspace(0.0, 1.0, n_quantiles)
    left_quantiles = torch.quantile(left.float(), grid)
    right_quantiles = torch.quantile(right.float(), grid)
    return float(torch.mean(torch.abs(left_quantiles - right_quantiles)).item())


def summarise_rows(rows: list[MetricRow]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Compute mean and sample standard deviation for numeric metrics by model."""
    models = sorted({str(row["model"]) for row in rows})
    summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    metric_names = [
        name for name in rows[0] if name not in {"model", "seed", "summary_path", "batch_path"}
    ]
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        summary[model] = {}
        for metric_name in metric_names:
            values = [
                float(row[metric_name])
                for row in model_rows
                if isinstance(row.get(metric_name), int | float)
            ]
            summary[model][metric_name] = summarise_values(values)
    return summary


def summarise_values(values: list[float]) -> dict[str, float | int | None]:
    """Summarise one metric vector."""
    if not values:
        return {"mean": None, "std": None, "n": 0}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.mean(values), "std": std, "n": len(values)}


def write_csv(path: Path, rows: list[MetricRow]) -> None:
    """Write per-seed aggregate rows to CSV."""
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON with stable formatting."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown(path: Path, aggregate: Mapping[str, Any]) -> None:
    """Write a compact Markdown aggregate table."""
    by_model = aggregate["by_model"]
    metrics = [
        "mmd",
        "swd",
        "terminal_w1",
        "volatility_w1",
        "drawdown_w1",
        "jump_count_w1",
        "inter_arrival_w1",
        "jump_size_w1",
        "var_01",
        "es_01",
        "sampled_active_codes",
        "sampled_code_perplexity",
    ]
    lines = [
        "# Hawkes Continuous-Ablation/Discrete Aggregate",
        "",
        "| Model | " + " | ".join(metrics) + " |",
        "|---|" + "|".join(["---"] * len(metrics)) + "|",
    ]
    for model, stats in by_model.items():
        cells = [model]
        cells.extend(format_mean_std(stats[metric]) for metric in metrics)
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_mean_std(stats: Mapping[str, Any]) -> str:
    """Format one mean/std pair for tables."""
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None:
        return "n/a"
    if std is None:
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


if __name__ == "__main__":
    main()
