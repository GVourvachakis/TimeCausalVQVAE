"""Aggregate Hawkes/SVMHJD compact token-prior robustness results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from time_causal_vae.evaluation.market_diagnostics import wasserstein_1d


@dataclass(frozen=True)
class RunSpec:
    """Filesystem locations for one evaluated prior seed."""

    model: str
    seed: int
    parameter_name: str
    evaluation_dir: Path
    train_runtime_path: Path


@dataclass(frozen=True)
class MetricSpec:
    """A scalar metric to report in the aggregate output."""

    key: str
    label: str


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("parameter_count", "parameters"),
    MetricSpec("train_runtime_seconds", "train runtime (s)"),
    MetricSpec("eval_runtime_seconds", "eval runtime (s)"),
    MetricSpec("mmd", "MMD"),
    MetricSpec("swd", "SWD"),
    MetricSpec("terminal_w1", "terminal W1"),
    MetricSpec("volatility_w1", "volatility W1"),
    MetricSpec("drawdown_w1", "drawdown W1"),
    MetricSpec("jump_count_w1", "jump-count W1"),
    MetricSpec("inter_arrival_w1", "inter-arrival W1"),
    MetricSpec("jump_size_w1", "jump-size W1"),
    MetricSpec("negative_jump_fraction", "negative jump fraction"),
    MetricSpec("var_1pct", "VaR 1%"),
    MetricSpec("es_1pct", "ES 1%"),
    MetricSpec("sampled_active_codes", "sampled active codes"),
    MetricSpec("sampled_perplexity", "sampled perplexity"),
    MetricSpec("transition_l1", "transition L1"),
    MetricSpec("run_length_w1", "run-length W1"),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Aggregate Hawkes/SVMHJD compact prior robustness metrics.",
    )
    parser.add_argument(
        "--robustness-root",
        default="outputs/hawkes_jump_logreturn_robustness",
        help="Root containing additive and k3 three-seed robustness outputs.",
    )
    parser.add_argument(
        "--compact-root",
        default="outputs/hawkes_jump_compact_conv_transformer",
        help="Root containing compact conv-transformer outputs.",
    )
    parser.add_argument(
        "--parameter-counts",
        default="outputs/hawkes_jump_compact_conv_transformer/parameter_counts_final.json",
        help="JSON produced by scripts/report_token_prior_parameter_counts.py.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/hawkes_jump_compact_conv_transformer/aggregate_prior_comparison.json",
        help="Path for aggregate JSON output.",
    )
    parser.add_argument(
        "--csv-output",
        default="outputs/hawkes_jump_compact_conv_transformer/aggregate_prior_runs.csv",
        help="Path for per-seed CSV output.",
    )
    parser.add_argument(
        "--markdown-output",
        default="outputs/hawkes_jump_compact_conv_transformer/aggregate_prior_comparison.md",
        help="Path for aggregate Markdown output.",
    )
    return parser


def main() -> None:
    """Aggregate all configured Hawkes compact-prior runs."""
    args = build_parser().parse_args()
    robustness_root = Path(args.robustness_root)
    compact_root = Path(args.compact_root)
    parameter_counts = load_parameter_counts(Path(args.parameter_counts))
    run_rows = [
        load_run_metrics(run_spec, parameter_counts)
        for run_spec in build_run_specs(robustness_root, compact_root)
    ]
    aggregate_rows = aggregate_by_model(run_rows)

    json_output = Path(args.json_output)
    csv_output = Path(args.csv_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)

    write_json(json_output, {"runs": run_rows, "aggregate": aggregate_rows})
    write_csv(csv_output, run_rows)
    markdown_output.write_text(markdown_table(aggregate_rows), encoding="utf-8")
    print(f"Wrote aggregate JSON: {json_output}")
    print(f"Wrote per-seed CSV: {csv_output}")
    print(f"Wrote aggregate Markdown: {markdown_output}")


def build_run_specs(robustness_root: Path, compact_root: Path) -> list[RunSpec]:
    """Return the fixed run set for additive, k3, and tiny priors."""
    runs: list[RunSpec] = []
    for seed in range(3):
        runs.append(
            RunSpec(
                model="additive_ar",
                seed=seed,
                parameter_name="hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive",
                evaluation_dir=robustness_root / "evaluations" / f"additive_seed{seed}",
                train_runtime_path=robustness_root
                / "priors"
                / "additive"
                / f"hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive_seed{seed}"
                / "runtime_summary.json",
            )
        )
        runs.append(
            RunSpec(
                model="conv_transformer_k3",
                seed=seed,
                parameter_name=(
                    "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer"
                ),
                evaluation_dir=robustness_root / "evaluations" / f"conv_transformer_seed{seed}",
                train_runtime_path=robustness_root
                / "priors"
                / "conv_transformer"
                / (
                    "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_"
                    f"conv_transformer_seed{seed}"
                )
                / "runtime_summary.json",
            )
        )

    tiny_eval_dirs = {
        0: compact_root / "evaluations" / "tiny_seed0",
        1: compact_root / "evaluations" / "tiny_seed1",
        2: compact_root / "evaluations" / "tiny_seed2",
    }
    tiny_train_runtime_paths = {
        0: compact_root
        / "priors"
        / "tiny"
        / "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_tiny_seed0"
        / "runtime_summary.json",
        1: compact_root
        / "priors"
        / "tiny_seed1"
        / "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_tiny_seed1"
        / "runtime_summary.json",
        2: compact_root
        / "priors"
        / "tiny_seed2"
        / "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_tiny_seed2"
        / "runtime_summary.json",
    }
    for seed in range(3):
        runs.append(
            RunSpec(
                model="conv_transformer_tiny",
                seed=seed,
                parameter_name=(
                    "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_tiny"
                ),
                evaluation_dir=tiny_eval_dirs[seed],
                train_runtime_path=tiny_train_runtime_paths[seed],
            )
        )
    return runs


def load_parameter_counts(path: Path) -> dict[str, int]:
    """Load parameter counts keyed by experiment name."""
    rows = cast(Sequence[Mapping[str, Any]], load_json(path))
    return {str(row["name"]): int(row["total_parameters"]) for row in rows}


def load_run_metrics(
    run_spec: RunSpec,
    parameter_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Load one run summary and flatten the metrics used for comparison."""
    summary_path = run_spec.evaluation_dir / "evaluation_summary.json"
    summary = cast(Mapping[str, Any], load_json(summary_path))
    smooth = cast(Mapping[str, Any], summary["smooth_metrics"])
    market = cast(Mapping[str, Any], summary["market_comparison"])
    jump_comparison = cast(Mapping[str, Any], summary["jump_comparison"])
    generated = cast(Mapping[str, Any], summary["jump_diagnostics"])["generated"]
    generated_jump = cast(Mapping[str, Any], generated)
    jump_sizes = cast(Mapping[str, Any], generated_jump["jump_sizes"])
    var_es = cast(Mapping[str, Any], generated_jump["var_es"])
    token = cast(Mapping[str, Any], summary["token_metrics"])
    runtime = cast(Mapping[str, Any] | None, summary.get("runtime_summary"))
    train_runtime = cast(Mapping[str, Any], load_json(run_spec.train_runtime_path))

    return {
        "model": run_spec.model,
        "seed": run_spec.seed,
        "parameter_count": parameter_counts[run_spec.parameter_name],
        "train_runtime_seconds": to_optional_float(train_runtime.get("elapsed_seconds")),
        "eval_runtime_seconds": to_optional_float(
            None if runtime is None else runtime.get("elapsed_seconds")
        ),
        "mmd": float(smooth["mmd"]),
        "swd": float(smooth["swd"]),
        "terminal_w1": float(smooth["terminal_return_wasserstein"]),
        "volatility_w1": float(smooth["volatility_wasserstein"]),
        "drawdown_w1": float(market["maximum_drawdown_wasserstein"]),
        "jump_count_w1": float(jump_comparison["detected_jump_count_wasserstein"]),
        "inter_arrival_w1": float(
            jump_comparison.get("detected_inter_arrival_wasserstein")
            or compute_inter_arrival_w1(run_spec.evaluation_dir / "evaluation_batch.pt")
        ),
        "jump_size_w1": float(jump_comparison["detected_jump_size_wasserstein"]),
        "negative_jump_fraction": float(jump_sizes["negative_jump_fraction"]),
        "var_1pct": float(var_es["lower_tail_var_q01"]),
        "es_1pct": float(var_es["lower_tail_es_q01"]),
        "sampled_active_codes": float(token["sampled_active_code_count"]),
        "sampled_perplexity": float(token["sampled_token_perplexity"]),
        "transition_l1": float(token["transition_matrix_l1"]),
        "run_length_w1": float(token.get("run_length_wasserstein", token["run_length_distance"])),
        "evaluation_summary_path": str(summary_path),
        "train_runtime_path": str(run_spec.train_runtime_path),
    }


def compute_inter_arrival_w1(batch_path: Path) -> float:
    """Compute inter-arrival W1 from saved jump-indicator tensors."""
    batch = torch.load(batch_path, map_location="cpu")
    real_gaps = inter_arrival_gaps(cast(Tensor, batch["real_jumps"]))
    generated_gaps = inter_arrival_gaps(cast(Tensor, batch["generated_jumps"]))
    if real_gaps.numel() == 0 or generated_gaps.numel() == 0:
        return 0.0
    return float(wasserstein_1d(real_gaps, generated_gaps))


def inter_arrival_gaps(jump_indicators: Tensor) -> Tensor:
    """Return concatenated within-path gaps between detected jumps."""
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


def aggregate_by_model(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed rows into mean/std rows."""
    models = sorted({str(row["model"]) for row in run_rows})
    aggregate_rows = []
    for model in models:
        model_rows = [row for row in run_rows if row["model"] == model]
        aggregate: dict[str, Any] = {
            "model": model,
            "seeds": [int(row["seed"]) for row in model_rows],
        }
        for metric in METRICS:
            values = [
                float(row[metric.key])
                for row in model_rows
                if row.get(metric.key) is not None and math.isfinite(float(row[metric.key]))
            ]
            aggregate[f"{metric.key}_mean"] = mean(values)
            aggregate[f"{metric.key}_std"] = std(values)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean or None when no values are available."""
    if not values:
        return None
    return float(statistics.fmean(values))


def std(values: Sequence[float]) -> float | None:
    """Return sample standard deviation for repeated seeds."""
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON document."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write per-seed scalar metrics as CSV."""
    fieldnames = [
        "model",
        "seed",
        *(metric.key for metric in METRICS),
        "evaluation_summary_path",
        "train_runtime_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def markdown_table(aggregate_rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the aggregate comparison as Markdown."""
    lines = [
        "# Hawkes/SVMHJD Compact Prior Aggregate",
        "",
        "| Model | Parameters | Train s | Eval s | MMD | SWD | Jump-count W1 | "
        "Inter-arrival W1 | Jump-size W1 | VaR 1% | ES 1% | Active codes | "
        "Perplexity | Transition L1 | Run-length W1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            f"{row['model']} | "
            f"{fmt_mean_std(row, 'parameter_count', digits=0)} | "
            f"{fmt_mean_std(row, 'train_runtime_seconds')} | "
            f"{fmt_mean_std(row, 'eval_runtime_seconds')} | "
            f"{fmt_mean_std(row, 'mmd')} | "
            f"{fmt_mean_std(row, 'swd')} | "
            f"{fmt_mean_std(row, 'jump_count_w1')} | "
            f"{fmt_mean_std(row, 'inter_arrival_w1')} | "
            f"{fmt_mean_std(row, 'jump_size_w1')} | "
            f"{fmt_mean_std(row, 'var_1pct')} | "
            f"{fmt_mean_std(row, 'es_1pct')} | "
            f"{fmt_mean_std(row, 'sampled_active_codes', digits=1)} | "
            f"{fmt_mean_std(row, 'sampled_perplexity')} | "
            f"{fmt_mean_std(row, 'transition_l1')} | "
            f"{fmt_mean_std(row, 'run_length_w1')} |"
        )
    return "\n".join(lines) + "\n"


def fmt_mean_std(row: Mapping[str, Any], key: str, *, digits: int = 4) -> str:
    """Format a mean and standard deviation cell."""
    mean_value = row.get(f"{key}_mean")
    std_value = row.get(f"{key}_std")
    if mean_value is None:
        return "n/a"
    mean_text = format_number(float(mean_value), digits=digits)
    if std_value is None:
        return mean_text
    return f"{mean_text} ± {format_number(float(std_value), digits=digits)}"


def format_number(value: float, *, digits: int) -> str:
    """Format scalar values compactly for Markdown."""
    if digits == 0:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def to_optional_float(value: object) -> float | None:
    """Convert a JSON scalar to float if present."""
    if value is None:
        return None
    return float(value)


def load_json(path: Path) -> Any:
    """Load a JSON document from disk."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    """Return finite values for a metric key."""
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


if __name__ == "__main__":
    main()
