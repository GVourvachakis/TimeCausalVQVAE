"""Score Hawkes/SVMHJD additive AR against the conv-transformer k3 prior."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ADDITIVE = "hidden128_logreturn_cb64_additive_ar"
CONV = "hidden128_logreturn_cb64_conv_transformer_k3"
CANDIDATES = (ADDITIVE, CONV)

AGGREGATE_MODEL_NAMES = {
    ADDITIVE: "cb64_additive_ar",
    CONV: "cb64_conv_transformer_k3",
}

DEFAULT_AGGREGATES = (
    Path("outputs/hawkes_jump_continuous_ablation_comparison/aggregate_summary.json"),
    Path("outputs/hawkes_jump_matched_continuous_comparison/aggregate_summary.json"),
)

SMOOTH_METRICS = (
    "mmd",
    "swd",
    "terminal_w1",
    "volatility_w1",
    "drawdown_w1",
)
JUMP_TIMING_METRICS = (
    "jump_count_w1",
    "inter_arrival_w1",
    "jump_size_w1",
)
TAIL_METRICS = (
    "var_01_error",
    "es_01_error",
    "negative_jump_fraction_error",
)
TOKEN_SCORE_METRICS = (
    "transition_matrix_l1",
    "run_length_w1",
)
TOKEN_DESCRIPTIVE_METRICS = ("sampled_code_perplexity",)

PROFILE_METRICS = {
    "smooth_profile": SMOOTH_METRICS,
    "jump_timing_profile": JUMP_TIMING_METRICS,
    "tail_profile": TAIL_METRICS,
    "token_profile": TOKEN_SCORE_METRICS,
}

METRIC_LABELS = {
    "mmd": "MMD",
    "swd": "SWD",
    "terminal_w1": "Terminal W1",
    "volatility_w1": "Volatility W1",
    "drawdown_w1": "Drawdown W1",
    "jump_count_w1": "Jump-count W1",
    "inter_arrival_w1": "Inter-arrival W1",
    "jump_size_w1": "Jump-size W1",
    "var_01": "VaR 1%",
    "es_01": "ES 1%",
    "var_01_error": "Abs VaR 1% error",
    "es_01_error": "Abs ES 1% error",
    "negative_jump_fraction": "Negative jump fraction",
    "negative_jump_fraction_error": "Abs negative jump fraction error",
    "transition_matrix_l1": "Transition L1",
    "run_length_w1": "Run-length W1",
    "sampled_code_perplexity": "Sampled perplexity",
}

FALLBACK_STATS = {
    ADDITIVE: {
        "mmd": (0.1567, 0.0644, 3),
        "swd": (0.0238, 0.0085, 3),
        "terminal_w1": (0.0320, 0.0152, 3),
        "volatility_w1": (0.0011, 0.0008, 3),
        "drawdown_w1": (0.0106, 0.0059, 3),
        "jump_count_w1": (0.0469, 0.0319, 3),
        "inter_arrival_w1": (6.3080, 4.2239, 3),
        "jump_size_w1": (0.0180, 0.0101, 3),
        "var_01": (-0.0745, 0.0028, 3),
        "es_01": (-0.1068, 0.0080, 3),
        "negative_jump_fraction": (0.9955, 0.0078, 3),
        "var_01_error": (0.00284186, 0.00209010, 3),
        "es_01_error": (0.00730964, 0.00344801, 3),
        "negative_jump_fraction_error": (0.03013245, 0.01745000, 3),
        "transition_matrix_l1": (0.4341, 0.0153, 3),
        "run_length_w1": (0.0031, 0.0033, 3),
        "sampled_code_perplexity": (44.36, 0.70, 3),
    },
    CONV: {
        "mmd": (0.1141, 0.0355, 3),
        "swd": (0.0186, 0.0060, 3),
        "terminal_w1": (0.0217, 0.0120, 3),
        "volatility_w1": (0.0010, 0.0010, 3),
        "drawdown_w1": (0.0111, 0.0052, 3),
        "jump_count_w1": (0.0576, 0.0324, 3),
        "inter_arrival_w1": (8.1888, 6.7270, 3),
        "jump_size_w1": (0.0177, 0.0101, 3),
        "var_01": (-0.0748, 0.0026, 3),
        "es_01": (-0.1069, 0.0080, 3),
        "negative_jump_fraction": (0.9989, 0.0019, 3),
        "var_01_error": (0.00275595, 0.00216408, 3),
        "es_01_error": (0.00618131, 0.00542595, 3),
        "negative_jump_fraction_error": (0.03355212, 0.01308300, 3),
        "transition_matrix_l1": (0.4345, 0.0177, 3),
        "run_length_w1": (0.0037, 0.0022, 3),
        "sampled_code_perplexity": (44.42, 0.80, 3),
    },
}


@dataclass(frozen=True)
class MetricStats:
    """Mean, optional standard deviation, and sample count for a metric."""

    mean: float
    std: float | None = None
    n: int | None = None


@dataclass
class SelectionInput:
    """Loaded candidate metrics and source metadata."""

    source: str
    metrics: dict[str, dict[str, MetricStats]] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Score Hawkes/SVMHJD additive AR against conv-transformer k3.",
    )
    parser.add_argument(
        "--aggregate",
        action="append",
        type=Path,
        help=(
            "Aggregate summary JSON to read. Can be passed more than once. "
            "Defaults to the local Hawkes continuous-ablation then matched-comparison outputs."
        ),
    )
    return parser


def main() -> int:
    """Run the scoring command."""
    args = build_parser().parse_args()
    selection_input = load_selection_input(tuple(args.aggregate or DEFAULT_AGGREGATES))
    print(render_report(selection_input))
    return 0


def load_selection_input(aggregate_paths: tuple[Path, ...]) -> SelectionInput:
    """Load aggregate JSON candidate rows or fall back to hard-coded model-card rows."""
    for aggregate_path in aggregate_paths:
        if aggregate_path.exists():
            return _load_from_aggregate(aggregate_path)
    return _load_fallback()


def render_report(selection_input: SelectionInput) -> str:
    """Render a markdown scoring report."""
    profile_scores = score_profiles(selection_input.metrics)
    balanced_scores = score_balanced_profile(profile_scores)
    lines = [
        "# Hawkes/SVMHJD additive-vs-conv prior selection score",
        "",
        f"Source: {selection_input.source}",
        "",
        "## Metric table",
        "",
        render_metric_table(
            selection_input.metrics,
            (*SMOOTH_METRICS, *JUMP_TIMING_METRICS, "var_01", "es_01", "negative_jump_fraction"),
        ),
        "",
        "## Tail reference-error table",
        "",
        render_metric_table(selection_input.metrics, TAIL_METRICS),
        "",
        "## Token-profile comparison",
        "",
        render_metric_table(
            selection_input.metrics,
            (*TOKEN_SCORE_METRICS, *TOKEN_DESCRIPTIVE_METRICS),
        ),
        "",
        "## Profile ranks",
        "",
        render_profile_table(profile_scores, balanced_scores),
        "",
        "## Decision",
        "",
        decision_text(profile_scores, balanced_scores),
    ]
    return "\n".join(lines)


def score_profiles(
    candidate_metrics: dict[str, dict[str, MetricStats]],
) -> dict[str, dict[str, float]]:
    """Score lower-is-better metric profiles by average rank."""
    return {
        profile_name: average_rank(candidate_metrics, metrics)
        for profile_name, metrics in PROFILE_METRICS.items()
    }


def score_balanced_profile(profile_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """Score the balanced profile from smooth, jump-timing, and tail ranks."""
    components = ("smooth_profile", "jump_timing_profile", "tail_profile")
    return {
        candidate: statistics.fmean(
            profile_scores[component][candidate] for component in components
        )
        for candidate in CANDIDATES
    }


def average_rank(
    candidate_metrics: dict[str, dict[str, MetricStats]],
    metrics: tuple[str, ...],
) -> dict[str, float]:
    """Compute average candidate rank across lower-is-better metrics."""
    metric_ranks = [rank_metric(candidate_metrics, metric) for metric in metrics]
    return {
        candidate: statistics.fmean(ranks[candidate] for ranks in metric_ranks)
        for candidate in CANDIDATES
    }


def rank_metric(
    candidate_metrics: dict[str, dict[str, MetricStats]],
    metric: str,
) -> dict[str, float]:
    """Rank candidates by a single lower-is-better metric, with average ranks for ties."""
    values = [(candidate, candidate_metrics[candidate][metric].mean) for candidate in CANDIDATES]
    sorted_values = sorted(values, key=lambda item: item[1])
    ranks: dict[str, float] = {}
    index = 0
    while index < len(sorted_values):
        tied = [sorted_values[index]]
        next_index = index + 1
        while next_index < len(sorted_values) and sorted_values[next_index][1] == tied[0][1]:
            tied.append(sorted_values[next_index])
            next_index += 1
        average_tie_rank = statistics.fmean(range(index + 1, next_index + 1))
        for candidate, _value in tied:
            ranks[candidate] = average_tie_rank
        index = next_index
    return ranks


def winner(scores: dict[str, float]) -> str:
    """Return the winning candidate name or 'tie'."""
    ordered = sorted(scores.items(), key=lambda item: item[1])
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return "tie"
    return ordered[0][0]


def decision_text(
    profile_scores: dict[str, dict[str, float]],
    balanced_scores: dict[str, float],
) -> str:
    """Render the explicit selection decision."""
    smooth_winner = winner(profile_scores["smooth_profile"])
    jump_winner = winner(profile_scores["jump_timing_profile"])
    tail_winner = winner(profile_scores["tail_profile"])
    token_winner = winner(profile_scores["token_profile"])
    balanced_winner = winner(balanced_scores)
    overall = CONV if balanced_winner == CONV or smooth_winner == CONV else ADDITIVE
    lines = [
        f"- Smooth-profile winner: `{smooth_winner}`.",
        f"- Jump-timing-profile winner: `{jump_winner}`.",
        f"- Tail-profile winner: `{tail_winner}`.",
        f"- Token-profile rank winner: `{token_winner}`; sampled perplexity is descriptive.",
        f"- Balanced-profile winner: `{balanced_winner}`.",
        f"- Overall selected research candidate: `{overall}`.",
        f"- Jump-count/inter-arrival specialised candidate: `{ADDITIVE}`.",
    ]
    if overall == CONV:
        lines.append(
            "- Decision: keep the conv-transformer k3 prior as the selected research candidate; "
            "retain additive AR as the required sparse-jump ablation."
        )
    else:
        lines.append(
            "- Decision: switch to additive AR only for applications that prioritise "
            "jump timing over smooth-path fidelity."
        )
    return "\n".join(lines)


def render_metric_table(
    candidate_metrics: dict[str, dict[str, MetricStats]],
    metrics: tuple[str, ...],
) -> str:
    """Render candidate metric means and standard deviations as a markdown table."""
    headers = ["Candidate", *(METRIC_LABELS[metric] for metric in metrics)]
    rows = [
        [candidate, *(format_stats(candidate_metrics[candidate][metric]) for metric in metrics)]
        for candidate in CANDIDATES
    ]
    return render_table(headers, rows)


def render_profile_table(
    profile_scores: dict[str, dict[str, float]],
    balanced_scores: dict[str, float],
) -> str:
    """Render profile rank scores as a markdown table."""
    headers = [
        "Candidate",
        "Smooth rank",
        "Jump-timing rank",
        "Tail rank",
        "Token rank",
        "Balanced rank",
    ]
    rows = [
        [
            candidate,
            f"{profile_scores['smooth_profile'][candidate]:.3f}",
            f"{profile_scores['jump_timing_profile'][candidate]:.3f}",
            f"{profile_scores['tail_profile'][candidate]:.3f}",
            f"{profile_scores['token_profile'][candidate]:.3f}",
            f"{balanced_scores[candidate]:.3f}",
        ]
        for candidate in CANDIDATES
    ]
    return render_table(headers, rows)


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple markdown table."""
    alignment = ["---", *(["---:"] * (len(headers) - 1))]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def format_stats(stats: MetricStats) -> str:
    """Format mean and standard deviation."""
    if stats.std is None:
        return f"{stats.mean:.6g}"
    return f"{stats.mean:.6g} / {stats.std:.6g}"


def _load_from_aggregate(aggregate_path: Path) -> SelectionInput:
    """Load candidate statistics from an aggregate JSON file."""
    raw = _load_json(aggregate_path)
    by_model = _expect_mapping(raw.get("by_model"), "by_model")
    rows = raw.get("rows", [])
    metrics: dict[str, dict[str, MetricStats]] = {}
    for candidate in CANDIDATES:
        aggregate_name = AGGREGATE_MODEL_NAMES[candidate]
        model_stats = _expect_mapping(by_model.get(aggregate_name), aggregate_name)
        metrics[candidate] = _stats_from_aggregate_model(model_stats)
        metrics[candidate].update(_tail_error_stats_from_rows(rows, aggregate_name))
    return SelectionInput(source=str(aggregate_path), metrics=metrics)


def _load_fallback() -> SelectionInput:
    """Load hard-coded model-card metric rows."""
    metrics = {
        candidate: {
            metric: MetricStats(mean=values[0], std=values[1], n=values[2])
            for metric, values in stats.items()
        }
        for candidate, stats in FALLBACK_STATS.items()
    }
    return SelectionInput(source="hard-coded model-card rows", metrics=metrics)


def _stats_from_aggregate_model(model_stats: dict[str, Any]) -> dict[str, MetricStats]:
    """Convert aggregate JSON model metrics to the canonical metric names."""
    key_map = {
        "mmd": "mmd",
        "swd": "swd",
        "terminal_w1": "terminal_w1",
        "volatility_w1": "volatility_w1",
        "drawdown_w1": "drawdown_w1",
        "jump_count_w1": "jump_count_w1",
        "inter_arrival_w1": "inter_arrival_w1",
        "jump_size_w1": "jump_size_w1",
        "var_01": "var_01",
        "es_01": "es_01",
        "negative_jump_fraction": "negative_jump_fraction",
        "transition_matrix_l1": "transition_matrix_l1",
        "run_length_w1": "run_length_w1",
        "sampled_code_perplexity": "sampled_code_perplexity",
    }
    return {
        canonical_key: _metric_stats(_expect_mapping(model_stats[aggregate_key], aggregate_key))
        for canonical_key, aggregate_key in key_map.items()
    }


def _tail_error_stats_from_rows(rows: Any, aggregate_name: str) -> dict[str, MetricStats]:
    """Compute per-seed tail errors against matched real references."""
    if not isinstance(rows, list):
        return {}
    errors: dict[str, list[float]] = {
        "var_01_error": [],
        "es_01_error": [],
        "negative_jump_fraction_error": [],
    }
    for row in rows:
        if not isinstance(row, dict) or row.get("model") != aggregate_name:
            continue
        summary_path = row.get("summary_path")
        if not isinstance(summary_path, str) or not Path(summary_path).exists():
            continue
        reference = _real_tail_reference(Path(summary_path))
        errors["var_01_error"].append(abs(float(row["var_01"]) - reference["var_01"]))
        errors["es_01_error"].append(abs(float(row["es_01"]) - reference["es_01"]))
        errors["negative_jump_fraction_error"].append(
            abs(float(row["negative_jump_fraction"]) - reference["negative_jump_fraction"]),
        )
    if any(not values for values in errors.values()):
        return {}
    return {metric: _stats_from_values(values) for metric, values in errors.items()}


def _real_tail_reference(summary_path: Path) -> dict[str, float]:
    """Read real VaR/ES and negative-jump fraction from one evaluation summary."""
    raw = _load_json(summary_path)
    real = _expect_mapping(
        _expect_mapping(raw["jump_diagnostics"], "jump_diagnostics")["real"], "real"
    )
    var_es = _expect_mapping(real["var_es"], "var_es")
    jump_sizes = _expect_mapping(real["jump_sizes"], "jump_sizes")
    return {
        "var_01": float(var_es["lower_tail_var_q01"]),
        "es_01": float(var_es["lower_tail_es_q01"]),
        "negative_jump_fraction": float(jump_sizes["negative_jump_fraction"]),
    }


def _metric_stats(raw_metric: dict[str, Any]) -> MetricStats:
    """Return metric statistics from an aggregate metric entry."""
    raw_std = raw_metric.get("std")
    return MetricStats(
        mean=float(raw_metric["mean"]),
        std=None if raw_std is None else float(raw_std),
        n=None if raw_metric.get("n") is None else int(raw_metric["n"]),
    )


def _stats_from_values(values: list[float]) -> MetricStats:
    """Return sample mean and sample standard deviation for values."""
    std = statistics.stdev(values) if len(values) > 1 else None
    return MetricStats(mean=statistics.fmean(values), std=std, n=len(values))


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return _expect_mapping(raw, str(path))


def _expect_mapping(value: Any, name: str) -> dict[str, Any]:
    """Validate that a value is a JSON object."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected {name} to be a JSON object.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
