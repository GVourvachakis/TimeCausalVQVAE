"""Run S&P500/VIX signature-conditioning token-prior ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

DEFAULT_WANDB_PROJECT = "time-causal-token-prior"
DEFAULT_WANDB_ENTITY = "tc_vae"
DEFAULT_TOKENIZER_DIR = "outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0"
DEFAULT_CONTINUOUS_CONFIG = "configs/experiments/sp500_vix_beta_cvae.yaml"
DEFAULT_CONTINUOUS_MODEL_DIR = "outputs/sp500_vix_continuous/final_model_unavailable"
MODEL_SELECTION_PROFILE_DOC = "docs/architecture/model_selection_profiles.md"
WANDB_RUN_URL_PATTERN = re.compile(r"https://wandb\.ai/[^\s]+/runs/[A-Za-z0-9_-]+")
WANDB_LIVE_ENV_DEFAULTS = {
    "MPLBACKEND": "Agg",
    "WANDB_DISABLE_SERVICE": "true",
    "WANDB_START_METHOD": "thread",
}
WANDB_FAILURE_HINT = (
    "W&B-enabled run failed. This runner keeps live W&B mode explicit and does "
    "not silently switch to WANDB_MODE=offline. If the failure is caused by "
    "socket, Tcl/Tk, or upstream CommError restrictions, rerun with --no-wandb "
    "and preserve the local JSON/CSV outputs."
)
PROFILE_RANK_METRICS = {
    "distributional": (
        "paper_mmd",
        "paper_swd",
        "paper_returns_wasserstein",
    ),
    "tail_risk": (
        "paper_terminal_return_wasserstein",
        "paper_volatility_wasserstein",
        "paper_maximum_drawdown_wasserstein",
    ),
    "sequential_dependence": (
        "paper_return_autocorrelation_l1",
        "paper_squared_return_autocorrelation_l1",
    ),
    "balanced_market": (
        "paper_swd",
        "paper_returns_wasserstein",
        "paper_terminal_return_wasserstein",
        "paper_volatility_wasserstein",
        "paper_maximum_drawdown_wasserstein",
        "paper_return_autocorrelation_l1",
        "paper_squared_return_autocorrelation_l1",
    ),
}
PAPER_STYLE_METRIC_KEYS = {
    "mmd": "paper_mmd",
    "swd": "paper_swd",
    "returns_wasserstein": "paper_returns_wasserstein",
    "terminal_return_wasserstein": "paper_terminal_return_wasserstein",
    "volatility_wasserstein": "paper_volatility_wasserstein",
    "maximum_drawdown_wasserstein": "paper_maximum_drawdown_wasserstein",
    "return_autocorrelation_l1": "paper_return_autocorrelation_l1",
    "squared_return_autocorrelation_l1": "paper_squared_return_autocorrelation_l1",
    "flattened_return_autocorrelation_l1": "paper_flattened_return_autocorrelation_l1",
    "flattened_squared_return_autocorrelation_l1": (
        "paper_flattened_squared_return_autocorrelation_l1"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the ablation-runner argument parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate S&P500/VIX signature-conditioning ablations.",
    )
    parser.add_argument("--configs", nargs="+", required=True, help="Token-prior YAML configs.")
    parser.add_argument("--output-dir", required=True, help="Base output directory under outputs/.")
    parser.add_argument("--base-data-dir", default="data/processed")
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--n-sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", help="Optional train/eval device override.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B for training runs.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B.")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument(
        "--run-paper-style",
        action="store_true",
        help="Run paper-style diagnostics after best-checkpoint evaluation.",
    )
    parser.add_argument("--continuous-config", default=DEFAULT_CONTINUOUS_CONFIG)
    parser.add_argument("--continuous-model-dir", default=DEFAULT_CONTINUOUS_MODEL_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/model wiring without training checkpoints or evaluation.",
    )
    return parser


def main() -> None:
    """Run the ablation coordinator."""
    args = build_parser().parse_args()
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_args(args)

    rows: list[dict[str, Any]] = []
    for config_path in [Path(path) for path in args.configs]:
        row = run_one_config(config_path, args=args, output_dir=output_dir)
        rows.append(row)
        write_aggregate_outputs(output_dir, rows)

    print("Signature-conditioning ablation runner complete.")
    print(f"output_dir: {output_dir}")
    for row in rows:
        print(
            f"{row['experiment_name']}: status={row['status']} "
            f"condition_dim={row['condition_dim']} feature_dim={row['feature_dimension']}"
        )


def validate_args(args: argparse.Namespace) -> None:
    """Validate scalar runner arguments."""
    if args.epochs is not None and args.epochs <= 0:
        raise SystemExit("--epochs must be positive when provided.")
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive.")
    if args.top_k is not None and args.top_k <= 0:
        raise SystemExit("--top-k must be positive when provided.")
    if args.wandb and args.no_wandb:
        raise SystemExit("--wandb and --no-wandb are mutually exclusive.")


def validate_output_dir(output_dir: str) -> Path:
    """Ensure generated ablation artifacts stay below ignored outputs/."""
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


def run_one_config(
    config_path: Path,
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one config through dry-run validation or train/evaluate steps."""
    raw_config = load_yaml(config_path)
    experiment = require_mapping(raw_config, "experiment")
    data = require_mapping(raw_config, "data")
    model = require_mapping(raw_config, "model")
    experiment_name = str(experiment["name"])
    seed = int(experiment.get("seed", 0))
    run_dir = output_dir / experiment_name
    train_output_dir = run_dir / "train"
    eval_output_dir = run_dir / "evaluation_best"
    paper_output_dir = run_dir / "paper_style"
    feature_dir = Path(str(data.get("condition_feature_dir", "")))
    feature_summary = load_feature_summary(feature_dir)
    condition_dim = int(model["condition_dim"])
    feature_dimension = int(feature_summary["feature_dimension"])
    expected_condition_dim = 1 + feature_dimension
    if condition_dim != expected_condition_dim:
        raise SystemExit(
            f"{config_path} sets condition_dim={condition_dim}, but feature summary "
            f"{feature_dir / 'signature_feature_summary.json'} implies "
            f"{expected_condition_dim}."
        )

    row: dict[str, Any] = {
        "config_path": str(config_path),
        "experiment_name": experiment_name,
        "seed": seed,
        "status": "pending",
        "condition_feature_dir": str(feature_dir),
        "feature_dimension": feature_dimension,
        "condition_dim": condition_dim,
        "context_length": int(feature_summary["context_length"]),
        "depth": int(feature_summary["depth"]),
        "lead_lag": bool(feature_summary["lead_lag"]),
        "include_time": bool(feature_summary["include_time"]),
        "include_vix": bool(feature_summary["include_vix"]),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "model_selection_profile_source": MODEL_SELECTION_PROFILE_DOC,
    }

    train_command = build_train_command(
        config_path,
        train_output_dir=train_output_dir,
        args=args,
        experiment_name=experiment_name,
    )
    if args.dry_run:
        train_command.append("--dry-run")
        result = run_command(train_command, env=wandb_compatible_env(args))
        row.update(
            {
                "status": "dry_run_validated",
                "train_returncode": result.returncode,
                "train_stdout_tail": tail_text(result.stdout),
                "train_stderr_tail": tail_text(result.stderr),
                "wandb_run_url": extract_wandb_run_url(result.stdout + "\n" + result.stderr),
                "profile_scores_available": False,
            }
        )
        return row

    train_result = run_command(train_command, env=wandb_compatible_env(args))
    best_model_dir = train_output_dir / f"{experiment_name}_seed{seed}" / "best_model"
    if not best_model_dir.exists():
        raise SystemExit(f"Expected best-model directory was not created: {best_model_dir}")

    eval_command = build_eval_command(
        config_path,
        prior_dir=best_model_dir,
        output_dir=eval_output_dir,
        args=args,
    )
    eval_result = run_command(eval_command, env=wandb_compatible_env(args))
    eval_summary = load_json(eval_output_dir / "token_prior_summary.json")
    metrics = cast(Mapping[str, Any], eval_summary["metrics"])
    row.update(
        {
            "status": "completed",
            "train_returncode": train_result.returncode,
            "eval_returncode": eval_result.returncode,
            "wandb_run_url": extract_wandb_run_url(
                train_result.stdout + "\n" + train_result.stderr
            ),
            "best_model_dir": str(best_model_dir),
            "evaluation_dir": str(eval_output_dir),
            "mmd": float(metrics["mmd"]),
            "swd": float(metrics["swd"]),
            "terminal_return_wasserstein": float(metrics["terminal_return_wasserstein"]),
            "volatility_wasserstein": float(metrics["volatility_wasserstein"]),
            "sampled_token_perplexity": float(metrics["sampled_token_codebook_perplexity"]),
        }
    )

    if args.run_paper_style:
        paper_result = run_command(
            build_paper_style_command(
                config_path,
                prior_dir=best_model_dir,
                output_dir=paper_output_dir,
                args=args,
            ),
            env=wandb_compatible_env(args),
        )
        row["paper_style_returncode"] = paper_result.returncode
        row["paper_style_dir"] = str(paper_output_dir)
        paper_summary_path = paper_output_dir / "paper_style_summary.json"
        if paper_summary_path.exists():
            row.update(extract_paper_style_metrics(load_json(paper_summary_path)))
            row["profile_scores_available"] = True

    return row


def build_train_command(
    config_path: Path,
    *,
    train_output_dir: Path,
    args: argparse.Namespace,
    experiment_name: str,
) -> list[str]:
    """Build a token-prior training command."""
    command = [
        "poetry",
        "run",
        "tcvae-train-token-prior",
        "--config",
        str(config_path),
        "--output-dir",
        str(train_output_dir),
    ]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.device:
        command.extend(["--device", str(args.device)])
    if args.wandb and not args.no_wandb:
        command.extend(
            [
                "--wandb",
                "--wandb-project",
                str(args.wandb_project),
                "--wandb-entity",
                str(args.wandb_entity),
                "--wandb-run-name",
                experiment_name,
            ]
        )
    else:
        command.append("--no-wandb")
    return command


def build_eval_command(
    config_path: Path,
    *,
    prior_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build a best-checkpoint evaluation command."""
    command = [
        "poetry",
        "run",
        "tcvae-evaluate-token-prior",
        "--config",
        str(config_path),
        "--prior-dir",
        str(prior_dir),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
        "--output-dir",
        str(output_dir),
        "--base-data-dir",
        str(args.base_data_dir),
        "--n-sample",
        str(args.n_sample),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
    ]
    if args.top_k is not None:
        command.extend(["--top-k", str(args.top_k)])
    if args.device:
        command.extend(["--device", str(args.device)])
    return command


def build_paper_style_command(
    config_path: Path,
    *,
    prior_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build a paper-style diagnostics command."""
    command = [
        "poetry",
        "run",
        "python",
        "scripts/evaluate_sp500_vix_paper_style.py",
        "--discrete-config",
        str(config_path),
        "--discrete-prior-dir",
        str(prior_dir),
        "--discrete-tokenizer-dir",
        str(args.tokenizer_dir),
        "--continuous-config",
        str(args.continuous_config),
        "--continuous-model-dir",
        str(args.continuous_model_dir),
        "--output-dir",
        str(output_dir),
        "--base-data-dir",
        str(args.base_data_dir),
        "--n-sample",
        str(args.n_sample),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
    ]
    if args.top_k is not None:
        command.extend(["--top-k", str(args.top_k)])
    return command


def wandb_compatible_env(args: argparse.Namespace) -> dict[str, str]:
    """Return environment variables for W&B in this socket-restricted sandbox."""
    env = dict(os.environ)
    if args.wandb and not args.no_wandb:
        env.pop("WANDB_MODE", None)
        for key, value in WANDB_LIVE_ENV_DEFAULTS.items():
            env.setdefault(key, value)
        env.setdefault("WANDB_PROJECT", str(args.wandb_project))
        env.setdefault("WANDB_ENTITY", str(args.wandb_entity))
    return env


def run_command(command: list[str], *, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    """Run one subprocess and echo a compact command header."""
    print("+ " + " ".join(command), flush=True)
    start = time.perf_counter()
    result = subprocess.run(
        command,
        check=False,
        cwd=Path.cwd(),
        env=dict(env),
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - start
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        wandb_hint = f"\n{WANDB_FAILURE_HINT}" if "--wandb" in command else ""
        raise SystemExit(
            f"Command failed with exit code {result.returncode} after {elapsed:.3f}s: "
            + " ".join(command)
            + wandb_hint
        )
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config as a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required config mapping."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def load_feature_summary(feature_dir: Path) -> dict[str, Any]:
    """Load the signature feature summary for one ablation."""
    summary_path = feature_dir / "signature_feature_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing signature feature summary: {summary_path}")
    summary = load_json(summary_path)
    if summary.get("status") != "success":
        raise SystemExit(f"Signature feature summary is not successful: {summary_path}")
    return summary


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file as a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"JSON file must contain a mapping: {path}")
    return cast(dict[str, Any], loaded)


def write_aggregate_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write aggregate ablation results as JSON and CSV."""
    annotated_rows = annotate_profile_rank_scores(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ablation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_selection_profiles": {
                    "source": MODEL_SELECTION_PROFILE_DOC,
                    "rank_score_convention": (
                        "Lower average rank is better. Scores are only populated "
                        "when --run-paper-style writes the required scalar metrics."
                    ),
                    "tail_penalty": (
                        "Not applied in the runner; inspect tail exceedance fields "
                        "from the paper-style summary separately."
                    ),
                },
                "runs": annotated_rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    csv_path = output_dir / "ablation_results.csv"
    fieldnames = sorted({key for row in annotated_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated_rows)


def extract_paper_style_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Extract discrete paper-style scalar metrics for aggregate reporting."""
    comparisons = summary.get("comparisons")
    if not isinstance(comparisons, dict):
        return {"paper_style_metrics_available": False}
    discrete = comparisons.get("discrete")
    if not isinstance(discrete, dict):
        return {"paper_style_metrics_available": False}

    metrics: dict[str, Any] = {"paper_style_metrics_available": True}
    for input_key, output_key in PAPER_STYLE_METRIC_KEYS.items():
        value = discrete.get(input_key)
        if isinstance(value, int | float):
            metrics[output_key] = float(value)

    tail_rates = discrete.get("tail_exceedance_rates")
    if isinstance(tail_rates, dict):
        for key, value in tail_rates.items():
            if isinstance(value, int | float):
                metrics[f"paper_tail_{key}"] = float(value)
    return metrics


def annotate_profile_rank_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add profile rank-score summaries where enough paper-style metrics exist."""
    annotated = [dict(row) for row in rows]
    for profile, metric_keys in PROFILE_RANK_METRICS.items():
        complete_rows = [
            row for row in annotated if all(is_finite_number(row.get(key)) for key in metric_keys)
        ]
        if not complete_rows:
            continue

        rank_maps: dict[str, dict[str, int]] = {}
        for metric_key in metric_keys:
            ordered = sorted(
                complete_rows,
                key=lambda row, key=metric_key: (
                    float(row[key]),
                    str(row["experiment_name"]),
                ),
            )
            rank_maps[metric_key] = {
                str(row["experiment_name"]): rank for rank, row in enumerate(ordered, start=1)
            }

        for row in complete_rows:
            experiment_name = str(row["experiment_name"])
            ranks = [rank_maps[key][experiment_name] for key in metric_keys]
            row[f"{profile}_profile_rank_score"] = sum(ranks) / len(ranks)
            row[f"{profile}_profile_rank_components"] = ",".join(metric_keys)
    return annotated


def is_finite_number(value: Any) -> bool:
    """Return whether a value is a finite scalar number."""
    return isinstance(value, int | float) and math.isfinite(float(value))


def extract_wandb_run_url(text: str) -> str | None:
    """Extract the last W&B cloud run URL emitted by a subprocess."""
    matches = WANDB_RUN_URL_PATTERN.findall(text)
    return matches[-1] if matches else None


def tail_text(text: str, *, n_lines: int = 20) -> str:
    """Return the last lines of captured subprocess text."""
    lines = text.strip().splitlines()
    return "\n".join(lines[-n_lines:])


if __name__ == "__main__":
    main()
