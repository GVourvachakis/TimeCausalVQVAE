"""Run S&P500/VIX conditional token-prior ablations through package CLIs."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

SUMMARY_FIELDS = [
    "config",
    "experiment_name",
    "seed",
    "run_dir",
    "final_evaluation_dir",
    "best_evaluation_dir",
    "dry_run",
    "train_status",
    "final_eval_status",
    "best_eval_status",
    "runtime_seconds",
    "best_epoch",
    "best_eval_cross_entropy",
    "best_eval_accuracy",
    "best_eval_perplexity",
    "final_eval_cross_entropy",
    "final_eval_accuracy",
    "final_eval_perplexity",
    "final_mmd",
    "final_swd",
    "final_volatility_wasserstein",
    "final_terminal_return_wasserstein",
    "best_mmd",
    "best_swd",
    "best_volatility_wasserstein",
    "best_terminal_return_wasserstein",
    "best_sampled_active_codes",
    "best_sampled_token_perplexity",
    "best_very_low_mmd",
    "best_low_mmd",
    "wandb_enabled",
    "wandb_project",
    "wandb_entity",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the S&P500/VIX conditional token-prior ablation parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate S&P500/VIX conditional token-prior ablations.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="One or more S&P500/VIX conditional token-prior configs.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory under outputs/.")
    parser.add_argument("--base-data-dir", default="data/processed", help="Market data root.")
    parser.add_argument("--epochs", type=int, help="Epoch override forwarded to training.")
    parser.add_argument("--n-sample", type=int, default=1000, help="Evaluation sample count.")
    parser.add_argument("--seed", type=int, help="Evaluation seed forwarded to evaluator.")
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate commands without training."
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B for training runs.")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B for all runs. Takes precedence over --wandb.",
    )
    parser.add_argument(
        "--wandb-project",
        default="time-causal-token-prior",
        help="W&B project for non-smoke runs.",
    )
    parser.add_argument("--wandb-entity", default="tc_vae", help="W&B entity.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode forwarded to training.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="sp500_vix_conditional_prior_ablation",
        help="Prefix for W&B run names.",
    )
    parser.add_argument(
        "--use-best-sampling",
        action="store_true",
        help="Evaluate with the best sampling parameters from the sampling ablation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used when --use-best-sampling is enabled.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k used when --use-best-sampling is enabled.",
    )
    return parser


def main() -> None:
    """Run all requested ablation configs and write aggregate summaries."""
    args = build_parser().parse_args()
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for config in [Path(value) for value in args.configs]:
        result = run_one_config(config, args=args, output_dir=output_dir)
        results.append(result)
        write_summaries(output_dir, results)
    print(f"wrote: {output_dir / 'sp500_vix_conditional_token_prior_ablation_summary.json'}")
    print(f"wrote: {output_dir / 'sp500_vix_conditional_token_prior_ablation_summary.csv'}")


def run_one_config(
    config_path: Path,
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate one config, or perform a dry run."""
    raw_config = load_yaml(config_path)
    experiment = require_mapping(raw_config, "experiment")
    training_seed = int(experiment.get("seed", 0))
    experiment_name = str(experiment["name"])
    run_dir = output_dir / f"{experiment_name}_seed{training_seed}"
    final_evaluation_dir = output_dir / f"{experiment_name}_seed{training_seed}_evaluation_final"
    best_evaluation_dir = output_dir / f"{experiment_name}_seed{training_seed}_evaluation_best"
    start_time = time.perf_counter()

    train_command = build_train_command(
        config_path=config_path,
        output_dir=output_dir,
        args=args,
        experiment_name=experiment_name,
        seed=training_seed,
    )
    final_eval_command = build_evaluation_command(
        raw_config=raw_config,
        config_path=config_path,
        prior_dir=run_dir,
        evaluation_dir=final_evaluation_dir,
        args=args,
    )
    best_eval_command = build_evaluation_command(
        raw_config=raw_config,
        config_path=config_path,
        prior_dir=run_dir / "best_model",
        evaluation_dir=best_evaluation_dir,
        args=args,
    )

    print(f"$ {shlex.join(train_command)}", flush=True)
    subprocess.run(train_command, check=True)

    final_eval_status = "skipped_dry_run"
    best_eval_status = "skipped_dry_run"
    metrics: dict[str, Any] = {}
    if not args.dry_run:
        metrics.update(load_training_metrics(run_dir))
        print(f"$ {shlex.join(final_eval_command)}", flush=True)
        subprocess.run(final_eval_command, check=True)
        final_eval_status = "passed"
        metrics.update(
            prefix_metrics(
                "final",
                load_evaluation_metrics(final_evaluation_dir / "token_prior_summary.json"),
            )
        )
        if (run_dir / "best_model").is_dir():
            print(f"$ {shlex.join(best_eval_command)}", flush=True)
            subprocess.run(best_eval_command, check=True)
            best_eval_status = "passed"
            metrics.update(
                prefix_metrics(
                    "best",
                    load_evaluation_metrics(best_evaluation_dir / "token_prior_summary.json"),
                )
            )
        else:
            best_eval_status = "missing_best_model"

    runtime_seconds = time.perf_counter() - start_time
    wandb_enabled = bool(args.wandb and not args.no_wandb and args.wandb_mode != "disabled")
    return {
        "config": str(config_path),
        "experiment_name": experiment_name,
        "seed": training_seed,
        "run_dir": str(run_dir),
        "final_evaluation_dir": str(final_evaluation_dir),
        "best_evaluation_dir": str(best_evaluation_dir),
        "dry_run": bool(args.dry_run),
        "train_status": "passed",
        "final_eval_status": final_eval_status,
        "best_eval_status": best_eval_status,
        "runtime_seconds": runtime_seconds,
        "wandb_enabled": wandb_enabled,
        "wandb_project": args.wandb_project if wandb_enabled else None,
        "wandb_entity": args.wandb_entity if wandb_enabled else None,
        **metrics,
    }


def build_train_command(
    *,
    config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    experiment_name: str,
    seed: int,
) -> list[str]:
    """Build one token-prior training command."""
    command = [
        sys.executable,
        "-m",
        "time_causal_vae.cli.train_token_prior",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
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
                f"{args.run_name_prefix}_{experiment_name}_seed{seed}",
            ]
        )
    else:
        command.append("--no-wandb")
    if args.wandb_mode:
        command.extend(["--wandb-mode", str(args.wandb_mode)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_evaluation_command(
    *,
    raw_config: Mapping[str, Any],
    config_path: Path,
    prior_dir: Path,
    evaluation_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build one token-prior evaluation command."""
    data = require_mapping(raw_config, "data")
    command = [
        sys.executable,
        "-m",
        "time_causal_vae.cli.evaluate_token_prior",
        "--config",
        str(config_path),
        "--prior-dir",
        str(prior_dir),
        "--tokenizer-dir",
        str(data["tokenizer_dir"]),
        "--output-dir",
        str(evaluation_dir),
        "--base-data-dir",
        str(args.base_data_dir),
        "--n-sample",
        str(args.n_sample),
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.device:
        command.extend(["--device", str(args.device)])
    if args.use_best_sampling:
        command.extend(["--temperature", str(args.temperature)])
        if args.top_k is not None:
            command.extend(["--top-k", str(args.top_k)])
    return command


def validate_output_dir(output_dir: str) -> Path:
    """Validate that ablation artifacts stay below ignored outputs/."""
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


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config as a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Expected mapping YAML config: {path}")
    return cast(dict[str, Any], loaded)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required mapping section."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Config requires mapping section {key!r}.")
    return cast(dict[str, Any], value)


def load_training_metrics(run_dir: Path) -> dict[str, Any]:
    """Load best-checkpoint and final token-likelihood metrics."""
    summary_path = run_dir / "best_checkpoint_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise SystemExit(f"Expected JSON object: {summary_path}")
        summary = cast(dict[str, Any], loaded)
        return {
            "best_epoch": summary.get("best_epoch"),
            "best_eval_cross_entropy": summary.get("best_eval_cross_entropy"),
            "best_eval_accuracy": summary.get("best_eval_accuracy"),
            "best_eval_perplexity": summary.get("best_eval_perplexity"),
            "final_eval_cross_entropy": summary.get("final_eval_cross_entropy"),
            "final_eval_accuracy": summary.get("final_eval_accuracy"),
            "final_eval_perplexity": summary.get("final_eval_perplexity"),
        }
    log_path = run_dir / "token_prior_training_log.jsonl"
    last_line = ""
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        raise SystemExit(f"Training log is empty: {log_path}")
    final_metrics = json.loads(last_line)
    if not isinstance(final_metrics, dict):
        raise SystemExit(f"Expected JSON object in training log: {log_path}")
    return {
        "final_eval_cross_entropy": final_metrics.get("eval_cross_entropy"),
        "final_eval_accuracy": final_metrics.get("eval_accuracy"),
        "final_eval_perplexity": final_metrics.get("eval_perplexity"),
    }


def load_evaluation_metrics(summary_path: Path) -> dict[str, Any]:
    """Load selected decoded and condition-bucket metrics."""
    with summary_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Expected JSON object: {summary_path}")
    metrics = cast(dict[str, Any], cast(dict[str, Any], loaded)["metrics"])
    condition_buckets = cast(list[dict[str, Any]], metrics.get("condition_buckets", []))
    very_low = find_bucket(condition_buckets, "very_low")
    low = find_bucket(condition_buckets, "low")
    return {
        "mmd": metrics.get("mmd"),
        "swd": metrics.get("swd"),
        "volatility_wasserstein": metrics.get("volatility_wasserstein"),
        "terminal_return_wasserstein": metrics.get("terminal_return_wasserstein"),
        "sampled_active_codes": metrics.get("sampled_token_active_code_count"),
        "sampled_token_perplexity": metrics.get("sampled_token_perplexity"),
        "very_low_mmd": very_low.get("mmd") if very_low else None,
        "low_mmd": low.get("mmd") if low else None,
    }


def find_bucket(
    condition_buckets: Sequence[Mapping[str, Any]],
    label: str,
) -> Mapping[str, Any] | None:
    """Return a named condition bucket if present."""
    for bucket in condition_buckets:
        if bucket.get("bucket_label") == label:
            return bucket
    return None


def prefix_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Prefix metric keys for final/best aggregate columns."""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_summaries(output_dir: Path, results: Sequence[Mapping[str, Any]]) -> None:
    """Write aggregate JSON and CSV summaries."""
    json_path = output_dir / "sp500_vix_conditional_token_prior_ablation_summary.json"
    csv_path = output_dir / "sp500_vix_conditional_token_prior_ablation_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"results": [dict(result) for result in results]}, handle, indent=2)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in SUMMARY_FIELDS})


if __name__ == "__main__":
    main()
