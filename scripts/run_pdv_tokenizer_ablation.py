"""Run causal VQ tokenizer ablations through the package CLIs."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import yaml

DEFAULT_WANDB_PROJECT = "time-causal-vq-tokenizer"
DEFAULT_WANDB_ENTITY = "tc_vae"


def build_parser() -> argparse.ArgumentParser:
    """Build the ablation runner parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate causal VQ tokenizer ablations.",
    )
    parser.add_argument("--configs", nargs="+", required=True, help="Tokenizer config paths.")
    parser.add_argument("--output-dir", required=True, help="Base output directory under outputs/.")
    parser.add_argument("--epochs", type=int, help="Override tokenizer training epochs.")
    parser.add_argument(
        "--n-sample-test",
        type=int,
        default=512,
        help="Number of paths used for each tokenizer evaluation.",
    )
    parser.add_argument("--seed", type=int, default=99, help="Evaluation seed.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory for datasets that require local files.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B for training runs.")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B for training runs. Takes precedence over --wandb.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="W&B project used when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=DEFAULT_WANDB_ENTITY,
        help="W&B entity used when --wandb is enabled.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="",
        help="Optional prefix added to train run names and artifact directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run tokenizer train dry-runs only, then write a setup summary.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip train/eval commands and collect metrics from existing run directories.",
    )
    return parser


def main() -> None:
    """Run all requested ablations and collect summary files."""
    args = build_parser().parse_args()
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for config_path_text in args.configs:
        config_path = Path(config_path_text)
        raw_config = load_yaml(config_path)
        experiment = require_mapping(raw_config, "experiment")
        experiment_name = str(experiment["name"])
        train_seed = int(experiment.get("seed", 0))
        run_name = build_run_name(
            experiment_name,
            train_seed=train_seed,
            prefix=args.run_name_prefix,
        )
        result = run_one_ablation(
            config_path=config_path,
            experiment_name=experiment_name,
            run_name=run_name,
            output_dir=output_dir,
            args=args,
        )
        results.append(result)

    write_json(output_dir / "tokenizer_ablation_summary.json", results)
    write_csv(output_dir / "tokenizer_ablation_summary.csv", results)
    print(f"Wrote {output_dir / 'tokenizer_ablation_summary.json'}")
    print(f"Wrote {output_dir / 'tokenizer_ablation_summary.csv'}")


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
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required YAML section as a dictionary."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def build_run_name(experiment_name: str, *, train_seed: int, prefix: str) -> str:
    """Build the expected tokenizer run directory name."""
    default_name = f"{experiment_name}_seed{train_seed}"
    if not prefix:
        return default_name
    return f"{prefix}_{default_name}"


def run_one_ablation(
    *,
    config_path: Path,
    experiment_name: str,
    run_name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one train/eval pair and return a flattened summary."""
    run_dir = output_dir / run_name
    eval_dir = output_dir / f"{run_name}_evaluation"
    started = time.perf_counter()
    result: dict[str, Any] = {
        "config": str(config_path),
        "experiment_name": experiment_name,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "evaluation_dir": str(eval_dir),
        "dry_run": bool(args.dry_run),
        "evaluation_seed": int(args.seed),
        "train_status": "pending",
        "eval_status": "pending",
    }

    if args.collect_only:
        result["train_status"] = "skipped_collect_only"
        result["eval_status"] = "skipped_collect_only"
        result["runtime_seconds"] = 0.0
        result.update(load_training_metrics(run_dir))
        result.update(load_evaluation_metrics(eval_dir))
        return result

    train_command = build_train_command(
        config_path=config_path,
        output_dir=output_dir,
        args=args,
        run_name=run_name,
    )
    result["train_command"] = shlex.join(train_command)
    run_command(train_command)
    result["train_status"] = "passed"

    if args.dry_run:
        result["eval_status"] = "skipped_dry_run"
        result["runtime_seconds"] = round(time.perf_counter() - started, 3)
        print(f"[dry-run] skipped evaluation for {experiment_name}")
        return result

    eval_command = build_eval_command(
        config_path=config_path,
        run_dir=run_dir,
        eval_dir=eval_dir,
        args=args,
    )
    result["eval_command"] = shlex.join(eval_command)
    run_command(eval_command)
    result["eval_status"] = "passed"
    result["runtime_seconds"] = round(time.perf_counter() - started, 3)
    result.update(load_training_metrics(run_dir))
    result.update(load_evaluation_metrics(eval_dir))
    return result


def build_train_command(
    *,
    config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    run_name: str,
) -> list[str]:
    """Build a tokenizer training command."""
    command = [
        sys.executable,
        "-m",
        "time_causal_vae.cli.train_tokenizer",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--base-data-dir",
        str(args.base_data_dir),
    ]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.dry_run:
        command.append("--dry-run")
    if args.run_name_prefix:
        command.extend(["--wandb-run-name", run_name])
    if args.wandb and not args.no_wandb:
        command.extend(
            [
                "--wandb",
                "--wandb-project",
                str(args.wandb_project),
                "--wandb-entity",
                str(args.wandb_entity),
            ]
        )
        if not args.run_name_prefix:
            command.extend(["--wandb-run-name", run_name])
    else:
        command.append("--no-wandb")
    return command


def build_eval_command(
    *,
    config_path: Path,
    run_dir: Path,
    eval_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build a tokenizer evaluation command."""
    return [
        sys.executable,
        "-m",
        "time_causal_vae.cli.evaluate_tokenizer",
        "--config",
        str(config_path),
        "--tokenizer-dir",
        str(run_dir),
        "--output-dir",
        str(eval_dir),
        "--base-data-dir",
        str(args.base_data_dir),
        "--n-sample-test",
        str(args.n_sample_test),
        "--seed",
        str(args.seed),
    ]


def run_command(command: list[str]) -> None:
    """Run a subprocess command with a readable echo."""
    print(f"$ {shlex.join(command)}")
    subprocess.run(command, check=True)


def load_training_metrics(run_dir: Path) -> dict[str, Any]:
    """Load selected training metrics if the tokenizer run wrote them."""
    metrics: dict[str, Any] = {}
    summary_path = run_dir / "codebook_summary.json"
    runtime_path = run_dir / "runtime_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        metrics.update(
            prefixed(
                "train",
                summary,
                (
                    "active_code_count",
                    "active_code_ratio",
                    "codebook_perplexity",
                    "mean_reconstruction_loss",
                    "mean_commitment_loss",
                    "mean_total_loss",
                ),
            )
        )
    if runtime_path.exists():
        runtime = load_json(runtime_path)
        metrics["train_runtime_seconds"] = runtime.get("runtime_seconds")
    return metrics


def load_evaluation_metrics(eval_dir: Path) -> dict[str, Any]:
    """Load selected evaluation and condition-bucket metrics."""
    summary = load_json(eval_dir / "tokenizer_summary.json")
    metric_source = summary.get("metrics", summary)
    if not isinstance(metric_source, dict):
        raise SystemExit(f"Expected metrics object in {eval_dir / 'tokenizer_summary.json'}")
    metrics = prefixed(
        "eval",
        metric_source,
        (
            "reconstruction_l1",
            "reconstruction_l2",
            "terminal_return_error",
            "volatility_reconstruction_error",
            "active_code_count",
            "active_code_ratio",
            "codebook_perplexity",
            "index_entropy",
        ),
    )
    buckets = metric_source.get("condition_buckets")
    if isinstance(buckets, list) and buckets:
        bucket_items = [bucket for bucket in buckets if isinstance(bucket, dict)]
        metrics["condition_bucket_count"] = len(bucket_items)
        metrics["condition_bucket_labels"] = [
            str(bucket.get("bucket_label")) for bucket in bucket_items
        ]
        metrics["min_bucket_active_code_count"] = min(
            int(bucket["active_code_count"]) for bucket in bucket_items
        )
        metrics["max_bucket_reconstruction_l1"] = max(
            float(bucket["reconstruction_l1"]) for bucket in bucket_items
        )
        metrics["max_bucket_volatility_reconstruction_error"] = max(
            float(bucket["volatility_reconstruction_error"]) for bucket in bucket_items
        )
        metrics["condition_buckets"] = bucket_items
    else:
        metrics["condition_bucket_count"] = 0
    return metrics


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Expected JSON object at {path}")
    return cast(dict[str, Any], loaded)


def prefixed(
    prefix: str,
    source: Mapping[str, Any],
    keys: Iterable[str],
) -> dict[str, Any]:
    """Copy selected keys with a prefix."""
    return {f"{prefix}_{key}": source.get(key) for key in keys if key in source}


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the aggregate JSON summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a flat CSV summary."""
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    """Serialise nested CSV values as JSON strings."""
    if isinstance(value, list | dict):
        return json.dumps(value, sort_keys=True)
    return value


if __name__ == "__main__":
    main()
