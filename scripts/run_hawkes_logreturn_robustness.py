"""Plan and run Hawkes log-return seed-robustness experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import yaml

TOKENIZER_CONFIGS: dict[int, Path] = {
    0: Path("configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64.yaml"),
    1: Path(
        "configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64_seed1.yaml"
    ),
    2: Path(
        "configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64_seed2.yaml"
    ),
}

ADDITIVE_CONFIGS: dict[int, Path] = {
    0: Path(
        "configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive.yaml"
    ),
    1: Path(
        "configs/experiments/"
        "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive_seed1.yaml"
    ),
    2: Path(
        "configs/experiments/"
        "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive_seed2.yaml"
    ),
}

CONV_CONFIGS: dict[int, Path] = {
    0: Path(
        "configs/experiments/"
        "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer.yaml"
    ),
    1: Path(
        "configs/experiments/"
        "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_seed1.yaml"
    ),
    2: Path(
        "configs/experiments/"
        "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_seed2.yaml"
    ),
}

CONTINUOUS_CONFIGS: dict[int, Path] = {
    0: Path("configs/experiments/hawkes_jump_beta_cvae_logreturn.yaml"),
    1: Path("configs/experiments/hawkes_jump_beta_cvae_logreturn_seed1.yaml"),
    2: Path("configs/experiments/hawkes_jump_beta_cvae_logreturn_seed2.yaml"),
}


@dataclass(frozen=True)
class CommandRecord:
    """One planned or executed command."""

    stage: str
    seed: int
    command: list[str]
    returncode: int | None = None
    runtime_seconds: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    skipped_reason: str = ""


@dataclass(frozen=True)
class SeedPaths:
    """Output paths for one robustness seed."""

    seed: int
    tokenizer_dir: Path
    token_dir: Path
    additive_prior_dir: Path
    conv_prior_dir: Path
    continuous_dir: Path
    additive_eval_dir: Path
    conv_eval_dir: Path


def build_parser() -> argparse.ArgumentParser:
    """Build the robustness-runner CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run or dry-run Hawkes/SVMHJD log-return seed robustness.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output-dir",
        default="outputs/hawkes_jump_logreturn_robustness",
        help="Output directory under outputs/.",
    )
    parser.add_argument("--epochs-tokenizer", type=int, default=50)
    parser.add_argument("--epochs-prior", type=int, default=50)
    parser.add_argument("--epochs-continuous", type=int, default=50)
    parser.add_argument("--n-sample", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--base-data-dir", default="data/processed")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", default="none")
    return parser


def main() -> None:
    """Run or plan the Hawkes log-return robustness workflow."""
    args = build_parser().parse_args()
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_args(args)

    records: list[CommandRecord] = []
    seed_summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        if seed not in TOKENIZER_CONFIGS:
            raise SystemExit(f"Unsupported seed {seed}. Config maps currently support 0, 1, 2.")
        paths = seed_paths(output_dir=output_dir, seed=seed)
        runtime_configs = write_runtime_configs(
            output_dir=output_dir,
            seed=seed,
            paths=paths,
            n_sample=int(args.n_sample),
        )
        seed_records = planned_commands(
            seed=seed,
            paths=paths,
            runtime_configs=runtime_configs,
            args=args,
        )
        if args.dry_run:
            records.extend(seed_records)
            seed_summaries.append(seed_summary(seed, paths, runtime_configs, status="planned"))
            continue

        executed = execute_seed_records(seed_records)
        records.extend(executed)
        seed_summaries.append(
            seed_summary(
                seed,
                paths,
                runtime_configs,
                status="complete" if all_successful(executed) else "failed",
            )
        )

    write_json(
        output_dir / "aggregate_summary.json",
        {
            "dry_run": bool(args.dry_run),
            "seeds": list(args.seeds),
            "n_sample": int(args.n_sample),
            "epochs": {
                "tokenizer": int(args.epochs_tokenizer),
                "prior": int(args.epochs_prior),
                "continuous": int(args.epochs_continuous),
            },
            "continuous_evaluation_status": continuous_evaluation_status(),
            "seed_summaries": seed_summaries,
            "commands": [asdict(record) for record in records],
        },
    )
    write_csv(output_dir / "aggregate_summary.csv", records)
    write_markdown_plan(output_dir / "command_plan.md", records, dry_run=bool(args.dry_run))
    print(f"Wrote Hawkes log-return robustness summary under {output_dir}")


def validate_output_dir(output_dir: str) -> Path:
    """Validate that generated artefacts stay below local outputs/."""
    path = Path(output_dir)
    resolved = path.resolve()
    outputs_root = (Path.cwd() / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise SystemExit(
            f"--output-dir must be under local outputs/. Received: {output_dir}"
        ) from exc
    return path


def validate_args(args: argparse.Namespace) -> None:
    """Validate common numeric CLI arguments."""
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")
    for field in ["epochs_tokenizer", "epochs_prior", "epochs_continuous"]:
        if int(getattr(args, field)) <= 0:
            raise SystemExit(f"--{field.replace('_', '-')} must be positive.")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive.")


def seed_paths(*, output_dir: Path, seed: int) -> SeedPaths:
    """Return canonical output paths for one seed."""
    tokenizer_name = "hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64"
    additive_name = "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive"
    conv_name = "hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer"
    continuous_name = "hawkes_jump_beta_cvae_logreturn"
    return SeedPaths(
        seed=seed,
        tokenizer_dir=output_dir / "tokenizers" / f"{tokenizer_name}_seed{seed}",
        token_dir=output_dir / "tokens" / f"{tokenizer_name}_seed{seed}",
        additive_prior_dir=output_dir / "priors" / "additive" / f"{additive_name}_seed{seed}",
        conv_prior_dir=output_dir / "priors" / "conv_transformer" / f"{conv_name}_seed{seed}",
        continuous_dir=output_dir / "continuous" / f"{continuous_name}_seed{seed}",
        additive_eval_dir=output_dir / "evaluations" / f"additive_seed{seed}",
        conv_eval_dir=output_dir / "evaluations" / f"conv_transformer_seed{seed}",
    )


def write_runtime_configs(
    *,
    output_dir: Path,
    seed: int,
    paths: SeedPaths,
    n_sample: int,
) -> dict[str, Path]:
    """Write seed-adjusted runtime configs under the output directory."""
    config_dir = output_dir / "run_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_yaml(TOKENIZER_CONFIGS[seed])
    set_nested_int(tokenizer, ("experiment", "seed"), seed)
    set_nested_int(tokenizer, ("data", "n_samples"), n_sample)
    set_nested_int(tokenizer, ("data", "data_params", "seed"), seed)

    additive = token_prior_runtime_config(
        source_path=ADDITIVE_CONFIGS[seed],
        seed=seed,
        paths=paths,
    )
    conv = token_prior_runtime_config(source_path=CONV_CONFIGS[seed], seed=seed, paths=paths)

    continuous = load_yaml(CONTINUOUS_CONFIGS[seed])
    set_nested_int(continuous, ("experiment", "seed"), seed)
    set_nested_int(continuous, ("data", "n_samples"), n_sample)
    set_nested_int(continuous, ("data", "params", "seed"), seed)
    set_nested_value(continuous, ("data", "params", "data_output"), "log_return")

    written = {
        "tokenizer": config_dir / f"tokenizer_seed{seed}.yaml",
        "additive": config_dir / f"prior_additive_seed{seed}.yaml",
        "conv": config_dir / f"prior_conv_transformer_seed{seed}.yaml",
        "continuous": config_dir / f"continuous_beta_cvae_seed{seed}.yaml",
    }
    write_yaml(written["tokenizer"], tokenizer)
    write_yaml(written["additive"], additive)
    write_yaml(written["conv"], conv)
    write_yaml(written["continuous"], continuous)
    return written


def token_prior_runtime_config(*, source_path: Path, seed: int, paths: SeedPaths) -> dict[str, Any]:
    """Return a token-prior config with seed-specific tokenizer and token paths."""
    config = load_yaml(source_path)
    set_nested_int(config, ("experiment", "seed"), seed)
    set_nested_value(config, ("data", "tokenizer_dir"), str(paths.tokenizer_dir))
    set_nested_value(config, ("data", "token_data_dir"), str(paths.token_dir))
    set_nested_value(config, ("data", "data_output"), "log_return")
    return config


def planned_commands(
    *,
    seed: int,
    paths: SeedPaths,
    runtime_configs: Mapping[str, Path],
    args: argparse.Namespace,
) -> list[CommandRecord]:
    """Build the ordered command list for one seed."""
    records = [
        CommandRecord(
            stage="train_tokenizer",
            seed=seed,
            command=[
                sys.executable,
                "-m",
                "time_causal_vae.cli.train_tokenizer",
                "--config",
                str(runtime_configs["tokenizer"]),
                "--output-dir",
                str(paths.tokenizer_dir.parent),
                "--epochs",
                str(args.epochs_tokenizer),
                "--base-data-dir",
                str(args.base_data_dir),
                *wandb_flags(args),
                *device_flags(args),
            ],
        ),
        CommandRecord(
            stage="extract_tokens",
            seed=seed,
            command=[
                sys.executable,
                "scripts/extract_token_indices.py",
                "--config",
                str(runtime_configs["tokenizer"]),
                "--tokenizer-dir",
                str(paths.tokenizer_dir),
                "--output-dir",
                str(paths.token_dir),
                "--n-sample",
                str(args.n_sample),
                "--seed",
                str(seed),
                "--base-data-dir",
                str(args.base_data_dir),
                *device_flags(args),
            ],
        ),
        token_prior_train_record(
            stage="train_additive_prior",
            seed=seed,
            config_path=runtime_configs["additive"],
            output_base=paths.additive_prior_dir.parent,
            epochs=int(args.epochs_prior),
            args=args,
        ),
        token_prior_eval_record(
            stage="evaluate_additive_prior",
            seed=seed,
            config_path=runtime_configs["additive"],
            prior_dir=paths.additive_prior_dir / "best_model",
            tokenizer_dir=paths.tokenizer_dir,
            output_dir=paths.additive_eval_dir,
            args=args,
        ),
        token_prior_train_record(
            stage="train_conv_prior",
            seed=seed,
            config_path=runtime_configs["conv"],
            output_base=paths.conv_prior_dir.parent,
            epochs=int(args.epochs_prior),
            args=args,
        ),
        token_prior_eval_record(
            stage="evaluate_conv_prior",
            seed=seed,
            config_path=runtime_configs["conv"],
            prior_dir=paths.conv_prior_dir / "best_model",
            tokenizer_dir=paths.tokenizer_dir,
            output_dir=paths.conv_eval_dir,
            args=args,
        ),
        CommandRecord(
            stage="train_continuous_logreturn",
            seed=seed,
            command=[
                sys.executable,
                "-m",
                "time_causal_vae.cli.train",
                "--config",
                str(runtime_configs["continuous"]),
                "--output-dir",
                str(paths.continuous_dir.parent),
                "--epochs",
                str(args.epochs_continuous),
                "--base-data-dir",
                str(args.base_data_dir),
                *wandb_flags(args),
                *device_flags(args),
            ],
        ),
        CommandRecord(
            stage="evaluate_continuous_logreturn",
            seed=seed,
            command=[],
            skipped_reason=continuous_evaluation_status(),
        ),
    ]
    return records


def token_prior_train_record(
    *,
    stage: str,
    seed: int,
    config_path: Path,
    output_base: Path,
    epochs: int,
    args: argparse.Namespace,
) -> CommandRecord:
    """Build one token-prior training record."""
    return CommandRecord(
        stage=stage,
        seed=seed,
        command=[
            sys.executable,
            "-m",
            "time_causal_vae.cli.train_token_prior",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_base),
            "--epochs",
            str(epochs),
            *wandb_flags(args),
            *device_flags(args),
        ],
    )


def token_prior_eval_record(
    *,
    stage: str,
    seed: int,
    config_path: Path,
    prior_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> CommandRecord:
    """Build one Hawkes-aware token-prior evaluation record."""
    return CommandRecord(
        stage=stage,
        seed=seed,
        command=[
            sys.executable,
            "scripts/evaluate_hawkes_jump_token_prior.py",
            "--config",
            str(config_path),
            "--prior-dir",
            str(prior_dir),
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--output-dir",
            str(output_dir),
            "--base-data-dir",
            str(args.base_data_dir),
            "--n-sample",
            str(args.n_sample),
            "--seed",
            str(seed),
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            *device_flags(args),
        ],
    )


def wandb_flags(args: argparse.Namespace) -> list[str]:
    """Return W&B CLI flags."""
    return ["--no-wandb"] if args.no_wandb else []


def device_flags(args: argparse.Namespace) -> list[str]:
    """Return device CLI flags."""
    return ["--device", str(args.device)] if args.device else []


def execute_seed_records(records: Sequence[CommandRecord]) -> list[CommandRecord]:
    """Execute all runnable commands for one seed in order."""
    executed: list[CommandRecord] = []
    for record in records:
        if record.skipped_reason:
            executed.append(record)
            continue
        result = run_command(record)
        executed.append(result)
        if result.returncode != 0:
            break
    return executed


def run_command(record: CommandRecord) -> CommandRecord:
    """Run a command record and return runtime details."""
    print("Running:", " ".join(record.command))
    started = time.perf_counter()
    completed = subprocess.run(record.command, check=False, capture_output=True, text=True)
    runtime = time.perf_counter() - started
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    return CommandRecord(
        stage=record.stage,
        seed=record.seed,
        command=record.command,
        returncode=int(completed.returncode),
        runtime_seconds=runtime,
        stdout_tail=tail(completed.stdout),
        stderr_tail=tail(completed.stderr),
    )


def all_successful(records: Sequence[CommandRecord]) -> bool:
    """Return true when all executed commands succeeded or were intentionally skipped."""
    return all(record.returncode in {0, None} for record in records)


def seed_summary(
    seed: int,
    paths: SeedPaths,
    runtime_configs: Mapping[str, Path],
    *,
    status: str,
) -> dict[str, Any]:
    """Return a compact per-seed summary."""
    return {
        "seed": seed,
        "status": status,
        "tokenizer_dir": str(paths.tokenizer_dir),
        "token_dir": str(paths.token_dir),
        "additive_prior_dir": str(paths.additive_prior_dir),
        "conv_prior_dir": str(paths.conv_prior_dir),
        "continuous_dir": str(paths.continuous_dir),
        "runtime_configs": {key: str(value) for key, value in runtime_configs.items()},
    }


def continuous_evaluation_status() -> str:
    """Describe the current continuous log-return evaluation limitation."""
    return (
        "pending_hawkes_specific_continuous_evaluator: generic tcvae-evaluate is not used "
        "because generated log returns must be converted to normalised prices before "
        "market and jump diagnostics"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a YAML mapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def set_nested_int(config: dict[str, Any], keys: Sequence[str], value: int) -> None:
    """Set a nested integer value."""
    set_nested_value(config, keys, int(value))


def set_nested_value(config: dict[str, Any], keys: Sequence[str], value: object) -> None:
    """Set a nested value in a YAML-derived dictionary."""
    target: dict[str, Any] = config
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"Expected mapping at {'.'.join(keys)}.")
        target = cast(dict[str, Any], child)
    target[keys[-1]] = value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, records: Sequence[CommandRecord]) -> None:
    """Write the command records as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed",
            "stage",
            "returncode",
            "runtime_seconds",
            "skipped_reason",
            "command",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "seed": record.seed,
                    "stage": record.stage,
                    "returncode": record.returncode,
                    "runtime_seconds": record.runtime_seconds,
                    "skipped_reason": record.skipped_reason,
                    "command": " ".join(record.command),
                }
            )


def write_markdown_plan(path: Path, records: Sequence[CommandRecord], *, dry_run: bool) -> None:
    """Write a human-readable command plan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Hawkes Log-Return Robustness Command Plan",
        "",
        f"Dry run: `{str(dry_run).lower()}`",
        "",
        "| Seed | Stage | Status | Command |",
        "| ---: | --- | --- | --- |",
    ]
    for record in records:
        status = record.skipped_reason or (
            "planned" if record.returncode is None else f"returncode {record.returncode}"
        )
        command = " ".join(record.command) if record.command else ""
        lines.append(f"| {record.seed} | `{record.stage}` | {status} | `{command}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tail(text: str, *, n_lines: int = 40) -> str:
    """Return the last ``n_lines`` of text."""
    return "\n".join(text.splitlines()[-n_lines:])


def to_jsonable(value: Any) -> Any:
    """Convert common path/dataclass containers to JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
