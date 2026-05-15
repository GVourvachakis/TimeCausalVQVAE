"""Run S&P500/VIX token-prior candidate robustness ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

DEFAULT_WANDB_PROJECT = "time-causal-token-prior"
DEFAULT_WANDB_ENTITY = "tc_vae"
DEFAULT_CONTINUOUS_CONFIG = "configs/experiments/sp500_vix_beta_cvae.yaml"
DEFAULT_CONTINUOUS_MODEL_DIR = (
    "outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model"
)
WANDB_ENVIRONMENT = {
    "MPLBACKEND": "Agg",
    "WANDB_DISABLE_SERVICE": "true",
    "WANDB_START_METHOD": "thread",
}
PROFILE_FORMULA = "MMD + SWD + volatility_wasserstein + terminal_return_wasserstein"
SUMMARY_FIELDS = [
    "config",
    "experiment_name",
    "training_seed",
    "evaluation_seed",
    "run_dir",
    "best_evaluation_dir",
    "paper_style_dir",
    "dry_run",
    "train_status",
    "best_eval_status",
    "paper_style_status",
    "runtime_seconds",
    "n_sample",
    "temperature",
    "top_k",
    "wandb_enabled",
    "wandb_project",
    "wandb_entity",
    "best_epoch",
    "best_eval_cross_entropy",
    "best_eval_accuracy",
    "best_eval_perplexity",
    "final_eval_cross_entropy",
    "final_eval_accuracy",
    "final_eval_perplexity",
    "mmd",
    "swd",
    "volatility_wasserstein",
    "terminal_return_wasserstein",
    "model_selection_profile_score",
    "sampled_active_codes",
    "sampled_token_perplexity",
    "marginal_code_l1",
    "transition_matrix_l1",
    "run_length_distance",
    "very_low_mmd",
    "low_mmd",
    "paper_mmd",
    "paper_swd",
    "paper_volatility_wasserstein",
    "paper_terminal_return_wasserstein",
    "paper_returns_wasserstein",
    "paper_maximum_drawdown_wasserstein",
    "paper_model_selection_profile_score",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the candidate-prior ablation parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate S&P500/VIX token-prior candidate ablations.",
    )
    parser.add_argument("--configs", nargs="+", required=True, help="Token-prior configs.")
    parser.add_argument("--output-dir", required=True, help="Output directory under outputs/.")
    parser.add_argument("--base-data-dir", default="data/processed", help="Market data root.")
    parser.add_argument("--epochs", type=int, help="Training epoch override.")
    parser.add_argument("--n-sample", type=int, default=1000, help="Evaluation sample count.")
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Evaluation and sampling seed. Training seeds come from each config.",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument(
        "--top-k",
        default="40",
        help="Top-k sampling cutoff. Use 'none' for unrestricted sampling.",
    )
    parser.add_argument("--device", help="Optional device override, for example cpu or cuda.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run token-prior train dry-runs and write a setup summary.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B for training runs.")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B for all training runs. Takes precedence over --wandb.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="W&B project used when --wandb is enabled.",
    )
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY, help="W&B entity.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode forwarded to training.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="sp500_vix_token_prior_candidate",
        help="Prefix for W&B display run names.",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Also run paper-style diagnostics for each best checkpoint.",
    )
    parser.add_argument("--continuous-config", default=DEFAULT_CONTINUOUS_CONFIG)
    parser.add_argument("--continuous-model-dir", default=DEFAULT_CONTINUOUS_MODEL_DIR)
    return parser


def main() -> None:
    """Run all requested token-prior candidate ablations."""
    args = build_parser().parse_args()
    output_dir = validate_output_dir(args.output_dir)
    validate_positive("n-sample", args.n_sample)
    if args.epochs is not None:
        validate_positive("epochs", args.epochs)
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive.")
    top_k = parse_top_k(args.top_k)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for config_text in args.configs:
        config_path = Path(config_text)
        raw_config = load_yaml(config_path)
        validate_candidate_config(raw_config, config_path=config_path)
        result = run_one_config(
            config_path=config_path,
            raw_config=raw_config,
            output_dir=output_dir,
            args=args,
            top_k=top_k,
        )
        results.append(result)
        write_summaries(output_dir, results, args=args, top_k=top_k)

    print(f"wrote: {output_dir / 'token_prior_candidate_ablation_summary.json'}")
    print(f"wrote: {output_dir / 'token_prior_candidate_ablation_summary.csv'}")


def run_one_config(
    *,
    config_path: Path,
    raw_config: Mapping[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    top_k: int | None,
) -> dict[str, Any]:
    """Run one train/evaluate pair, or validate it in dry-run mode."""
    experiment = require_mapping(raw_config, "experiment")
    data = require_mapping(raw_config, "data")
    experiment_name = str(experiment["name"])
    training_seed = int(experiment.get("seed", 0))
    run_name = f"{experiment_name}_seed{training_seed}"
    run_dir = output_dir / run_name
    best_evaluation_dir = output_dir / f"{run_name}_evaluation_best"
    paper_style_dir = output_dir / f"{run_name}_paper_style"
    started = time.perf_counter()

    train_command = build_train_command(
        config_path=config_path,
        output_dir=output_dir,
        args=args,
        experiment_name=experiment_name,
        training_seed=training_seed,
    )
    best_eval_command = build_evaluation_command(
        config_path=config_path,
        tokenizer_dir=Path(str(data["tokenizer_dir"])),
        prior_dir=run_dir / "best_model",
        evaluation_dir=best_evaluation_dir,
        args=args,
        top_k=top_k,
    )
    paper_style_command = build_paper_style_command(
        config_path=config_path,
        tokenizer_dir=Path(str(data["tokenizer_dir"])),
        prior_dir=run_dir / "best_model",
        output_dir=paper_style_dir,
        args=args,
        top_k=top_k,
    )

    print(f"$ {shlex.join(train_command)}", flush=True)
    run_command(train_command)

    metrics: dict[str, Any] = {}
    best_eval_status = "skipped_dry_run"
    paper_style_status = "skipped_dry_run"
    if not args.dry_run:
        metrics.update(load_training_metrics(run_dir))
        print(f"$ {shlex.join(best_eval_command)}", flush=True)
        run_command(best_eval_command)
        best_eval_status = "passed"
        metrics.update(load_evaluation_metrics(best_evaluation_dir / "token_prior_summary.json"))
        if args.paper_style:
            print(f"$ {shlex.join(paper_style_command)}", flush=True)
            run_command(paper_style_command)
            paper_style_status = "passed"
            metrics.update(load_paper_style_metrics(paper_style_dir / "paper_style_summary.json"))
        else:
            paper_style_status = "skipped_not_requested"

    runtime_seconds = round(time.perf_counter() - started, 3)
    wandb_enabled = bool(args.wandb and not args.no_wandb and args.wandb_mode != "disabled")
    return {
        "config": str(config_path),
        "experiment_name": experiment_name,
        "training_seed": training_seed,
        "evaluation_seed": int(args.seed),
        "run_dir": str(run_dir),
        "best_evaluation_dir": str(best_evaluation_dir),
        "paper_style_dir": str(paper_style_dir),
        "dry_run": bool(args.dry_run),
        "train_status": "passed",
        "best_eval_status": best_eval_status,
        "paper_style_status": paper_style_status,
        "runtime_seconds": runtime_seconds,
        "n_sample": int(args.n_sample),
        "temperature": float(args.temperature),
        "top_k": top_k_label(top_k),
        "wandb_enabled": wandb_enabled,
        "wandb_project": args.wandb_project if wandb_enabled else None,
        "wandb_entity": args.wandb_entity if wandb_enabled else None,
        "train_command": shlex.join(train_command),
        "best_eval_command": shlex.join(best_eval_command),
        "paper_style_command": shlex.join(paper_style_command),
        **empty_metrics(),
        **metrics,
    }


def build_train_command(
    *,
    config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    experiment_name: str,
    training_seed: int,
) -> list[str]:
    """Build a token-prior training command."""
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
                f"{args.run_name_prefix}_{experiment_name}_seed{training_seed}",
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
    config_path: Path,
    tokenizer_dir: Path,
    prior_dir: Path,
    evaluation_dir: Path,
    args: argparse.Namespace,
    top_k: int | None,
) -> list[str]:
    """Build a best-checkpoint token-prior evaluation command."""
    command = [
        sys.executable,
        "-m",
        "time_causal_vae.cli.evaluate_token_prior",
        "--config",
        str(config_path),
        "--prior-dir",
        str(prior_dir),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--output-dir",
        str(evaluation_dir),
        "--base-data-dir",
        str(args.base_data_dir),
        "--n-sample",
        str(args.n_sample),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
    ]
    if args.device:
        command.extend(["--device", str(args.device)])
    if top_k is not None:
        command.extend(["--top-k", str(top_k)])
    return command


def build_paper_style_command(
    *,
    config_path: Path,
    tokenizer_dir: Path,
    prior_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    top_k: int | None,
) -> list[str]:
    """Build a paper-style evaluation command."""
    return [
        sys.executable,
        "scripts/evaluate_sp500_vix_paper_style.py",
        "--discrete-config",
        str(config_path),
        "--discrete-prior-dir",
        str(prior_dir),
        "--discrete-tokenizer-dir",
        str(tokenizer_dir),
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
        "--top-k",
        top_k_label(top_k),
    ]


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


def validate_positive(name: str, value: int) -> None:
    """Validate a positive integer argument."""
    if value <= 0:
        raise SystemExit(f"--{name} must be positive.")


def parse_top_k(raw_value: str) -> int | None:
    """Parse the optional top-k argument."""
    normalised = raw_value.strip().lower()
    if normalised in {"none", "null", "unrestricted"}:
        return None
    value = int(raw_value)
    if value <= 0:
        raise SystemExit("--top-k must be a positive integer or 'none'.")
    return value


def top_k_label(top_k: int | None) -> str:
    """Return a stable top-k label."""
    if top_k is None:
        return "none"
    return str(top_k)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required YAML section."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Config requires mapping section {key!r}.")
    return cast(dict[str, Any], value)


def validate_candidate_config(raw_config: Mapping[str, Any], *, config_path: Path) -> None:
    """Guard this runner to scalar-conditioned S&P500/VIX additive token priors."""
    experiment = require_mapping(raw_config, "experiment")
    data = require_mapping(raw_config, "data")
    model = require_mapping(raw_config, "model")
    checks = {
        "experiment.dataset": experiment.get("dataset") == "sp500_vix",
        "model.family": model.get("family") == "causal_token_prior",
        "model.condition_dim": int(model.get("condition_dim", -1)) == 1,
        "model.condition_injection": model.get("condition_injection") == "additive",
        "data.tokenizer_dir": Path(str(data.get("tokenizer_dir", ""))).is_dir(),
        "data.token_data_dir": Path(str(data.get("token_data_dir", ""))).is_dir(),
    }
    failed = [field for field, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(
            f"{config_path} is outside the controlled S&P500/VIX token-prior surface. "
            f"Failed checks: {', '.join(failed)}"
        )


def run_command(command: Sequence[str]) -> None:
    """Run one subprocess command with the W&B-compatible environment."""
    subprocess.run(list(command), check=True, env=build_subprocess_environment())


def build_subprocess_environment() -> dict[str, str]:
    """Return subprocess environment with WANDB_MODE unset and backend-safe plotting."""
    environment = os.environ.copy()
    environment.pop("WANDB_MODE", None)
    environment.update(WANDB_ENVIRONMENT)
    return environment


def empty_metrics() -> dict[str, Any]:
    """Return stable metric keys for aggregate outputs."""
    excluded = {
        "config",
        "experiment_name",
        "training_seed",
        "evaluation_seed",
        "run_dir",
        "best_evaluation_dir",
        "paper_style_dir",
        "dry_run",
        "train_status",
        "best_eval_status",
        "paper_style_status",
        "runtime_seconds",
        "n_sample",
        "temperature",
        "top_k",
        "wandb_enabled",
        "wandb_project",
        "wandb_entity",
    }
    return {field: None for field in SUMMARY_FIELDS if field not in excluded}


def load_training_metrics(run_dir: Path) -> dict[str, Any]:
    """Load best-checkpoint and final token-likelihood metrics."""
    summary_path = run_dir / "best_checkpoint_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing training summary: {summary_path}")
    summary = load_json(summary_path)
    return {
        "best_epoch": summary.get("best_epoch"),
        "best_eval_cross_entropy": summary.get("best_eval_cross_entropy"),
        "best_eval_accuracy": summary.get("best_eval_accuracy"),
        "best_eval_perplexity": summary.get("best_eval_perplexity"),
        "final_eval_cross_entropy": summary.get("final_eval_cross_entropy"),
        "final_eval_accuracy": summary.get("final_eval_accuracy"),
        "final_eval_perplexity": summary.get("final_eval_perplexity"),
    }


def load_evaluation_metrics(summary_path: Path) -> dict[str, Any]:
    """Load decoded token-prior metrics and model-selection profile score."""
    summary = load_json(summary_path)
    metrics = cast(dict[str, Any], summary["metrics"])
    condition_buckets = cast(list[dict[str, Any]], metrics.get("condition_buckets", []))
    very_low = find_bucket(condition_buckets, "very_low")
    low = find_bucket(condition_buckets, "low")
    profile_score = profile(
        metrics.get("mmd"),
        metrics.get("swd"),
        metrics.get("volatility_wasserstein"),
        metrics.get("terminal_return_wasserstein"),
    )
    return {
        "mmd": metrics.get("mmd"),
        "swd": metrics.get("swd"),
        "volatility_wasserstein": metrics.get("volatility_wasserstein"),
        "terminal_return_wasserstein": metrics.get("terminal_return_wasserstein"),
        "model_selection_profile_score": profile_score,
        "sampled_active_codes": metrics.get("sampled_token_active_code_count"),
        "sampled_token_perplexity": metrics.get("sampled_token_perplexity"),
        "marginal_code_l1": metrics.get("marginal_code_l1"),
        "transition_matrix_l1": metrics.get("transition_matrix_l1"),
        "run_length_distance": metrics.get("run_length_distance"),
        "very_low_mmd": very_low.get("mmd") if very_low else None,
        "low_mmd": low.get("mmd") if low else None,
    }


def load_paper_style_metrics(summary_path: Path) -> dict[str, Any]:
    """Load paper-style discrete metrics and profile score."""
    summary = load_json(summary_path)
    comparisons = cast(dict[str, Any], summary["comparisons"])
    discrete = cast(dict[str, Any], comparisons["discrete"])
    profile_score = profile(
        discrete.get("mmd"),
        discrete.get("swd"),
        discrete.get("volatility_wasserstein"),
        discrete.get("terminal_return_wasserstein"),
    )
    return {
        "paper_mmd": discrete.get("mmd"),
        "paper_swd": discrete.get("swd"),
        "paper_volatility_wasserstein": discrete.get("volatility_wasserstein"),
        "paper_terminal_return_wasserstein": discrete.get("terminal_return_wasserstein"),
        "paper_returns_wasserstein": discrete.get("returns_wasserstein"),
        "paper_maximum_drawdown_wasserstein": discrete.get("maximum_drawdown_wasserstein"),
        "paper_model_selection_profile_score": profile_score,
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


def profile(*values: object) -> float | None:
    """Compute the model-selection profile score when all values are present."""
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return cast(dict[str, Any], loaded)


def write_summaries(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    top_k: int | None,
) -> None:
    """Write aggregate JSON and CSV summaries."""
    metadata = {
        "score_formula": PROFILE_FORMULA,
        "wandb_environment": {
            "WANDB_MODE": "unset",
            **WANDB_ENVIRONMENT,
        },
        "paper_style_requested": bool(args.paper_style),
        "n_sample": int(args.n_sample),
        "evaluation_seed": int(args.seed),
        "temperature": float(args.temperature),
        "top_k": top_k_label(top_k),
    }
    json_path = output_dir / "token_prior_candidate_ablation_summary.json"
    csv_path = output_dir / "token_prior_candidate_ablation_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "results": [dict(result) for result in results]},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in SUMMARY_FIELDS})


if __name__ == "__main__":
    main()
