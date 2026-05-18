"""Run Hawkes-jump tokenizer utilisation ablations without training token priors."""

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

import torch
from torch import Tensor

from time_causal_vae.data.hawkes_jump import HawkesJumpDataset
from time_causal_vae.evaluation.tokenizer import summarise_code_usage
from time_causal_vae.models.discrete.priors.data import (
    load_tokenizer_experiment_config,
)


@dataclass(frozen=True)
class CommandResult:
    """Compact record for one subprocess invocation."""

    command: list[str]
    returncode: int
    runtime_seconds: float
    stdout_tail: str
    stderr_tail: str


def build_parser() -> argparse.ArgumentParser:
    """Build the Hawkes tokenizer ablation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train and inspect Hawkes-jump tokenizer ablations.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Tokenizer experiment YAML configs to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hawkes_jump_tokenizer_ablation",
        help="Output directory under outputs/ for aggregate ablation artefacts.",
    )
    parser.add_argument("--epochs", type=int, help="Override tokenizer epoch count.")
    parser.add_argument("--dry-run", action="store_true", help="Build data/model only.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B for training runs.")
    parser.add_argument("--device", help="Device override passed through to tokenizer CLIs.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory for datasets that require local files.",
    )
    return parser


def main() -> None:
    """Run the configured Hawkes tokenizer ablations."""
    args = build_parser().parse_args()
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for config_path in args.configs:
        summary = run_single_config(
            config_path=Path(config_path),
            output_dir=output_dir,
            epochs=args.epochs,
            dry_run=bool(args.dry_run),
            no_wandb=bool(args.no_wandb),
            device=cast(str | None, args.device),
            base_data_dir=str(args.base_data_dir),
        )
        summaries.append(summary)

    write_json(output_dir / "aggregate_summary.json", {"runs": summaries})
    write_csv(output_dir / "aggregate_summary.csv", summaries)
    print(f"Wrote aggregate summaries under {output_dir}")


def validate_output_dir(output_dir: str) -> Path:
    """Validate that ablation outputs stay below local outputs/."""
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


def run_single_config(
    *,
    config_path: Path,
    output_dir: Path,
    epochs: int | None,
    dry_run: bool,
    no_wandb: bool,
    device: str | None,
    base_data_dir: str,
) -> dict[str, Any]:
    """Train or dry-run one tokenizer config, then collect utilisation diagnostics."""
    raw_config = load_tokenizer_experiment_config(config_path)
    experiment = require_mapping(raw_config, "experiment")
    data = require_mapping(raw_config, "data")
    model = require_mapping(raw_config, "model")
    training = require_mapping(raw_config, "training")

    experiment_name = str(experiment["name"])
    seed = int(experiment.get("seed", 0))
    n_samples = int(data["n_samples"])
    codebook_size = int(model["codebook_size"])
    effective_epochs = int(epochs if epochs is not None else training.get("epochs", 20))
    tokenizer_base = output_dir / "tokenizers"
    tokenizer_dir = tokenizer_base / f"{experiment_name}_seed{seed}"
    evaluation_dir = output_dir / "evaluations" / experiment_name
    token_dir = output_dir / "tokens" / experiment_name
    diagnostic_dir = output_dir / "diagnostics"

    command_results: list[CommandResult] = []
    train_command = tokenizer_train_command(
        config_path=config_path,
        tokenizer_base=tokenizer_base,
        epochs=effective_epochs,
        dry_run=dry_run,
        no_wandb=no_wandb,
        device=device,
        base_data_dir=base_data_dir,
    )
    command_results.append(run_command(train_command))

    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "experiment_name": experiment_name,
        "status": "dry_run_complete" if dry_run else "complete",
        "seed": seed,
        "n_samples": n_samples,
        "n_timesteps": int(data["n_timesteps"]),
        "data_output": str(require_mapping(data, "data_params").get("data_output", "price")),
        "simulation_scheme": str(
            require_mapping(data, "data_params").get("simulation_scheme", "fixed_grid")
        ),
        "codebook_size": codebook_size,
        "encoder_hidden_dim": int(model["encoder_hidden_dim"]),
        "decoder_hidden_dim": int(model["decoder_hidden_dim"]),
        "codebook_dim": int(model["codebook_dim"]),
        "commitment_weight": float(model["commitment_weight"]),
        "epochs": effective_epochs,
        "tokenizer_dir": str(tokenizer_dir),
        "evaluation_dir": str(evaluation_dir),
        "token_dir": str(token_dir),
        "commands": [asdict(result) for result in command_results],
    }
    if command_results[-1].returncode != 0:
        summary["status"] = "failed_train"
        return summary
    if dry_run:
        summary["planned_diagnostics"] = [
            "active_codes",
            "perplexity",
            "jump_vs_nonjump_code_usage",
            "rare_code_activation",
            "token_change_rate_around_jumps",
        ]
        return summary

    eval_command = tokenizer_eval_command(
        config_path=config_path,
        tokenizer_dir=tokenizer_dir,
        evaluation_dir=evaluation_dir,
        n_samples=n_samples,
        seed=seed,
        device=device,
        base_data_dir=base_data_dir,
    )
    command_results.append(run_command(eval_command))
    if command_results[-1].returncode != 0:
        summary["status"] = "failed_evaluation"
        summary["commands"] = [asdict(result) for result in command_results]
        return summary

    extract_command = token_extract_command(
        config_path=config_path,
        tokenizer_dir=tokenizer_dir,
        token_dir=token_dir,
        n_samples=n_samples,
        seed=seed,
        device=device,
        base_data_dir=base_data_dir,
    )
    command_results.append(run_command(extract_command))
    summary["commands"] = [asdict(result) for result in command_results]
    if command_results[-1].returncode != 0:
        summary["status"] = "failed_extraction"
        return summary

    summary.update(load_tokenizer_metrics(evaluation_dir, token_dir))
    alignment = compute_jump_code_alignment(
        raw_config=raw_config,
        token_dir=token_dir,
        codebook_size=codebook_size,
        n_samples=n_samples,
        base_data_dir=base_data_dir,
    )
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    write_json(diagnostic_dir / f"{experiment_name}_jump_code_alignment.json", alignment)
    summary.update(flatten_alignment_summary(alignment))
    return summary


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required mapping section."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def tokenizer_train_command(
    *,
    config_path: Path,
    tokenizer_base: Path,
    epochs: int,
    dry_run: bool,
    no_wandb: bool,
    device: str | None,
    base_data_dir: str,
) -> list[str]:
    """Build the tokenizer training command."""
    command = [
        sys.executable,
        "-m",
        "time_causal_vae.cli.train_tokenizer",
        "--config",
        str(config_path),
        "--output-dir",
        str(tokenizer_base),
        "--epochs",
        str(epochs),
        "--base-data-dir",
        base_data_dir,
    ]
    if dry_run:
        command.append("--dry-run")
    if no_wandb:
        command.append("--no-wandb")
    if device is not None:
        command.extend(["--device", device])
    return command


def tokenizer_eval_command(
    *,
    config_path: Path,
    tokenizer_dir: Path,
    evaluation_dir: Path,
    n_samples: int,
    seed: int,
    device: str | None,
    base_data_dir: str,
) -> list[str]:
    """Build the tokenizer evaluation command."""
    command = [
        sys.executable,
        "-m",
        "time_causal_vae.cli.evaluate_tokenizer",
        "--config",
        str(config_path),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--output-dir",
        str(evaluation_dir),
        "--n-sample-test",
        str(n_samples),
        "--seed",
        str(seed),
        "--base-data-dir",
        base_data_dir,
    ]
    if device is not None:
        command.extend(["--device", device])
    return command


def token_extract_command(
    *,
    config_path: Path,
    tokenizer_dir: Path,
    token_dir: Path,
    n_samples: int,
    seed: int,
    device: str | None,
    base_data_dir: str,
) -> list[str]:
    """Build the frozen token extraction command."""
    command = [
        sys.executable,
        "scripts/extract_token_indices.py",
        "--config",
        str(config_path),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--output-dir",
        str(token_dir),
        "--n-sample",
        str(n_samples),
        "--seed",
        str(seed),
        "--base-data-dir",
        base_data_dir,
    ]
    if device is not None:
        command.extend(["--device", device])
    return command


def run_command(command: Sequence[str]) -> CommandResult:
    """Run a subprocess and keep compact stdout/stderr tails."""
    print("Running:", " ".join(command))
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    runtime = time.perf_counter() - started
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    return CommandResult(
        command=list(command),
        returncode=int(completed.returncode),
        runtime_seconds=runtime,
        stdout_tail=tail(completed.stdout),
        stderr_tail=tail(completed.stderr),
    )


def tail(text: str, *, n_lines: int = 40) -> str:
    """Return the last ``n_lines`` from command output."""
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


def load_tokenizer_metrics(evaluation_dir: Path, token_dir: Path) -> dict[str, Any]:
    """Load primary tokenizer evaluation and extraction summaries."""
    metrics: dict[str, Any] = {}
    eval_summary_path = evaluation_dir / "tokenizer_summary.json"
    if eval_summary_path.exists():
        eval_summary = load_json(eval_summary_path)
        for key in [
            "reconstruction_l1",
            "reconstruction_l2",
            "terminal_return_error",
            "volatility_reconstruction_error",
            "active_code_count",
            "active_code_ratio",
            "codebook_perplexity",
            "index_entropy",
        ]:
            if key in eval_summary:
                metrics[f"eval_{key}"] = eval_summary[key]
    token_summary_path = token_dir / "token_dataset_summary.json"
    if token_summary_path.exists():
        token_summary = load_json(token_summary_path)
        combined = cast(Mapping[str, Any], token_summary.get("combined", {}))
        for key in ["active_code_count", "active_code_ratio", "codebook_perplexity"]:
            if key in combined:
                metrics[f"extracted_combined_{key}"] = combined[key]
    return metrics


def compute_jump_code_alignment(
    *,
    raw_config: Mapping[str, Any],
    token_dir: Path,
    codebook_size: int,
    n_samples: int,
    base_data_dir: str,
) -> dict[str, Any]:
    """Compute oracle jump/code alignment from extracted indices and simulator metadata."""
    train_dataset, eval_dataset = build_hawkes_oracle_datasets(raw_config, n_samples=n_samples)
    split_summaries = {
        "train": split_alignment(
            indices=load_token_indices(token_dir / "train_tokens.pt"),
            jumps=oracle_jump_indicators(train_dataset),
            codebook_size=codebook_size,
        ),
        "eval": split_alignment(
            indices=load_token_indices(token_dir / "eval_tokens.pt"),
            jumps=oracle_jump_indicators(eval_dataset),
            codebook_size=codebook_size,
        ),
    }
    return {
        "jump_window_radius": 1,
        "split_summaries": split_summaries,
        "combined": combine_split_alignment(split_summaries),
    }


def build_hawkes_oracle_datasets(
    raw_config: Mapping[str, Any],
    *,
    n_samples: int,
) -> tuple[HawkesJumpDataset, HawkesJumpDataset]:
    """Build Hawkes datasets directly so oracle simulator metadata is preserved."""
    experiment = require_mapping(raw_config, "experiment")
    data = require_mapping(raw_config, "data")
    if str(experiment["dataset"]) not in {"hawkes_jump", "HawkesJump"}:
        raise ValueError("Jump/code alignment currently supports only Hawkes-jump configs.")
    n_timesteps = int(data["n_timesteps"])
    data_params = dict(require_mapping(data, "data_params"))
    train_dataset = HawkesJumpDataset(n_samples, n_timesteps, **data_params)
    eval_params = dict(data_params)
    if eval_params.get("seed") is not None:
        eval_params["seed"] = int(eval_params["seed"]) + 1
    eval_dataset = HawkesJumpDataset(n_samples, n_timesteps, **eval_params)
    return train_dataset, eval_dataset


def load_token_indices(path: Path) -> Tensor:
    """Load extracted tokenizer indices from a token payload file."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "indices" not in payload:
        raise ValueError(f"Token payload does not contain indices: {path}")
    indices = payload["indices"]
    if not isinstance(indices, Tensor):
        raise TypeError(f"Token indices must be a tensor: {path}")
    return indices.long()


def oracle_jump_indicators(dataset: Any) -> Tensor:
    """Read oracle jump indicators from a Hawkes-jump dataset object."""
    indicators = getattr(dataset, "jump_indicators", None)
    if indicators is None:
        raise ValueError("Dataset does not expose oracle jump_indicators.")
    if not isinstance(indicators, Tensor):
        indicators = torch.as_tensor(indicators)
    return indicators.detach().cpu().float()


def split_alignment(*, indices: Tensor, jumps: Tensor, codebook_size: int) -> dict[str, Any]:
    """Summarise token usage in oracle jump and non-jump windows for one split."""
    if indices.ndim != 2:
        return {
            "supported": False,
            "note": f"Expected [sample, timestep] vector tokens; got {list(indices.shape)}.",
        }
    jump_mask = jump_window_mask(jumps).to(torch.bool)
    if jump_mask.shape != indices.shape:
        raise ValueError(
            f"Jump mask shape {list(jump_mask.shape)} does not match indices {list(indices.shape)}."
        )
    nonjump_mask = ~jump_mask
    all_counts = code_counts(indices, torch.ones_like(jump_mask, dtype=torch.bool), codebook_size)
    rare_codes = rare_active_codes(all_counts)
    jump_counts = code_counts(indices, jump_mask, codebook_size)
    nonjump_counts = code_counts(indices, nonjump_mask, codebook_size)
    return {
        "supported": True,
        "jump_positions": int(jump_mask.sum().item()),
        "nonjump_positions": int(nonjump_mask.sum().item()),
        "jump_code_usage": summarise_code_usage(jump_counts),
        "nonjump_code_usage": summarise_code_usage(nonjump_counts),
        "rare_code_count": int(rare_codes.sum().item()),
        "rare_activation_jump": activation_fraction(indices, jump_mask, rare_codes),
        "rare_activation_nonjump": activation_fraction(indices, nonjump_mask, rare_codes),
        "rare_activation_lift": safe_ratio(
            activation_fraction(indices, jump_mask, rare_codes),
            activation_fraction(indices, nonjump_mask, rare_codes),
        ),
        "token_change_rate_jump": token_change_rate(indices, jump_mask),
        "token_change_rate_nonjump": token_change_rate(indices, nonjump_mask),
        "jump_nonjump_code_l1": distribution_l1(jump_counts, nonjump_counts),
    }


def jump_window_mask(jumps: Tensor, *, radius: int = 1) -> Tensor:
    """Return positions within ``radius`` timesteps of any oracle jump."""
    squeezed = jumps.squeeze(-1) if jumps.ndim == 3 and jumps.shape[-1] == 1 else jumps
    mask = squeezed > 0.0
    expanded = mask.clone()
    for offset in range(1, radius + 1):
        expanded[:, offset:] |= mask[:, :-offset]
        expanded[:, :-offset] |= mask[:, offset:]
    return expanded


def code_counts(indices: Tensor, mask: Tensor, codebook_size: int) -> Tensor:
    """Return code counts under a boolean mask."""
    selected = indices[mask]
    if selected.numel() == 0:
        return torch.zeros(codebook_size, dtype=torch.long)
    return torch.bincount(selected.reshape(-1), minlength=codebook_size)[:codebook_size]


def rare_active_codes(counts: Tensor, *, max_frequency: float = 0.01) -> Tensor:
    """Return active codes whose frequency is at most ``max_frequency``."""
    total = float(counts.sum().item())
    if total <= 0.0:
        return torch.zeros_like(counts, dtype=torch.bool)
    frequencies = counts.float() / total
    return (counts > 0) & (frequencies <= max_frequency)


def activation_fraction(indices: Tensor, mask: Tensor, code_mask: Tensor) -> float:
    """Return the fraction of selected positions assigned to selected codes."""
    selected = indices[mask]
    if selected.numel() == 0 or int(code_mask.sum().item()) == 0:
        return 0.0
    return float(code_mask[selected].float().mean().item())


def token_change_rate(indices: Tensor, mask: Tensor) -> float:
    """Return adjacent token-change rate for masked destination positions."""
    if indices.shape[1] <= 1:
        return 0.0
    changes = indices[:, 1:] != indices[:, :-1]
    destination_mask = mask[:, 1:]
    if int(destination_mask.sum().item()) == 0:
        return 0.0
    return float(changes[destination_mask].float().mean().item())


def distribution_l1(left_counts: Tensor, right_counts: Tensor) -> float:
    """Return L1 distance between two empirical code distributions."""
    left_total = float(left_counts.sum().item())
    right_total = float(right_counts.sum().item())
    if left_total <= 0.0 or right_total <= 0.0:
        return 0.0
    left = left_counts.float() / left_total
    right = right_counts.float() / right_total
    return float((left - right).abs().sum().item())


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio with zero denominator mapped to zero."""
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def combine_split_alignment(split_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return a compact cross-split alignment summary."""
    supported = [summary for summary in split_summaries.values() if summary.get("supported")]
    if not supported:
        return {"supported": False}
    jump_positions = sum(int(summary["jump_positions"]) for summary in supported)
    nonjump_positions = sum(int(summary["nonjump_positions"]) for summary in supported)
    weighted_jump_change = weighted_average(supported, "token_change_rate_jump", "jump_positions")
    weighted_nonjump_change = weighted_average(
        supported,
        "token_change_rate_nonjump",
        "nonjump_positions",
    )
    return {
        "supported": True,
        "jump_positions": jump_positions,
        "nonjump_positions": nonjump_positions,
        "token_change_rate_jump": weighted_jump_change,
        "token_change_rate_nonjump": weighted_nonjump_change,
        "token_change_rate_lift": safe_ratio(weighted_jump_change, weighted_nonjump_change),
        "mean_jump_nonjump_code_l1": float(
            sum(float(summary["jump_nonjump_code_l1"]) for summary in supported) / len(supported)
        ),
        "mean_rare_activation_lift": float(
            sum(float(summary["rare_activation_lift"]) for summary in supported) / len(supported)
        ),
    }


def weighted_average(
    summaries: Sequence[Mapping[str, Any]],
    value_key: str,
    weight_key: str,
) -> float:
    """Return a weighted average for split summaries."""
    total_weight = sum(float(summary[weight_key]) for summary in summaries)
    if total_weight <= 0.0:
        return 0.0
    return float(
        sum(float(summary[value_key]) * float(summary[weight_key]) for summary in summaries)
        / total_weight
    )


def flatten_alignment_summary(alignment: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the combined alignment summary into aggregate CSV columns."""
    combined = cast(Mapping[str, Any], alignment.get("combined", {}))
    return {
        f"alignment_{key}": value
        for key, value in combined.items()
        if isinstance(value, bool | int | float | str)
    }


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], loaded)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write aggregate scalar run summaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_rows = [{key: value for key, value in row.items() if is_scalar(value)} for row in rows]
    fieldnames = sorted({key for row in scalar_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def is_scalar(value: Any) -> bool:
    """Return whether a value is suitable for one CSV cell."""
    return isinstance(value, str | int | float | bool) or value is None


def to_jsonable(value: Any) -> Any:
    """Convert tensors and paths into JSON-safe Python values."""
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
