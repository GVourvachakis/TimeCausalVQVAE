"""Build a dry-run command plan for per-experiment discrete model selection."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "experiments"
OUTPUTS_ROOT = REPO_ROOT / "outputs"

ExperimentName = Literal["black_scholes", "heston", "pdv", "sp500_vix"]
ConfigKind = Literal["tokenizer", "token_prior"]


@dataclass(frozen=True)
class CandidateSpec:
    """One tokenizer and token-prior candidate pair."""

    experiment: ExperimentName
    candidate: str
    tokenizer_config: str
    prior_config: str
    conditional: bool
    description: str


@dataclass(frozen=True)
class CommandPlan:
    """Dry-run commands and validation state for one candidate."""

    experiment: str
    candidate: str
    conditional: bool
    tokenizer_config: str
    prior_config: str
    tokenizer_dir: str
    token_data_dir: str
    tokenizer_train_command: list[str]
    token_extract_command: list[str]
    prior_train_command: list[str]
    validation_status: str
    smoke_status: str
    notes: str


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        experiment="black_scholes",
        candidate="standard_vq_additive_ar",
        tokenizer_config="black_scholes_causal_vq_tokenizer_codebook64_codebookdim16.yaml",
        prior_config="black_scholes_causal_token_prior_additive.yaml",
        conditional=False,
        description="Standard VQ tokenizer with the single-code causal AR prior.",
    ),
    CandidateSpec(
        experiment="black_scholes",
        candidate="hidden128_additive_ar",
        tokenizer_config="black_scholes_causal_vq_tokenizer_hidden128.yaml",
        prior_config="black_scholes_causal_token_prior_hidden128_additive.yaml",
        conditional=False,
        description="Hidden128 tokenizer with the single-code causal AR prior.",
    ),
    CandidateSpec(
        experiment="black_scholes",
        candidate="hidden128_conv_transformer_k3",
        tokenizer_config="black_scholes_causal_vq_tokenizer_hidden128.yaml",
        prior_config="black_scholes_causal_token_prior_hidden128_conv_transformer.yaml",
        conditional=False,
        description="Hidden128 tokenizer with causal conv-transformer k3 prior.",
    ),
    CandidateSpec(
        experiment="heston",
        candidate="standard_vq_additive_ar",
        tokenizer_config="heston_causal_vq_tokenizer.yaml",
        prior_config="heston_causal_token_prior_additive.yaml",
        conditional=False,
        description="Standard VQ tokenizer with the single-code causal AR prior.",
    ),
    CandidateSpec(
        experiment="heston",
        candidate="hidden128_additive_ar",
        tokenizer_config="heston_causal_vq_tokenizer_hidden128.yaml",
        prior_config="heston_causal_token_prior_hidden128_additive.yaml",
        conditional=False,
        description="Hidden128 tokenizer with the single-code causal AR prior.",
    ),
    CandidateSpec(
        experiment="heston",
        candidate="hidden128_conv_transformer_k3",
        tokenizer_config="heston_causal_vq_tokenizer_hidden128.yaml",
        prior_config="heston_causal_token_prior_hidden128_conv_transformer.yaml",
        conditional=False,
        description="Hidden128 tokenizer with causal conv-transformer k3 prior.",
    ),
    CandidateSpec(
        experiment="pdv",
        candidate="conditional_standard_vq_additive_ar",
        tokenizer_config="pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml",
        prior_config="pdv_causal_token_prior_additive_seed1.yaml",
        conditional=True,
        description="Conditional standard VQ tokenizer with additive scalar prior.",
    ),
    CandidateSpec(
        experiment="pdv",
        candidate="conditional_hidden128_additive_ar",
        tokenizer_config="pdv_causal_vq_tokenizer_hidden128.yaml",
        prior_config="pdv_causal_token_prior_hidden128_additive.yaml",
        conditional=True,
        description="Conditional hidden128 tokenizer with additive scalar prior.",
    ),
    CandidateSpec(
        experiment="pdv",
        candidate="conditional_hidden128_conv_transformer_k3",
        tokenizer_config="pdv_causal_vq_tokenizer_hidden128.yaml",
        prior_config="pdv_causal_token_prior_hidden128_conv_transformer.yaml",
        conditional=True,
        description="Conditional hidden128 tokenizer with causal conv-transformer k3 prior.",
    ),
    CandidateSpec(
        experiment="sp500_vix",
        candidate="conditional_standard_vq_additive_ar",
        tokenizer_config="sp500_vix_causal_vq_tokenizer.yaml",
        prior_config="sp500_vix_causal_token_prior_additive.yaml",
        conditional=True,
        description="Promoted public S&P500/VIX standard VQ plus additive prior.",
    ),
    CandidateSpec(
        experiment="sp500_vix",
        candidate="conditional_hidden128_conv_transformer_k3",
        tokenizer_config="sp500_vix_causal_vq_tokenizer_hidden128.yaml",
        prior_config="sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml",
        conditional=True,
        description="Conditional hidden128 tokenizer with causal conv-transformer k3 prior.",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the runner CLI parser."""
    parser = argparse.ArgumentParser(
        description="Create a per-experiment model-selection dry-run plan.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("black_scholes", "heston", "pdv", "sp500_vix"),
        default=("black_scholes", "heston", "pdv", "sp500_vix"),
        help="Experiments to include.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/per_experiment_selection_dry",
        help="Ignored output directory for aggregate plan JSON/CSV files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the command plan without executing any training commands.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Execute tokenizer CLI dry-runs only. Token-prior dry-runs are listed.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epoch override for command plans.")
    parser.add_argument(
        "--n-sample",
        type=int,
        default=128,
        help="Sample count for token extraction and later evaluation commands.",
    )
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory for market-data commands.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Add --no-wandb to plan commands.")
    parser.add_argument(
        "--wandb-entity",
        default="tc_vae",
        help="W&B entity to use when W&B is enabled.",
    )
    return parser


def main() -> None:
    """Create and optionally smoke-check the per-experiment selection setup."""
    args = build_parser().parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")
    if args.smoke and args.dry_run:
        raise SystemExit("Use either --dry-run or --smoke, not both.")

    output_dir = validate_output_dir(args.output_dir)
    selected = select_candidates(cast(Sequence[ExperimentName], args.experiments))
    plans = [build_command_plan(candidate, args=args) for candidate in selected]
    if args.smoke:
        plans = run_tokenizer_smoke(plans)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "selection_dry_run_plan.json", plans, args=args)
    write_csv(output_dir / "selection_dry_run_plan.csv", plans)
    print_summary(plans, output_dir=output_dir)


def validate_output_dir(raw_output_dir: str) -> Path:
    """Validate that aggregate plan files stay below ignored outputs/."""
    path = Path(raw_output_dir)
    resolved = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(OUTPUTS_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"--output-dir must be under ignored outputs/. Received: {raw_output_dir}"
        ) from exc
    return resolved


def select_candidates(experiments: Sequence[ExperimentName]) -> list[CandidateSpec]:
    """Return candidate specs for the requested experiments."""
    requested = set(experiments)
    return [candidate for candidate in CANDIDATES if candidate.experiment in requested]


def build_command_plan(candidate: CandidateSpec, *, args: argparse.Namespace) -> CommandPlan:
    """Validate one candidate and build dry-run commands."""
    tokenizer_path = config_path(candidate.tokenizer_config)
    prior_path = config_path(candidate.prior_config)
    tokenizer_config = load_config(tokenizer_path, kind="tokenizer")
    prior_config = load_config(prior_path, kind="token_prior")
    validate_candidate_config(
        candidate, tokenizer_config=tokenizer_config, prior_config=prior_config
    )

    prior_data = require_mapping(prior_config, "data", prior_path)
    tokenizer_dir = str(prior_data["tokenizer_dir"])
    token_data_dir = str(prior_data["token_data_dir"])
    tokenizer_output_dir = str(Path(tokenizer_dir).parent)
    prior_output_dir = str(Path(token_data_dir).parent / "prior")
    tokenizer_command = tokenizer_train_command(
        config=tokenizer_path,
        output_dir=Path(tokenizer_output_dir),
        args=args,
    )
    extract_command = token_extract_command(
        config=tokenizer_path,
        tokenizer_dir=Path(tokenizer_dir),
        output_dir=Path(token_data_dir),
        args=args,
    )
    prior_command = prior_train_command(
        config=prior_path, output_dir=Path(prior_output_dir), args=args
    )
    note = candidate.description
    if not candidate.conditional:
        note = f"{note} Synthetic labels are not used as prior conditions."
    return CommandPlan(
        experiment=candidate.experiment,
        candidate=candidate.candidate,
        conditional=candidate.conditional,
        tokenizer_config=relative_to_repo(tokenizer_path),
        prior_config=relative_to_repo(prior_path),
        tokenizer_dir=tokenizer_dir,
        token_data_dir=token_data_dir,
        tokenizer_train_command=tokenizer_command,
        token_extract_command=extract_command,
        prior_train_command=prior_command,
        validation_status="ok",
        smoke_status="not_run",
        notes=note,
    )


def config_path(filename: str) -> Path:
    """Return the absolute path for one experiment config, requiring it to exist."""
    path = CONFIG_DIR / filename
    if not path.exists():
        raise SystemExit(f"Missing required config: {path}")
    return path


def load_config(path: Path, *, kind: ConfigKind) -> dict[str, Any]:
    """Load and validate one tokenizer or token-prior config."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must contain a YAML mapping.")
    for section in ("experiment", "data", "model", "training"):
        require_mapping(cast(Mapping[str, Any], loaded), section, path)
    model = require_mapping(cast(Mapping[str, Any], loaded), "model", path)
    family = str(model.get("family"))
    expected = "causal_vq_tokenizer" if kind == "tokenizer" else "causal_token_prior"
    if family != expected:
        raise SystemExit(f"{path} model.family must be {expected!r}; got {family!r}.")
    return cast(dict[str, Any], loaded)


def require_mapping(raw_config: Mapping[str, Any], key: str, path: Path) -> dict[str, Any]:
    """Return a required YAML mapping section."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"{path} requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def validate_candidate_config(
    candidate: CandidateSpec,
    *,
    tokenizer_config: Mapping[str, Any],
    prior_config: Mapping[str, Any],
) -> None:
    """Validate cross-config assumptions without loading model code."""
    tokenizer_model = require_mapping(
        tokenizer_config, "model", config_path(candidate.tokenizer_config)
    )
    prior_model = require_mapping(prior_config, "model", config_path(candidate.prior_config))
    tokenizer_experiment = require_mapping(
        tokenizer_config,
        "experiment",
        config_path(candidate.tokenizer_config),
    )
    prior_experiment = require_mapping(
        prior_config, "experiment", config_path(candidate.prior_config)
    )
    if tokenizer_experiment.get("dataset") != prior_experiment.get("dataset"):
        raise SystemExit(
            f"{candidate.candidate} dataset mismatch: "
            f"{tokenizer_experiment.get('dataset')} vs {prior_experiment.get('dataset')}"
        )
    if int(tokenizer_model["codebook_size"]) != int(prior_model["codebook_size"]):
        raise SystemExit(
            f"{candidate.candidate} codebook mismatch: "
            f"{tokenizer_model['codebook_size']} vs {prior_model['codebook_size']}"
        )
    if int(tokenizer_model["data_length"]) != int(prior_model["sequence_length"]):
        raise SystemExit(
            f"{candidate.candidate} sequence mismatch: "
            f"{tokenizer_model['data_length']} vs {prior_model['sequence_length']}"
        )
    prior_type = str(prior_model.get("prior_type", "single_code"))
    if prior_type not in {"single_code", "causal_conv_transformer"}:
        raise SystemExit(f"{candidate.prior_config} uses excluded or unsupported prior_type.")
    if candidate.conditional and str(prior_model.get("condition_injection")) != "additive":
        raise SystemExit(f"{candidate.prior_config} must use additive conditioning.")
    if not candidate.conditional and "condition_injection" in prior_model:
        raise SystemExit(f"{candidate.prior_config} must remain unconditioned.")


def tokenizer_train_command(
    *, config: Path, output_dir: Path, args: argparse.Namespace
) -> list[str]:
    """Return the tokenizer training dry-run command."""
    command = [
        "poetry",
        "run",
        "tcvae-train-tokenizer",
        "--config",
        relative_to_repo(config),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--base-data-dir",
        str(args.base_data_dir),
        "--dry-run",
    ]
    append_wandb_args(command, project="time-causal-vq-tokenizer", args=args)
    return command


def token_extract_command(
    *,
    config: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Return the token extraction command to run after a tokenizer checkpoint exists."""
    return [
        "poetry",
        "run",
        "python",
        "scripts/extract_token_indices.py",
        "--config",
        relative_to_repo(config),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--output-dir",
        str(output_dir),
        "--base-data-dir",
        str(args.base_data_dir),
        "--n-sample",
        str(args.n_sample),
    ]


def prior_train_command(*, config: Path, output_dir: Path, args: argparse.Namespace) -> list[str]:
    """Return the token-prior training dry-run command."""
    command = [
        "poetry",
        "run",
        "tcvae-train-token-prior",
        "--config",
        relative_to_repo(config),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--dry-run",
    ]
    append_wandb_args(command, project="time-causal-token-prior", args=args)
    return command


def append_wandb_args(command: list[str], *, project: str, args: argparse.Namespace) -> None:
    """Append W&B profile arguments to one command."""
    if bool(args.no_wandb):
        command.append("--no-wandb")
        return
    command.extend(
        ["--wandb", "--wandb-project", project, "--wandb-entity", str(args.wandb_entity)]
    )


def run_tokenizer_smoke(plans: Sequence[CommandPlan]) -> list[CommandPlan]:
    """Execute tokenizer CLI dry-runs and return plans with smoke status."""
    updated = []
    for plan in plans:
        result = subprocess.run(
            plan.tokenizer_train_command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        status = "ok" if result.returncode == 0 else f"failed:{result.returncode}"
        updated.append(
            CommandPlan(
                **{
                    **asdict(plan),
                    "smoke_status": status,
                    "notes": smoke_note(plan.notes, result),
                }
            )
        )
    return updated


def smoke_note(note: str, result: subprocess.CompletedProcess[str]) -> str:
    """Append compact smoke output to the candidate note."""
    if result.returncode == 0:
        return f"{note} Tokenizer CLI dry-run succeeded."
    stderr = result.stderr.strip().splitlines()
    message = stderr[-1] if stderr else "no stderr"
    return f"{note} Tokenizer CLI dry-run failed: {message}"


def write_json(path: Path, plans: Sequence[CommandPlan], *, args: argparse.Namespace) -> None:
    """Write the aggregate command plan JSON file."""
    payload = {
        "script": "scripts/run_per_experiment_model_selection.py",
        "mode": "dry_run" if bool(args.dry_run) else "smoke" if bool(args.smoke) else "plan",
        "epochs": int(args.epochs),
        "n_sample": int(args.n_sample),
        "no_wandb": bool(args.no_wandb),
        "candidates": [serialise_plan(plan) for plan in plans],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, plans: Sequence[CommandPlan]) -> None:
    """Write the aggregate command plan CSV file."""
    fieldnames = [
        "experiment",
        "candidate",
        "conditional",
        "tokenizer_config",
        "prior_config",
        "tokenizer_dir",
        "token_data_dir",
        "tokenizer_train_command",
        "token_extract_command",
        "prior_train_command",
        "validation_status",
        "smoke_status",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for plan in plans:
            writer.writerow(serialise_plan(plan))


def serialise_plan(plan: CommandPlan) -> dict[str, Any]:
    """Return a JSON/CSV-friendly plan mapping."""
    payload = asdict(plan)
    payload["tokenizer_train_command"] = shlex.join(plan.tokenizer_train_command)
    payload["token_extract_command"] = shlex.join(plan.token_extract_command)
    payload["prior_train_command"] = shlex.join(plan.prior_train_command)
    return payload


def print_summary(plans: Sequence[CommandPlan], *, output_dir: Path) -> None:
    """Print a compact dry-run summary."""
    print("Per-experiment model-selection setup")
    print(f"candidates: {len(plans)}")
    print(f"output_dir: {output_dir}")
    for plan in plans:
        print(
            f"- {plan.experiment}/{plan.candidate}: "
            f"{plan.validation_status}, smoke={plan.smoke_status}"
        )
    print("files: selection_dry_run_plan.json, selection_dry_run_plan.csv")


def relative_to_repo(path: Path) -> str:
    """Return a repository-relative path string."""
    return str(path.resolve().relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
