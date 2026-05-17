"""Plan and optionally run final per-experiment model evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from time_causal_vae.experiments.selection_profiles import (
    ProfileName,
    score_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
SELECTION_RESULTS_PATH = OUTPUTS_ROOT / "per_experiment_selection" / "selection_results.json"

ExperimentName = Literal["black_scholes", "heston", "pdv", "sp500_vix"]
TargetKind = Literal["continuous", "discrete", "no_leakage"]

CONTINUOUS_BASELINES: dict[ExperimentName, str] = {
    "black_scholes": "configs/experiments/black_scholes_beta_cvae.yaml",
    "heston": "configs/experiments/heston_info_cvae.yaml",
    "pdv": "configs/experiments/pdv_info_cvae.yaml",
    "sp500_vix": "configs/experiments/sp500_vix_beta_cvae.yaml",
}

PUBLIC_DISCRETE_PRIORS: dict[ExperimentName, str | None] = {
    "black_scholes": "configs/experiments/black_scholes_causal_token_prior.yaml",
    "heston": "configs/experiments/heston_causal_token_prior_additive.yaml",
    "pdv": "configs/experiments/pdv_causal_token_prior_additive_seed1.yaml",
    "sp500_vix": "configs/experiments/sp500_vix_causal_token_prior_additive.yaml",
}

STATIC_PROVISIONAL_SELECTIONS: dict[ExperimentName, str] = {
    "black_scholes": "hidden128_conv_transformer_k3",
    "heston": "standard_vq_additive_ar",
    "pdv": "conditional_standard_vq_additive_ar",
    "sp500_vix": "conditional_hidden128_conv_transformer_k3",
}

PATH_METRICS = (
    "mmd",
    "swd",
    "returns_wasserstein",
    "terminal_return_wasserstein",
    "volatility_wasserstein",
    "maximum_drawdown_wasserstein",
    "return_autocorrelation_within_path_l1",
    "squared_return_autocorrelation_within_path_l1",
)

GENERIC_TOKEN_PRIOR_METRICS = (
    "mmd",
    "swd",
    "terminal_return_wasserstein",
    "volatility_wasserstein",
)

SP500_PAPER_STYLE_METRICS = (*PATH_METRICS, "vix_bucket_path_diagnostics")


@dataclass(frozen=True)
class SelectionCandidate:
    """One provisional or baseline discrete candidate."""

    experiment: ExperimentName
    candidate: str
    tokenizer_config: str
    prior_config: str
    tokenizer_dir: str
    token_data_dir: str
    prior_dir: str
    conditional: bool


@dataclass(frozen=True)
class EvaluationTarget:
    """One final-evaluation or no-leakage target."""

    experiment: str
    target: str
    kind: TargetKind
    roles: list[str]
    command: list[str]
    output_dir: str
    status: str
    supported_metrics: list[str]
    missing_metrics: list[str]
    warnings: list[str] = field(default_factory=list)
    runtime_seconds: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""


def build_parser() -> argparse.ArgumentParser:
    """Build the final-evaluation runner parser."""
    parser = argparse.ArgumentParser(
        description="Plan or run final per-experiment evaluation for provisional selections.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("black_scholes", "heston", "pdv", "sp500_vix"),
        default=("black_scholes", "heston", "pdv", "sp500_vix"),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/per_experiment_final_evaluation",
        help="Ignored output directory for aggregate evaluation JSON/CSV files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write a plan without evaluation.")
    parser.add_argument("--n-sample", type=int, default=1000)
    parser.add_argument(
        "--profile",
        choices=("distributional", "tail_risk", "sequential_dependence", "balanced_market"),
        default="balanced_market",
    )
    parser.add_argument("--base-data-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", default="none")
    parser.add_argument(
        "--continuous-model-dir",
        action="append",
        default=[],
        metavar="EXPERIMENT=PATH",
        help="Optional final_model directory override for a continuous baseline.",
    )
    return parser


def main() -> None:
    """Plan or execute final evaluation targets."""
    args = build_parser().parse_args()
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive.")

    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = cast(Sequence[ExperimentName], args.experiments)
    continuous_model_dirs = parse_continuous_model_dirs(args.continuous_model_dir)
    candidates = load_selection_candidates(SELECTION_RESULTS_PATH)
    targets = build_targets(
        experiments=experiments,
        candidates=candidates,
        output_dir=output_dir,
        args=args,
        continuous_model_dirs=continuous_model_dirs,
    )
    if not args.dry_run:
        targets = execute_targets(targets, profile=cast(ProfileName, args.profile))

    write_json(output_dir / "final_evaluation_plan.json", targets, args=args)
    write_csv(output_dir / "final_evaluation_plan.csv", targets)
    print_summary(targets, output_dir=output_dir)
    failed = [target for target in targets if target.status.startswith("failed")]
    if failed:
        raise SystemExit(f"{len(failed)} final-evaluation target(s) failed.")


def validate_output_dir(raw_output_dir: str) -> Path:
    """Validate that aggregate files stay below ignored outputs/."""
    path = Path(raw_output_dir)
    resolved = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(OUTPUTS_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"--output-dir must be under ignored outputs/. Received: {raw_output_dir}"
        ) from exc
    return resolved


def parse_continuous_model_dirs(raw_values: Sequence[str]) -> dict[ExperimentName, str]:
    """Parse ``EXPERIMENT=PATH`` continuous model-dir overrides."""
    parsed: dict[ExperimentName, str] = {}
    valid = set(CONTINUOUS_BASELINES)
    for value in raw_values:
        if "=" not in value:
            raise SystemExit("--continuous-model-dir must use EXPERIMENT=PATH.")
        experiment, path = value.split("=", 1)
        if experiment not in valid:
            raise SystemExit(f"Unknown experiment in --continuous-model-dir: {experiment}")
        parsed[cast(ExperimentName, experiment)] = path
    return parsed


def load_selection_candidates(path: Path) -> dict[tuple[ExperimentName, str], SelectionCandidate]:
    """Load candidate directories and configs from the aggregate selection JSON."""
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"Selection results must be a mapping: {path}")
    raw_candidates = loaded.get("candidates", [])
    candidates: dict[tuple[ExperimentName, str], SelectionCandidate] = {}
    if not isinstance(raw_candidates, list):
        raise SystemExit(f"Selection results candidates must be a list: {path}")
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            continue
        experiment = cast(ExperimentName, str(raw_candidate["experiment"]))
        candidate = str(raw_candidate["candidate"])
        candidates[(experiment, candidate)] = SelectionCandidate(
            experiment=experiment,
            candidate=candidate,
            tokenizer_config=str(raw_candidate["tokenizer_config"]),
            prior_config=str(raw_candidate["prior_config"]),
            tokenizer_dir=str(raw_candidate["tokenizer_dir"]),
            token_data_dir=str(raw_candidate["token_data_dir"]),
            prior_dir=str(raw_candidate["prior_dir"]),
            conditional=bool(raw_candidate["conditional"]),
        )
    return candidates


def build_targets(
    *,
    experiments: Sequence[ExperimentName],
    candidates: Mapping[tuple[ExperimentName, str], SelectionCandidate],
    output_dir: Path,
    args: argparse.Namespace,
    continuous_model_dirs: Mapping[ExperimentName, str],
) -> list[EvaluationTarget]:
    """Build final-evaluation and no-leakage targets."""
    targets: list[EvaluationTarget] = []
    for experiment in experiments:
        targets.append(
            continuous_target(
                experiment=experiment,
                output_dir=output_dir,
                args=args,
                continuous_model_dirs=continuous_model_dirs,
            )
        )
        discrete_targets = unique_discrete_targets(experiment, candidates)
        for candidate, roles in discrete_targets:
            targets.append(
                discrete_target(
                    candidate=candidate,
                    roles=roles,
                    output_dir=output_dir,
                    args=args,
                )
            )
            targets.extend(
                no_leakage_targets(
                    candidate=candidate,
                    output_dir=output_dir,
                    args=args,
                )
            )
    return targets


def continuous_target(
    *,
    experiment: ExperimentName,
    output_dir: Path,
    args: argparse.Namespace,
    continuous_model_dirs: Mapping[ExperimentName, str],
) -> EvaluationTarget:
    """Build a continuous-baseline evaluation target."""
    config = CONTINUOUS_BASELINES[experiment]
    model_dir = continuous_model_dirs.get(experiment) or discover_continuous_model_dir(experiment)
    target_output = output_dir / experiment / "continuous"
    command: list[str] = []
    status = "not_available"
    warnings = ["continuous final_model directory is required before path evaluation."]
    if model_dir is not None:
        command = [
            "poetry",
            "run",
            "tcvae-evaluate",
            "--config",
            config,
            "--model-dir",
            model_dir,
            "--output-dir",
            str(target_output),
            "--base-data-dir",
            str(args.base_data_dir),
            "--n-sample-test",
            str(args.n_sample),
            "--seed",
            str(args.seed),
        ]
        status = "dry_run" if bool(args.dry_run) else "pending"
        warnings = []
    return EvaluationTarget(
        experiment=experiment,
        target="continuous_selected_baseline",
        kind="continuous",
        roles=["continuous_selected_baseline"],
        command=command,
        output_dir=str(target_output),
        status=status,
        supported_metrics=["legacy_hyper_metric"],
        missing_metrics=list(PATH_METRICS),
        warnings=warnings,
    )


def discover_continuous_model_dir(experiment: ExperimentName) -> str | None:
    """Return the newest local reproduction final_model directory when present."""
    candidates = sorted(
        (OUTPUTS_ROOT / "reproduction" / experiment).glob("*/final_model"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
    )
    if candidates:
        return str(candidates[-1])
    sp500_default = OUTPUTS_ROOT / "sp500_vix_continuous" / "beta_cvae" / "final_model"
    if experiment == "sp500_vix" and sp500_default.is_dir():
        return str(sp500_default)
    return None


def unique_discrete_targets(
    experiment: ExperimentName,
    candidates: Mapping[tuple[ExperimentName, str], SelectionCandidate],
) -> list[tuple[SelectionCandidate, list[str]]]:
    """Return public and provisional discrete targets without duplicates."""
    target_roles: dict[str, list[str]] = {}
    public = public_candidate(experiment, candidates)
    if public is not None:
        target_roles.setdefault(public.candidate, []).append("public_discrete_baseline")
    provisional_name = provisional_selection(experiment, candidates)
    provisional = candidates.get((experiment, provisional_name))
    if provisional is not None:
        target_roles.setdefault(provisional.candidate, []).append("provisional_best_discrete")
    targets: list[tuple[SelectionCandidate, list[str]]] = []
    for candidate_name, roles in target_roles.items():
        candidate = candidates.get((experiment, candidate_name))
        if candidate is not None:
            targets.append((candidate, roles))
    return targets


def public_candidate(
    experiment: ExperimentName,
    candidates: Mapping[tuple[ExperimentName, str], SelectionCandidate],
) -> SelectionCandidate | None:
    """Return the public or standard discrete comparison candidate when available."""
    public_prior = PUBLIC_DISCRETE_PRIORS[experiment]
    if public_prior is None:
        return None
    for (candidate_experiment, _candidate_name), candidate in candidates.items():
        if candidate_experiment == experiment and candidate.prior_config == public_prior:
            return candidate
    if experiment == "black_scholes" and Path(public_prior).exists():
        return candidate_from_prior_config(
            experiment=experiment,
            candidate="public_standard_vq_smoke_baseline",
            prior_config=public_prior,
            tokenizer_config="configs/experiments/black_scholes_causal_vq_tokenizer.yaml",
        )
    return None


def candidate_from_prior_config(
    *,
    experiment: ExperimentName,
    candidate: str,
    prior_config: str,
    tokenizer_config: str,
) -> SelectionCandidate:
    """Build a candidate from a token-prior config when no aggregate run exists."""
    raw_config = load_yaml_mapping(prior_config)
    data = require_mapping(raw_config, "data", prior_config)
    prior_experiment = require_mapping(raw_config, "experiment", prior_config)
    prior_output_dir = Path(str(data["token_data_dir"])).parent / "prior"
    prior_run_name = f"{prior_experiment['name']}_seed{prior_experiment.get('seed', 0)}"
    prior_dir = prior_output_dir / prior_run_name
    model = require_mapping(raw_config, "model", prior_config)
    return SelectionCandidate(
        experiment=experiment,
        candidate=candidate,
        tokenizer_config=tokenizer_config,
        prior_config=prior_config,
        tokenizer_dir=str(data["tokenizer_dir"]),
        token_data_dir=str(data["token_data_dir"]),
        prior_dir=str(prior_dir),
        conditional=str(model.get("condition_injection", "none")) != "none",
    )


def provisional_selection(
    experiment: ExperimentName,
    candidates: Mapping[tuple[ExperimentName, str], SelectionCandidate],
) -> str:
    """Return the provisional selected candidate name."""
    scored: list[tuple[float, str]] = []
    loaded = load_selection_results_payload(SELECTION_RESULTS_PATH)
    for raw_candidate in loaded:
        if raw_candidate.get("experiment") != experiment:
            continue
        metrics = parse_json_field(raw_candidate.get("metrics"), default={})
        prior = metrics.get("prior_best_checkpoint")
        if not isinstance(prior, dict):
            continue
        ce = prior.get("best_eval_cross_entropy")
        if isinstance(ce, int | float):
            scored.append((float(ce), str(raw_candidate["candidate"])))
    if scored:
        return sorted(scored)[0][1]
    if (experiment, STATIC_PROVISIONAL_SELECTIONS[experiment]) in candidates:
        return STATIC_PROVISIONAL_SELECTIONS[experiment]
    return STATIC_PROVISIONAL_SELECTIONS[experiment]


def discrete_target(
    *,
    candidate: SelectionCandidate,
    roles: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> EvaluationTarget:
    """Build a discrete path-evaluation target."""
    target_output = output_dir / candidate.experiment / candidate.candidate / "path_metrics"
    supported = list(GENERIC_TOKEN_PRIOR_METRICS)
    missing = [metric for metric in PATH_METRICS if metric not in supported]
    command = generic_token_prior_command(candidate, output_dir=target_output, args=args)
    warnings: list[str] = []
    status = "dry_run" if bool(args.dry_run) else "pending"
    if candidate.experiment == "sp500_vix":
        supported = list(SP500_PAPER_STYLE_METRICS)
        missing = []
        continuous_model_dir = discover_continuous_model_dir("sp500_vix")
        command = sp500_paper_style_command(
            candidate,
            output_dir=target_output,
            args=args,
            continuous_model_dir=continuous_model_dir or "<continuous-model-dir>",
        )
        if continuous_model_dir is None and not bool(args.dry_run):
            status = "not_available"
            warnings.append("S&P500/VIX paper-style evaluation requires a continuous model dir.")
    return EvaluationTarget(
        experiment=candidate.experiment,
        target=candidate.candidate,
        kind="discrete",
        roles=roles,
        command=command,
        output_dir=str(target_output),
        status=status,
        supported_metrics=supported,
        missing_metrics=missing,
        warnings=warnings,
    )


def generic_token_prior_command(
    candidate: SelectionCandidate,
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Return a generic token-prior path-evaluation command."""
    command = [
        "poetry",
        "run",
        "tcvae-evaluate-token-prior",
        "--config",
        candidate.prior_config,
        "--prior-dir",
        candidate.prior_dir,
        "--tokenizer-dir",
        candidate.tokenizer_dir,
        "--output-dir",
        str(output_dir),
        "--n-sample",
        str(args.n_sample),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--base-data-dir",
        str(args.base_data_dir),
    ]
    top_k = optional_top_k(args.top_k)
    if top_k is not None:
        command.extend(["--top-k", str(top_k)])
    return command


def sp500_paper_style_command(
    candidate: SelectionCandidate,
    *,
    output_dir: Path,
    args: argparse.Namespace,
    continuous_model_dir: str,
) -> list[str]:
    """Return the S&P500/VIX paper-style evaluation command."""
    command = [
        "poetry",
        "run",
        "python",
        "scripts/evaluate_sp500_vix_paper_style.py",
        "--discrete-config",
        candidate.prior_config,
        "--discrete-prior-dir",
        candidate.prior_dir,
        "--discrete-tokenizer-dir",
        candidate.tokenizer_dir,
        "--continuous-config",
        CONTINUOUS_BASELINES["sp500_vix"],
        "--continuous-model-dir",
        continuous_model_dir,
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
        str(args.top_k),
    ]
    return command


def no_leakage_targets(
    *,
    candidate: SelectionCandidate,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[EvaluationTarget]:
    """Build no-leakage check targets for one discrete candidate."""
    targets = [
        no_leakage_target(
            experiment=candidate.experiment,
            target=f"{candidate.candidate}_causal_conv",
            command=["poetry", "run", "python", "scripts/check_causal_conv_no_leakage.py"],
            output_dir=output_dir / candidate.experiment / candidate.candidate / "no_leakage",
            args=args,
            warnings=["source-level causal convolution check, not a checkpoint-specific test."],
        ),
        no_leakage_target(
            experiment=candidate.experiment,
            target=f"{candidate.candidate}_token_prior",
            command=[
                "poetry",
                "run",
                "python",
                "scripts/check_conditional_token_prior_no_leakage.py",
                "--config",
                candidate.prior_config,
                "--prior-dir",
                candidate.prior_dir,
                "--token-data-dir",
                candidate.token_data_dir,
                "--device",
                "cpu",
            ],
            output_dir=output_dir / candidate.experiment / candidate.candidate / "no_leakage",
            args=args,
        ),
    ]
    if candidate.conditional:
        targets.append(
            no_leakage_target(
                experiment=candidate.experiment,
                target=f"{candidate.candidate}_tokenizer",
                command=[
                    "poetry",
                    "run",
                    "python",
                    "scripts/check_conditional_vq_tokenizer_no_leakage.py",
                    "--config",
                    candidate.tokenizer_config,
                    "--tokenizer-dir",
                    candidate.tokenizer_dir,
                    "--base-data-dir",
                    str(args.base_data_dir),
                    "--device",
                    "cpu",
                ],
                output_dir=output_dir / candidate.experiment / candidate.candidate / "no_leakage",
                args=args,
            )
        )
    else:
        targets.append(
            EvaluationTarget(
                experiment=candidate.experiment,
                target=f"{candidate.candidate}_tokenizer",
                kind="no_leakage",
                roles=["tokenizer_no_leakage"],
                command=[],
                output_dir=str(
                    output_dir / candidate.experiment / candidate.candidate / "no_leakage"
                ),
                status="not_available",
                supported_metrics=[],
                missing_metrics=[],
                warnings=[
                    "trained unconditioned tokenizer no-leakage script is not available; "
                    "only the generic tokenizer source smoke check exists."
                ],
            )
        )
    return targets


def no_leakage_target(
    *,
    experiment: str,
    target: str,
    command: list[str],
    output_dir: Path,
    args: argparse.Namespace,
    warnings: list[str] | None = None,
) -> EvaluationTarget:
    """Build one no-leakage check target."""
    return EvaluationTarget(
        experiment=experiment,
        target=target,
        kind="no_leakage",
        roles=["no_leakage"],
        command=command,
        output_dir=str(output_dir),
        status="dry_run" if bool(args.dry_run) else "pending",
        supported_metrics=[],
        missing_metrics=[],
        warnings=warnings or [],
    )


def execute_targets(
    targets: Sequence[EvaluationTarget],
    *,
    profile: ProfileName,
) -> list[EvaluationTarget]:
    """Execute available targets and load resulting metrics when present."""
    updated: list[EvaluationTarget] = []
    for target in targets:
        if target.status == "not_available":
            updated.append(target)
            continue
        start = time.perf_counter()
        result = subprocess.run(
            target.command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        runtime = time.perf_counter() - start
        status = "ok" if result.returncode == 0 else f"failed:{result.returncode}"
        metrics = load_target_metrics(target)
        profile_payload: dict[str, Any] = {}
        warnings = list(target.warnings)
        if target.kind == "discrete" and metrics:
            profile_score = score_profile(
                cast(Mapping[str, int | float], numeric_metrics(metrics)),
                profile,
            )
            profile_payload = asdict(profile_score)
            warnings.extend(profile_score.warnings)
        updated.append(
            EvaluationTarget(
                **{
                    **asdict(target),
                    "status": status,
                    "runtime_seconds": runtime,
                    "metrics": metrics,
                    "profile": profile_payload,
                    "stdout_tail": tail_text(result.stdout),
                    "stderr_tail": tail_text(result.stderr),
                    "warnings": warnings,
                }
            )
        )
    return updated


def load_target_metrics(target: EvaluationTarget) -> dict[str, Any]:
    """Load metrics emitted by an evaluation target."""
    output_dir = Path(target.output_dir)
    if target.kind == "continuous":
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            return load_json_mapping(summary_path)
    if target.kind == "discrete":
        paper_summary = output_dir / "paper_style_summary.json"
        if paper_summary.exists():
            summary = load_json_mapping(paper_summary)
            comparison = require_mapping(summary, "comparisons", str(paper_summary))
            discrete = comparison.get("discrete")
            if isinstance(discrete, dict):
                return cast(dict[str, Any], discrete)
            return summary
        token_summary = output_dir / "token_prior_summary.json"
        if token_summary.exists():
            summary = load_json_mapping(token_summary)
            metrics = summary.get("metrics")
            if isinstance(metrics, dict):
                return cast(dict[str, Any], metrics)
            return summary
    return {}


def numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, int | float]:
    """Return numeric top-level metrics suitable for profile scoring."""
    return {key: value for key, value in metrics.items() if isinstance(value, int | float)}


def load_selection_results_payload(path: Path) -> list[dict[str, Any]]:
    """Return raw candidate payloads from the aggregate selection JSON."""
    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return []
    candidates = loaded.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Expected YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def load_json_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON mapping."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"Expected JSON mapping: {path}")
    return cast(dict[str, Any], loaded)


def require_mapping(raw_config: Mapping[str, Any], key: str, path: str) -> dict[str, Any]:
    """Return a required mapping section."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"{path} requires mapping section {key!r}.")
    return cast(dict[str, Any], value)


def parse_json_field(raw_value: Any, *, default: dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON string field from aggregate CSV/JSON-compatible rows."""
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str) or not raw_value:
        return default
    loaded = json.loads(raw_value)
    if not isinstance(loaded, dict):
        return default
    return cast(dict[str, Any], loaded)


def optional_top_k(raw_value: Any) -> int | None:
    """Parse an optional top-k value."""
    normalised = str(raw_value).strip().lower()
    if normalised in {"none", "null", "unrestricted", ""}:
        return None
    value = int(normalised)
    if value <= 0:
        raise SystemExit("--top-k must be positive or 'none'.")
    return value


def tail_text(text: str, *, max_lines: int = 20) -> str:
    """Return a compact subprocess output tail."""
    return "\n".join(text.strip().splitlines()[-max_lines:])


def write_json(
    path: Path,
    targets: Sequence[EvaluationTarget],
    *,
    args: argparse.Namespace,
) -> None:
    """Write aggregate final-evaluation plan JSON."""
    payload = {
        "script": "scripts/run_per_experiment_final_evaluation.py",
        "mode": "dry_run" if bool(args.dry_run) else "execute",
        "profile": str(args.profile),
        "n_sample": int(args.n_sample),
        "targets": [serialise_target(target) for target in targets],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, targets: Sequence[EvaluationTarget]) -> None:
    """Write aggregate final-evaluation plan CSV."""
    fieldnames = [
        "experiment",
        "target",
        "kind",
        "roles",
        "status",
        "command",
        "output_dir",
        "supported_metrics",
        "missing_metrics",
        "warnings",
        "runtime_seconds",
        "metrics",
        "profile",
        "stdout_tail",
        "stderr_tail",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for target in targets:
            writer.writerow(serialise_target(target))


def serialise_target(target: EvaluationTarget) -> dict[str, Any]:
    """Return a JSON/CSV-friendly target mapping."""
    payload = asdict(target)
    payload["command"] = shlex.join(target.command)
    payload["roles"] = json.dumps(target.roles, sort_keys=True)
    payload["supported_metrics"] = json.dumps(target.supported_metrics, sort_keys=True)
    payload["missing_metrics"] = json.dumps(target.missing_metrics, sort_keys=True)
    payload["warnings"] = json.dumps(target.warnings, sort_keys=True)
    payload["metrics"] = json.dumps(target.metrics, sort_keys=True)
    payload["profile"] = json.dumps(target.profile, sort_keys=True)
    return payload


def print_summary(targets: Sequence[EvaluationTarget], *, output_dir: Path) -> None:
    """Print a compact final-evaluation summary."""
    print("Per-experiment final-evaluation setup")
    print(f"targets: {len(targets)}")
    print(f"output_dir: {output_dir}")
    for target in targets:
        role_text = ",".join(target.roles)
        print(f"- {target.experiment}/{target.target}: {target.kind}, {role_text}, {target.status}")
    print("files: final_evaluation_plan.json, final_evaluation_plan.csv")


if __name__ == "__main__":
    main()
