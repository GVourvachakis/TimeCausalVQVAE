"""Run curated notebook previews with runtime tracking."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import nbformat
import yaml

BUCKET_ESTIMATES_SECONDS = {
    "short": (60, 5 * 60),
    "medium": (5 * 60, 15 * 60),
    "long": (15 * 60, 45 * 60),
}
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_LOG_PATH = Path("NOTEBOOK_EXECUTION_LOG.md")
DEFAULT_SUMMARY_PATH = Path("notebooks/preview_execution_summary.json")
TRACEBACK_DIR = Path("notebooks/preview_tracebacks")
PREVIEW_ENV = {
    "MPLBACKEND": "Agg",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLE_SERVICE": "true",
}
PARAMETER_OVERRIDES = {
    "preview": {
        "RUN_FULL": False,
        "RUN_HEAVY": False,
        "RUN_TRAINING": False,
        "RUN_EVALUATION": False,
        "RUN_EXPENSIVE_METRICS": False,
        "RUN_SIGNATURE_KERNEL": False,
        "RUN_ADAPTED_WASSERSTEIN": False,
        "ALLOW_MISSING_OUTPUTS": True,
    },
    "full-preview": {
        "RUN_FULL": True,
        "RUN_HEAVY": True,
        "RUN_TRAINING": False,
        "RUN_EVALUATION": False,
        "RUN_EXPENSIVE_METRICS": False,
        "RUN_SIGNATURE_KERNEL": False,
        "RUN_ADAPTED_WASSERSTEIN": False,
        "ALLOW_MISSING_OUTPUTS": True,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("notebooks/preview_notebook_manifest.yaml"),
        help="YAML manifest listing notebooks to execute.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing later notebooks after a notebook failure.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Validate the manifest and print estimates without executing notebooks.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Markdown execution log path.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="JSON execution summary path.",
    )
    parser.add_argument(
        "--parameter-mode",
        choices=["none", "preview", "full-preview"],
        default="none",
        help="Override existing notebook parameter assignments before execution.",
    )
    parser.add_argument(
        "--max-total-runtime-hours",
        type=float,
        default=None,
        help="Stop after a notebook if projected total runtime exceeds this limit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    """Return the current UTC timestamp for logs."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def format_duration(seconds: float | None) -> str:
    """Format a duration in seconds for human-readable logs."""
    if seconds is None:
        return "unknown"
    rounded = round(seconds)
    minutes, remainder = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {remainder}s"
    if minutes:
        return f"{minutes}m {remainder}s"
    return f"{remainder}s"


def estimate_label(bucket: str) -> str:
    """Return the human-readable estimate range for a runtime bucket."""
    if bucket in BUCKET_ESTIMATES_SECONDS:
        low, high = BUCKET_ESTIMATES_SECONDS[bucket]
        return f"{format_duration(low)} to {format_duration(high)}"
    return "unknown until after the first observed run"


def midpoint_estimate(bucket: str) -> float | None:
    """Return the midpoint estimate for a runtime bucket."""
    if bucket not in BUCKET_ESTIMATES_SECONDS:
        return None
    low, high = BUCKET_ESTIMATES_SECONDS[bucket]
    return (low + high) / 2


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the top-level manifest shape."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Manifest must contain a mapping: {path}")
    notebooks = loaded.get("notebooks")
    if not isinstance(notebooks, list) or not notebooks:
        raise ValueError("Manifest must contain a non-empty 'notebooks' list.")
    return loaded


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate notebook entries and return them in priority order."""
    notebooks = sorted(manifest["notebooks"], key=lambda item: item.get("priority", 0))
    seen_paths: set[str] = set()
    for item in notebooks:
        path = item.get("path")
        if not isinstance(path, str) or not path.endswith(".ipynb"):
            raise ValueError(f"Notebook path must be an .ipynb string: {item!r}")
        if path in seen_paths:
            raise ValueError(f"Duplicate notebook in manifest: {path}")
        seen_paths.add(path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Notebook listed in manifest does not exist: {path}")
        if item.get("training_allowed") is not False:
            raise ValueError(f"Preview manifest must set training_allowed: false for {path}")
        if item.get("evaluation_allowed", False) is not False:
            raise ValueError(f"Preview manifest must set evaluation_allowed: false for {path}")
        if item.get("expensive_metrics_allowed") is not False:
            raise ValueError(
                f"Preview manifest must set expensive_metrics_allowed: false for {path}"
            )
        bucket = item.get("runtime_bucket", "unknown")
        if bucket not in {*BUCKET_ESTIMATES_SECONDS, "unknown"}:
            raise ValueError(f"Unknown runtime bucket for {path}: {bucket}")
    return notebooks


def remaining_eta_seconds(
    remaining: list[dict[str, Any]],
    observed_by_bucket: dict[str, list[float]],
    observed_all: list[float],
) -> float | None:
    """Estimate remaining runtime from bucket observations and midpoint defaults."""
    total = 0.0
    for item in remaining:
        bucket = str(item.get("runtime_bucket", "unknown"))
        if observed_by_bucket[bucket]:
            total += mean(observed_by_bucket[bucket])
            continue
        if bucket == "unknown":
            if not observed_all:
                return None
            total += mean(observed_all)
            continue
        estimate = midpoint_estimate(bucket)
        if estimate is None:
            return None
        total += estimate
    return total


def projected_runtime_seconds(
    elapsed_seconds: float,
    eta_seconds: float | None,
) -> float | None:
    """Return elapsed plus estimated remaining runtime."""
    if eta_seconds is None:
        return None
    return elapsed_seconds + eta_seconds


def initial_estimate_seconds(notebooks: list[dict[str, Any]]) -> float | None:
    """Estimate total runtime before execution from manifest runtime buckets."""
    total = 0.0
    for item in notebooks:
        estimate = midpoint_estimate(str(item.get("runtime_bucket", "unknown")))
        if estimate is None:
            return None
        total += estimate
    return total


def format_reason(reason: str | None) -> str:
    """Format a reason string for Markdown tables."""
    if not reason:
        return ""
    return reason.replace("|", "\\|")


def apply_parameter_mode(notebook_path: Path, mode: str) -> dict[str, list[str]]:
    """Override existing top-level notebook boolean parameters in place."""
    overrides = PARAMETER_OVERRIDES.get(mode)
    if not overrides:
        return {"applied": [], "missing": sorted(PARAMETER_OVERRIDES["preview"])}

    notebook = nbformat.read(notebook_path, as_version=4)
    applied: set[str] = set()
    modified = False
    patterns = {
        name: re.compile(rf"^({re.escape(name)}\s*=\s*)(True|False)(\s*(?:#.*)?)$")
        for name in overrides
    }

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        lines = cell.get("source", "").splitlines(keepends=True)
        new_lines: list[str] = []
        for line in lines:
            newline = "\n" if line.endswith("\n") else ""
            body = line[:-1] if newline else line
            for name, pattern in patterns.items():
                match = pattern.match(body)
                if match is None:
                    continue
                replacement_value = "True" if overrides[name] else "False"
                body = f"{match.group(1)}{replacement_value}{match.group(3)}"
                applied.add(name)
                modified = True
                break
            new_lines.append(f"{body}{newline}")
        cell["source"] = "".join(new_lines)

    if modified:
        nbformat.write(notebook, notebook_path)

    return {
        "applied": sorted(applied),
        "missing": sorted(set(overrides) - applied),
    }


def write_failure_trace(
    notebook_path: str,
    command: list[str],
    started_at: str,
    ended_at: str,
    exc: BaseException | None,
    result: subprocess.CompletedProcess[str] | None,
) -> str:
    """Write stdout, stderr, and exception details for a failed notebook."""
    TRACEBACK_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = notebook_path.replace("/", "__").replace(".ipynb", "")
    trace_path = TRACEBACK_DIR / f"{safe_name}.log"
    lines = [
        f"notebook: {notebook_path}",
        f"started_at: {started_at}",
        f"ended_at: {ended_at}",
        f"command: {' '.join(command)}",
        "",
    ]
    if result is not None:
        lines.extend([
            f"returncode: {result.returncode}",
            "",
            "stdout:",
            result.stdout or "",
            "",
            "stderr:",
            result.stderr or "",
        ])
    if exc is not None:
        lines.extend(["", "python_traceback:", "".join(traceback.format_exception(exc))])
    trace_path.write_text("\n".join(lines), encoding="utf-8")
    return str(trace_path)


def write_markdown_log(
    path: Path,
    *,
    manifest_path: Path,
    started_at: str,
    entries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    eta_seconds: float | None,
    run_status: str,
    initial_eta_seconds: float | None,
    max_total_runtime_hours: float | None,
    parameter_mode: str,
    message: str | None = None,
) -> None:
    """Write the live Markdown notebook execution log."""
    lines = [
        "# Notebook Execution Log",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Parameter mode: `{parameter_mode}`",
        f"- Started: {started_at}",
        f"- Last update: {utc_now()}",
        f"- Status: {run_status}",
        f"- Initial estimated total: {format_duration(initial_eta_seconds)}",
        f"- Remaining ETA: {format_duration(eta_seconds)}",
        f"- Runtime guard: {max_total_runtime_hours or 'not set'} hours",
        "",
        (
            "| Priority | Notebook | Bucket | Status | Runtime | Started | Ended | "
            "ETA after | Reason | Expensive metrics | Traceback |"
        ),
        "| ---: | --- | --- | --- | ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    if message:
        lines.extend(["", "## Message", "", message, ""])
    for entry in entries:
        traceback_path = entry.get("traceback_path") or ""
        traceback_cell = f"`{traceback_path}`" if traceback_path else ""
        expensive_metric_status = entry.get("expensive_metric_status", "")
        reason = format_reason(entry.get("reason"))
        row_template = (
            "| {priority} | `{path}` | {bucket} | {status} | {runtime} | {started} | "
            "{ended} | {eta_after} | {reason} | {expensive_metrics} | {trace} |"
        )
        lines.append(
            row_template.format(
                priority=entry["priority"],
                path=entry["path"],
                bucket=entry["runtime_bucket"],
                status=entry["status"],
                runtime=format_duration(entry.get("runtime_seconds")),
                started=entry.get("started_at", ""),
                ended=entry.get("ended_at", ""),
                eta_after=format_duration(entry.get("remaining_eta_seconds")),
                reason=reason,
                expensive_metrics=expensive_metric_status,
                trace=traceback_cell,
            )
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append("- `{path}` failed; traceback/log: `{traceback_path}`".format(**failure))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    *,
    manifest_path: Path,
    started_at: str,
    entries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    eta_seconds: float | None,
    run_status: str,
    initial_eta_seconds: float | None,
    max_total_runtime_hours: float | None,
    parameter_mode: str,
    message: str | None = None,
) -> None:
    """Write the machine-readable execution summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(manifest_path),
        "started_at": started_at,
        "last_update": utc_now(),
        "complete": run_status == "complete",
        "status": run_status,
        "remaining_eta_seconds": eta_seconds,
        "remaining_eta_human": format_duration(eta_seconds),
        "initial_estimate_seconds": initial_eta_seconds,
        "initial_estimate_human": format_duration(initial_eta_seconds),
        "max_total_runtime_hours": max_total_runtime_hours,
        "parameter_mode": parameter_mode,
        "message": message,
        "entries": entries,
        "failures": failures,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_notebook(notebook: dict[str, Any], parameter_mode: str) -> dict[str, Any]:
    """Execute one notebook with nbconvert and return its log entry."""
    path = str(notebook["path"])
    parameter_result = apply_parameter_mode(Path(path), parameter_mode)
    command = ["jupyter", "nbconvert", "--execute", "--inplace", path]
    timeout = int(notebook.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    expensive_metric_status = (
        "allowed" if notebook.get("expensive_metrics_allowed") is True else "skipped by manifest"
    )
    env = os.environ.copy()
    env.update(PREVIEW_ENV)
    started_at = utc_now()
    started = time.monotonic()
    result: subprocess.CompletedProcess[str] | None = None
    exc: BaseException | None = None
    status = "passed"
    traceback_path = None
    reason = ""

    try:
        result = subprocess.run(
            command,
            check=False,
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            status = "failed"
            reason = "nbconvert returned a non-zero exit code"
    except subprocess.TimeoutExpired as err:
        status = "timeout"
        exc = err
        reason = f"timeout_seconds={timeout} exceeded"
    except Exception as err:  # pragma: no cover - defensive logging for unexpected launch errors.
        status = "failed"
        exc = err
        reason = "unexpected runner failure"
    ended_at = utc_now()
    runtime_seconds = time.monotonic() - started

    if status != "passed":
        traceback_path = write_failure_trace(path, command, started_at, ended_at, exc, result)

    return {
        "priority": notebook.get("priority"),
        "path": path,
        "category": notebook.get("category"),
        "runtime_bucket": notebook.get("runtime_bucket", "unknown"),
        "estimated_runtime": estimate_label(str(notebook.get("runtime_bucket", "unknown"))),
        "timeout_seconds": timeout,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": runtime_seconds,
        "remaining_eta_seconds": None,
        "reason": reason,
        "expensive_metric_status": expensive_metric_status,
        "parameter_mode": parameter_mode,
        "parameters_applied": parameter_result["applied"],
        "parameters_missing": parameter_result["missing"],
        "traceback_path": traceback_path,
    }


def print_estimates(notebooks: list[dict[str, Any]]) -> float | None:
    """Print preflight runtime estimates without executing notebooks."""
    print("Estimate-only preflight: no notebooks will be executed.")
    for item in notebooks:
        bucket = str(item.get("runtime_bucket", "unknown"))
        print(
            "[{priority:>3}] {path} ({bucket}): {estimate}".format(
                priority=item.get("priority", ""),
                path=item["path"],
                bucket=bucket,
                estimate=estimate_label(bucket),
            )
        )
    total = initial_estimate_seconds(notebooks)
    total_label = format_duration(total)
    print(f"Estimated total runtime: {total_label}")
    return total


def main() -> int:
    """Run the preview notebook command-line workflow."""
    args = parse_args()
    manifest = load_manifest(args.manifest)
    notebooks = validate_manifest(manifest)
    initial_eta_seconds = initial_estimate_seconds(notebooks)

    if args.estimate_only:
        print_estimates(notebooks)
        message = "Estimate-only preflight: no notebooks were executed."
        write_markdown_log(
            args.log_path,
            manifest_path=args.manifest,
            started_at=utc_now(),
            entries=[],
            failures=[],
            eta_seconds=initial_eta_seconds,
            run_status="estimate_only",
            initial_eta_seconds=initial_eta_seconds,
            max_total_runtime_hours=args.max_total_runtime_hours,
            parameter_mode=args.parameter_mode,
            message=message,
        )
        write_summary(
            args.summary_path,
            manifest_path=args.manifest,
            started_at=utc_now(),
            entries=[],
            failures=[],
            eta_seconds=initial_eta_seconds,
            run_status="estimate_only",
            initial_eta_seconds=initial_eta_seconds,
            max_total_runtime_hours=args.max_total_runtime_hours,
            parameter_mode=args.parameter_mode,
            message=message,
        )
        return 0

    started_at = utc_now()
    started_monotonic = time.monotonic()
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    observed_by_bucket: dict[str, list[float]] = defaultdict(list)
    observed_all: list[float] = []
    run_status = "running"
    message: str | None = None

    for index, notebook in enumerate(notebooks):
        bucket = str(notebook.get("runtime_bucket", "unknown"))
        print(
            "Running {path} ({bucket}, estimated {estimate})".format(
                path=notebook["path"],
                bucket=bucket,
                estimate=estimate_label(bucket),
            ),
            flush=True,
        )
        entry = run_notebook(notebook, args.parameter_mode)
        entries.append(entry)
        if entry["status"] == "passed":
            observed_by_bucket[bucket].append(float(entry["runtime_seconds"]))
            observed_all.append(float(entry["runtime_seconds"]))
        else:
            failures.append({
                "path": entry["path"],
                "status": entry["status"],
                "traceback_path": entry["traceback_path"],
            })

        remaining = notebooks[index + 1 :]
        eta_seconds = remaining_eta_seconds(remaining, observed_by_bucket, observed_all)
        entry["remaining_eta_seconds"] = eta_seconds
        elapsed_seconds = time.monotonic() - started_monotonic
        projected_seconds = projected_runtime_seconds(elapsed_seconds, eta_seconds)
        runtime_limit_seconds = (
            args.max_total_runtime_hours * 3600 if args.max_total_runtime_hours else None
        )
        should_stop_for_runtime = (
            runtime_limit_seconds is not None
            and projected_seconds is not None
            and projected_seconds > runtime_limit_seconds
            and bool(remaining)
        )
        if should_stop_for_runtime:
            run_status = "stopped_by_runtime_guard"
            message = (
                "Stopped because elapsed plus estimated remaining runtime "
                f"({format_duration(projected_seconds)}) exceeds "
                f"{args.max_total_runtime_hours} hours."
            )
            for skipped in remaining:
                entries.append({
                    "priority": skipped.get("priority"),
                    "path": skipped["path"],
                    "category": skipped.get("category"),
                    "runtime_bucket": skipped.get("runtime_bucket", "unknown"),
                    "estimated_runtime": estimate_label(
                        str(skipped.get("runtime_bucket", "unknown"))
                    ),
                    "timeout_seconds": int(
                        skipped.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS
                    ),
                    "status": "skipped",
                    "started_at": "",
                    "ended_at": "",
                    "runtime_seconds": None,
                    "remaining_eta_seconds": eta_seconds,
                    "reason": "stopped_by_runtime_guard",
                    "expensive_metric_status": (
                        "allowed"
                        if skipped.get("expensive_metrics_allowed") is True
                        else "skipped by manifest"
                    ),
                    "parameter_mode": args.parameter_mode,
                    "parameters_applied": [],
                    "parameters_missing": [],
                    "traceback_path": None,
                })
        elif not remaining:
            run_status = "complete"
        elif bool(failures) and not args.continue_on_error:
            run_status = "failed"
        else:
            run_status = "running"
        write_markdown_log(
            args.log_path,
            manifest_path=args.manifest,
            started_at=started_at,
            entries=entries,
            failures=failures,
            eta_seconds=eta_seconds,
            run_status=run_status,
            initial_eta_seconds=initial_eta_seconds,
            max_total_runtime_hours=args.max_total_runtime_hours,
            parameter_mode=args.parameter_mode,
            message=message,
        )
        write_summary(
            args.summary_path,
            manifest_path=args.manifest,
            started_at=started_at,
            entries=entries,
            failures=failures,
            eta_seconds=eta_seconds,
            run_status=run_status,
            initial_eta_seconds=initial_eta_seconds,
            max_total_runtime_hours=args.max_total_runtime_hours,
            parameter_mode=args.parameter_mode,
            message=message,
        )

        print(
            "Finished {path}: {status} in {runtime}; remaining ETA {eta}".format(
                path=entry["path"],
                status=entry["status"],
                runtime=format_duration(entry["runtime_seconds"]),
                eta=format_duration(eta_seconds),
            ),
            flush=True,
        )
        if should_stop_for_runtime:
            return 2
        if failures and not args.continue_on_error:
            return 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
