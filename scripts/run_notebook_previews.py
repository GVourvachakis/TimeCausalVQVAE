"""Run curated notebook previews with runtime tracking."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

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
    complete: bool,
) -> None:
    """Write the live Markdown notebook execution log."""
    status = "complete" if complete else "running"
    lines = [
        "# Notebook Execution Log",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Started: {started_at}",
        f"- Last update: {utc_now()}",
        f"- Status: {status}",
        f"- Remaining ETA: {format_duration(eta_seconds)}",
        "",
        "| Priority | Notebook | Bucket | Status | Runtime | Started | Ended | Traceback |",
        "| ---: | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for entry in entries:
        traceback_path = entry.get("traceback_path") or ""
        traceback_cell = f"`{traceback_path}`" if traceback_path else ""
        row_template = (
            "| {priority} | `{path}` | {bucket} | {status} | {runtime} | "
            "{started} | {ended} | {trace} |"
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
    complete: bool,
) -> None:
    """Write the machine-readable execution summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(manifest_path),
        "started_at": started_at,
        "last_update": utc_now(),
        "complete": complete,
        "remaining_eta_seconds": eta_seconds,
        "remaining_eta_human": format_duration(eta_seconds),
        "entries": entries,
        "failures": failures,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    """Execute one notebook with nbconvert and return its log entry."""
    path = str(notebook["path"])
    command = ["jupyter", "nbconvert", "--execute", "--inplace", path]
    timeout = int(notebook.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    env = os.environ.copy()
    env.update(PREVIEW_ENV)
    started_at = utc_now()
    started = time.monotonic()
    result: subprocess.CompletedProcess[str] | None = None
    exc: BaseException | None = None
    status = "passed"
    traceback_path = None

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
    except subprocess.TimeoutExpired as err:
        status = "timeout"
        exc = err
    except Exception as err:  # pragma: no cover - defensive logging for unexpected launch errors.
        status = "failed"
        exc = err
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
        "traceback_path": traceback_path,
    }


def print_estimates(notebooks: list[dict[str, Any]]) -> None:
    """Print preflight runtime estimates without executing notebooks."""
    print("Estimate-only preflight: no notebooks will be executed.")
    total = 0.0
    unknown = False
    for item in notebooks:
        bucket = str(item.get("runtime_bucket", "unknown"))
        estimate = midpoint_estimate(bucket)
        if estimate is None:
            unknown = True
        else:
            total += estimate
        print(
            "[{priority:>3}] {path} ({bucket}): {estimate}".format(
                priority=item.get("priority", ""),
                path=item["path"],
                bucket=bucket,
                estimate=estimate_label(bucket),
            )
        )
    total_label = (
        format_duration(total) if not unknown else f"{format_duration(total)} plus unknowns"
    )
    print(f"Estimated total runtime: {total_label}")


def main() -> int:
    """Run the preview notebook command-line workflow."""
    args = parse_args()
    manifest = load_manifest(args.manifest)
    notebooks = validate_manifest(manifest)

    if args.estimate_only:
        print_estimates(notebooks)
        return 0

    started_at = utc_now()
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    observed_by_bucket: dict[str, list[float]] = defaultdict(list)
    observed_all: list[float] = []

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
        entry = run_notebook(notebook)
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
        complete = not remaining or (bool(failures) and not args.continue_on_error)
        write_markdown_log(
            args.log_path,
            manifest_path=args.manifest,
            started_at=started_at,
            entries=entries,
            failures=failures,
            eta_seconds=eta_seconds,
            complete=complete,
        )
        write_summary(
            args.summary_path,
            manifest_path=args.manifest,
            started_at=started_at,
            entries=entries,
            failures=failures,
            eta_seconds=eta_seconds,
            complete=complete,
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
        if failures and not args.continue_on_error:
            return 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
