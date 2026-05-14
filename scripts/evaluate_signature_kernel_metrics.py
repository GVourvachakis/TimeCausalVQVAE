"""Evaluate optional signature-kernel metrics for path batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

from time_causal_vae.evaluation.signature_kernel_metrics import (
    SIGKERNEL_INSTALL_HINT,
    OptionalDependencyError,
    SignatureKernelConfig,
    compute_signature_kernel_mmd,
    result_to_dict,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the signature-kernel metric CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate optional sigkernel-based path metrics.",
    )
    parser.add_argument("--real-paths")
    parser.add_argument("--generated-paths")
    parser.add_argument("--paper-style-batch")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--dyadic-order", type=int, default=1)
    parser.add_argument("--rbf-sigma", type=float, default=1.0)
    parser.add_argument("--include-time", action="store_true")
    parser.add_argument("--use-lead-lag", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    return parser


def main() -> None:
    """Run signature-kernel metric evaluation."""
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = perf_counter()
    config = SignatureKernelConfig(
        dyadic_order=args.dyadic_order,
        rbf_sigma=args.rbf_sigma,
        include_time=args.include_time,
        use_lead_lag=args.use_lead_lag,
    )

    try:
        real_paths, generated_paths, input_manifest = load_input_paths(args)
        result = compute_signature_kernel_mmd(real_paths, generated_paths, config=config)
    except OptionalDependencyError as exc:
        summary = missing_dependency_summary(args, str(exc), perf_counter() - start_time)
        write_summary_files(output_dir, summary)
        print("Optional dependency missing: sigkernel")
        print(str(exc))
        print(f"Wrote missing-dependency summary to {output_dir}")
        return

    elapsed = perf_counter() - start_time
    summary = {
        "status": "ok",
        "sigkernel_installed": True,
        "elapsed_seconds": elapsed,
        "inputs": input_manifest,
        "result": result_to_dict(result),
    }
    write_summary_files(output_dir, summary)
    print("Signature-kernel metric evaluation complete.")
    print(f"output_dir: {output_dir}")
    print(f"mmd_biased: {summary['result']['mmd_biased']:.8f}")


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.dyadic_order < 0:
        raise ValueError("--dyadic-order must be non-negative.")
    if args.rbf_sigma <= 0.0:
        raise ValueError("--rbf-sigma must be positive.")
    modes = [
        bool(args.synthetic),
        bool(args.paper_style_batch),
        bool(args.real_paths or args.generated_paths),
    ]
    if sum(modes) != 1:
        raise ValueError(
            "Choose exactly one input mode: --synthetic, --paper-style-batch, "
            "or both --real-paths and --generated-paths.",
        )
    if bool(args.real_paths) != bool(args.generated_paths):
        raise ValueError("--real-paths and --generated-paths must be provided together.")


def load_input_paths(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load real and generated path batches from the requested source."""
    if args.synthetic:
        real_paths, generated_paths = synthetic_paths(args.max_samples)
        return (
            real_paths,
            generated_paths,
            {
                "mode": "synthetic",
                "real_shape": list(real_paths.shape),
                "generated_shape": list(generated_paths.shape),
                "max_samples": args.max_samples,
            },
        )
    if args.paper_style_batch:
        batch_path = resolve_paper_style_batch(Path(args.paper_style_batch))
        payload = torch.load(batch_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("--paper-style-batch must load a dictionary.")
        real_paths = tensor_to_numpy(get_required_tensor(payload, "real_paths"), name="real_paths")
        generated_paths = tensor_to_numpy(
            get_required_tensor(payload, "decoded_paths"),
            name="decoded_paths",
        )
        real_paths, generated_paths = limit_samples(real_paths, generated_paths, args.max_samples)
        return (
            real_paths,
            generated_paths,
            {
                "mode": "paper_style_batch",
                "source": str(batch_path),
                "real_shape": list(real_paths.shape),
                "generated_shape": list(generated_paths.shape),
                "max_samples": args.max_samples,
            },
        )

    real_paths = load_array_path(Path(args.real_paths))
    generated_paths = load_array_path(Path(args.generated_paths))
    real_paths, generated_paths = limit_samples(real_paths, generated_paths, args.max_samples)
    return (
        real_paths,
        generated_paths,
        {
            "mode": "path_files",
            "real_source": args.real_paths,
            "generated_source": args.generated_paths,
            "real_shape": list(real_paths.shape),
            "generated_shape": list(generated_paths.shape),
            "max_samples": args.max_samples,
        },
    )


def resolve_paper_style_batch(path: Path) -> Path:
    """Resolve a paper-style batch path or output directory."""
    if path.is_dir():
        candidate = path / "discrete_paper_style_batch.pt"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{path} does not contain discrete_paper_style_batch.pt.")
    return path


def get_required_tensor(payload: dict[Any, Any], key: str) -> Tensor:
    """Return a tensor payload value."""
    value = payload.get(key)
    if not isinstance(value, Tensor):
        raise ValueError(f"paper-style batch is missing tensor key {key!r}.")
    return value


def load_array_path(path: Path) -> np.ndarray:
    """Load a path batch from ``.npy``, ``.npz``, or ``.pt``."""
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    if path.suffix == ".npz":
        with np.load(path) as payload:
            for key in ("paths", "decoded_paths", "generated_paths", "real_paths", "arr_0"):
                if key in payload:
                    return np.asarray(payload[key], dtype=np.float64)
            raise ValueError(f"{path} does not contain a recognised path array key.")
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, Tensor):
            return tensor_to_numpy(payload, name=str(path))
        if isinstance(payload, dict):
            for key in ("paths", "decoded_paths", "generated_paths", "real_paths"):
                value = payload.get(key)
                if isinstance(value, Tensor):
                    return tensor_to_numpy(value, name=f"{path}:{key}")
                if value is not None:
                    return np.asarray(value, dtype=np.float64)
        raise ValueError(f"{path} does not contain a recognised tensor or path array.")
    raise ValueError(f"Unsupported path file extension: {path.suffix}")


def tensor_to_numpy(tensor: Tensor, *, name: str) -> np.ndarray:
    """Convert a tensor to a float64 NumPy array."""
    array = tensor.detach().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return np.asarray(array, dtype=np.float64)


def limit_samples(
    real_paths: np.ndarray,
    generated_paths: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a common sample cap to two path batches."""
    n_samples = min(len(real_paths), len(generated_paths), max_samples)
    if n_samples <= 0:
        raise ValueError("path batches must contain at least one common sample.")
    return real_paths[:n_samples], generated_paths[:n_samples]


def synthetic_paths(max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic positive synthetic price paths."""
    n_samples = max(2, max_samples)
    n_steps = 16
    time = np.linspace(0.0, 1.0, num=n_steps, dtype=np.float64)
    real_paths = []
    generated_paths = []
    for index in range(n_samples):
        phase = index / max(n_samples - 1, 1)
        real_log = 0.03 * time + 0.02 * np.sin(2.0 * np.pi * (time + phase))
        generated_log = 0.028 * time + 0.018 * np.sin(2.0 * np.pi * (time + phase + 0.03))
        real_paths.append(np.exp(real_log))
        generated_paths.append(np.exp(generated_log))
    return np.asarray(real_paths, dtype=np.float64)[..., None], np.asarray(
        generated_paths,
        dtype=np.float64,
    )[..., None]


def missing_dependency_summary(
    args: argparse.Namespace,
    message: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build a JSON-serialisable missing-dependency summary."""
    return {
        "status": "missing_dependency",
        "sigkernel_installed": False,
        "dependency": "sigkernel",
        "message": message,
        "install_hint": SIGKERNEL_INSTALL_HINT,
        "elapsed_seconds": elapsed_seconds,
        "requested": vars(args),
    }


def write_summary_files(output_dir: Path, summary: dict[str, Any]) -> None:
    """Write JSON and Markdown summaries."""
    json_path = output_dir / "signature_kernel_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = output_dir / "signature_kernel_summary.md"
    md_path.write_text(summary_to_markdown(summary), encoding="utf-8")


def summary_to_markdown(summary: dict[str, Any]) -> str:
    """Render a concise Markdown summary."""
    lines = ["# Signature-Kernel Metric Summary", ""]
    lines.append(f"- status: `{summary['status']}`")
    lines.append(f"- sigkernel installed: `{summary['sigkernel_installed']}`")
    lines.append(f"- elapsed seconds: `{float(summary['elapsed_seconds']):.6f}`")
    if summary["status"] == "missing_dependency":
        lines.extend(
            [
                "",
                "## Missing Dependency",
                "",
                str(summary["message"]),
                "",
                "Install hint:",
                "",
                "```bash",
                str(summary["install_hint"]),
                "```",
            ],
        )
        return "\n".join(lines) + "\n"

    result = summary["result"]
    checks = result["kxx_checks"], result["kyy_checks"], result["kxy_checks"]
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"- biased signature-kernel MMD: `{float(result['mmd_biased']):.8f}`",
            f"- preprocessing channels: `{', '.join(result['preprocessing']['channels'])}`",
            f"- include time: `{result['preprocessing']['include_time']}`",
            f"- lead-lag: `{result['preprocessing']['use_lead_lag']}`",
            "",
            "| Gram | Finite | Symmetric | Positive diagonal | Shape |",
            "| --- | --- | --- | --- | --- |",
        ],
    )
    for name, check in zip(("Kxx", "Kyy", "Kxy"), checks, strict=True):
        lines.append(
            f"| {name} | `{check['finite']}` | `{check['symmetric']}` | "
            f"`{check['positive_diagonal']}` | `{check['shape']}` |",
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
