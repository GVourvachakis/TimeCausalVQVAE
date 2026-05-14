"""Extract optional log-signature context features for S&P500/VIX experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from time_causal_vae.evaluation.signature_features import (
    DEFAULT_STANDARDIZATION_EPSILON,
    OptionalDependencyError,
    SignatureFeatureConfig,
    apply_feature_standardization,
    compute_signature_feature_batch,
    feature_standardization_statistics,
    metadata_to_dict,
)

DEFAULT_SP500_VIX_TARGET_LENGTH = 60
DEFAULT_SYNTHETIC_SAMPLE_COUNT = 16
DEFAULT_SYNTHETIC_TARGET_LENGTH = 12


def build_parser() -> argparse.ArgumentParser:
    """Build the signature-feature extraction CLI parser."""
    parser = argparse.ArgumentParser(
        description="Extract optional historical log-signature features for S&P500/VIX.",
    )
    parser.add_argument("--dataset", default="sp500_vix", choices=["sp500_vix"])
    parser.add_argument("--base-data-dir", default="data/processed")
    parser.add_argument(
        "--output-dir",
        default="outputs/sp500_vix_discrete/signature_features/logsig_l2",
    )
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--use-lead-lag", action="store_true")
    parser.add_argument("--include-time", action="store_true")
    parser.add_argument("--include-vix", action="store_true")
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Fit train-set feature mean/std and apply the transform to train and eval features.",
    )
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--synthetic", action="store_true")
    return parser


def main() -> None:
    """Run signature-feature extraction."""
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = perf_counter()
    config = SignatureFeatureConfig(
        depth=args.depth,
        use_lead_lag=args.use_lead_lag,
        include_time=args.include_time,
        include_vix=args.include_vix,
        use_log_signature=True,
    )
    try:
        data_payload = build_input_payload(args)
        features, feature_metadata = compute_signature_feature_batch(
            data_payload["price_contexts"],
            vix_contexts=data_payload.get("vix_contexts"),
            config=config,
        )
    except OptionalDependencyError as exc:
        summary = missing_dependency_summary(args, str(exc), elapsed_seconds=start_time)
        write_summary_files(output_dir, summary)
        print("Optional dependency missing: iisignature")
        print(str(exc))
        print(f"Wrote missing-dependency summary to {output_dir}")
        return

    elapsed = perf_counter() - start_time
    labels = data_payload["labels"]
    sample_indices = data_payload["sample_indices"]
    metadata = metadata_to_dict(feature_metadata)
    standardization = build_standardization_payload(features, enabled=bool(args.standardize))
    if args.standardize:
        features = apply_feature_standardization(
            features,
            mean=standardization["mean"],
            std=standardization["std"],
        )
    metadata["preprocessing"]["standardized"] = bool(args.standardize)
    metadata["standardization"] = standardization_summary(standardization, features=features)
    common_npz_payload = {
        "features": features,
        "labels": labels,
        "sample_indices": sample_indices,
        "metadata": np.array(json.dumps(metadata, sort_keys=True)),
    }
    np.savez_compressed(output_dir / "train_signature_features.npz", **common_npz_payload)
    np.savez_compressed(output_dir / "eval_signature_features.npz", **common_npz_payload)
    if args.standardize:
        np.savez_compressed(
            output_dir / "signature_feature_standardization.npz",
            mean=standardization["mean"],
            std=standardization["std"],
            epsilon=np.asarray([standardization["epsilon"]], dtype=np.float64),
        )

    summary = success_summary(
        args,
        data_payload=data_payload,
        metadata=metadata,
        features=features,
        standardization=standardization,
        elapsed_seconds=elapsed,
    )
    write_summary_files(output_dir, summary)

    print("Signature feature extraction complete.")
    print(f"output_dir: {output_dir}")
    print(f"features_shape: {list(features.shape)}")
    print(f"finite: {summary['finite']}")
    print(f"standardized: {summary['standardization']['enabled']}")
    print("files: train_signature_features.npz, eval_signature_features.npz")


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.depth <= 0:
        raise SystemExit("--depth must be positive.")
    if args.context_length <= 0:
        raise SystemExit("--context-length must be positive.")


def validate_output_dir(output_dir: str) -> Path:
    """Validate that generated feature artifacts stay below ignored outputs/."""
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


def build_input_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build historical-context arrays for synthetic or real S&P500/VIX data."""
    if args.synthetic:
        return build_synthetic_payload(args.context_length, args.seed)
    return build_sp500_vix_payload(Path(args.base_data_dir), args.context_length)


def build_synthetic_payload(context_length: int, seed: int) -> dict[str, Any]:
    """Build deterministic synthetic contexts for smoke testing."""
    sample_count = DEFAULT_SYNTHETIC_SAMPLE_COUNT
    total_length = sample_count + DEFAULT_SYNTHETIC_TARGET_LENGTH
    phase = (seed % 17) / 17.0
    grid = np.arange(total_length, dtype=np.float64)
    price = 1.0 + 0.002 * grid + 0.015 * np.sin(grid / 3.0 + phase)
    vix = 0.2 + 0.03 * np.cos(grid / 5.0 + phase)
    return build_context_payload(
        price,
        vix,
        context_length=context_length,
        target_length=DEFAULT_SYNTHETIC_TARGET_LENGTH,
        sample_count=sample_count,
        source="synthetic",
    )


def build_sp500_vix_payload(base_data_dir: Path, context_length: int) -> dict[str, Any]:
    """Load local S&P500/VIX data and build aligned historical contexts."""
    data_path = base_data_dir / "sp500vix" / "sp500vix_normalized.npy"
    if not data_path.exists():
        raise SystemExit(f"Missing S&P500/VIX data file: {data_path}")
    array = np.load(data_path)
    if array.ndim != 2 or array.shape[1] < 2:
        raise SystemExit(f"Expected S&P500/VIX array with shape [time, >=2], got {array.shape}.")
    price = np.asarray(array[:, 0], dtype=np.float64)
    vix = np.asarray(array[:, 1], dtype=np.float64)
    sample_count = len(price) - DEFAULT_SP500_VIX_TARGET_LENGTH + 1
    if sample_count <= 0:
        raise SystemExit("S&P500/VIX array is shorter than the target window length.")
    return build_context_payload(
        price,
        vix,
        context_length=context_length,
        target_length=DEFAULT_SP500_VIX_TARGET_LENGTH,
        sample_count=sample_count,
        source=str(data_path),
    )


def build_context_payload(
    price: np.ndarray,
    vix: np.ndarray,
    *,
    context_length: int,
    target_length: int,
    sample_count: int,
    source: str,
) -> dict[str, Any]:
    """Build left-padded historical context arrays for target-window starts."""
    del target_length
    sample_indices = np.arange(sample_count, dtype=np.int64)
    price_contexts = np.stack(
        [historical_context(price, int(index), context_length) for index in sample_indices],
        axis=0,
    )
    vix_contexts = np.stack(
        [historical_context(vix, int(index), context_length) for index in sample_indices],
        axis=0,
    )
    labels = vix[sample_indices][:, None].astype(np.float64)
    return {
        "price_contexts": price_contexts,
        "vix_contexts": vix_contexts,
        "labels": labels,
        "sample_indices": sample_indices,
        "source": source,
        "padding": "left-pad with the first available historical value; index 0 uses series[0].",
        "sample_count": sample_count,
    }


def historical_context(series: np.ndarray, target_start: int, context_length: int) -> np.ndarray:
    """Return historical context ending before ``target_start`` with left padding."""
    if target_start < 0:
        raise ValueError("target_start must be non-negative.")
    values = np.asarray(series, dtype=np.float64)
    start = max(0, target_start - context_length)
    context = values[start:target_start]
    if context.size == 0:
        context = values[:1]
    if context.size < context_length:
        pad_value = context[0]
        padding = np.full(context_length - context.size, pad_value, dtype=np.float64)
        context = np.concatenate([padding, context], axis=0)
    return context.astype(np.float64, copy=False)


def build_standardization_payload(
    train_features: np.ndarray,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Build fitted standardisation statistics from train features."""
    if not enabled:
        return {
            "enabled": False,
            "epsilon": DEFAULT_STANDARDIZATION_EPSILON,
            "mean": np.empty((0,), dtype=np.float64),
            "std": np.empty((0,), dtype=np.float64),
        }
    mean, std = feature_standardization_statistics(
        train_features,
        epsilon=DEFAULT_STANDARDIZATION_EPSILON,
    )
    return {
        "enabled": True,
        "epsilon": DEFAULT_STANDARDIZATION_EPSILON,
        "mean": mean,
        "std": std,
    }


def standardization_summary(
    standardization: dict[str, Any],
    *,
    features: np.ndarray,
) -> dict[str, Any]:
    """Return JSON-safe standardisation metadata."""
    enabled = bool(standardization["enabled"])
    summary: dict[str, Any] = {
        "enabled": enabled,
        "epsilon": float(standardization["epsilon"]),
        "fit_split": "train",
        "applied_to": ["train", "eval"] if enabled else [],
    }
    if not enabled:
        return summary
    mean = np.asarray(standardization["mean"], dtype=np.float64)
    std = np.asarray(standardization["std"], dtype=np.float64)
    summary.update(
        {
            "stats_file": "signature_feature_standardization.npz",
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "mean_abs_max": float(np.max(np.abs(mean))),
            "std_min": float(np.min(std)),
            "std_max": float(np.max(std)),
            "post_transform_mean_abs_max": float(np.max(np.abs(np.mean(features, axis=0)))),
            "post_transform_std_min": float(np.min(np.std(features, axis=0))),
            "post_transform_std_max": float(np.max(np.std(features, axis=0))),
        }
    )
    return summary


def success_summary(
    args: argparse.Namespace,
    *,
    data_payload: dict[str, Any],
    metadata: dict[str, Any],
    features: np.ndarray,
    standardization: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build a successful extraction summary."""
    standardization_metadata = standardization_summary(standardization, features=features)
    outputs = [
        "train_signature_features.npz",
        "eval_signature_features.npz",
        "signature_feature_summary.json",
        "signature_feature_summary.md",
    ]
    if standardization_metadata["enabled"]:
        outputs.append("signature_feature_standardization.npz")
    return {
        "status": "success",
        "dataset": args.dataset,
        "synthetic": bool(args.synthetic),
        "source": data_payload["source"],
        "depth": args.depth,
        "context_length": args.context_length,
        "lead_lag": bool(args.use_lead_lag),
        "include_time": bool(args.include_time),
        "include_vix": bool(args.include_vix),
        "seed": args.seed,
        "feature_shape": list(features.shape),
        "feature_dimension": int(features.shape[1]),
        "sample_count": int(features.shape[0]),
        "finite": bool(np.isfinite(features).all()),
        "metadata": metadata,
        "standardization": standardization_metadata,
        "padding": data_payload["padding"],
        "elapsed_seconds": elapsed_seconds,
        "outputs": outputs,
    }


def missing_dependency_summary(
    args: argparse.Namespace,
    message: str,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build a missing-optional-dependency summary."""
    return {
        "status": "missing_optional_dependency",
        "dataset": args.dataset,
        "synthetic": bool(args.synthetic),
        "depth": args.depth,
        "context_length": args.context_length,
        "lead_lag": bool(args.use_lead_lag),
        "include_time": bool(args.include_time),
        "include_vix": bool(args.include_vix),
        "standardization": {
            "enabled": bool(getattr(args, "standardize", False)),
            "fit_split": "train",
        },
        "seed": args.seed,
        "iisignature_installed": False,
        "message": message,
        "elapsed_seconds": perf_counter() - elapsed_seconds,
        "outputs": [
            "signature_feature_summary.json",
            "signature_feature_summary.md",
        ],
    }


def write_summary_files(output_dir: Path, summary: dict[str, Any]) -> None:
    """Write JSON and Markdown summaries for a signature feature run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "signature_feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (output_dir / "signature_feature_summary.md").write_text(
        summary_to_markdown(summary),
        encoding="utf-8",
    )


def summary_to_markdown(summary: dict[str, Any]) -> str:
    """Render a compact Markdown summary."""
    lines = [
        "# Signature Feature Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Dataset: `{summary['dataset']}`",
        f"- Synthetic: `{summary['synthetic']}`",
        f"- Depth: `{summary['depth']}`",
        f"- Context length: `{summary['context_length']}`",
        f"- Lead-lag: `{summary['lead_lag']}`",
        f"- Include time: `{summary['include_time']}`",
        f"- Include VIX: `{summary['include_vix']}`",
        f"- Standardized: `{summary['standardization']['enabled']}`",
        f"- Seed: `{summary['seed']}`",
    ]
    if summary["status"] == "success":
        lines.extend(
            [
                f"- Feature shape: `{summary['feature_shape']}`",
                f"- Feature dimension: `{summary['feature_dimension']}`",
                f"- Finite: `{summary['finite']}`",
                f"- Source: `{summary['source']}`",
                f"- Padding: {summary['padding']}",
            ]
        )
    else:
        lines.extend(
            [
                f"- `iisignature` installed: `{summary['iisignature_installed']}`",
                f"- Message: {summary['message']}",
            ]
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.extend([f"- `{name}`" for name in summary["outputs"]])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
