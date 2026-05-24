"""Plot qualitative feature geometry for real and generated path samples."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "time_causal_vae_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from time_causal_vae.evaluation.sample_geometry import (
    DatasetType,
    PathFeatureMatrix,
    feature_matrix_to_json,
    fit_tsne_projection,
    kde_or_ecdf_summary,
    path_feature_matrix,
    projection_to_json,
)

REAL_KEYS = (
    "real_paths",
    "real_data",
    "real_decoder_space",
    "paths",
    "data",
    "batch",
    "samples",
)
GENERATED_KEYS = (
    "decoded_paths",
    "generated_paths",
    "fake_paths",
    "fake_data",
    "generated_decoder_space",
    "decoded_decoder_space",
    "paths",
    "samples",
    "data",
    "batch",
)
DEFAULT_PLOT_FEATURES = (
    "terminal_return",
    "realised_volatility",
    "maximum_drawdown",
    "return_skewness",
    "return_excess_kurtosis",
    "return_autocorr_lag_1",
    "squared_return_autocorr_lag_1",
    "detected_jump_count",
    "return_var_01",
    "return_expected_shortfall_01",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the sample-geometry diagnostics parser."""
    parser = argparse.ArgumentParser(
        description="Create qualitative t-SNE and KDE/ECDF diagnostics for generated path samples.",
    )
    parser.add_argument("--real-batch", required=True, help="Path to a real path tensor payload.")
    parser.add_argument(
        "--generated-batch",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="Generated path tensor payload as name:path. Repeat for multiple models.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory under outputs/.")
    parser.add_argument(
        "--dataset",
        choices=("sp500_vix", "hawkes_jump", "generic"),
        default="generic",
        help="Dataset convention used for optional features.",
    )
    parser.add_argument("--tsne", action="store_true", help="Create a qualitative t-SNE plot.")
    parser.add_argument("--kde", action="store_true", help="Create ECDF and optional KDE plots.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for t-SNE initialisation.")
    return parser


def main() -> None:
    """Run sample-geometry diagnostics."""
    args = build_parser().parse_args()
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_type = cast(DatasetType, args.dataset)

    real_paths = load_path_tensor(args.real_batch, preferred_keys=REAL_KEYS)
    generated_paths = {
        name: load_path_tensor(path, preferred_keys=GENERATED_KEYS)
        for name, path in parse_generated_batches(args.generated_batch).items()
    }
    if not generated_paths:
        raise SystemExit("At least one --generated-batch name:path pair is required.")

    real_features = path_feature_matrix(real_paths, dataset_type=dataset_type)
    generated_features = {
        name: path_feature_matrix(paths, dataset_type=dataset_type)
        for name, paths in generated_paths.items()
    }

    torch.save(
        {
            "real": {
                "values": real_features.values,
                "feature_names": real_features.feature_names,
                "dataset_type": real_features.dataset_type,
            },
            "generated": {
                name: {
                    "values": features.values,
                    "feature_names": features.feature_names,
                    "dataset_type": features.dataset_type,
                }
                for name, features in generated_features.items()
            },
        },
        output_dir / "sample_geometry_features.pt",
    )

    summary: dict[str, Any] = {
        "dataset_type": dataset_type,
        "real_batch": str(args.real_batch),
        "generated_batches": parse_generated_batches(args.generated_batch),
        "feature_names": real_features.feature_names,
        "real_shape": list(real_features.values.shape),
        "generated_shapes": {
            name: list(features.values.shape) for name, features in generated_features.items()
        },
        "notes": {
            "tsne": "qualitative_only",
            "outputs": "local figures are written under outputs/ and are not model artefacts.",
        },
    }
    generated_plots: list[str] = []

    write_json(output_dir / "real_feature_matrix.json", feature_matrix_to_json(real_features))
    write_json(
        output_dir / "generated_feature_matrices.json",
        {name: feature_matrix_to_json(features) for name, features in generated_features.items()},
    )

    if args.tsne:
        projection = fit_tsne_projection(
            real_features,
            generated_features,
            random_state=int(args.seed),
        )
        write_json(output_dir / "sample_geometry_projection.json", projection_to_json(projection))
        plot_projection(output_dir / "sample_geometry_projection.png", projection)
        generated_plots.append("sample_geometry_projection.png")
        summary["projection"] = {
            "method": projection.method,
            "metadata": projection.metadata,
        }

    if args.kde:
        distribution_summary = {
            "real": kde_or_ecdf_summary(real_features),
            "generated": {
                name: kde_or_ecdf_summary(features) for name, features in generated_features.items()
            },
        }
        write_json(output_dir / "sample_geometry_kde_ecdf_summary.json", distribution_summary)
        feature_names = selected_plot_features(real_features.feature_names)
        plot_ecdf_grid(
            output_dir / "sample_geometry_ecdf.png",
            real_features=real_features,
            generated_features=generated_features,
            feature_names=feature_names,
        )
        generated_plots.append("sample_geometry_ecdf.png")
        kde_plotted = plot_kde_grid(
            output_dir / "sample_geometry_kde.png",
            distribution_summary=distribution_summary,
            feature_names=feature_names,
        )
        if kde_plotted:
            generated_plots.append("sample_geometry_kde.png")
        summary["distribution_summary"] = {
            "feature_names": feature_names,
            "ecdf": True,
            "kde": kde_plotted,
            "fallback": None
            if kde_plotted
            else "ECDF only; scipy KDE was unavailable or feature densities were degenerate.",
        }

    summary["generated_plots"] = generated_plots
    write_json(output_dir / "sample_geometry_summary.json", summary)

    print("Sample-geometry diagnostics complete.")
    print(f"output_dir: {output_dir}")
    print(f"dataset_type: {dataset_type}")
    print(f"real_shape: {summary['real_shape']}")
    print(f"generated_shapes: {summary['generated_shapes']}")
    print(f"plots: {', '.join(generated_plots) if generated_plots else 'none requested'}")


def parse_generated_batches(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``name:path`` generated batch arguments."""
    parsed: dict[str, str] = {}
    for raw_value in values:
        if ":" not in raw_value:
            raise SystemExit(
                f"--generated-batch values must use name:path format. Received: {raw_value!r}"
            )
        name, path = raw_value.split(":", 1)
        if not name.strip() or not path.strip():
            raise SystemExit(
                "--generated-batch values must include both a name and a path. "
                f"Received: {raw_value!r}"
            )
        if name in parsed:
            raise SystemExit(f"Duplicate generated model name: {name!r}.")
        parsed[name] = path
    return parsed


def load_path_tensor(path: str | Path, *, preferred_keys: Sequence[str]) -> Tensor:
    """Load a path tensor from a tensor file or dictionary payload."""
    payload_path = Path(path)
    if not payload_path.exists():
        raise SystemExit(f"Path batch does not exist: {payload_path}")
    payload = torch.load(payload_path, map_location="cpu")
    if isinstance(payload, Tensor):
        return payload.detach().float()
    if not isinstance(payload, Mapping):
        raise SystemExit(f"Expected a tensor or dictionary payload in {payload_path}.")

    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, Tensor):
            return value.detach().float()

    tensor_candidates = [
        (key, value)
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, Tensor) and value.ndim in {2, 3}
    ]
    if len(tensor_candidates) == 1:
        return tensor_candidates[0][1].detach().float()
    keys = ", ".join(str(key) for key in payload)
    expected = ", ".join(preferred_keys)
    raise SystemExit(
        f"Could not select a path tensor from {payload_path}. "
        f"Expected one of [{expected}] or a single 2D/3D tensor. Keys: {keys}"
    )


def selected_plot_features(feature_names: Sequence[str], *, max_features: int = 6) -> list[str]:
    """Choose a compact, stable set of feature panels for reports."""
    selected = [name for name in DEFAULT_PLOT_FEATURES if name in feature_names]
    if len(selected) < max_features:
        selected.extend(name for name in feature_names if name not in selected)
    return selected[:max_features]


def plot_projection(path: Path, projection: Any) -> None:
    """Plot a two-dimensional projection coloured by source label."""
    coordinates = projection.coordinates.detach().cpu()
    labels = projection.labels
    unique_labels = list(dict.fromkeys(labels))
    figure, axis = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    for label in unique_labels:
        indices = [index for index, observed in enumerate(labels) if observed == label]
        points = coordinates[indices]
        axis.scatter(points[:, 0], points[:, 1], s=18, alpha=0.75, label=label)
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    axis.set_title(f"{projection.method.upper()} feature projection")
    axis.legend(frameon=False, fontsize="small")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_ecdf_grid(
    path: Path,
    *,
    real_features: PathFeatureMatrix,
    generated_features: Mapping[str, PathFeatureMatrix],
    feature_names: Sequence[str],
) -> None:
    """Plot empirical CDF overlays for selected features."""
    figure, axes = make_feature_grid(len(feature_names))
    for axis, feature_name in zip(axes, feature_names, strict=False):
        plot_single_ecdf(axis, real_features, feature_name, label="real", linewidth=2.0)
        for model_name, features in generated_features.items():
            plot_single_ecdf(axis, features, feature_name, label=model_name, linewidth=1.35)
        axis.set_title(feature_name)
        axis.set_xlabel("feature value")
        axis.set_ylabel("ECDF")
    axes[0].legend(frameon=False, fontsize="small")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_single_ecdf(
    axis: Any,
    features: PathFeatureMatrix,
    feature_name: str,
    *,
    label: str,
    linewidth: float,
) -> None:
    """Plot one ECDF line."""
    column = feature_column(features, feature_name)
    sorted_values = column.sort().values
    y_values = torch.arange(1, sorted_values.numel() + 1, dtype=torch.float32)
    y_values = y_values / float(sorted_values.numel())
    axis.plot(sorted_values.numpy(), y_values.numpy(), label=label, linewidth=linewidth)


def plot_kde_grid(
    path: Path,
    *,
    distribution_summary: Mapping[str, Any],
    feature_names: Sequence[str],
) -> bool:
    """Plot KDE overlays for selected features when density summaries exist."""
    figure, axes = make_feature_grid(len(feature_names))
    plotted = False
    for axis, feature_name in zip(axes, feature_names, strict=False):
        plotted |= plot_single_kde(axis, distribution_summary["real"], feature_name, label="real")
        for model_name, summary in distribution_summary["generated"].items():
            plotted |= plot_single_kde(axis, summary, feature_name, label=str(model_name))
        axis.set_title(feature_name)
        axis.set_xlabel("feature value")
        axis.set_ylabel("density")
    axes[0].legend(frameon=False, fontsize="small")
    if plotted:
        figure.savefig(path, dpi=160)
    plt.close(figure)
    return plotted


def plot_single_kde(
    axis: Any, summary: Mapping[str, Any], feature_name: str, *, label: str
) -> bool:
    """Plot one KDE line if it is available in a summary payload."""
    feature_summary = summary["features"].get(feature_name, {})
    kde = feature_summary.get("kde")
    if not isinstance(kde, Mapping):
        return False
    x_values = kde.get("x")
    density = kde.get("density")
    if not isinstance(x_values, list) or not isinstance(density, list):
        return False
    axis.plot(x_values, density, label=label, linewidth=1.5)
    return True


def make_feature_grid(panel_count: int) -> tuple[Any, list[Any]]:
    """Create a compact feature-panel grid."""
    column_count = 2 if panel_count > 1 else 1
    row_count = (panel_count + column_count - 1) // column_count
    figure, axes_array = plt.subplots(
        row_count,
        column_count,
        figsize=(6.2 * column_count, 3.6 * row_count),
        constrained_layout=True,
        squeeze=False,
    )
    axes = [axis for row in axes_array for axis in row]
    for axis in axes[panel_count:]:
        axis.set_visible(False)
    return figure, axes[:panel_count]


def feature_column(features: PathFeatureMatrix, feature_name: str) -> Tensor:
    """Return one named feature column."""
    try:
        index = features.feature_names.index(feature_name)
    except ValueError as exc:
        raise SystemExit(f"Feature {feature_name!r} is unavailable.") from exc
    return features.values[:, index].detach().float().cpu()


def validate_output_dir(output_dir: str) -> Path:
    """Validate that sample-geometry artifacts stay below local outputs/."""
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic JSON artifact."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
