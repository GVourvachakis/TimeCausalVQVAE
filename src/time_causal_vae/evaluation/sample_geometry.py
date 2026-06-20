"""Feature-space diagnostics for real and generated path samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from time_causal_vae.evaluation.market_diagnostics import (
    DEFAULT_AUTOCORRELATION_LAGS,
    compute_log_returns,
    maximum_drawdown,
    squeeze_paths,
    terminal_returns,
)

DatasetType = Literal["sp500_vix", "hawkes_jump", "generic"]


@dataclass(frozen=True)
class PathFeatureMatrix:
    """Named feature matrix for one batch of paths."""

    values: Tensor
    feature_names: list[str]
    dataset_type: DatasetType


@dataclass(frozen=True)
class ProjectionResult:
    """Two-dimensional feature projection for qualitative visual diagnostics."""

    coordinates: Tensor
    labels: list[str]
    feature_names: list[str]
    method: str
    metadata: dict[str, Any]


def path_feature_matrix(
    paths: Tensor,
    dataset_type: DatasetType,
    *,
    autocorrelation_lags: Sequence[int] = DEFAULT_AUTOCORRELATION_LAGS,
) -> PathFeatureMatrix:
    """Build one financial feature vector per path.

    The feature matrix is intended for qualitative geometry diagnostics, not for
    model selection. Features are computed independently per path.
    """
    if dataset_type not in {"sp500_vix", "hawkes_jump", "generic"}:
        raise ValueError(f"Unsupported dataset_type: {dataset_type!r}.")
    paths_2d = squeeze_paths(paths)
    is_hawkes_return_series = dataset_type == "hawkes_jump" and not bool((paths_2d > 0.0).all())
    returns = _returns_for_dataset(paths, dataset_type=dataset_type)
    if returns.ndim != 2:
        raise ValueError(f"Expected per-path return matrix; got {tuple(returns.shape)}.")

    drawdown_paths = _price_paths_from_returns(returns) if is_hawkes_return_series else paths
    terminal_column = returns.sum(dim=1) if is_hawkes_return_series else terminal_returns(paths)
    feature_columns: list[Tensor] = [
        terminal_column,
        returns.std(dim=1, unbiased=False),
        maximum_drawdown(drawdown_paths),
        _skewness_per_path(returns),
        _excess_kurtosis_per_path(returns),
        returns.mean(dim=1),
        returns.abs().max(dim=1).values,
    ]
    feature_names = [
        "terminal_return",
        "realised_volatility",
        "maximum_drawdown",
        "return_skewness",
        "return_excess_kurtosis",
        "mean_return",
        "max_abs_return",
    ]

    for lag in autocorrelation_lags:
        autocorr = _autocorrelation_per_path(returns, lag=lag)
        squared_autocorr = _autocorrelation_per_path(returns.square(), lag=lag)
        feature_columns.extend([autocorr, squared_autocorr])
        feature_names.extend([f"return_autocorr_lag_{lag}", f"squared_return_autocorr_lag_{lag}"])

    if dataset_type == "hawkes_jump":
        jump_indicators = _detect_jumps_from_return_matrix(returns).float()
        jump_counts = jump_indicators.sum(dim=1)
        feature_columns.extend([
            jump_counts,
            jump_indicators.mean(dim=1),
            _value_at_risk_per_path(returns, level=0.01),
            _expected_shortfall_per_path(returns, level=0.01),
            paths_2d[:, -1] - paths_2d[:, 0],
        ])
        feature_names.extend([
            "detected_jump_count",
            "detected_jump_fraction",
            "return_var_01",
            "return_expected_shortfall_01",
            "path_increment",
        ])

    values = torch.stack([column.detach().float().cpu() for column in feature_columns], dim=1)
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return PathFeatureMatrix(values=values, feature_names=feature_names, dataset_type=dataset_type)


def _returns_for_dataset(paths: Tensor, *, dataset_type: DatasetType) -> Tensor:
    paths_2d = squeeze_paths(paths)
    if dataset_type == "hawkes_jump" and not bool((paths_2d > 0.0).all()):
        return paths_2d
    return compute_log_returns(paths)


def fit_tsne_projection(
    real_features: PathFeatureMatrix | Tensor,
    generated_features_by_model: Mapping[str, PathFeatureMatrix | Tensor],
    *,
    random_state: int = 0,
) -> ProjectionResult:
    """Fit a qualitative two-dimensional t-SNE projection when available.

    If scikit-learn's t-SNE is unavailable or too few samples are provided, the
    function returns a deterministic PCA-style projection using torch SVD.
    """
    real_values, feature_names = _feature_values_and_names(real_features)
    feature_blocks = [real_values]
    labels = ["real"] * real_values.shape[0]
    for model_name, features in generated_features_by_model.items():
        values, model_feature_names = _feature_values_and_names(features)
        if model_feature_names and feature_names and model_feature_names != feature_names:
            raise ValueError(f"Feature names for {model_name!r} do not match real features.")
        feature_blocks.append(values)
        labels.extend([str(model_name)] * values.shape[0])

    combined = torch.cat(feature_blocks, dim=0)
    standardised = _standardise_features(combined)
    sample_count = standardised.shape[0]
    metadata: dict[str, Any] = {
        "sample_count": int(sample_count),
        "feature_count": int(standardised.shape[1]),
        "qualitative_only": True,
    }

    if sample_count >= 4:
        try:
            from sklearn.manifold import TSNE

            perplexity = min(30.0, max(2.0, float((sample_count - 1) // 3)))
            projection_array = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=random_state,
            ).fit_transform(standardised.numpy())
            metadata["perplexity"] = perplexity
            return ProjectionResult(
                coordinates=torch.as_tensor(projection_array, dtype=torch.float32),
                labels=labels,
                feature_names=feature_names,
                method="tsne",
                metadata=metadata,
            )
        except ImportError:
            metadata["fallback_reason"] = "sklearn_unavailable"

    metadata.setdefault("fallback_reason", "too_few_samples_for_tsne")
    return ProjectionResult(
        coordinates=_pca_projection(standardised),
        labels=labels,
        feature_names=feature_names,
        method="pca_fallback",
        metadata=metadata,
    )


def kde_or_ecdf_summary(
    features: PathFeatureMatrix | Tensor,
    *,
    max_ecdf_points: int = 200,
) -> dict[str, Any]:
    """Summarise feature distributions with ECDF points and optional KDE values."""
    values, feature_names = _feature_values_and_names(features)
    summaries: dict[str, Any] = {}
    for index in range(values.shape[1]):
        name = feature_names[index] if feature_names else f"feature_{index}"
        column = values[:, index].detach().float().cpu()
        sorted_values = column.sort().values
        ecdf_x, ecdf_y = _subsample_ecdf(sorted_values, max_points=max_ecdf_points)
        summary: dict[str, Any] = {
            "mean": float(column.mean().item()),
            "std": float(column.std(unbiased=False).item()),
            "min": float(column.min().item()),
            "max": float(column.max().item()),
            "q01": float(torch.quantile(column, 0.01).item()),
            "q05": float(torch.quantile(column, 0.05).item()),
            "q50": float(torch.quantile(column, 0.50).item()),
            "q95": float(torch.quantile(column, 0.95).item()),
            "q99": float(torch.quantile(column, 0.99).item()),
            "ecdf": {
                "x": [float(value) for value in ecdf_x],
                "y": [float(value) for value in ecdf_y],
            },
        }
        kde = _kde_summary(column)
        if kde is not None:
            summary["kde"] = kde
        summaries[name] = summary
    return {
        "feature_names": feature_names,
        "feature_count": int(values.shape[1]),
        "sample_count": int(values.shape[0]),
        "features": summaries,
    }


def _skewness_per_path(returns: Tensor) -> Tensor:
    centred = returns - returns.mean(dim=1, keepdim=True)
    std = returns.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-12)
    return (centred / std).pow(3).mean(dim=1)


def _excess_kurtosis_per_path(returns: Tensor) -> Tensor:
    centred = returns - returns.mean(dim=1, keepdim=True)
    std = returns.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-12)
    return (centred / std).pow(4).mean(dim=1) - 3.0


def _autocorrelation_per_path(returns: Tensor, *, lag: int) -> Tensor:
    if lag <= 0:
        raise ValueError("autocorrelation lag must be positive.")
    if lag >= returns.shape[1]:
        return torch.zeros(returns.shape[0], dtype=returns.dtype, device=returns.device)
    centred = returns - returns.mean(dim=1, keepdim=True)
    variance = centred.square().mean(dim=1).clamp_min(1e-12)
    autocovariance = (centred[:, :-lag] * centred[:, lag:]).mean(dim=1)
    return autocovariance / variance


def _value_at_risk_per_path(returns: Tensor, *, level: float) -> Tensor:
    return torch.quantile(returns.detach().float(), level, dim=1)


def _expected_shortfall_per_path(returns: Tensor, *, level: float) -> Tensor:
    var = _value_at_risk_per_path(returns, level=level)
    mask = returns <= var[:, None]
    counts = mask.sum(dim=1).clamp_min(1)
    tail_sums = torch.where(mask, returns, torch.zeros_like(returns)).sum(dim=1)
    return tail_sums / counts


def _price_paths_from_returns(returns: Tensor, *, initial_price: float = 1.0) -> Tensor:
    return float(initial_price) * torch.exp(torch.cumsum(returns.float(), dim=1))


def _detect_jumps_from_return_matrix(
    returns: Tensor,
    *,
    threshold_multiplier: float = 4.0,
    min_abs_return: float = 0.0,
) -> Tensor:
    median = returns.median()
    mad = (returns - median).abs().median()
    robust_scale = (1.4826 * mad).clamp_min(1e-8)
    threshold = max(float(threshold_multiplier * robust_scale.item()), float(min_abs_return))
    return (returns - median).abs() >= threshold


def _feature_values_and_names(features: PathFeatureMatrix | Tensor) -> tuple[Tensor, list[str]]:
    if isinstance(features, PathFeatureMatrix):
        return features.values.detach().float().cpu(), list(features.feature_names)
    if features.ndim != 2:
        raise ValueError(f"Expected a [sample, feature] matrix; got {tuple(features.shape)}.")
    return features.detach().float().cpu(), []


def _standardise_features(values: Tensor) -> Tensor:
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return (values - mean) / std


def _pca_projection(values: Tensor) -> Tensor:
    if values.shape[1] == 0:
        return torch.zeros(values.shape[0], 2)
    if values.shape[0] <= 1:
        return torch.zeros(values.shape[0], 2)
    _u, _s, vh = torch.linalg.svd(values, full_matrices=False)
    components = vh[: min(2, vh.shape[0])]
    projected = values @ components.T
    if projected.shape[1] == 1:
        projected = torch.cat([projected, torch.zeros(projected.shape[0], 1)], dim=1)
    return torch.as_tensor(projected[:, :2]).float()


def _subsample_ecdf(sorted_values: Tensor, *, max_points: int) -> tuple[list[float], list[float]]:
    n_values = sorted_values.numel()
    if n_values == 0:
        return [], []
    if n_values <= max_points:
        indices = torch.arange(n_values)
    else:
        indices = torch.linspace(0, n_values - 1, max_points).round().long()
    selected = sorted_values.index_select(0, indices)
    y_values = (indices.float() + 1.0) / float(n_values)
    return selected.tolist(), y_values.tolist()


def _kde_summary(values: Tensor, *, grid_size: int = 128) -> dict[str, list[float]] | None:
    if values.numel() < 2:
        return None
    std = values.std(unbiased=False)
    if float(std.item()) <= 1e-12:
        return None
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return None

    grid = torch.linspace(float(values.min().item()), float(values.max().item()), grid_size)
    kde = gaussian_kde(values.numpy())
    density = kde(grid.numpy())
    return {
        "x": [float(value) for value in grid.tolist()],
        "density": [float(value) for value in density.tolist()],
    }


def projection_to_json(result: ProjectionResult) -> dict[str, Any]:
    """Return a JSON-serialisable projection payload."""
    return {
        "method": result.method,
        "labels": result.labels,
        "feature_names": result.feature_names,
        "coordinates": result.coordinates.detach().cpu().tolist(),
        "metadata": result.metadata,
    }


def feature_matrix_to_json(features: PathFeatureMatrix) -> dict[str, Any]:
    """Return a JSON-serialisable feature matrix payload."""
    return {
        "dataset_type": features.dataset_type,
        "feature_names": features.feature_names,
        "values": features.values.detach().cpu().tolist(),
        "shape": list(features.values.shape),
    }
