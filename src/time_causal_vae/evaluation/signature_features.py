"""Optional signature and log-signature feature extraction utilities.

The functions in this module use :mod:`iisignature` only when a caller requests
signature features. The package is intentionally not a hard project dependency.
These helpers build historical-context features for later conditioning ablations;
callers must pass context paths that end before the target/generated window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

import numpy as np

IISIGNATURE_INSTALL_HINT = (
    "Optional signature feature extraction requires 'iisignature'. It is not installed by "
    "default. In a temporary or opt-in environment, install NumPy first and then try: "
    "pip install iisignature --no-build-isolation"
)
DEFAULT_STANDARDIZATION_EPSILON = 1e-8


class OptionalDependencyError(ImportError):
    """Raised when an optional signature dependency is unavailable."""


@dataclass(frozen=True)
class SignatureFeatureConfig:
    """Configuration for one signature feature extraction pass."""

    depth: int
    use_lead_lag: bool = False
    include_time: bool = False
    include_vix: bool = False
    use_log_signature: bool = True


@dataclass(frozen=True)
class SignatureFeatureMetadata:
    """Metadata describing a computed signature feature matrix."""

    depth: int
    input_channels: list[str]
    lead_lag: bool
    feature_dimension: int
    preprocessing: dict[str, bool | str]
    finite: bool
    package: str
    package_version: str


def load_iisignature() -> Any:
    """Import ``iisignature`` or raise a clear optional-dependency error."""
    try:
        import iisignature
    except ImportError as exc:
        raise OptionalDependencyError(IISIGNATURE_INSTALL_HINT) from exc
    return iisignature


def iisignature_version() -> str:
    """Return the installed ``iisignature`` version when available."""
    try:
        return metadata.version("iisignature")
    except metadata.PackageNotFoundError:
        return "unknown"


def as_1d_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    """Return ``values`` as a finite one-dimensional float64 array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or singleton two-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def normalised_price_path(price_context: np.ndarray) -> np.ndarray:
    """Return a price path normalised by its first context value."""
    prices = as_1d_float_array(price_context, name="price_context")
    first = prices[0]
    if not np.isfinite(first) or abs(first) <= np.finfo(np.float64).eps:
        raise ValueError("price_context first value must be finite and non-zero.")
    normalised = np.asarray(prices / first, dtype=np.float64)
    return normalised[:, None]


def log_return_path(normalised_prices: np.ndarray) -> np.ndarray:
    """Return one-step log returns with a leading zero for alignment."""
    prices = as_1d_float_array(normalised_prices, name="normalised_prices")
    if (prices <= 0.0).any():
        raise ValueError("normalised_prices must be strictly positive for log returns.")
    log_prices = np.log(prices)
    returns = np.asarray(np.diff(log_prices, prepend=log_prices[0]), dtype=np.float64)
    return returns[:, None]


def cumulative_log_return_path(normalised_prices: np.ndarray) -> np.ndarray:
    """Return cumulative log returns from the first context value."""
    prices = as_1d_float_array(normalised_prices, name="normalised_prices")
    if (prices <= 0.0).any():
        raise ValueError("normalised_prices must be strictly positive for cumulative returns.")
    cumulative = np.asarray(np.log(prices / prices[0]), dtype=np.float64)
    return cumulative[:, None]


def time_channel(length: int) -> np.ndarray:
    """Return a `[0, 1]` time channel of ``length`` rows."""
    if length <= 0:
        raise ValueError("length must be positive.")
    if length == 1:
        return np.zeros((1, 1), dtype=np.float64)
    return np.linspace(0.0, 1.0, num=length, dtype=np.float64)[:, None]


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Apply a lead-lag transform to a two-dimensional path array."""
    path_2d = validate_path_array(path, name="path")
    repeated = np.repeat(path_2d, repeats=2, axis=0)
    return np.concatenate([repeated[:-1], repeated[1:]], axis=1)


def validate_path_array(path: np.ndarray, *, name: str) -> np.ndarray:
    """Return ``path`` as a finite two-dimensional float64 array."""
    array = np.asarray(path, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [time, channels].")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have positive time and channel dimensions.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def build_context_path(
    price_context: np.ndarray,
    *,
    vix_context: np.ndarray | None = None,
    include_time: bool = False,
    include_vix: bool = False,
    use_lead_lag: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build a finite signature input path from historical context only."""
    normalised_prices = normalised_price_path(price_context).reshape(-1)
    channels = [
        normalised_prices[:, None],
        log_return_path(normalised_prices),
        cumulative_log_return_path(normalised_prices),
    ]
    channel_names = ["normalised_price", "log_return", "cumulative_log_return"]

    if include_time:
        channels.append(time_channel(len(normalised_prices)))
        channel_names.append("time")

    if include_vix:
        if vix_context is None:
            raise ValueError("include_vix=True requires vix_context.")
        vix = as_1d_float_array(vix_context, name="vix_context")
        if len(vix) != len(normalised_prices):
            raise ValueError("vix_context length must match price_context length.")
        channels.append(vix[:, None])
        channel_names.append("vix")

    path = validate_path_array(np.concatenate(channels, axis=1), name="context_path")
    if use_lead_lag:
        path = lead_lag_transform(path)
        channel_names = [f"{name}_lead" for name in channel_names] + [
            f"{name}_lag" for name in channel_names
        ]
    return path, channel_names


def compute_truncated_signature(
    path: np.ndarray,
    *,
    depth: int,
    log_signature: bool = True,
) -> np.ndarray:
    """Compute a truncated signature or log-signature for one path."""
    if depth <= 0:
        raise ValueError("depth must be positive.")
    iisignature = load_iisignature()
    path_2d = validate_path_array(path, name="path")
    if log_signature:
        prepared = iisignature.prepare(path_2d.shape[1], depth)
        feature = iisignature.logsig(path_2d, prepared)
    else:
        feature = iisignature.sig(path_2d, depth)
    feature = np.asarray(feature, dtype=np.float64)
    if feature.ndim != 1:
        raise ValueError("signature feature must be one-dimensional.")
    if not np.isfinite(feature).all():
        raise ValueError("signature feature contains non-finite values.")
    return feature


def compute_signature_feature_batch(
    price_contexts: np.ndarray,
    *,
    vix_contexts: np.ndarray | None = None,
    config: SignatureFeatureConfig,
) -> tuple[np.ndarray, SignatureFeatureMetadata]:
    """Compute signature features for a batch of historical contexts."""
    prices = validate_context_batch(price_contexts, name="price_contexts")
    vix_batch = None
    if config.include_vix:
        if vix_contexts is None:
            raise ValueError("config.include_vix=True requires vix_contexts.")
        vix_batch = validate_context_batch(vix_contexts, name="vix_contexts")
        if vix_batch.shape != prices.shape:
            raise ValueError("vix_contexts must match price_contexts shape.")

    features: list[np.ndarray] = []
    input_channels: list[str] | None = None
    for index, price_context in enumerate(prices):
        vix_context = None if vix_batch is None else vix_batch[index]
        path, channel_names = build_context_path(
            price_context,
            vix_context=vix_context,
            include_time=config.include_time,
            include_vix=config.include_vix,
            use_lead_lag=config.use_lead_lag,
        )
        feature = compute_truncated_signature(
            path,
            depth=config.depth,
            log_signature=config.use_log_signature,
        )
        features.append(feature)
        if input_channels is None:
            input_channels = channel_names

    feature_matrix = np.stack(features, axis=0)
    finite = bool(np.isfinite(feature_matrix).all())
    metadata_obj = SignatureFeatureMetadata(
        depth=config.depth,
        input_channels=input_channels or [],
        lead_lag=config.use_lead_lag,
        feature_dimension=int(feature_matrix.shape[1]),
        preprocessing={
            "include_time": config.include_time,
            "include_vix": config.include_vix,
            "feature_type": "log_signature" if config.use_log_signature else "signature",
        },
        finite=finite,
        package="iisignature",
        package_version=iisignature_version(),
    )
    return feature_matrix, metadata_obj


def feature_standardization_statistics(
    train_features: np.ndarray,
    *,
    epsilon: float = DEFAULT_STANDARDIZATION_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Return guarded train-set mean and standard deviation for feature scaling."""
    features = validate_feature_matrix(train_features, name="train_features")
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    guarded_std = np.where(std < epsilon, 1.0, std)
    return mean.astype(np.float64, copy=False), guarded_std.astype(np.float64, copy=False)


def apply_feature_standardization(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply fitted feature standardisation statistics to a feature matrix."""
    feature_matrix = validate_feature_matrix(features, name="features")
    mean_vector = as_1d_float_array(mean, name="mean")
    std_vector = as_1d_float_array(std, name="std")
    if mean_vector.shape[0] != feature_matrix.shape[1]:
        raise ValueError("mean length must match feature dimension.")
    if std_vector.shape[0] != feature_matrix.shape[1]:
        raise ValueError("std length must match feature dimension.")
    if (std_vector <= 0.0).any():
        raise ValueError("std values must be strictly positive.")
    standardized = np.asarray(
        (feature_matrix - mean_vector[None, :]) / std_vector[None, :],
        dtype=np.float64,
    )
    if not np.isfinite(standardized).all():
        raise ValueError("standardized features contain non-finite values.")
    return standardized.astype(np.float64, copy=False)


def validate_feature_matrix(features: np.ndarray, *, name: str) -> np.ndarray:
    """Return ``features`` as a finite `[sample, feature]` float64 matrix."""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [sample, feature].")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have positive sample and feature dimensions.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def validate_context_batch(contexts: np.ndarray, *, name: str) -> np.ndarray:
    """Return ``contexts`` as a finite `[batch, time]` float64 array."""
    array = np.asarray(contexts, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, time] or [batch, time, 1].")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have positive batch and time dimensions.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def metadata_to_dict(metadata_obj: SignatureFeatureMetadata) -> dict[str, Any]:
    """Convert signature feature metadata to a JSON-serialisable dictionary."""
    return asdict(metadata_obj)
