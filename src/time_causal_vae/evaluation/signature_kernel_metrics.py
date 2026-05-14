"""Optional signature-kernel path metrics.

This module uses :mod:`sigkernel` only when callers request signature-kernel
evaluation. The package is intentionally not a hard project dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

import numpy as np
import torch
from torch import Tensor

SIGKERNEL_INSTALL_HINT = (
    "Optional signature-kernel metrics require 'sigkernel'. It is not installed by default. "
    "In a temporary or opt-in environment, try: pip install Cython && "
    "pip install 'git+https://github.com/crispitagorico/sigkernel.git' --no-build-isolation"
)


class OptionalDependencyError(ImportError):
    """Raised when an optional signature-kernel dependency is unavailable."""


@dataclass(frozen=True)
class SignatureKernelConfig:
    """Configuration for one signature-kernel metric pass."""

    dyadic_order: int = 1
    rbf_sigma: float = 1.0
    include_time: bool = False
    use_lead_lag: bool = False
    max_batch: int = 100


@dataclass(frozen=True)
class GramChecks:
    """Numerical checks for a Gram matrix."""

    finite: bool
    symmetric: bool | None
    positive_diagonal: bool | None
    shape: tuple[int, int]


@dataclass(frozen=True)
class SignatureKernelResult:
    """Result of one signature-kernel MMD computation."""

    mmd_biased: float
    kxx_checks: GramChecks
    kyy_checks: GramChecks
    kxy_checks: GramChecks
    preprocessing: dict[str, bool | list[str]]
    config: dict[str, int | float | bool]
    package: str
    package_version: str


def load_sigkernel() -> Any:
    """Import ``sigkernel`` or raise a clear optional-dependency error."""
    try:
        import sigkernel
    except ImportError as exc:
        raise OptionalDependencyError(SIGKERNEL_INSTALL_HINT) from exc
    return sigkernel


def sigkernel_version() -> str:
    """Return the installed ``sigkernel`` distribution version when available."""
    try:
        return metadata.version("sigkernel")
    except metadata.PackageNotFoundError:
        return "unknown"


def validate_price_batch(paths: np.ndarray | Tensor, *, name: str) -> np.ndarray:
    """Return paths as a finite float64 ``[batch, time]`` price array."""
    array = paths.detach().cpu().numpy() if isinstance(paths, Tensor) else np.asarray(paths)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, time] or [batch, time, 1].")
    if array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError(f"{name} must have positive batch size and at least two time steps.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    if (array <= 0.0).any():
        raise ValueError(f"{name} must contain strictly positive price values.")
    return array


def time_channel(length: int) -> np.ndarray:
    """Return a ``[0, 1]`` time channel."""
    if length <= 0:
        raise ValueError("length must be positive.")
    if length == 1:
        return np.zeros((1, 1), dtype=np.float64)
    return np.linspace(0.0, 1.0, num=length, dtype=np.float64)[:, None]


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Apply a lead-lag transform to a two-dimensional path."""
    path_2d = validate_path(path, name="path")
    repeated = np.repeat(path_2d, repeats=2, axis=0)
    return np.concatenate([repeated[:-1], repeated[1:]], axis=1)


def validate_path(path: np.ndarray, *, name: str) -> np.ndarray:
    """Return a finite two-dimensional float64 path."""
    array = np.asarray(path, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [time, channels].")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have positive time and channel dimensions.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def preprocess_price_paths(
    paths: np.ndarray | Tensor,
    *,
    include_time: bool = False,
    use_lead_lag: bool = False,
) -> tuple[Tensor, list[str]]:
    """Build signature-kernel input paths from normalised price paths."""
    prices = validate_price_batch(paths, name="paths")
    processed_paths: list[np.ndarray] = []
    channel_names = ["normalised_price", "log_return", "cumulative_log_return"]
    if include_time:
        channel_names.append("time")

    for price_path in prices:
        normalised = price_path / price_path[0]
        log_prices = np.log(normalised)
        log_returns = np.diff(log_prices, prepend=log_prices[0])
        cumulative_returns = np.log(normalised / normalised[0])
        channels = [
            normalised[:, None],
            log_returns[:, None],
            cumulative_returns[:, None],
        ]
        if include_time:
            channels.append(time_channel(len(normalised)))
        path = validate_path(np.concatenate(channels, axis=1), name="preprocessed_path")
        if use_lead_lag:
            path = lead_lag_transform(path)
        processed_paths.append(path)

    if use_lead_lag:
        channel_names = [f"{name}_lead" for name in channel_names] + [
            f"{name}_lag" for name in channel_names
        ]
    stacked = np.stack(processed_paths, axis=0)
    return torch.as_tensor(stacked, dtype=torch.float64), channel_names


def compute_signature_kernel_gram(
    x_paths: Tensor,
    y_paths: Tensor,
    *,
    config: SignatureKernelConfig,
    symmetric: bool = False,
) -> Tensor:
    """Compute a signature-kernel Gram matrix using ``sigkernel``."""
    if config.dyadic_order < 0:
        raise ValueError("dyadic_order must be non-negative.")
    if config.rbf_sigma <= 0.0:
        raise ValueError("rbf_sigma must be positive.")
    if config.max_batch <= 0:
        raise ValueError("max_batch must be positive.")
    sigkernel = load_sigkernel()
    static_kernel = sigkernel.RBFKernel(sigma=config.rbf_sigma)
    kernel = sigkernel.SigKernel(static_kernel, config.dyadic_order)
    gram = kernel.compute_Gram(x_paths, y_paths, sym=symmetric, max_batch=config.max_batch)
    if not isinstance(gram, Tensor):
        gram = torch.as_tensor(gram, dtype=torch.float64)
    return gram.detach().cpu().to(dtype=torch.float64)


def gram_checks(gram: Tensor, *, symmetric: bool) -> GramChecks:
    """Return finite, symmetry, and positive-diagonal checks for ``gram``."""
    if gram.ndim != 2:
        raise ValueError("gram must be a matrix.")
    finite = bool(torch.isfinite(gram).all().item())
    symmetric_value = None
    positive_diagonal = None
    if symmetric:
        symmetric_value = bool(torch.allclose(gram, gram.T, rtol=1e-5, atol=1e-7))
        positive_diagonal = bool((torch.diag(gram) > 0.0).all().item())
    return GramChecks(
        finite=finite,
        symmetric=symmetric_value,
        positive_diagonal=positive_diagonal,
        shape=(int(gram.shape[0]), int(gram.shape[1])),
    )


def compute_signature_kernel_mmd(
    real_paths: np.ndarray | Tensor,
    generated_paths: np.ndarray | Tensor,
    *,
    config: SignatureKernelConfig,
) -> SignatureKernelResult:
    """Compute biased signature-kernel MMD for two path batches."""
    real_tensor, channel_names = preprocess_price_paths(
        real_paths,
        include_time=config.include_time,
        use_lead_lag=config.use_lead_lag,
    )
    generated_tensor, generated_channel_names = preprocess_price_paths(
        generated_paths,
        include_time=config.include_time,
        use_lead_lag=config.use_lead_lag,
    )
    if generated_channel_names != channel_names:
        raise ValueError("real and generated preprocessing channels differ.")

    kxx = compute_signature_kernel_gram(real_tensor, real_tensor, config=config, symmetric=True)
    kyy = compute_signature_kernel_gram(
        generated_tensor,
        generated_tensor,
        config=config,
        symmetric=True,
    )
    kxy = compute_signature_kernel_gram(real_tensor, generated_tensor, config=config)
    mmd = torch.mean(kxx) + torch.mean(kyy) - 2.0 * torch.mean(kxy)
    mmd_value = float(mmd.item())
    if not np.isfinite(mmd_value):
        raise ValueError("signature-kernel MMD is not finite.")

    return SignatureKernelResult(
        mmd_biased=mmd_value,
        kxx_checks=gram_checks(kxx, symmetric=True),
        kyy_checks=gram_checks(kyy, symmetric=True),
        kxy_checks=gram_checks(kxy, symmetric=False),
        preprocessing={
            "channels": channel_names,
            "include_time": config.include_time,
            "use_lead_lag": config.use_lead_lag,
        },
        config=asdict(config),
        package="sigkernel",
        package_version=sigkernel_version(),
    )


def result_to_dict(result: SignatureKernelResult) -> dict[str, Any]:
    """Convert a signature-kernel result to a JSON-serialisable dictionary."""
    return asdict(result)
