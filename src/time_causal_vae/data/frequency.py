"""Causal low/high-frequency decomposition utilities."""

from __future__ import annotations

import math
from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor

Alpha: TypeAlias = float | int
FrequencyComponent: TypeAlias = Literal["low", "high"]


def causal_ema(path: Tensor, alpha: Alpha) -> Tensor:
    """Return the causal exponential moving average of a path tensor.

    The time dimension is dimension 0 for ``[time]`` tensors and dimension 1
    for batched ``[batch, time]`` or ``[batch, time, channels]`` tensors.
    """
    smoothing = _validate_alpha(alpha)
    _validate_path(path)
    time_dim = _time_dim(path)
    length = path.shape[time_dim]
    if length == 0:
        raise ValueError("path time dimension must be non-empty.")

    low = torch.empty_like(path)
    _set_time_slice(low, time_dim, 0, _get_time_slice(path, time_dim, 0))
    for step in range(1, length):
        previous = _get_time_slice(low, time_dim, step - 1)
        current = _get_time_slice(path, time_dim, step)
        _set_time_slice(
            low,
            time_dim,
            step,
            smoothing * current + (1.0 - smoothing) * previous,
        )
    return low


def causal_ema_decompose(path: Tensor, alpha: Alpha) -> tuple[Tensor, Tensor]:
    """Return causal EMA low-frequency and residual high-frequency components."""
    low = causal_ema(path, alpha)
    high = path - low
    return low, high


def compose_low_high(low: Tensor, high: Tensor) -> Tensor:
    """Recompose a path from low-frequency and high-frequency components."""
    if low.shape != high.shape:
        raise ValueError(f"low and high shapes must match; got {low.shape} and {high.shape}.")
    return low + high


def causal_ema_frequency_channels(path: Tensor, alpha: Alpha) -> Tensor:
    """Return two-channel ``[low, high]`` EMA decomposition for one-channel paths."""
    if path.ndim != 3 or path.shape[-1] != 1:
        raise ValueError(
            f"frequency decomposition expects [batch, time, 1] paths; got {tuple(path.shape)}."
        )
    low, high = causal_ema_decompose(path, alpha)
    return torch.cat([low, high], dim=-1)


def causal_ema_frequency_component(
    path: Tensor,
    alpha: Alpha,
    component: FrequencyComponent,
) -> Tensor:
    """Return one causal EMA frequency component for one-channel paths."""
    channels = causal_ema_frequency_channels(path, alpha)
    low, high = split_low_high_channels(channels)
    if component == "low":
        return low
    if component == "high":
        return high
    raise ValueError(f"frequency component must be 'low' or 'high'; got {component!r}.")


def causal_ema_frequency_transform(
    path: Tensor,
    alpha: Alpha,
    component: str | None = None,
) -> Tensor:
    """Return joint or component-specific causal EMA frequency data.

    When ``component`` is ``None``, this preserves the existing joint-tokenizer
    behaviour and returns ``[low, high]`` channels. When it is ``low`` or
    ``high``, the returned tensor keeps the one-channel ``[batch, time, 1]``
    shape required by separate tokenizers.
    """
    selected = normalise_frequency_component(component)
    if selected is None:
        return causal_ema_frequency_channels(path, alpha)
    return causal_ema_frequency_component(path, alpha, selected)


def normalise_frequency_component(value: object) -> FrequencyComponent | None:
    """Normalise an optional frequency component config value."""
    if value is None:
        return None
    text = str(value).lower()
    if text in {"none", "null", "both", "joint", "all"}:
        return None
    if text in {"low", "high"}:
        return cast(FrequencyComponent, text)
    raise ValueError(f"data.frequency_component must be null, 'low', or 'high'; got {value!r}.")


def split_low_high_channels(channels: Tensor) -> tuple[Tensor, Tensor]:
    """Split a two-channel frequency tensor into low and high components."""
    if channels.ndim != 3 or channels.shape[-1] != 2:
        raise ValueError(
            f"frequency channels must have shape [batch, time, 2]; got {tuple(channels.shape)}."
        )
    return channels[..., 0:1], channels[..., 1:2]


def compose_frequency_channels(channels: Tensor) -> Tensor:
    """Compose two-channel frequency paths into one-channel path space."""
    low, high = split_low_high_channels(channels)
    return compose_low_high(low, high)


def _validate_alpha(alpha: Alpha) -> float:
    """Validate and normalise the EMA smoothing parameter."""
    value = float(alpha)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1.")
    return value


def _validate_path(path: Tensor) -> None:
    """Validate path tensor rank and dtype."""
    if not isinstance(path, Tensor):
        raise TypeError("path must be a torch.Tensor.")
    if path.ndim not in {1, 2, 3}:
        raise ValueError(
            "path must have shape [time], [batch, time], or [batch, time, channels]; "
            f"got {tuple(path.shape)}."
        )
    if not path.is_floating_point():
        raise TypeError("path must be a floating-point tensor to preserve EMA dtype.")


def _time_dim(path: Tensor) -> int:
    """Return the time dimension for supported path layouts."""
    return 0 if path.ndim == 1 else 1


def _get_time_slice(path: Tensor, time_dim: int, step: int) -> Tensor:
    """Return one time slice from a supported path tensor."""
    return path.select(time_dim, step)


def _set_time_slice(path: Tensor, time_dim: int, step: int, value: Tensor) -> None:
    """Set one time slice on a supported path tensor."""
    path.select(time_dim, step).copy_(value)
