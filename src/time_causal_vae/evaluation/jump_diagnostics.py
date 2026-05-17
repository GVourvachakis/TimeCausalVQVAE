"""Jump-specific diagnostics for Hawkes-jump benchmark paths."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import torch
from torch import Tensor

from time_causal_vae.evaluation.market_diagnostics import (
    compute_log_returns,
    distribution_summary,
    squeeze_paths,
    wasserstein_1d,
)


def detect_jumps_from_returns(
    paths: Tensor,
    *,
    threshold_multiplier: float = 4.0,
    min_abs_return: float = 0.0,
) -> Tensor:
    """Detect jump steps from robust one-step return outliers.

    The returned tensor has shape ``[batch, length, 1]`` and is aligned to path
    time indices. Index zero is always false because no return ends there.
    """
    if threshold_multiplier <= 0.0:
        raise ValueError("threshold_multiplier must be positive.")
    returns = compute_log_returns(paths)
    median = returns.median()
    mad = (returns - median).abs().median()
    robust_scale = (1.4826 * mad).clamp_min(1e-8)
    threshold = max(float(threshold_multiplier * robust_scale.item()), float(min_abs_return))
    jump_returns = (returns - median).abs() >= threshold
    leading = torch.zeros(
        (jump_returns.shape[0], 1),
        dtype=torch.bool,
        device=jump_returns.device,
    )
    return torch.cat([leading, jump_returns], dim=1).unsqueeze(-1)


def detected_jump_sizes(paths: Tensor, jump_indicators: Tensor) -> Tensor:
    """Return detected jump sizes aligned to path time indices."""
    returns = compute_log_returns(paths)
    indicators = _as_bool_time_tensor(jump_indicators, expected_length=returns.shape[1] + 1)
    sizes = torch.zeros_like(indicators, dtype=returns.dtype)
    sizes[:, 1:] = torch.where(indicators[:, 1:], returns, torch.zeros_like(returns))
    return sizes.unsqueeze(-1)


def jump_count_distribution(
    jump_indicators: Tensor | None = None,
    *,
    jump_counts: Tensor | None = None,
) -> dict[str, Any]:
    """Summarise the per-path jump-count distribution."""
    count_tensor = _count_tensor(jump_indicators=jump_indicators, jump_counts=jump_counts)
    per_path = count_tensor.sum(dim=1).float()
    return {
        "path_count": int(per_path.numel()),
        "total_jumps": int(count_tensor.sum().item()),
        "paths_with_jump_count": int((per_path > 0.0).sum().item()),
        "paths_with_jump_fraction": float((per_path > 0.0).float().mean().item()),
        "per_path": distribution_summary(per_path),
    }


def inter_arrival_distribution(
    jump_indicators: Tensor | None = None,
    *,
    jump_counts: Tensor | None = None,
) -> dict[str, Any]:
    """Summarise distances between consecutive jump steps within each path."""
    indicators = _indicator_tensor(jump_indicators=jump_indicators, jump_counts=jump_counts)
    gaps: list[float] = []
    paths_with_multiple_jumps = 0
    for path_indicators in indicators:
        event_positions = torch.nonzero(path_indicators, as_tuple=False).flatten().float()
        if event_positions.numel() > 1:
            paths_with_multiple_jumps += 1
            gaps.extend(event_positions.diff().tolist())
    gap_tensor = torch.tensor(gaps, dtype=torch.float32)
    return {
        "gap_count": int(gap_tensor.numel()),
        "paths_with_multiple_jumps": paths_with_multiple_jumps,
        "gaps": distribution_summary(gap_tensor),
    }


def jump_size_distribution(jump_sizes: Tensor | None) -> dict[str, Any]:
    """Summarise aggregate jump sizes when oracle or detected sizes are available."""
    if jump_sizes is None:
        return {"available": False}
    sizes = _as_float_time_tensor(jump_sizes)
    nonzero = sizes[sizes.abs() > 0.0]
    positive = nonzero[nonzero > 0.0]
    negative = nonzero[nonzero < 0.0]
    return {
        "available": True,
        "jump_step_count": int(nonzero.numel()),
        "positive_jump_step_count": int(positive.numel()),
        "negative_jump_step_count": int(negative.numel()),
        "negative_jump_fraction": float(
            (negative.numel() / nonzero.numel()) if nonzero.numel() else 0.0
        ),
        "all": distribution_summary(nonzero),
        "positive": distribution_summary(positive),
        "negative": distribution_summary(negative),
        "absolute": distribution_summary(nonzero.abs()),
    }


def jump_clustering_score(
    jump_indicators: Tensor | None = None,
    *,
    jump_counts: Tensor | None = None,
) -> dict[str, float | int]:
    """Measure simple same-path jump clustering and count over-dispersion."""
    indicators = _indicator_tensor(jump_indicators=jump_indicators, jump_counts=jump_counts)
    counts = _count_tensor(jump_indicators=jump_indicators, jump_counts=jump_counts).float()
    adjacent_pairs = (indicators[:, 1:] & indicators[:, :-1]).sum(dim=1).float()
    jump_steps = indicators.sum(dim=1).float()
    jump_counts_per_path = counts.sum(dim=1)
    total_jump_steps = float(jump_steps.sum().item())
    mean_count = jump_counts_per_path.mean()
    variance_count = jump_counts_per_path.var(unbiased=False)
    same_step_multi_jump_fraction = float((counts > 1.0).float().mean().item())
    return {
        "adjacent_jump_pair_count": int(adjacent_pairs.sum().item()),
        "paths_with_adjacent_jumps": int((adjacent_pairs > 0.0).sum().item()),
        "paths_with_adjacent_jump_fraction": float((adjacent_pairs > 0.0).float().mean().item()),
        "adjacent_pair_per_jump_step": float(
            adjacent_pairs.sum().item() / max(total_jump_steps, 1.0)
        ),
        "same_step_multi_jump_fraction": same_step_multi_jump_fraction,
        "count_overdispersion": float((variance_count / mean_count.clamp_min(1e-8)).item()),
    }


def return_tail_thresholds(
    paths: Tensor,
    *,
    levels: Iterable[float] = (0.001, 0.01, 0.99, 0.999),
) -> dict[str, float]:
    """Estimate return-tail thresholds from a reference path sample."""
    returns = compute_log_returns(paths).flatten()
    thresholds: dict[str, float] = {}
    for level in levels:
        _validate_probability(level, name="tail level")
        thresholds[_threshold_key(level)] = float(torch.quantile(returns, level).item())
    return thresholds


def tail_exceedance(
    paths: Tensor,
    *,
    thresholds: Mapping[str, float],
) -> dict[str, float | int]:
    """Count return exceedances beyond named lower and upper thresholds."""
    returns = compute_log_returns(paths).flatten()
    result: dict[str, float | int] = {"return_count": int(returns.numel())}
    for name, threshold in thresholds.items():
        threshold_value = float(threshold)
        if _is_lower_threshold_name(name):
            mask = returns <= threshold_value
            prefix = "below"
        else:
            mask = returns >= threshold_value
            prefix = "above"
        count = int(mask.sum().item())
        result[f"{prefix}_{name}_count"] = count
        result[f"{prefix}_{name}_fraction"] = float(mask.float().mean().item())
    return result


def value_at_risk(
    values: Tensor,
    *,
    level: float = 0.01,
    tail: Literal["left", "right"] = "left",
) -> float:
    """Return lower- or upper-tail empirical value at risk for raw returns."""
    _validate_probability(level, name="level")
    flattened = values.detach().float().reshape(-1)
    if flattened.numel() == 0:
        return 0.0
    quantile = level if tail == "left" else 1.0 - level
    return float(torch.quantile(flattened, quantile).item())


def expected_shortfall(
    values: Tensor,
    *,
    level: float = 0.01,
    tail: Literal["left", "right"] = "left",
) -> float:
    """Return empirical expected shortfall beyond the selected VaR threshold."""
    var = value_at_risk(values, level=level, tail=tail)
    flattened = values.detach().float().reshape(-1)
    if flattened.numel() == 0:
        return 0.0
    tail_values = flattened[flattened <= var] if tail == "left" else flattened[flattened >= var]
    if tail_values.numel() == 0:
        return var
    return float(tail_values.mean().item())


def var_es_summary(
    paths: Tensor,
    *,
    levels: Iterable[float] = (0.01, 0.05),
) -> dict[str, float]:
    """Return lower-tail VaR and ES for one-step log returns."""
    returns = compute_log_returns(paths).flatten()
    summary: dict[str, float] = {}
    for level in levels:
        key = _threshold_key(level)
        summary[f"lower_tail_var_{key}"] = value_at_risk(returns, level=level, tail="left")
        summary[f"lower_tail_es_{key}"] = expected_shortfall(returns, level=level, tail="left")
    return summary


def jump_diagnostic_summary(
    paths: Tensor,
    *,
    jump_indicators: Tensor | None = None,
    jump_counts: Tensor | None = None,
    jump_sizes: Tensor | None = None,
    tail_reference_paths: Tensor | None = None,
) -> dict[str, Any]:
    """Return a combined jump, tail, and risk diagnostic summary."""
    if jump_indicators is None and jump_counts is None:
        jump_indicators = detect_jumps_from_returns(paths)
    if jump_sizes is None:
        indicator_for_sizes = _indicator_tensor(
            jump_indicators=jump_indicators,
            jump_counts=jump_counts,
        )
        jump_sizes = detected_jump_sizes(paths, indicator_for_sizes)
    reference_paths = paths if tail_reference_paths is None else tail_reference_paths
    thresholds = return_tail_thresholds(reference_paths)
    return {
        "path_shape": list(squeeze_paths(paths).shape),
        "jump_counts": jump_count_distribution(
            jump_indicators=jump_indicators,
            jump_counts=jump_counts,
        ),
        "inter_arrivals": inter_arrival_distribution(
            jump_indicators=jump_indicators,
            jump_counts=jump_counts,
        ),
        "jump_sizes": jump_size_distribution(jump_sizes),
        "clustering": jump_clustering_score(
            jump_indicators=jump_indicators,
            jump_counts=jump_counts,
        ),
        "tail_thresholds": thresholds,
        "tail_exceedance": tail_exceedance(paths, thresholds=thresholds),
        "var_es": var_es_summary(paths),
    }


def jump_count_wasserstein(
    first_counts: Tensor,
    second_counts: Tensor,
) -> float:
    """Return W1 distance between per-path jump-count distributions."""
    first = _as_float_time_tensor(first_counts).sum(dim=1)
    second = _as_float_time_tensor(second_counts).sum(dim=1)
    return wasserstein_1d(first, second)


def _indicator_tensor(
    jump_indicators: Tensor | None = None,
    *,
    jump_counts: Tensor | None = None,
) -> Tensor:
    if jump_counts is not None:
        return _as_float_time_tensor(jump_counts) > 0.0
    if jump_indicators is None:
        raise ValueError("jump_indicators or jump_counts must be provided.")
    return _as_bool_time_tensor(jump_indicators)


def _count_tensor(
    jump_indicators: Tensor | None = None,
    *,
    jump_counts: Tensor | None = None,
) -> Tensor:
    if jump_counts is not None:
        return _as_float_time_tensor(jump_counts)
    if jump_indicators is None:
        raise ValueError("jump_indicators or jump_counts must be provided.")
    return _as_bool_time_tensor(jump_indicators).float()


def _as_bool_time_tensor(values: Tensor, *, expected_length: int | None = None) -> Tensor:
    if values.ndim == 3 and values.shape[-1] == 1:
        squeezed = values[..., 0]
    elif values.ndim == 2:
        squeezed = values
    else:
        raise ValueError(f"Expected [batch, length] or [batch, length, 1]; got {values.shape}.")
    if expected_length is not None and squeezed.shape[1] != expected_length:
        raise ValueError(f"Expected length {expected_length}; got {squeezed.shape[1]}.")
    return squeezed.detach().bool().cpu()


def _as_float_time_tensor(values: Tensor) -> Tensor:
    if values.ndim == 3 and values.shape[-1] == 1:
        squeezed = values[..., 0]
    elif values.ndim == 2:
        squeezed = values
    else:
        raise ValueError(f"Expected [batch, length] or [batch, length, 1]; got {values.shape}.")
    return squeezed.detach().float().cpu()


def _validate_probability(value: float, *, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1).")


def _threshold_key(level: float) -> str:
    text = f"{level:.4f}".split(".")[1].rstrip("0")
    return "q" + text


def _is_lower_threshold_name(name: str) -> bool:
    digits = "".join(character for character in name if character.isdigit())
    if not digits:
        return True
    try:
        numeric = float("0." + digits)
    except ValueError:
        return True
    return numeric < 0.5
