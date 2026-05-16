"""Market-style diagnostics for generated one-dimensional financial paths.

References
----------
    [tcvae_2024], [deepvol_2022], [aotnumerics] in README.md.
Borrowed idea:
    Compare generated paths through return, volatility, autocorrelation, and downstream-risk views.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor

DEFAULT_AUTOCORRELATION_LAGS = (1, 2, 5, 10, 20)


def compute_log_returns(paths: Tensor) -> Tensor:
    """Return one-step log returns when paths are positive, else increments.

    The S&P500/VIX normalised windows are expected to be positive. The fallback
    keeps diagnostics usable for synthetic or transformed data that may include
    non-positive values.
    """
    paths_2d = squeeze_paths(paths)
    if bool((paths_2d > 0.0).all()):
        return paths_2d.clamp_min(1e-8).log().diff(dim=1)
    return paths_2d.diff(dim=1)


def terminal_returns(paths: Tensor) -> Tensor:
    """Return terminal simple returns for each path."""
    paths_2d = squeeze_paths(paths)
    denominator = paths_2d[:, 0].abs().clamp_min(1e-8)
    return (paths_2d[:, -1] - paths_2d[:, 0]) / denominator


def volatility_per_path(paths: Tensor) -> Tensor:
    """Return per-path realised volatility from one-step returns."""
    returns = compute_log_returns(paths)
    return returns.std(dim=1, unbiased=False)


def return_autocorrelation(
    paths: Tensor,
    *,
    lags: Iterable[int] = DEFAULT_AUTOCORRELATION_LAGS,
    squared: bool = False,
) -> dict[str, float]:
    """Compute within-path return autocorrelation for selected lags."""
    return return_autocorrelation_within_path(paths, lags=lags, squared=squared)


def return_autocorrelation_flattened(
    paths: Tensor,
    *,
    lags: Iterable[int] = DEFAULT_AUTOCORRELATION_LAGS,
    squared: bool = False,
) -> dict[str, float]:
    """Compute flattened autocorrelation, connecting adjacent path windows.

    This diagnostic is retained for backwards comparison only. Paper-style
    metrics should prefer :func:`return_autocorrelation_within_path`, which
    does not connect the end of one generated path to the start of the next.
    """
    returns = compute_log_returns(paths).flatten()
    if squared:
        returns = returns.square()
    centred = returns - returns.mean()
    variance = centred.square().mean().clamp_min(1e-12)
    values: dict[str, float] = {}
    for lag in lags:
        if lag <= 0:
            raise ValueError("autocorrelation lags must be positive.")
        if lag >= centred.numel():
            values[str(lag)] = 0.0
            continue
        autocovariance = (centred[:-lag] * centred[lag:]).mean()
        values[str(lag)] = float((autocovariance / variance).detach().cpu())
    return values


def return_autocorrelation_within_path(
    paths: Tensor,
    *,
    lags: Iterable[int] = DEFAULT_AUTOCORRELATION_LAGS,
    squared: bool = False,
) -> dict[str, float]:
    """Compute autocorrelation per path, then average across paths."""
    returns = compute_log_returns(paths)
    if squared:
        returns = returns.square()
    centred = returns - returns.mean(dim=1, keepdim=True)
    variance = centred.square().mean(dim=1).clamp_min(1e-12)
    values: dict[str, float] = {}
    path_length = centred.shape[1]
    for lag in lags:
        if lag <= 0:
            raise ValueError("autocorrelation lags must be positive.")
        if lag >= path_length:
            values[str(lag)] = 0.0
            continue
        autocovariance = (centred[:, :-lag] * centred[:, lag:]).mean(dim=1)
        values[str(lag)] = float((autocovariance / variance).mean().detach().cpu())
    return values


def max_abs_return_per_path(paths: Tensor) -> Tensor:
    """Return the maximum absolute one-step return for each path."""
    return compute_log_returns(paths).abs().max(dim=1).values


def max_rolling_volatility_per_path(paths: Tensor, *, window: int = 5) -> Tensor:
    """Return the maximum rolling realised volatility for each path."""
    if window <= 1:
        raise ValueError("rolling volatility window must be greater than one.")
    returns = compute_log_returns(paths)
    if returns.shape[1] < window:
        return returns.std(dim=1, unbiased=False)
    windows = returns.unfold(dimension=1, size=window, step=1)
    return windows.std(dim=2, unbiased=False).max(dim=1).values


def decoded_path_bounds(paths: Tensor) -> dict[str, float | int]:
    """Return decoded path min/max and non-positive path counts."""
    paths_2d = squeeze_paths(paths)
    non_positive_mask = (paths_2d <= 0.0).any(dim=1)
    return {
        "path_min": float(paths_2d.min().detach().cpu()),
        "path_max": float(paths_2d.max().detach().cpu()),
        "non_positive_path_count": int(non_positive_mask.sum().detach().cpu()),
        "non_positive_path_fraction": float(non_positive_mask.float().mean().detach().cpu()),
    }


def return_tail_thresholds(real_paths: Tensor) -> dict[str, float]:
    """Return real-data one-step return tail thresholds."""
    real_returns = compute_log_returns(real_paths).flatten()
    quantiles = torch.quantile(
        real_returns,
        torch.tensor([0.001, 0.01, 0.99, 0.999], dtype=real_returns.dtype),
    )
    return {
        "q001": float(quantiles[0].detach().cpu()),
        "q01": float(quantiles[1].detach().cpu()),
        "q99": float(quantiles[2].detach().cpu()),
        "q999": float(quantiles[3].detach().cpu()),
    }


def tail_exceedance_rates(
    paths: Tensor,
    *,
    thresholds: Mapping[str, float],
) -> dict[str, float | int]:
    """Count returns beyond thresholds estimated from real data."""
    returns = compute_log_returns(paths).flatten()
    count = returns.numel()
    below_q001 = returns < float(thresholds["q001"])
    below_q01 = returns < float(thresholds["q01"])
    above_q99 = returns > float(thresholds["q99"])
    above_q999 = returns > float(thresholds["q999"])
    return {
        "return_count": int(count),
        "below_real_q001_count": int(below_q001.sum().detach().cpu()),
        "below_real_q001_fraction": float(below_q001.float().mean().detach().cpu()),
        "below_real_q01_count": int(below_q01.sum().detach().cpu()),
        "below_real_q01_fraction": float(below_q01.float().mean().detach().cpu()),
        "above_real_q99_count": int(above_q99.sum().detach().cpu()),
        "above_real_q99_fraction": float(above_q99.float().mean().detach().cpu()),
        "above_real_q999_count": int(above_q999.sum().detach().cpu()),
        "above_real_q999_fraction": float(above_q999.float().mean().detach().cpu()),
    }


def outlier_path_metadata(
    paths: Tensor,
    *,
    top_k: int = 10,
    rolling_window: int = 5,
) -> dict[str, Any]:
    """Summarise extreme paths by return and rolling-volatility criteria."""
    paths_2d = squeeze_paths(paths)
    max_abs_returns = max_abs_return_per_path(paths)
    max_volatility = max_rolling_volatility_per_path(paths, window=rolling_window)
    terminal = terminal_returns(paths)
    realised_volatility = volatility_per_path(paths)
    k = min(top_k, paths_2d.shape[0])
    return {
        "path_shape": list(paths.shape),
        "bounds": decoded_path_bounds(paths),
        "max_abs_return": distribution_summary(max_abs_returns),
        "max_rolling_volatility": distribution_summary(max_volatility),
        "top_by_max_abs_return": _top_path_records(
            paths_2d=paths_2d,
            scores=max_abs_returns,
            terminal=terminal,
            volatility=realised_volatility,
            k=k,
            score_name="max_abs_return",
        ),
        "top_by_max_rolling_volatility": _top_path_records(
            paths_2d=paths_2d,
            scores=max_volatility,
            terminal=terminal,
            volatility=realised_volatility,
            k=k,
            score_name="max_rolling_volatility",
        ),
    }


def _top_path_records(
    *,
    paths_2d: Tensor,
    scores: Tensor,
    terminal: Tensor,
    volatility: Tensor,
    k: int,
    score_name: str,
) -> list[dict[str, float | int]]:
    """Return compact metadata for the top-scoring paths."""
    indices = torch.topk(scores, k=k).indices.detach().cpu()
    records: list[dict[str, float | int]] = []
    for rank, index_tensor in enumerate(indices):
        index = int(index_tensor.item())
        path = paths_2d[index]
        records.append(
            {
                "rank": rank + 1,
                "path_index": index,
                score_name: float(scores[index].detach().cpu()),
                "terminal_return": float(terminal[index].detach().cpu()),
                "volatility": float(volatility[index].detach().cpu()),
                "path_min": float(path.min().detach().cpu()),
                "path_max": float(path.max().detach().cpu()),
            }
        )
    return records


def skewness(values: Tensor) -> float:
    """Return population skewness for a tensor flattened to one dimension."""
    flattened = values.detach().float().reshape(-1)
    if flattened.numel() == 0:
        return 0.0
    centred = flattened - flattened.mean()
    std = flattened.std(unbiased=False).clamp_min(1e-12)
    return float((centred / std).pow(3).mean().detach().cpu())


def kurtosis(values: Tensor) -> float:
    """Return population excess kurtosis for a tensor flattened to one dimension."""
    flattened = values.detach().float().reshape(-1)
    if flattened.numel() == 0:
        return 0.0
    centred = flattened - flattened.mean()
    std = flattened.std(unbiased=False).clamp_min(1e-12)
    return float(((centred / std).pow(4).mean() - 3.0).detach().cpu())


def maximum_drawdown(paths: Tensor) -> Tensor:
    """Return maximum drawdown for each path."""
    paths_2d = squeeze_paths(paths)
    running_peak = torch.cummax(paths_2d, dim=1).values.abs().clamp_min(1e-8)
    drawdown = (running_peak - paths_2d) / running_peak
    return drawdown.max(dim=1).values


def wasserstein_1d(first: Tensor, second: Tensor) -> float:
    """Return equal-weight one-dimensional Wasserstein distance."""
    first_flat = first.detach().float().reshape(-1)
    second_flat = second.detach().float().reshape(-1)
    if first_flat.numel() == 0 or second_flat.numel() == 0:
        return 0.0
    n_values = min(first_flat.numel(), second_flat.numel())
    return float(
        (first_flat.sort().values[:n_values] - second_flat.sort().values[:n_values])
        .abs()
        .mean()
        .detach()
        .cpu()
    )


def distribution_summary(values: Tensor) -> dict[str, float]:
    """Summarise a one-dimensional diagnostic distribution."""
    flattened = values.detach().float().reshape(-1)
    if flattened.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "q01": 0.0,
            "q05": 0.0,
            "q50": 0.0,
            "q95": 0.0,
            "q99": 0.0,
        }
    quantiles = torch.quantile(
        flattened,
        torch.tensor([0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999], dtype=flattened.dtype),
    )
    return {
        "mean": float(flattened.mean().detach().cpu()),
        "std": float(flattened.std(unbiased=False).detach().cpu()),
        "skewness": skewness(flattened),
        "excess_kurtosis": kurtosis(flattened),
        "q001": float(quantiles[0].detach().cpu()),
        "q01": float(quantiles[1].detach().cpu()),
        "q05": float(quantiles[2].detach().cpu()),
        "q50": float(quantiles[3].detach().cpu()),
        "q95": float(quantiles[4].detach().cpu()),
        "q99": float(quantiles[5].detach().cpu()),
        "q999": float(quantiles[6].detach().cpu()),
    }


def market_style_summary(
    paths: Tensor,
    *,
    lags: Iterable[int] = DEFAULT_AUTOCORRELATION_LAGS,
) -> dict[str, Any]:
    """Compute market-style summaries for a batch of paths."""
    returns = compute_log_returns(paths)
    terminal = terminal_returns(paths)
    volatility = volatility_per_path(paths)
    drawdowns = maximum_drawdown(paths)
    autocorr = return_autocorrelation_within_path(paths, lags=lags)
    squared_autocorr = return_autocorrelation_within_path(paths, lags=lags, squared=True)
    flattened_autocorr = return_autocorrelation_flattened(paths, lags=lags)
    flattened_squared_autocorr = return_autocorrelation_flattened(
        paths,
        lags=lags,
        squared=True,
    )
    return {
        "path_shape": list(paths.shape),
        "return_mode": "log_return" if bool((squeeze_paths(paths) > 0.0).all()) else "increment",
        "returns": distribution_summary(returns),
        "terminal_returns": distribution_summary(terminal),
        "volatility": distribution_summary(volatility),
        "maximum_drawdown": distribution_summary(drawdowns),
        "max_abs_return_per_path": distribution_summary(max_abs_return_per_path(paths)),
        "max_rolling_volatility_per_path": distribution_summary(
            max_rolling_volatility_per_path(paths)
        ),
        "decoded_path_bounds": decoded_path_bounds(paths),
        "return_autocorrelation": autocorr,
        "squared_return_autocorrelation": squared_autocorr,
        "return_autocorrelation_within_path": autocorr,
        "squared_return_autocorrelation_within_path": squared_autocorr,
        "return_autocorrelation_flattened": flattened_autocorr,
        "squared_return_autocorrelation_flattened": flattened_squared_autocorr,
    }


def compare_market_summaries(
    *,
    real_paths: Tensor,
    generated_paths: Tensor,
    lags: Iterable[int] = DEFAULT_AUTOCORRELATION_LAGS,
) -> dict[str, Any]:
    """Compare generated paths against real paths with market diagnostics."""
    real_returns = compute_log_returns(real_paths)
    generated_returns = compute_log_returns(generated_paths)
    real_terminal = terminal_returns(real_paths)
    generated_terminal = terminal_returns(generated_paths)
    real_volatility = volatility_per_path(real_paths)
    generated_volatility = volatility_per_path(generated_paths)
    real_drawdown = maximum_drawdown(real_paths)
    generated_drawdown = maximum_drawdown(generated_paths)
    real_autocorr = return_autocorrelation_within_path(real_paths, lags=lags)
    generated_autocorr = return_autocorrelation_within_path(generated_paths, lags=lags)
    real_squared_autocorr = return_autocorrelation_within_path(real_paths, lags=lags, squared=True)
    generated_squared_autocorr = return_autocorrelation_within_path(
        generated_paths,
        lags=lags,
        squared=True,
    )
    real_flattened_autocorr = return_autocorrelation_flattened(real_paths, lags=lags)
    generated_flattened_autocorr = return_autocorrelation_flattened(
        generated_paths,
        lags=lags,
    )
    real_flattened_squared_autocorr = return_autocorrelation_flattened(
        real_paths,
        lags=lags,
        squared=True,
    )
    generated_flattened_squared_autocorr = return_autocorrelation_flattened(
        generated_paths,
        lags=lags,
        squared=True,
    )
    thresholds = return_tail_thresholds(real_paths)
    return {
        "summary": market_style_summary(generated_paths, lags=lags),
        "returns_wasserstein": wasserstein_1d(real_returns, generated_returns),
        "terminal_return_wasserstein": wasserstein_1d(real_terminal, generated_terminal),
        "volatility_wasserstein": wasserstein_1d(real_volatility, generated_volatility),
        "maximum_drawdown_wasserstein": wasserstein_1d(real_drawdown, generated_drawdown),
        "return_autocorrelation_l1": autocorrelation_l1(real_autocorr, generated_autocorr),
        "squared_return_autocorrelation_l1": autocorrelation_l1(
            real_squared_autocorr,
            generated_squared_autocorr,
        ),
        "return_autocorrelation_within_path_l1": autocorrelation_l1(
            real_autocorr,
            generated_autocorr,
        ),
        "squared_return_autocorrelation_within_path_l1": autocorrelation_l1(
            real_squared_autocorr,
            generated_squared_autocorr,
        ),
        "return_autocorrelation_flattened_l1": autocorrelation_l1(
            real_flattened_autocorr,
            generated_flattened_autocorr,
        ),
        "squared_return_autocorrelation_flattened_l1": autocorrelation_l1(
            real_flattened_squared_autocorr,
            generated_flattened_squared_autocorr,
        ),
        "tail_thresholds_from_real": thresholds,
        "tail_exceedance_rates": tail_exceedance_rates(
            generated_paths,
            thresholds=thresholds,
        ),
    }


def autocorrelation_l1(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    """Return mean absolute distance between matching autocorrelation lags."""
    keys = sorted(set(first).intersection(second), key=int)
    if not keys:
        return 0.0
    return float(sum(abs(first[key] - second[key]) for key in keys) / len(keys))


def squeeze_paths(paths: Tensor) -> Tensor:
    """Convert path tensors to ``[batch, length]``."""
    if paths.ndim == 3 and paths.shape[-1] == 1:
        return paths[..., 0].detach().float()
    if paths.ndim == 2:
        return paths.detach().float()
    raise ValueError(
        f"Expected paths with shape [batch, length, 1] or [batch, length]; got {paths.shape}."
    )
