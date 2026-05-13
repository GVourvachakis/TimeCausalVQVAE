"""Volatility diagnostics for generated path inspection."""

from __future__ import annotations

from typing import Any

import torch
from scipy import stats
from torch import Tensor


def prices_to_returns(prices: Tensor) -> Tensor:
    """Convert price paths to simple returns.

    Parameters
    ----------
    prices:
        Tensor of shape ``(n_paths, n_steps, data_dim)``.

    Returns
    -------
    Tensor
        Simple returns with one fewer time step.
    """
    return prices[:, 1:] / prices[:, :-1] - 1


def volatility_samples(price_paths: Tensor, horizon: float) -> Tensor:
    """Compute per-path realised volatility using the upstream formula.

    Parameters
    ----------
    price_paths:
        Price paths with shape ``(n_paths, n_steps, data_dim)``.
    horizon:
        Time horizon used in the upstream drift-volatility diagnostic.

    Returns
    -------
    Tensor
        One volatility sample per path and data dimension.
    """
    returns = prices_to_returns(price_paths.detach().cpu().float())
    variance = torch.sum(returns**2, dim=1) / horizon
    return torch.sqrt(variance)


def volatility_diagnostics(real_data: Tensor, fake_data: Tensor, horizon: float) -> dict[str, Any]:
    """Summarise volatility differences without changing generated paths.

    Parameters
    ----------
    real_data:
        Real price paths.
    fake_data:
        Generated price paths.
    horizon:
        Time horizon used to compute realised volatility.

    Returns
    -------
    dict[str, Any]
        Scalar summary diagnostics suitable for notebook display and candidate
        inspection. These values are diagnostic only and never rescale generated paths.
    """
    real_vol = volatility_samples(real_data, horizon).flatten()
    fake_vol = volatility_samples(fake_data, horizon).flatten()
    real_np = real_vol.numpy()
    fake_np = fake_vol.numpy()
    real_mean = float(real_vol.mean())
    fake_mean = float(fake_vol.mean())
    summary: dict[str, Any] = {
        "real_volatility_mean": real_mean,
        "fake_volatility_mean": fake_mean,
        "real_volatility_std": float(real_vol.std()),
        "fake_volatility_std": float(fake_vol.std()),
        "absolute_mean_difference": abs(real_mean - fake_mean),
        "wasserstein_1": float(stats.wasserstein_distance(real_np, fake_np)),
    }
    ks_result = stats.ks_2samp(real_np, fake_np)
    summary["ks_statistic"] = float(ks_result.statistic)
    summary["ks_pvalue"] = float(ks_result.pvalue)
    return summary
