"""Plotting helpers compatible with the legacy visualisation utilities."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def visualize_data(data: Any, dim: int = 0) -> Any:
    """Plot sample paths for one data dimension.

    Parameters
    ----------
    data:
        Tensor-like or NumPy array with path data.
    dim:
        Final-axis dimension to plot.

    Returns
    -------
    Any
        Matplotlib figure.
    """
    if not isinstance(data, np.ndarray):
        data = data.numpy()
    fig, ax = plt.subplots(1, 1)
    ax.plot(data[:1000][..., dim].T, alpha=0.3, marker="o", linewidth=1, markersize=1)
    return fig


def visualize_real_recon_fake(
    real: Any, recon: Any, fake: Any, dim: int = 0, n_sample: int = 1000
) -> Any:
    """Plot real, reconstructed, and generated paths side by side."""
    fig, ax = plt.subplots(1, 3, figsize=[16, 4], sharex=True, sharey=True)
    ax[0].plot(real[:n_sample][..., dim].numpy().T, alpha=0.3)
    ax[1].plot(recon[:n_sample][..., dim].numpy().T, alpha=0.3)
    ax[2].plot(fake[:n_sample][..., dim].numpy().T, alpha=0.3)
    return fig


def visualize_real_recon_fake_2d(real: Any, recon: Any, fake: Any) -> Any:
    """Scatter real, reconstructed, and generated two-dimensional samples."""
    fig, ax = plt.subplots(1, 3, figsize=[16, 4], sharex=True, sharey=True)
    for i, data in enumerate([real, recon, fake]):
        data = data.flatten(start_dim=1)
        ax[i].scatter(data[:, 0], data[:, 1], s=1)
    return fig


def visualize_data_2d(data: Any) -> None:
    """Scatter two-dimensional samples and show the figure."""
    data = data.flatten(start_dim=1)
    _fig, ax = plt.subplots(1, 1)
    ax.scatter(data[:, 0], data[:, 1], s=1)
    plt.show()
