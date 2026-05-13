"""Plotting helpers for target evaluation workflows."""

from __future__ import annotations

from typing import Any


def visualize_real_recon_fake(*args: Any, **kwargs: Any) -> Any:
    """Delegate the selected reconstruction plot to the legacy plotting helper."""
    from time_causal_vae.utils.plotting import visualize_real_recon_fake as _visualize

    return _visualize(*args, **kwargs)


def visualize_real_recon_fake_2d(*args: Any, **kwargs: Any) -> Any:
    """Delegate the 2D reconstruction plot to the legacy plotting helper."""
    from time_causal_vae.utils.plotting import visualize_real_recon_fake_2d as _visualize

    return _visualize(*args, **kwargs)
