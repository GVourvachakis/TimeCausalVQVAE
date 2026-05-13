"""Plotting styles for source-style and clean diagnostic figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_source_style() -> None:
    """Apply the upstream notebook plotting style.

    The upstream notebooks call ``seaborn.set_theme()`` before diagnostic figures.
    Keeping that call in one helper makes source-style figures consistent without
    changing any metric or sampling semantics.
    """
    import seaborn as sns

    sns.set_theme()


def apply_clean_style() -> None:
    """Restore Matplotlib's default plotting style."""
    plt.style.use("default")
