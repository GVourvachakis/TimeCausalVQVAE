"""Randomness helpers for reproducible target runs."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy, PyTorch, and CUDA seeds.

    Parameters
    ----------
    seed:
        Seed applied to ``random``, ``numpy``, ``torch``, and all CUDA devices.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
