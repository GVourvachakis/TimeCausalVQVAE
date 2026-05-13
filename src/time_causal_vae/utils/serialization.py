"""Serialization helpers compatible with the legacy utility functions."""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import torch


def save_obj(obj: Any, filepath: str) -> int:
    """Save an object based on the filepath extension.

    Parameters
    ----------
    obj:
        Object to save.
    filepath:
        Destination path ending in ``.pkl``, ``.pt``, ``.json``, or ``.npy``.

    Returns
    -------
    int
        Legacy-compatible success code, always ``0``.
    """
    if filepath.endswith("pkl"):
        with open(filepath, "wb") as handle:
            pickle.dump(obj, handle)
        return 0
    elif filepath.endswith("pt"):
        with open(filepath, "wb") as handle:
            torch.save(obj, handle)
        return 0
    elif filepath.endswith("json"):
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(obj, handle)
        return 0
    elif filepath.endswith("npy"):
        np.save(filepath, obj)
        return 0
    else:
        raise NotImplementedError(f"No suitable saver for the path: {filepath}")


def load_obj(filepath: str) -> Any:
    """Load an object based on the filepath extension.

    Parameters
    ----------
    filepath:
        Source path ending in ``.pkl``, ``.pt``, ``.json``, or ``.npy``.

    Returns
    -------
    Any
        Loaded object.
    """
    if filepath.endswith("pkl"):
        with open(filepath, "rb") as handle:
            return pickle.load(handle)
    elif filepath.endswith("pt"):
        with open(filepath, "rb") as handle:
            return torch.load(handle, weights_only=True)
    elif filepath.endswith("json"):
        with open(filepath, "rb") as handle:
            return json.load(handle)
    elif filepath.endswith("npy"):
        return np.load(filepath)
    else:
        raise NotImplementedError(f"No suitable loader for the path: {filepath}")
