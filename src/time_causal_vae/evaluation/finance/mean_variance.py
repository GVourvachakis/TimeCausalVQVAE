"""Lazy wrapper for optional external mean-variance evaluation helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXTERNAL_MODULE = "time_causal_vae.evaluation.external.mean_variance"


def __getattr__(name: str) -> Any:
    """Load optional mean-variance helpers on demand."""
    return getattr(import_module(_EXTERNAL_MODULE), name)
