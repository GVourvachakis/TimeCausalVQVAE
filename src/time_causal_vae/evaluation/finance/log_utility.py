"""Lazy wrapper for optional external log-utility evaluation helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXTERNAL_MODULE = "time_causal_vae.evaluation.external.log_utility"


def __getattr__(name: str) -> Any:
    """Load optional log-utility helpers on demand."""
    return getattr(import_module(_EXTERNAL_MODULE), name)
