"""Lazy wrappers for optional external unconditional evaluation helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXTERNAL_MODULE = "time_causal_vae.evaluation.external.unconditional"


def __getattr__(name: str) -> Any:
    """Load optional unconditional helpers on demand."""
    return getattr(import_module(_EXTERNAL_MODULE), name)
