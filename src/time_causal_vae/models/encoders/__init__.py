# mypy: ignore-errors
# ruff: noqa
"""Encoder modules and compatibility aliases."""

from time_causal_vae.models.encoders.base import BaseEncoder
from time_causal_vae.models.encoders.lstm import (
    CLSTMEncoder,
    CLSTMResEncoder,
    ConditionalResidualLSTMEncoder,
    LSTMEncoder,
    LSTMResEncoder,
)
from time_causal_vae.models.encoders.mlp import CMLPEncoder, IdEncoder, MLPEncoder

__all__ = [
    "BaseEncoder",
    "CLSTMEncoder",
    "CLSTMResEncoder",
    "CMLPEncoder",
    "ConditionalResidualLSTMEncoder",
    "IdEncoder",
    "LSTMEncoder",
    "LSTMResEncoder",
    "MLPEncoder",
]
