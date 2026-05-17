# mypy: ignore-errors
# ruff: noqa
"""Encoder modules and compatibility aliases."""

from time_causal_vae.models.continuous.encoders.base import BaseEncoder
from time_causal_vae.models.continuous.encoders.lstm import (
    CLSTMEncoder,
    CLSTMResEncoder,
    ConditionalResidualLSTMEncoder,
    LSTMEncoder,
    LSTMResEncoder,
)
from time_causal_vae.models.continuous.encoders.mlp import CMLPEncoder, IdEncoder, MLPEncoder

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
