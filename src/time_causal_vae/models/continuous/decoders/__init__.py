# mypy: ignore-errors
# ruff: noqa
"""Decoder modules and compatibility aliases."""

from time_causal_vae.models.continuous.decoders.base import BaseDecoder
from time_causal_vae.models.continuous.decoders.lstm import (
    CLSTMResDecoder,
    ConditionalResidualLSTMDecoder,
    LSTMDecoder,
    LSTMResDecoder,
)
from time_causal_vae.models.continuous.decoders.mlp import (
    CAddMLPDecoder,
    CMLPDecoder,
    IdDecoder,
    MLPDecoder,
)
from time_causal_vae.models.continuous.decoders.neural_sde import CRSigDecoder, NeuralSDEDecoder

__all__ = [
    "BaseDecoder",
    "CAddMLPDecoder",
    "CLSTMResDecoder",
    "CMLPDecoder",
    "CRSigDecoder",
    "ConditionalResidualLSTMDecoder",
    "IdDecoder",
    "LSTMDecoder",
    "LSTMResDecoder",
    "MLPDecoder",
    "NeuralSDEDecoder",
]
