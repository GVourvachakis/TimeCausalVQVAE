# mypy: ignore-errors
# ruff: noqa
"""Objective modules and compatibility aliases."""

from time_causal_vae.models.objectives.beta_cvae import (
    BetaConditionalVAE,
    BetaCVAE,
    BetaCVAEConfig,
    BetaVAE,
    BetaVAEConfig,
)
from time_causal_vae.models.objectives.info_cvae import (
    InfoConditionalVAE,
    InfoCVAE,
    InfoCVAEConfig,
    InfoVAE,
    InfoVAEConfig,
)
from time_causal_vae.models.objectives.vae import CVAE, VAE, CVAEConfig, VAEConfig

__all__ = [
    "CVAE",
    "VAE",
    "BetaCVAE",
    "BetaCVAEConfig",
    "BetaConditionalVAE",
    "BetaVAE",
    "BetaVAEConfig",
    "CVAEConfig",
    "InfoCVAE",
    "InfoCVAEConfig",
    "InfoConditionalVAE",
    "InfoVAE",
    "InfoVAEConfig",
    "VAEConfig",
]
