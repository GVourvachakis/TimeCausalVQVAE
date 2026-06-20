# mypy: ignore-errors
"""Model namespaces for TimeCausalVAE."""

from __future__ import annotations

import importlib
import sys

from time_causal_vae.models.continuous.factory import ModelFactory, NetworkPipeline

_COMPAT_ALIASES = {
    "base": "continuous.base",
    "config": "continuous.config",
    "distances": "continuous.distances",
    "factory": "continuous.factory",
    "initialisation": "continuous.initialisation",
    "losses": "continuous.losses",
    "transforms": "continuous.transforms",
    "conditioners": "continuous.conditioners",
    "conditioners.base": "continuous.conditioners.base",
    "conditioners.identity": "continuous.conditioners.identity",
    "decoders": "continuous.decoders",
    "decoders.base": "continuous.decoders.base",
    "decoders.lstm": "continuous.decoders.lstm",
    "decoders.mlp": "continuous.decoders.mlp",
    "decoders.neural_sde": "continuous.decoders.neural_sde",
    "encoders": "continuous.encoders",
    "encoders.base": "continuous.encoders.base",
    "encoders.lstm": "continuous.encoders.lstm",
    "encoders.mlp": "continuous.encoders.mlp",
    "objectives": "continuous.objectives",
    "objectives.beta_cvae": "continuous.objectives.beta_cvae",
    "objectives.info_cvae": "continuous.objectives.info_cvae",
    "objectives.vae": "continuous.objectives.vae",
    "priors": "continuous.priors",
    "priors.base": "continuous.priors.base",
    "priors.gaussian": "continuous.priors.gaussian",
    "priors.realnvp": "continuous.priors.realnvp",
}


def _register_compat_aliases() -> None:
    """Keep old continuous import paths pointing at the canonical modules."""
    package = sys.modules[__name__]
    for old_suffix, new_suffix in _COMPAT_ALIASES.items():
        old_name = f"{__name__}.{old_suffix}"
        new_name = f"{__name__}.{new_suffix}"
        module = importlib.import_module(new_name)
        sys.modules.setdefault(old_name, module)
        if "." not in old_suffix:
            setattr(package, old_suffix, module)


_register_compat_aliases()

__all__ = ["ModelFactory", "NetworkPipeline"]
