"""Compatibility helpers for legacy checkpoint metadata."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

MODEL_CLASS_MAP = {
    "BetaCVAE": "BetaConditionalVAE",
    "InfoCVAE": "InfoConditionalVAE",
    "VAE": "VAE",
}

MODEL_CONFIG_NAME_MAP = {
    # Legacy selected BetaCVAE checkpoints may store the beta objective config
    # under this name. The target factory can still rebuild the selected model
    # from exp_config.yaml without changing state tensors.
    "BetaVAEConfig": "BetaCVAEConfig",
    "BetaCVAEConfig": "BetaCVAEConfig",
    "InfoCVAEConfig": "InfoCVAEConfig",
    "VAEConfig": "VAEConfig",
}

ENCODER_CLASS_MAP = {
    "CLSTMRes": "ConditionalResidualLSTMEncoder",
    "CLSTM": "CLSTMEncoder",
    "LSTM": "LSTMEncoder",
    "MLP": "MLPEncoder",
    "CMLP": "CMLPEncoder",
    "IdEncoder": "IdEncoder",
}

DECODER_CLASS_MAP = {
    "CLSTMRes": "ConditionalResidualLSTMDecoder",
    "LSTMRes": "LSTMResDecoder",
    "LSTM": "LSTMDecoder",
    "MLP": "MLPDecoder",
    "CMLP": "CMLPDecoder",
    "CAddMLP": "CAddMLPDecoder",
    "CRSigDecoder": "CRSigDecoder",
    "IdDecoder": "IdDecoder",
}

PRIOR_CLASS_MAP = {
    "RealNVP": "RealNVPPrior",
    "Gaussian": "GaussianPrior",
}


def target_name(kind: str, legacy_name: str) -> str:
    """Return the target class/config name for a legacy checkpoint component."""
    maps = {
        "model": MODEL_CLASS_MAP,
        "model_config": MODEL_CONFIG_NAME_MAP,
        "encoder": ENCODER_CLASS_MAP,
        "decoder": DECODER_CLASS_MAP,
        "prior": PRIOR_CLASS_MAP,
    }
    return maps[kind].get(legacy_name, legacy_name)


def compatibility_summary(exp_config: Any) -> dict[str, str]:
    """Summarise legacy-to-target names without mutating the experiment config."""
    return {
        "model": target_name("model", exp_config.model),
        "encoder": target_name("encoder", exp_config.encoder),
        "decoder": target_name("decoder", exp_config.decoder),
        "prior": target_name("prior", exp_config.prior),
    }


def load_legacy_exp_config(exp_config_path: Path, *, base_data_dir: str | None = None) -> Any:
    """Load the legacy ``exp_config.yaml`` required next to ``final_model``."""
    with exp_config_path.open(encoding="utf-8") as file:
        exp_config = yaml.load(file, Loader=yaml.UnsafeLoader)

    exp_config = deepcopy(exp_config)
    if base_data_dir is not None:
        exp_config.base_data_dir = base_data_dir
    return exp_config
