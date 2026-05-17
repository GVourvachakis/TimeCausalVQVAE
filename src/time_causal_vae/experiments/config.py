"""Experiment configuration loading and continuous-backend adaptation utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import ml_collections
import yaml

from time_causal_vae.typing import PathLike

DATASET_NAME_MAP = {
    "black_scholes": "BSprice",
    "heston": "Hestonprice",
    "hawkes_jump": "HawkesJump",
    "path_dependent_volatility": "PDVPriceConFeature",
    "sp500_vix": "SP500VIX",
}

MODEL_NAME_MAP = {
    "beta_cvae": "BetaCVAE",
    "info_cvae": "InfoCVAE",
}

ENCODER_NAME_MAP = {
    "conditional_residual_lstm": "CLSTMRes",
}

DECODER_NAME_MAP = {
    "conditional_residual_lstm": "CLSTMRes",
}

CONDITIONER_NAME_MAP = {
    "identity": "Id",
}

PRIOR_NAME_MAP = {
    "real_nvp": "RealNVP",
}

OPTIMIZER_NAME_MAP = {
    "adam": "Adam",
}

SUPPORTED_DATA_PARAMS = {
    "black_scholes": frozenset({"mu", "sigma", "dt"}),
    "heston": frozenset({"r", "kappa", "theta", "v_0", "rho", "xi", "dt"}),
    "hawkes_jump": frozenset(
        {
            "baseline_intensity",
            "brownian_volatility",
            "data_output",
            "decay",
            "drift",
            "dt",
            "excitation",
            "mark_excitation",
            "max_intensity",
            "max_jumps_per_step",
            "max_volatility",
            "negative_jump_mean",
            "negative_jump_probability",
            "negative_jump_std",
            "positive_jump_mean",
            "positive_jump_std",
            "seed",
            "severe_jump_mean",
            "severe_jump_probability",
            "severe_jump_std",
            "volatility_decay",
            "volatility_excitation",
            "volatility_excitation_scale",
        }
    ),
}


@dataclass(frozen=True)
class FieldMapping:
    """Document one portable-config to backend-config field mapping."""

    backend_name: str
    portable_path: str
    description: str


FIELD_MAPPINGS = (
    FieldMapping("model", "model.objective", "Map selected objective names to backend names."),
    FieldMapping("dataset", "experiment.dataset", "Map selected dataset names to backend names."),
    FieldMapping("encoder", "model.encoder", "Map selected encoder names to backend names."),
    FieldMapping("decoder", "model.decoder", "Map selected decoder names to backend names."),
    FieldMapping(
        "conditioner",
        "model.conditioner",
        "Map selected conditioner names to backend names.",
    ),
    FieldMapping("prior", "model.prior", "Map selected prior names to backend names."),
    FieldMapping("n_sample", "data.n_samples", "Backend dataset sample count."),
    FieldMapping("n_timestep", "data.n_timesteps", "Backend dataset time-step count."),
    FieldMapping("data_dim", "model.data_dim", "Observed data dimension used by models."),
    FieldMapping("data_length", "model.data_length", "Observed path length used by models."),
    FieldMapping("latent_dim", "model.latent_dim", "Latent path channel count."),
    FieldMapping("latent_length", "model.latent_length", "Latent path length."),
    FieldMapping("condition_dim", "data.condition_dim", "Scalar condition dimension."),
    FieldMapping("beta", "model.beta", "BetaCVAE and InfoCVAE KL weighting."),
    FieldMapping("alpha", "model.alpha", "InfoCVAE MMD weighting when present."),
    FieldMapping("E_hidden_dim", "model.encoder_hidden_dim", "Encoder hidden width."),
    FieldMapping("E_num_layers", "model.encoder_num_layers", "Encoder LSTM layers."),
    FieldMapping("D_hidden_dim", "model.decoder_hidden_dim", "Decoder hidden width."),
    FieldMapping("D_num_layers", "model.decoder_num_layers", "Decoder LSTM layers."),
    FieldMapping("P_num_flows", "model.prior_num_flows", "RealNVP flow count."),
    FieldMapping("P_hidden_dim", "model.prior_hidden_dim", "RealNVP hidden width."),
    FieldMapping("lr", "training.learning_rate", "Trainer learning rate field."),
    FieldMapping(
        "train_batch_size",
        "training.train_batch_size",
        "Per-device train batch size.",
    ),
    FieldMapping(
        "eval_batch_size",
        "training.eval_batch_size",
        "Per-device evaluation batch size.",
    ),
    FieldMapping("epochs", "training.epochs", "Trainer epoch count."),
    FieldMapping("transform", "data.transform", "Input transform name."),
    FieldMapping("inv_transform", "data.inverse_transform", "Inverse transform name."),
    FieldMapping("seed", "experiment.seed", "Random seed."),
    FieldMapping("steps_predict", "training.steps_predict", "Prediction cadence."),
    FieldMapping("steps_saving", "training.steps_saving", "Checkpoint cadence."),
)


def load_experiment_config(path: PathLike) -> dict[str, Any]:
    """Load an experiment configuration from YAML.

    Parameters
    ----------
    path
        Path to the experiment YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return cast(dict[str, Any], yaml.safe_load(handle))


def load_selected_config(path: PathLike) -> dict[str, Any]:
    """Load a selected experiment YAML file."""
    data = load_experiment_config(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level mapping")
    return data


def adapt_selected_config(
    config: Mapping[str, Any],
    *,
    output_dir: PathLike,
    device: str | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    wandb: bool = False,
    base_data_dir: PathLike = "data/processed",
) -> ml_collections.ConfigDict:
    """Convert a selected YAML config mapping into the continuous backend config object."""
    experiment = _required_mapping(config, "experiment")
    data = _required_mapping(config, "data")
    model = _required_mapping(config, "model")
    training = _required_mapping(config, "training")
    dataset_name = _required_value(experiment, "dataset", "experiment.dataset")
    data_params = _validated_data_params(data, dataset_name)

    output_path = Path(output_dir)
    backend_config = {
        "experiment_name": _required_value(experiment, "name", "experiment.name"),
        "model": _map_value(model, "objective", MODEL_NAME_MAP, "model.objective"),
        "dataset": _map_value(experiment, "dataset", DATASET_NAME_MAP, "experiment.dataset"),
        "encoder": _map_value(model, "encoder", ENCODER_NAME_MAP, "model.encoder"),
        "decoder": _map_value(model, "decoder", DECODER_NAME_MAP, "model.decoder"),
        "conditioner": _map_value(
            model,
            "conditioner",
            CONDITIONER_NAME_MAP,
            "model.conditioner",
        ),
        "prior": _map_value(model, "prior", PRIOR_NAME_MAP, "model.prior"),
        "n_sample": _required_value(data, "n_samples", "data.n_samples"),
        "n_timestep": _required_value(data, "n_timesteps", "data.n_timesteps"),
        "data_dim": _required_value(model, "data_dim", "model.data_dim"),
        "data_length": _required_value(model, "data_length", "model.data_length"),
        "latent_dim": _required_value(model, "latent_dim", "model.latent_dim"),
        "latent_length": _required_value(model, "latent_length", "model.latent_length"),
        "condition_dim": _required_value(data, "condition_dim", "data.condition_dim"),
        "beta": _required_value(model, "beta", "model.beta"),
        "alpha": model.get("alpha"),
        "E_hidden_dim": _required_value(
            model,
            "encoder_hidden_dim",
            "model.encoder_hidden_dim",
        ),
        "E_num_layers": _required_value(
            model,
            "encoder_num_layers",
            "model.encoder_num_layers",
        ),
        "D_hidden_dim": _required_value(
            model,
            "decoder_hidden_dim",
            "model.decoder_hidden_dim",
        ),
        "D_num_layers": _required_value(
            model,
            "decoder_num_layers",
            "model.decoder_num_layers",
        ),
        "P_num_flows": _required_value(model, "prior_num_flows", "model.prior_num_flows"),
        "P_hidden_dim": _required_value(model, "prior_hidden_dim", "model.prior_hidden_dim"),
        "lr": _required_value(training, "learning_rate", "training.learning_rate"),
        "train_batch_size": _required_value(
            training,
            "train_batch_size",
            "training.train_batch_size",
        ),
        "eval_batch_size": _required_value(
            training,
            "eval_batch_size",
            "training.eval_batch_size",
        ),
        "epochs": epochs
        if epochs is not None
        else _required_value(training, "epochs", "training.epochs"),
        "transform": _required_value(data, "transform", "data.transform"),
        "inv_transform": _required_value(data, "inverse_transform", "data.inverse_transform"),
        "seed": seed
        if seed is not None
        else _required_value(experiment, "seed", "experiment.seed"),
        "steps_predict": _required_value(training, "steps_predict", "training.steps_predict"),
        "steps_saving": _required_value(training, "steps_saving", "training.steps_saving"),
        "optimizer": _map_value(training, "optimizer", OPTIMIZER_NAME_MAP, "training.optimizer"),
        "wandb": wandb,
        "device_name": device,
        "output_dir": str(output_path),
        "base_output_dir": str(output_path),
        "base_data_dir": str(base_data_dir),
        "data_params": data_params,
        "discriminator": None,
        "comment": None,
        "ploter": "path",
        "C_input_dim": 0,
        "C_hidden_dim": 0,
        "C_num_layers": 0,
        "C_output_dim": 0,
        "E_input_dim": _required_value(model, "data_dim", "model.data_dim"),
        "E_output_dim": _required_value(model, "latent_dim", "model.latent_dim"),
        "D_input_dim": _required_value(model, "latent_dim", "model.latent_dim"),
        "D_output_dim": _required_value(model, "data_dim", "model.data_dim"),
        "P_latent_dim": None,
    }

    return ml_collections.ConfigDict(backend_config)


def load_continuous_backend_config(
    path: PathLike,
    *,
    output_dir: PathLike,
    device: str | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    wandb: bool = False,
    base_data_dir: PathLike = "data/processed",
) -> ml_collections.ConfigDict:
    """Load and adapt a selected continuous YAML config in one step.

    The returned field names match the existing continuous TC-VAE backend. The public
    selector remains `time_causal_vae.experiments.model_registry`.
    """
    return adapt_selected_config(
        load_selected_config(path),
        output_dir=output_dir,
        device=device,
        epochs=epochs,
        seed=seed,
        wandb=wandb,
        base_data_dir=base_data_dir,
    )


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected '{key}' mapping in selected experiment config")
    return value


def _required_value(config: Mapping[str, Any], key: str, dotted_path: str) -> Any:
    value = config.get(key)
    if value is None:
        raise ValueError(f"Missing required config value '{dotted_path}'")
    return value


def _map_value(
    config: Mapping[str, Any],
    key: str,
    value_map: Mapping[str, str],
    dotted_path: str,
) -> str:
    raw_value = _required_value(config, key, dotted_path)
    if not isinstance(raw_value, str):
        raise ValueError(f"Expected string config value '{dotted_path}'")
    try:
        return value_map[raw_value]
    except KeyError as exc:
        valid_values = ", ".join(sorted(value_map))
        raise ValueError(
            f"Unsupported value {raw_value!r} for '{dotted_path}'. Expected one of: {valid_values}"
        ) from exc


def _validated_data_params(data: Mapping[str, Any], dataset_name: str) -> dict[str, Any]:
    params = data.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ValueError("Expected optional 'data.params' to be a mapping")

    params_dict = dict(params)
    if not params_dict:
        return {}

    supported = SUPPORTED_DATA_PARAMS.get(dataset_name)
    if supported is None:
        raise ValueError(
            f"Unsupported custom data.params for dataset '{dataset_name}'. "
            "Custom synthetic parameters are currently supported only for "
            "black_scholes, heston, and hawkes_jump."
        )

    unsupported = sorted(set(params_dict) - supported)
    if unsupported:
        supported_text = ", ".join(sorted(supported))
        unsupported_text = ", ".join(unsupported)
        raise ValueError(
            f"Unsupported data.params for dataset '{dataset_name}': {unsupported_text}. "
            f"Supported parameters: {supported_text}."
        )

    return params_dict
