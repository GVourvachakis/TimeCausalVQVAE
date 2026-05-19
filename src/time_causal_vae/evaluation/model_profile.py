"""Model-size and inference-time profiling helpers for report notebooks."""

from __future__ import annotations

import statistics
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import ml_collections
import torch
import yaml
from torch import nn

from time_causal_vae.models.continuous.factory import ModelFactory
from time_causal_vae.models.discrete.priors import CausalTokenPriorConfig, build_token_prior_model
from time_causal_vae.models.discrete.tokenizers import CausalVQTokenizer, VQTokenizerConfig

ModelFamily = Literal["continuous", "discrete"]

_CONTINUOUS_MODEL_NAMES = {
    "beta_cvae": "BetaCVAE",
    "info_cvae": "InfoCVAE",
}
_CONTINUOUS_ENCODER_NAMES = {
    "conditional_residual_lstm": "CLSTMRes",
}
_CONTINUOUS_DECODER_NAMES = {
    "conditional_residual_lstm": "CLSTMRes",
}
_CONTINUOUS_CONDITIONER_NAMES = {
    "identity": "Id",
}
_CONTINUOUS_PRIOR_NAMES = {
    "gaussian": "Gaussian",
    "real_nvp": "RealNVP",
}


@dataclass(frozen=True)
class ModelProfileSpec:
    """Configuration paths and labels for one report model profile."""

    role: str
    family: ModelFamily
    model: str
    config: str | Path | None = None
    tokenizer_config: str | Path | None = None
    prior_config: str | Path | None = None
    sampling: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ModelProfileResult:
    """Profile output for one continuous model or one tokenizer-plus-prior pair."""

    role: str
    family: ModelFamily
    model: str
    parameters: int
    tokenizer_parameters: int | None
    token_prior_parameters: int | None
    batch_size: int
    inference_mean_seconds: float
    inference_std_seconds: float
    device: str
    timed_operation: str
    source: str

    def as_row(self) -> dict[str, Any]:
        """Return a notebook-friendly table row."""
        mean_ms = self.inference_mean_seconds * 1000.0
        std_ms = self.inference_std_seconds * 1000.0
        return {
            "role": self.role,
            "family": self.family,
            "model": self.model,
            "parameters": self.parameters,
            "parameters_millions": self.parameters / 1_000_000.0,
            "tokenizer_parameters": self.tokenizer_parameters,
            "token_prior_parameters": self.token_prior_parameters,
            "benchmark_batch_size": self.batch_size,
            "inference_mean_ms": mean_ms,
            "inference_std_ms": std_ms,
            "inference_ms_per_path": mean_ms / self.batch_size,
            "device": self.device,
            "timed_operation": self.timed_operation,
            "source": self.source,
        }


def profile_model_specs(
    specs: Sequence[ModelProfileSpec],
    *,
    repo_root: str | Path = ".",
    batch_size: int = 64,
    warmup_runs: int = 1,
    repeats: int = 3,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Profile model specs and return rows suitable for a pandas DataFrame."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    root = Path(repo_root)
    device_obj = torch.device(device)
    rows = []
    for spec in specs:
        if spec.family == "continuous":
            result = _profile_continuous_spec(
                spec,
                repo_root=root,
                batch_size=batch_size,
                warmup_runs=warmup_runs,
                repeats=repeats,
                device=device_obj,
                seed=seed,
            )
        elif spec.family == "discrete":
            result = _profile_discrete_spec(
                spec,
                repo_root=root,
                batch_size=batch_size,
                warmup_runs=warmup_runs,
                repeats=repeats,
                device=device_obj,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported model family: {spec.family!r}.")
        rows.append(result.as_row())
    return rows


def _profile_continuous_spec(
    spec: ModelProfileSpec,
    *,
    repo_root: Path,
    batch_size: int,
    warmup_runs: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> ModelProfileResult:
    if spec.config is None:
        raise ValueError(f"{spec.role} requires config.")

    torch.manual_seed(seed)
    raw_config = _load_yaml_mapping(_resolve_path(spec.config, repo_root))
    model = _build_continuous_model(raw_config).to(device)
    model.eval()
    model.device = device
    condition_dim = int(cast(Mapping[str, Any], raw_config["data"]).get("condition_dim", 0))
    conditions = _zero_conditions(
        batch_size=batch_size,
        condition_dim=condition_dim,
        device=device,
    )

    def run_generation() -> torch.Tensor:
        if conditions is None:
            return cast(torch.Tensor, model.generation(batch_size))
        return cast(torch.Tensor, model.generation(batch_size, c=conditions))

    mean_seconds, std_seconds = _time_inference(
        run_generation,
        warmup_runs=warmup_runs,
        repeats=repeats,
        device=device,
    )
    return ModelProfileResult(
        role=spec.role,
        family=spec.family,
        model=spec.model,
        parameters=_count_parameters(model),
        tokenizer_parameters=None,
        token_prior_parameters=None,
        batch_size=batch_size,
        inference_mean_seconds=mean_seconds,
        inference_std_seconds=std_seconds,
        device=str(device),
        timed_operation="continuous generation",
        source="YAML architecture, randomly initialised weights",
    )


def _profile_discrete_spec(
    spec: ModelProfileSpec,
    *,
    repo_root: Path,
    batch_size: int,
    warmup_runs: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> ModelProfileResult:
    if spec.tokenizer_config is None or spec.prior_config is None:
        raise ValueError(f"{spec.role} requires tokenizer_config and prior_config.")

    torch.manual_seed(seed)
    tokenizer_config = _build_tokenizer_config(
        _load_yaml_mapping(_resolve_path(spec.tokenizer_config, repo_root))
    )
    prior_config = _build_prior_config(
        _load_yaml_mapping(_resolve_path(spec.prior_config, repo_root))
    )
    tokenizer = CausalVQTokenizer(tokenizer_config).to(device)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="enable_nested_tensor is True.*",
            category=UserWarning,
        )
        prior = build_token_prior_model(prior_config).to(device)
    tokenizer.eval()
    prior.eval()

    prior_conditions = _zero_conditions(
        batch_size=batch_size,
        condition_dim=prior_config.condition_dim,
        device=device,
    )
    tokenizer_conditions = _zero_conditions(
        batch_size=batch_size,
        condition_dim=tokenizer_config.condition_dim,
        device=device,
    )
    sampling = dict(spec.sampling or {})
    temperature = float(sampling.get("temperature", 1.0))
    top_k_raw = sampling.get("top_k")
    top_k = None if top_k_raw is None else int(top_k_raw)

    def run_generation() -> torch.Tensor:
        tokens = prior.sample(
            batch_size=batch_size,
            device=device,
            temperature=temperature,
            top_k=top_k,
            conditions=prior_conditions,
        )
        return tokenizer.decode_indices(tokens, tokenizer_conditions)

    mean_seconds, std_seconds = _time_inference(
        run_generation,
        warmup_runs=warmup_runs,
        repeats=repeats,
        device=device,
    )
    tokenizer_parameters = _count_parameters(tokenizer)
    token_prior_parameters = _count_parameters(prior)
    return ModelProfileResult(
        role=spec.role,
        family=spec.family,
        model=spec.model,
        parameters=tokenizer_parameters + token_prior_parameters,
        tokenizer_parameters=tokenizer_parameters,
        token_prior_parameters=token_prior_parameters,
        batch_size=batch_size,
        inference_mean_seconds=mean_seconds,
        inference_std_seconds=std_seconds,
        device=str(device),
        timed_operation="token-prior sampling plus tokenizer decode",
        source="YAML architecture, randomly initialised weights",
    )


def _build_continuous_model(raw_config: Mapping[str, Any]) -> nn.Module:
    experiment = _required_mapping(raw_config, "experiment")
    data = _required_mapping(raw_config, "data")
    model = _required_mapping(raw_config, "model")
    training = _required_mapping(raw_config, "training")
    exp_config = ml_collections.ConfigDict(
        {
            "experiment_name": str(experiment["name"]),
            "model": _mapped_name(model, "objective", _CONTINUOUS_MODEL_NAMES),
            "dataset": str(experiment["dataset"]),
            "encoder": _mapped_name(model, "encoder", _CONTINUOUS_ENCODER_NAMES),
            "decoder": _mapped_name(model, "decoder", _CONTINUOUS_DECODER_NAMES),
            "conditioner": _mapped_name(model, "conditioner", _CONTINUOUS_CONDITIONER_NAMES),
            "prior": _mapped_name(model, "prior", _CONTINUOUS_PRIOR_NAMES),
            "n_sample": int(data["n_samples"]),
            "n_timestep": int(data["n_timesteps"]),
            "data_dim": int(model["data_dim"]),
            "data_length": int(model["data_length"]),
            "latent_dim": int(model["latent_dim"]),
            "latent_length": int(model["latent_length"]),
            "condition_dim": int(data.get("condition_dim", 0)),
            "beta": float(model.get("beta", 1.0)),
            "alpha": float(model.get("alpha", 1.0)),
            "E_hidden_dim": int(model["encoder_hidden_dim"]),
            "E_num_layers": int(model["encoder_num_layers"]),
            "D_hidden_dim": int(model["decoder_hidden_dim"]),
            "D_num_layers": int(model["decoder_num_layers"]),
            "P_num_flows": int(model.get("prior_num_flows", 0)),
            "P_hidden_dim": int(model.get("prior_hidden_dim", 1)),
            "lr": float(training.get("learning_rate", 1e-3)),
            "train_batch_size": int(training.get("train_batch_size", 64)),
            "eval_batch_size": int(training.get("eval_batch_size", 64)),
            "epochs": int(training.get("epochs", 1)),
            "transform": str(data.get("transform", "")),
            "inv_transform": str(data.get("inverse_transform", "")),
            "seed": int(experiment.get("seed", 0)),
            "steps_predict": int(training.get("steps_predict", 1)),
            "steps_saving": int(training.get("steps_saving", 1)),
        }
    )
    return cast(nn.Module, ModelFactory()(exp_config))


def _build_tokenizer_config(raw_config: Mapping[str, Any]) -> VQTokenizerConfig:
    data = _required_mapping(raw_config, "data")
    model = _required_mapping(raw_config, "model")
    return VQTokenizerConfig(
        data_dim=int(model.get("data_dim", data["data_dim"])),
        data_length=int(model.get("data_length", data["n_timesteps"])),
        embedding_dim=int(model["embedding_dim"]),
        codebook_size=int(model["codebook_size"]),
        commitment_weight=float(model["commitment_weight"]),
        encoder_hidden_dim=int(model["encoder_hidden_dim"]),
        decoder_hidden_dim=int(model["decoder_hidden_dim"]),
        num_layers=int(model["num_layers"]),
        dilations=tuple(int(value) for value in cast(Sequence[Any], model["dilations"])),
        dropout=float(model.get("dropout", 0.0)),
        condition_dim=int(model.get("condition_dim", data.get("condition_dim", 0))),
        kmeans_init=bool(model.get("kmeans_init", False)),
        kmeans_iters=int(model.get("kmeans_iters", 10)),
        use_cosine_sim=bool(model.get("use_cosine_sim", False)),
        codebook_dim=_optional_int(model.get("codebook_dim")),
        threshold_ema_dead_code=float(model.get("threshold_ema_dead_code", 0.0)),
        decay=float(model.get("decay", 0.8)),
        usage_regularization_weight=float(model.get("usage_regularization_weight", 0.0)),
        usage_regularization_type=cast(
            Literal["none", "entropy"],
            str(model.get("usage_regularization_type", "none")),
        ),
        quantizer_type=cast(
            Literal["vector", "residual_vq", "grouped_residual_vq"],
            str(model.get("quantizer_type", "vector")),
        ),
        num_quantizers=int(model.get("num_quantizers", 1)),
        groups=int(model.get("groups", 1)),
        shared_codebook=bool(model.get("shared_codebook", False)),
        stochastic_sample_codes=bool(model.get("stochastic_sample_codes", False)),
        sample_codebook_temp=float(model.get("sample_codebook_temp", 0.0)),
    )


def _build_prior_config(raw_config: Mapping[str, Any]) -> CausalTokenPriorConfig:
    model = _required_mapping(raw_config, "model")
    return CausalTokenPriorConfig(
        codebook_size=int(model["codebook_size"]),
        sequence_length=int(model["sequence_length"]),
        token_embedding_dim=int(model["token_embedding_dim"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        mlp_hidden_dim=int(model["mlp_hidden_dim"]),
        dropout=float(model.get("dropout", 0.0)),
        bos_token_id=_optional_int(model.get("bos_token_id")),
        pad_token_id=_optional_int(model.get("pad_token_id")),
        prediction_convention=str(model.get("prediction_convention", "bos_shifted_next_token")),
        condition_dim=int(model.get("condition_dim", 0)),
        condition_injection=cast(
            Literal["none", "additive", "adaln_lite"],
            str(model.get("condition_injection", "none")),
        ),
        condition_hidden_dim=_optional_int(model.get("condition_hidden_dim")),
        adaln_hidden_dim=_optional_int(model.get("adaln_hidden_dim")),
        prior_type=cast(
            Literal[
                "single_code",
                "causal_conv_transformer",
                "factorised_multi_code",
                "hierarchical_rvq_q2",
            ],
            str(model.get("prior_type", "single_code")),
        ),
        index_shape=_optional_int_list(model.get("index_shape")),
        num_quantizers=int(model.get("num_quantizers", 1)),
        groups=int(model.get("groups", 1)),
        component_loss_weights=_optional_float_list(model.get("component_loss_weights")),
        conv_num_layers=int(model.get("conv_num_layers", 0)),
        conv_kernel_size=int(model.get("conv_kernel_size", 3)),
        conv_dilations=_optional_int_list(model.get("conv_dilations")),
        conv_dropout=float(model.get("conv_dropout", 0.0)),
    )


def _time_inference(
    operation: Callable[[], object],
    *,
    warmup_runs: int,
    repeats: int,
    device: torch.device,
) -> tuple[float, float]:
    timings = []
    with torch.inference_mode():
        for _ in range(warmup_runs):
            operation()
        _synchronise_if_needed(device)
        for _ in range(repeats):
            start = time.perf_counter()
            operation()
            _synchronise_if_needed(device)
            timings.append(time.perf_counter() - start)
    mean_seconds = statistics.fmean(timings)
    std_seconds = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return mean_seconds, std_seconds


def _zero_conditions(
    *,
    batch_size: int,
    condition_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if condition_dim <= 0:
        return None
    return torch.zeros(batch_size, condition_dim, device=device)


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}.")
    return cast(dict[str, Any], payload)


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping section {key!r}.")
    return value


def _mapped_name(config: Mapping[str, Any], key: str, name_map: Mapping[str, str]) -> str:
    raw_name = config.get(key)
    if not isinstance(raw_name, str):
        raise ValueError(f"Expected string config value {key!r}.")
    try:
        return name_map[raw_name]
    except KeyError as exc:
        valid_names = ", ".join(sorted(name_map))
        raise ValueError(
            f"Unsupported {key}={raw_name!r}. Expected one of: {valid_names}."
        ) from exc


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Expected a list of integers.")
    return [int(item) for item in value]


def _optional_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Expected a list of floats.")
    return [float(item) for item in value]


def _synchronise_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
