"""Check multi-code token-prior prefix causality and tokenizer decoding."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import Tensor, nn

from time_causal_vae.evaluation.token_prior import load_trained_token_prior
from time_causal_vae.evaluation.tokenizer import load_trained_tokenizer
from time_causal_vae.models.discrete.priors import CausalTokenPriorConfig, build_token_prior_model
from time_causal_vae.models.discrete.priors.causal_transformer import (
    FactorisedMultiCodeTokenPrior,
    HierarchicalRVQ2TokenPrior,
    assert_hierarchical_rvq_prior_no_future_leakage,
    assert_multicode_token_prior_no_future_leakage,
)
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the script parser."""
    parser = argparse.ArgumentParser(
        description="Check q2 multi-code token-prior no-leakage and decode smoke.",
    )
    parser.add_argument("--config", required=True, help="Multi-code token-prior YAML config.")
    parser.add_argument(
        "--prior-dir",
        help="Optional trained prior directory; omitted means source-level random init.",
    )
    parser.add_argument(
        "--token-data-dir",
        help="Optional token data override; defaults to data.token_data_dir from config.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        help="Optional tokenizer dir override; defaults to data.tokenizer_dir from config.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the check.")
    parser.add_argument("--cutoff", type=int, default=29, help="Inclusive output-logit cutoff.")
    parser.add_argument("--seed", type=int, default=99, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Device for the check.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, help="Optional top-k sampling.")
    return parser


def main() -> int:
    """Run the multi-code prior no-leakage smoke."""
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    raw_config = load_yaml_mapping(args.config)
    prior_config = build_prior_config(raw_config)
    if prior_config.prior_type not in {"factorised_multi_code", "hierarchical_rvq_q2"}:
        raise SystemExit(
            "check_multicode_token_prior_no_leakage requires prior_type "
            "'factorised_multi_code' or 'hierarchical_rvq_q2'."
        )
    model, prior_config = load_or_build_prior(
        prior_config=prior_config,
        prior_dir=args.prior_dir,
        device=device,
    )
    tokens, conditions, token_source = load_or_make_batch(
        raw_config,
        token_data_dir=args.token_data_dir,
        prior_config=prior_config,
        batch_size=args.batch_size,
        device=device,
    )
    output = model(tokens, conditions=conditions)
    loss = cast(Tensor, output["loss"])
    logits = cast(Tensor, output["logits"])
    if not torch.isfinite(loss):
        raise AssertionError(f"Expected finite loss; got {loss.item()}.")
    expected_logits_shape = (
        tokens.shape[0],
        prior_config.sequence_length,
        prior_config.component_count,
        prior_config.codebook_size,
    )
    if tuple(logits.shape) != expected_logits_shape:
        raise AssertionError(
            f"Expected logits shape {expected_logits_shape}; got {tuple(logits.shape)}."
        )

    reference_logits, changed_logits = assert_no_future_leakage(
        model,
        tokens,
        cutoff=args.cutoff,
        conditions=conditions,
    )
    prefix = slice(None, args.cutoff + 1)
    max_prefix_diff = float(
        (reference_logits[:, prefix] - changed_logits[:, prefix]).abs().max().item()
    )
    reference_predictions = reference_logits[:, prefix].argmax(dim=-1)
    changed_predictions = changed_logits[:, prefix].argmax(dim=-1)
    if not torch.equal(reference_predictions, changed_predictions):
        raise AssertionError("Prefix predictions changed after future-token perturbation.")

    sampled = model.sample(
        batch_size=tokens.shape[0],
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditions=conditions,
    )
    expected_sample_shape = (
        tokens.shape[0],
        prior_config.sequence_length,
        *prior_config.component_shape,
    )
    if tuple(sampled.shape) != expected_sample_shape:
        raise AssertionError(
            f"Expected sampled shape {expected_sample_shape}; got {tuple(sampled.shape)}."
        )

    tokenizer_dir = tokenizer_dir_from_config(raw_config, override=args.tokenizer_dir)
    tokenizer, _tokenizer_config, _checkpoint = load_trained_tokenizer(tokenizer_dir, device=device)
    tokenizer.eval()
    quantized = tokenizer.quantizer.decode_indices(sampled)
    decoded = tokenizer.decode_indices(sampled, conditions=conditions)
    if decoded.shape[:2] != sampled.shape[:2]:
        raise AssertionError(
            f"Decoded paths must share batch/time shape with samples; got {tuple(decoded.shape)}."
        )

    print("PASS multi-code token-prior no-leakage check")
    print(f"config={args.config}")
    print(f"prior_dir={args.prior_dir or 'random_initialization'}")
    print(f"token_data_dir={token_source}")
    print(f"tokenizer_dir={tokenizer_dir}")
    print(f"prior_type={prior_config.prior_type}")
    print(f"tokens={tuple(tokens.shape)}")
    print(f"conditions={None if conditions is None else tuple(conditions.shape)}")
    print(f"logits={tuple(logits.shape)}")
    print(f"samples={tuple(sampled.shape)}")
    print(f"quantized={tuple(quantized.shape)}")
    print(f"decoded={tuple(decoded.shape)}")
    print(f"cutoff={args.cutoff}")
    print(f"cross_entropy={float(cast(Tensor, output['cross_entropy']).detach().cpu()):.8f}")
    print(f"accuracy={float(cast(Tensor, output['accuracy']).detach().cpu()):.8f}")
    print(f"perplexity={float(cast(Tensor, output['perplexity']).detach().cpu()):.8f}")
    print(f"max_prefix_diff={max_prefix_diff:.8e}")
    return 0


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def build_prior_config(raw_config: Mapping[str, Any]) -> CausalTokenPriorConfig:
    """Build a token-prior config from a YAML mapping."""
    model = require_mapping(raw_config, "model")
    return CausalTokenPriorConfig(
        codebook_size=int(model["codebook_size"]),
        sequence_length=int(model["sequence_length"]),
        token_embedding_dim=int(model["token_embedding_dim"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        mlp_hidden_dim=int(model["mlp_hidden_dim"]),
        dropout=float(model.get("dropout", 0.0)),
        bos_token_id=optional_int(model.get("bos_token_id")),
        pad_token_id=optional_int(model.get("pad_token_id")),
        prediction_convention=str(model.get("prediction_convention", "bos_shifted_next_token")),
        condition_dim=int(model.get("condition_dim", 0)),
        condition_injection=str(model.get("condition_injection", "none")),  # type: ignore[arg-type]
        condition_hidden_dim=optional_int(model.get("condition_hidden_dim")),
        adaln_hidden_dim=optional_int(model.get("adaln_hidden_dim")),
        prior_type=str(model.get("prior_type", "single_code")),  # type: ignore[arg-type]
        index_shape=optional_int_list(model.get("index_shape")),
        num_quantizers=int(model.get("num_quantizers", 1)),
        groups=int(model.get("groups", 1)),
        component_loss_weights=optional_float_list(model.get("component_loss_weights")),
        conv_num_layers=int(model.get("conv_num_layers", 0)),
        conv_kernel_size=int(model.get("conv_kernel_size", 3)),
        conv_dilations=optional_int_list(model.get("conv_dilations")),
        conv_dropout=float(model.get("conv_dropout", 0.0)),
    )


def load_or_build_prior(
    *,
    prior_config: CausalTokenPriorConfig,
    prior_dir: str | None,
    device: torch.device,
) -> tuple[nn.Module, CausalTokenPriorConfig]:
    """Load a trained prior or construct a source-level random model."""
    if prior_dir is None:
        return build_token_prior_model(prior_config).to(device), prior_config
    prior, loaded_config, _checkpoint = load_trained_token_prior(Path(prior_dir), device=device)
    return prior, loaded_config


def assert_no_future_leakage(
    model: nn.Module,
    tokens: Tensor,
    *,
    cutoff: int,
    conditions: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Dispatch to the appropriate multi-code no-leakage assertion."""
    if isinstance(model, FactorisedMultiCodeTokenPrior):
        return assert_multicode_token_prior_no_future_leakage(
            model,
            tokens,
            cutoff,
            conditions=conditions,
        )
    if isinstance(model, HierarchicalRVQ2TokenPrior):
        return assert_hierarchical_rvq_prior_no_future_leakage(
            model,
            tokens,
            cutoff,
            conditions=conditions,
        )
    raise TypeError(f"Unsupported multi-code prior module: {type(model).__name__}")


def load_or_make_batch(
    raw_config: Mapping[str, Any],
    *,
    token_data_dir: str | None,
    prior_config: CausalTokenPriorConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor | None, str]:
    """Load extracted q2 token data or create a synthetic q2 batch."""
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    directory = (
        Path(token_data_dir)
        if token_data_dir is not None
        else token_data_dir_from_config(raw_config)
    )
    if directory is None:
        tokens = torch.randint(
            low=0,
            high=prior_config.codebook_size,
            size=(batch_size, prior_config.sequence_length, *prior_config.component_shape),
            dtype=torch.long,
            device=device,
        )
        conditions = synthetic_conditions(prior_config, batch_size=batch_size, device=device)
        return tokens, conditions, "synthetic_random_tokens"

    payload_path = directory / "train_tokens.pt"
    if not payload_path.exists():
        raise SystemExit(f"Missing token payload: {payload_path}")
    payload = cast(
        Mapping[str, Tensor],
        torch.load(payload_path, map_location="cpu", weights_only=True),
    )
    tokens = payload["indices"][:batch_size].long().to(device)
    expected_token_shape = (batch_size, prior_config.sequence_length, *prior_config.component_shape)
    if tuple(tokens.shape) != expected_token_shape:
        raise SystemExit(f"Expected token batch {expected_token_shape}; got {tuple(tokens.shape)}.")
    conditions = None
    if prior_config.condition_injection != "none":
        if "labels" not in payload:
            raise SystemExit(f"Conditional prior requires labels in {payload_path}.")
        conditions = payload["labels"][:batch_size].to(device=device, dtype=torch.float32)
        if conditions.ndim == 1:
            conditions = conditions[:, None]
        if conditions.shape[-1] != prior_config.condition_dim:
            raise SystemExit(
                f"Expected condition_dim={prior_config.condition_dim}; got {conditions.shape[-1]}."
            )
    return tokens, conditions, str(payload_path)


def synthetic_conditions(
    prior_config: CausalTokenPriorConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor | None:
    """Return synthetic scalar conditions when conditioning is enabled."""
    if prior_config.condition_injection == "none":
        return None
    return torch.randn(batch_size, prior_config.condition_dim, device=device)


def token_data_dir_from_config(raw_config: Mapping[str, Any]) -> Path | None:
    """Return data.token_data_dir when present."""
    data = raw_config.get("data")
    if not isinstance(data, Mapping):
        return None
    raw_path = data.get("token_data_dir")
    if raw_path is None:
        return None
    return Path(str(raw_path))


def tokenizer_dir_from_config(raw_config: Mapping[str, Any], *, override: str | None) -> Path:
    """Return the tokenizer directory for decode smoke."""
    if override is not None:
        return Path(override)
    data = require_mapping(raw_config, "data")
    raw_path = data.get("tokenizer_dir")
    if raw_path is None:
        raise SystemExit("Config requires data.tokenizer_dir or --tokenizer-dir for decode smoke.")
    return Path(str(raw_path))


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required YAML section."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def optional_int(value: Any) -> int | None:
    """Return an optional integer."""
    if value is None:
        return None
    return int(value)


def optional_int_list(value: Any) -> list[int] | None:
    """Return an optional integer list."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise SystemExit("Expected a list of integers.")
    return [int(item) for item in value]


def optional_float_list(value: Any) -> list[float] | None:
    """Return an optional float list."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise SystemExit("Expected a list of floats.")
    return [float(item) for item in value]


if __name__ == "__main__":
    raise SystemExit(main())
