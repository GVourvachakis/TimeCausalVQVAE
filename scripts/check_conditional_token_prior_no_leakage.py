"""Check conditional causal token-prior prefix causality."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from torch import Tensor

from time_causal_vae.evaluation.token_prior import load_trained_token_prior
from time_causal_vae.models.discrete.priors import (
    CausalTokenPriorConfig,
    CausalTokenTransformerPrior,
)
from time_causal_vae.models.discrete.priors.causal_transformer import (
    assert_token_prior_no_future_leakage,
)
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the script parser."""
    parser = argparse.ArgumentParser(
        description="Check scalar or temporal conditional causal token-prior no leakage.",
    )
    parser.add_argument("--config", help="Optional token-prior YAML config.")
    parser.add_argument(
        "--prior-dir",
        help="Optional trained prior directory containing token_prior.pt.",
    )
    parser.add_argument(
        "--token-data-dir",
        help="Optional extracted token dataset directory containing train_tokens.pt.",
    )
    parser.add_argument("--cutoff", type=int, default=29, help="Inclusive output-logit cutoff.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the check.")
    parser.add_argument("--seed", type=int, default=99, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Device for the check.")
    parser.add_argument(
        "--condition-injection",
        choices=["none", "additive", "adaln_lite"],
        help="Override condition injection for synthetic or source smoke checks.",
    )
    return parser


def main() -> int:
    """Run the conditional token-prior no-leakage check."""
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    config = load_prior_config(args.config, condition_injection_override=args.condition_injection)
    model: CausalTokenTransformerPrior
    if args.prior_dir is None:
        model = CausalTokenTransformerPrior(config).to(device)
    else:
        model, config, _checkpoint = load_trained_token_prior(Path(args.prior_dir), device=device)
        if args.condition_injection is not None:
            raise SystemExit("--condition-injection cannot override a trained checkpoint.")
    tokens, conditions, source = load_or_make_batch(
        token_data_dir=args.token_data_dir,
        config=config,
        batch_size=args.batch_size,
        device=device,
    )
    model.eval()

    try:
        output = model(tokens, conditions=conditions)
        if not torch.isfinite(cast(Tensor, output.loss)):
            raise AssertionError(f"Expected finite loss; got {output.loss.item()}.")
        if tuple(output.logits.shape) != (
            tokens.shape[0],
            config.sequence_length,
            config.codebook_size,
        ):
            raise AssertionError(f"Unexpected logits shape: {tuple(output.logits.shape)}.")
        reference_logits, changed_logits = assert_token_prior_no_future_leakage(
            model,
            tokens,
            args.cutoff,
            conditions=conditions,
            atol=1e-6,
            rtol=1e-5,
        )
        max_prefix_diff = (
            reference_logits[:, : args.cutoff + 1] - changed_logits[:, : args.cutoff + 1]
        ).abs()
        max_prefix_diff_value = float(max_prefix_diff.max().item())
        samples = model.sample(
            batch_size=tokens.shape[0],
            device=device,
            temperature=1.0,
            top_k=None,
            conditions=conditions,
        )
    except Exception as exc:
        print(f"FAIL conditional token-prior no-leakage check: {exc}")
        return 1

    print("PASS conditional token-prior no-leakage check")
    print(f"source={source}")
    print(f"config={args.config or 'synthetic_default'}")
    print(f"prior_dir={args.prior_dir or 'random_initialization'}")
    print(f"token_data_dir={args.token_data_dir or 'synthetic_random_tokens'}")
    print(f"tokens={tuple(tokens.shape)}")
    print(f"conditions={tuple(conditions.shape) if conditions is not None else None}")
    print(f"condition_injection={config.condition_injection}")
    print(f"condition_dim={config.condition_dim}")
    print(f"logits={tuple(output.logits.shape)}")
    print(f"samples={tuple(samples.shape)}")
    print(f"cutoff={args.cutoff}")
    print(f"cross_entropy={output.cross_entropy.item():.8f}")
    print(f"accuracy={output.accuracy.item():.8f}")
    print(f"perplexity={output.perplexity.item():.8f}")
    print(f"max_prefix_diff={max_prefix_diff_value:.8e}")
    return 0


def load_prior_config(
    config_path: str | None,
    *,
    condition_injection_override: str | None,
) -> CausalTokenPriorConfig:
    """Load config from YAML or return a synthetic scalar conditional config."""
    if config_path is None:
        return synthetic_config(condition_injection_override)
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Token-prior config must be a mapping: {config_path}")
    model = require_mapping(cast(Mapping[str, Any], loaded), "model")
    config = CausalTokenPriorConfig(
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
        condition_injection=condition_injection(model.get("condition_injection", "none")),
        condition_hidden_dim=optional_int(model.get("condition_hidden_dim")),
        adaln_hidden_dim=optional_int(model.get("adaln_hidden_dim")),
    )
    if condition_injection_override is None:
        return config
    override = condition_injection(condition_injection_override)
    if override == "none":
        return replace(config, condition_injection=override, condition_dim=0)
    condition_dim = config.condition_dim if config.condition_dim > 0 else 1
    return replace(config, condition_injection=override, condition_dim=condition_dim)


def synthetic_config(condition_injection_override: str | None) -> CausalTokenPriorConfig:
    """Return a small scalar conditional config for source-level smoke tests."""
    injection = condition_injection(condition_injection_override or "additive")
    condition_dim = 0 if injection == "none" else 1
    return CausalTokenPriorConfig(
        codebook_size=64,
        sequence_length=60,
        token_embedding_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_hidden_dim=128,
        dropout=0.0,
        condition_dim=condition_dim,
        condition_injection=injection,
    )


def load_or_make_batch(
    *,
    token_data_dir: str | None,
    config: CausalTokenPriorConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor | None, str]:
    """Load extracted tokens/labels or create a synthetic scalar-conditional batch."""
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if token_data_dir is None:
        tokens = torch.randint(
            low=0,
            high=config.codebook_size,
            size=(batch_size, config.sequence_length),
            dtype=torch.long,
            device=device,
        )
        conditions = None
        if config.condition_injection != "none":
            conditions = torch.randn(batch_size, config.condition_dim, device=device)
        return tokens, conditions, "synthetic_random"

    payload_path = Path(token_data_dir) / "train_tokens.pt"
    if not payload_path.exists():
        raise SystemExit(f"Missing token payload: {payload_path}")
    payload = cast(
        Mapping[str, Tensor],
        torch.load(payload_path, map_location="cpu", weights_only=True),
    )
    tokens = payload["indices"][:batch_size].long().to(device)
    conditions = None
    if config.condition_injection != "none":
        if "labels" not in payload:
            raise SystemExit("Token payload does not contain labels required for conditioning.")
        conditions = payload["labels"][:batch_size].to(device=device, dtype=torch.float32)
    return tokens, conditions, str(payload_path)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required YAML section as a dictionary."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Token-prior config requires a mapping section named {key!r}.")
    return cast(dict[str, Any], value)


def optional_int(value: Any) -> int | None:
    """Return an optional integer config value."""
    if value is None:
        return None
    return int(value)


def condition_injection(value: Any) -> Literal["none", "additive", "adaln_lite"]:
    """Return a validated condition-injection value."""
    typed_value = str(value)
    if typed_value not in {"none", "additive", "adaln_lite"}:
        raise SystemExit("condition_injection must be 'none', 'additive', or 'adaln_lite'.")
    return cast(Literal["none", "additive", "adaln_lite"], typed_value)


if __name__ == "__main__":
    raise SystemExit(main())
