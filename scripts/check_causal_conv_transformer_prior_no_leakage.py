"""Check causal-convolution transformer prior prefix causality."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from time_causal_vae.token_prior import (
    CausalConvTransformerPrior,
    CausalTokenPriorConfig,
    assert_token_prior_no_future_leakage,
)
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the no-leakage check parser."""
    parser = argparse.ArgumentParser(
        description="Check causal conv-transformer token-prior no-future-leakage.",
    )
    parser.add_argument("--config", required=True, help="Token-prior YAML config.")
    parser.add_argument("--batch-size", type=int, default=8, help="Synthetic batch size.")
    parser.add_argument("--cutoff", type=int, default=29, help="Inclusive output-logit cutoff.")
    parser.add_argument("--seed", type=int, default=99, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Device for the check.")
    return parser


def main() -> int:
    """Run the synthetic no-leakage check."""
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    config = load_prior_config(args.config)
    if config.prior_type != "causal_conv_transformer":
        raise SystemExit("This check requires prior_type='causal_conv_transformer'.")
    if config.condition_injection != "additive" or config.condition_dim != 1:
        raise SystemExit("This check expects additive scalar VIX conditioning.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    model = CausalConvTransformerPrior(config).to(device)
    tokens = torch.randint(
        low=0,
        high=config.codebook_size,
        size=(args.batch_size, config.sequence_length),
        dtype=torch.long,
        device=device,
    )
    conditions = torch.randn(args.batch_size, config.condition_dim, device=device)
    model.eval()

    try:
        output = model(tokens, conditions=conditions)
        reference_logits, changed_logits = assert_token_prior_no_future_leakage(
            model,
            tokens,
            args.cutoff,
            conditions=conditions,
            atol=1e-6,
            rtol=1e-5,
        )
    except Exception as exc:
        print(f"FAIL causal conv-transformer prior no-leakage check: {exc}")
        return 1

    prefix = slice(None, args.cutoff + 1)
    max_prefix_diff = (reference_logits[:, prefix, :] - changed_logits[:, prefix, :]).abs()
    print("PASS causal conv-transformer prior no-leakage check")
    print(f"config={args.config}")
    print(f"prior_type={config.prior_type}")
    print(f"tokens={tuple(tokens.shape)}")
    print(f"conditions={tuple(conditions.shape)}")
    print(f"logits={tuple(output.logits.shape)}")
    print(f"cutoff={args.cutoff}")
    print(f"conv_num_layers={config.conv_num_layers}")
    print(f"conv_kernel_size={config.conv_kernel_size}")
    print(f"conv_dilations={config.conv_dilations}")
    print(f"conv_dropout={config.conv_dropout}")
    print(f"cross_entropy={output.cross_entropy.item():.8f}")
    print(f"accuracy={output.accuracy.item():.8f}")
    print(f"perplexity={output.perplexity.item():.8f}")
    print(f"max_prefix_diff={float(max_prefix_diff.max().item()):.8e}")
    return 0


def load_prior_config(path: str | Path) -> CausalTokenPriorConfig:
    """Load a causal conv-transformer token-prior config from YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {config_path}")
    model_config = loaded.get("model")
    if not isinstance(model_config, dict):
        raise SystemExit(f"Config requires a model section: {config_path}")
    return CausalTokenPriorConfig(
        codebook_size=int(model_config["codebook_size"]),
        sequence_length=int(model_config["sequence_length"]),
        token_embedding_dim=int(model_config["token_embedding_dim"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        mlp_hidden_dim=int(model_config["mlp_hidden_dim"]),
        dropout=float(model_config.get("dropout", 0.0)),
        bos_token_id=optional_int(model_config.get("bos_token_id")),
        pad_token_id=optional_int(model_config.get("pad_token_id")),
        prediction_convention=str(
            model_config.get("prediction_convention", "bos_shifted_next_token")
        ),
        condition_dim=int(model_config.get("condition_dim", 0)),
        condition_injection=parse_condition_injection(
            model_config.get("condition_injection", "none")
        ),
        condition_hidden_dim=optional_int(model_config.get("condition_hidden_dim")),
        adaln_hidden_dim=optional_int(model_config.get("adaln_hidden_dim")),
        prior_type=parse_prior_type(model_config.get("prior_type", "single_code")),
        index_shape=optional_int_list(model_config.get("index_shape")),
        num_quantizers=int(model_config.get("num_quantizers", 1)),
        groups=int(model_config.get("groups", 1)),
        component_loss_weights=optional_float_list(model_config.get("component_loss_weights")),
        conv_num_layers=int(model_config.get("conv_num_layers", 0)),
        conv_kernel_size=int(model_config.get("conv_kernel_size", 3)),
        conv_dilations=optional_int_list(model_config.get("conv_dilations")),
        conv_dropout=float(model_config.get("conv_dropout", 0.0)),
    )


def optional_int(value: Any) -> int | None:
    """Return an optional integer config value."""
    if value is None:
        return None
    return int(value)


def optional_int_list(value: Any) -> list[int] | None:
    """Return an optional list of integers."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise SystemExit("Expected a list of integers.")
    return [int(item) for item in value]


def optional_float_list(value: Any) -> list[float] | None:
    """Return an optional list of floats."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise SystemExit("Expected a list of floats.")
    return [float(item) for item in value]


def parse_prior_type(
    value: Any,
) -> Literal["single_code", "causal_conv_transformer"]:
    """Parse the supported prior type for this check."""
    parsed = str(value)
    if parsed not in {"single_code", "causal_conv_transformer"}:
        raise SystemExit("prior_type must be 'single_code' or 'causal_conv_transformer'.")
    return cast(Literal["single_code", "causal_conv_transformer"], parsed)


def parse_condition_injection(value: Any) -> Literal["none", "additive", "adaln_lite"]:
    """Parse the supported condition-injection mode."""
    parsed = str(value)
    if parsed not in {"none", "additive", "adaln_lite"}:
        raise SystemExit("condition_injection must be 'none', 'additive', or 'adaln_lite'.")
    return cast(Literal["none", "additive", "adaln_lite"], parsed)


if __name__ == "__main__":
    raise SystemExit(main())
