"""Check factorised multi-code token-prior prefix causality."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from torch import Tensor

from time_causal_vae.token_prior import (
    CausalTokenPriorConfig,
    FactorisedMultiCodeTokenPrior,
    assert_multicode_token_prior_no_future_leakage,
)
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the no-leakage check parser."""
    parser = argparse.ArgumentParser(
        description="Check factorised multi-code causal token-prior no-future-leakage.",
    )
    parser.add_argument("--config", help="Optional token-prior YAML config.")
    parser.add_argument("--token-data-dir", help="Optional extracted token artifact directory.")
    parser.add_argument("--cutoff", type=int, default=29, help="Inclusive zero-indexed cutoff.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the check.")
    parser.add_argument("--seed", type=int, default=99, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Device for the check.")
    return parser


def main() -> int:
    """Run a synthetic or token-data backed multi-code no-leakage check."""
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.config is not None and Path(args.config).exists():
        config = load_prior_config(args.config)
    elif args.config is not None:
        print(f"SKIP config-backed check because config does not exist: {args.config}")
        config = synthetic_prior_config()
    else:
        config = synthetic_prior_config()

    if config.prior_type != "factorised_multi_code":
        raise SystemExit("This check requires prior_type='factorised_multi_code'.")
    model = FactorisedMultiCodeTokenPrior(config).to(device)

    tokens, conditions, source = load_or_make_batch(
        config=config,
        token_data_dir=args.token_data_dir,
        batch_size=args.batch_size,
        device=device,
    )
    try:
        output = model(tokens, conditions=conditions)
        reference_logits, changed_logits = assert_multicode_token_prior_no_future_leakage(
            model,
            tokens,
            args.cutoff,
            conditions=conditions,
        )
    except Exception as exc:
        print(f"FAIL factorised multi-code token-prior no-leakage check: {exc}")
        return 1

    prefix = slice(None, args.cutoff + 1)
    max_prefix_diff = (reference_logits[:, prefix, :, :] - changed_logits[:, prefix, :, :]).abs()
    component_shapes = {
        name: list(logits.shape)
        for name, logits in cast(dict[str, Tensor], output.logits_by_component).items()
    }
    print("PASS factorised multi-code token-prior no-leakage check")
    print(f"source={source}")
    print(f"prior_type={config.prior_type}")
    print(f"tokens={tuple(tokens.shape)}")
    print(f"conditions={None if conditions is None else tuple(conditions.shape)}")
    print(f"logits={tuple(reference_logits.shape)}")
    print(f"logits_by_component={component_shapes}")
    print(f"cutoff={args.cutoff}")
    print(f"loss={output.loss.item():.8f}")
    print(f"cross_entropy={output.cross_entropy.item():.8f}")
    print(f"accuracy={output.accuracy.item():.8f}")
    print(f"perplexity={output.perplexity.item():.8f}")
    for component_name in cast(dict[str, Tensor], output.logits_by_component):
        print(
            f"{component_name}: "
            f"ce={output[f'component_cross_entropy_{component_name}'].item():.8f} "
            f"accuracy={output[f'component_accuracy_{component_name}'].item():.8f} "
            f"perplexity={output[f'component_perplexity_{component_name}'].item():.8f}"
        )
    print(f"max_prefix_diff={float(max_prefix_diff.max().item()):.8e}")
    return 0


def synthetic_prior_config() -> CausalTokenPriorConfig:
    """Return a small RVQ q2-style synthetic config."""
    return CausalTokenPriorConfig(
        codebook_size=64,
        sequence_length=60,
        token_embedding_dim=128,
        num_layers=2,
        num_heads=4,
        mlp_hidden_dim=256,
        dropout=0.1,
        condition_dim=1,
        condition_injection="additive",
        prior_type="factorised_multi_code",
        index_shape=[60, 2],
        num_quantizers=2,
        groups=1,
    )


def load_prior_config(path: str | Path) -> CausalTokenPriorConfig:
    """Load a factorised token-prior config from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    model_config = loaded.get("model")
    if not isinstance(model_config, dict):
        raise SystemExit(f"Config requires a model section: {path}")
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
    )


def load_or_make_batch(
    *,
    config: CausalTokenPriorConfig,
    token_data_dir: str | None,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor | None, str]:
    """Load a real token batch if available, otherwise create synthetic data."""
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if token_data_dir is not None:
        train_path = Path(token_data_dir) / "train_tokens.pt"
        if not train_path.exists():
            raise SystemExit(f"Missing train token artifact: {train_path}")
        loaded = torch.load(train_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or "indices" not in loaded:
            raise SystemExit(f"Token artifact must contain indices: {train_path}")
        tokens = cast(Tensor, loaded["indices"]).long()[:batch_size].to(device)
        labels = cast(Tensor | None, loaded.get("labels"))
        conditions = None
        if config.condition_injection != "none":
            if labels is None:
                raise SystemExit(f"Conditional prior requires labels in {train_path}")
            conditions = labels.float()[:batch_size]
            if conditions.ndim == 1:
                conditions = conditions[:, None]
            conditions = conditions.to(device)
        return tokens, conditions, f"token_data:{token_data_dir}"

    tokens = torch.randint(
        low=0,
        high=config.codebook_size,
        size=(batch_size, config.sequence_length, *config.component_shape),
        dtype=torch.long,
        device=device,
    )
    conditions = None
    if config.condition_injection != "none":
        conditions = torch.randn(batch_size, config.condition_dim, device=device)
    return tokens, conditions, "synthetic_random_tokens"


def optional_int(value: Any) -> int | None:
    """Return an optional integer config value."""
    if value is None:
        return None
    return int(value)


def optional_int_list(value: Any) -> list[int] | None:
    """Return an optional integer list."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise SystemExit("index_shape must be a list when provided.")
    return [int(item) for item in value]


def optional_float_list(value: Any) -> list[float] | None:
    """Return an optional float list."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise SystemExit("component_loss_weights must be a list when provided.")
    return [float(item) for item in value]


def parse_prior_type(value: Any) -> Literal["single_code", "factorised_multi_code"]:
    """Parse the supported prior type."""
    parsed = str(value)
    if parsed not in {"single_code", "factorised_multi_code"}:
        raise SystemExit("prior_type must be 'single_code' or 'factorised_multi_code'.")
    return cast(Literal["single_code", "factorised_multi_code"], parsed)


def parse_condition_injection(value: Any) -> Literal["none", "additive", "adaln_lite"]:
    """Parse the supported condition-injection mode."""
    parsed = str(value)
    if parsed not in {"none", "additive", "adaln_lite"}:
        raise SystemExit("condition_injection must be 'none', 'additive', or 'adaln_lite'.")
    return cast(Literal["none", "additive", "adaln_lite"], parsed)


if __name__ == "__main__":
    raise SystemExit(main())
