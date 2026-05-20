"""Report parameter counts for causal token-prior experiment configs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from torch import nn

from time_causal_vae.cli.train_token_prior import build_prior_config, load_token_prior_yaml
from time_causal_vae.models.discrete.priors import build_token_prior_model


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Report parameter counts for causal token-prior YAML configs.",
    )
    parser.add_argument("configs", nargs="+", help="Token-prior YAML config paths.")
    parser.add_argument(
        "--json-output",
        help="Optional JSON output path for machine-readable parameter-count rows.",
    )
    return parser


def main() -> None:
    """Load each config, instantiate its prior, and print a Markdown table."""
    parser = build_parser()
    args = parser.parse_args()
    rows = [parameter_count_row(Path(config_path)) for config_path in args.configs]
    print(markdown_table(rows))
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True)
            handle.write("\n")


def parameter_count_row(config_path: Path) -> dict[str, Any]:
    """Return one parameter-count row for a YAML config."""
    raw_config = load_token_prior_yaml(config_path)
    prior_config = build_prior_config({"model": raw_config["model"]})
    model = build_token_prior_model(prior_config)
    experiment = cast(Mapping[str, Any], raw_config["experiment"])
    model_config = cast(Mapping[str, Any], raw_config["model"])
    return {
        "name": str(experiment["name"]),
        "config": str(config_path),
        "prior_type": prior_config.prior_type,
        "token_embedding_dim": prior_config.token_embedding_dim,
        "num_layers": prior_config.num_layers,
        "mlp_hidden_dim": prior_config.mlp_hidden_dim,
        "conv_blocks": conv_block_count(model),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "tokenizer_dir": str(cast(Mapping[str, Any], raw_config["data"])["tokenizer_dir"]),
        "token_data_dir": str(cast(Mapping[str, Any], raw_config["data"])["token_data_dir"]),
        "conv_kernel_size": model_config.get("conv_kernel_size"),
        "condition_dim": model_config.get("condition_dim", 0),
    }


def conv_block_count(model: nn.Module) -> int:
    """Return the number of causal convolution blocks actually instantiated."""
    conv_preprocessor = getattr(model, "conv_preprocessor", None)
    blocks = getattr(conv_preprocessor, "blocks", None)
    if blocks is None:
        return 0
    return len(blocks)


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render rows as a compact Markdown table."""
    lines = [
        "| Config | Prior | Embed | Layers | MLP | Conv blocks | Parameters |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['name']}` | "
            f"`{row['prior_type']}` | "
            f"{row['token_embedding_dim']} | "
            f"{row['num_layers']} | "
            f"{row['mlp_hidden_dim']} | "
            f"{row['conv_blocks']} | "
            f"{int(row['total_parameters']):,} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
