"""Extract frozen tokenizer index datasets for causal token-prior training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from time_causal_vae.token_prior.data import (
    build_tokenizer_datasets,
    extract_dataset_tokens,
    load_frozen_tokenizer,
    load_tokenizer_experiment_config,
    save_token_dataset,
    token_dataset_summary,
)
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the token-index extraction CLI parser."""
    parser = argparse.ArgumentParser(
        description="Extract train/eval tokenizer indices for causal token-prior training.",
    )
    parser.add_argument("--config", required=True, help="Tokenizer experiment YAML config.")
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        help="Directory containing tokenizer.pt and tokenizer_config.json.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory under outputs/.")
    parser.add_argument(
        "--n-sample",
        type=int,
        help="Optional override for train and eval dataset sample count.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data generation.")
    parser.add_argument("--device", help="Device override, for example cpu or cuda.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory for datasets that require local files.",
    )
    return parser


def main() -> None:
    """Run token-index extraction."""
    args = build_parser().parse_args()
    if args.n_sample is not None and args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive when provided.")

    output_dir = validate_output_dir(args.output_dir)
    set_seed(args.seed)
    device = select_device(args.device)
    raw_config = load_tokenizer_experiment_config(args.config)
    train_dataset, eval_dataset = build_tokenizer_datasets(
        raw_config,
        n_sample=args.n_sample,
        base_data_dir=args.base_data_dir,
    )
    tokenizer, tokenizer_config = load_frozen_tokenizer(args.tokenizer_dir, device=device)
    train_payload = extract_dataset_tokens(tokenizer, train_dataset, device=device)
    eval_payload = extract_dataset_tokens(tokenizer, eval_dataset, device=device)
    summary = token_dataset_summary(
        tokenizer_dir=args.tokenizer_dir,
        config_path=args.config,
        tokenizer_config=tokenizer_config,
        train_payload=train_payload,
        eval_payload=eval_payload,
        seed=args.seed,
        n_sample=args.n_sample,
    )
    save_token_dataset(
        output_dir,
        train_payload=train_payload,
        eval_payload=eval_payload,
        summary=summary,
    )

    print("Token index extraction complete.")
    print(f"tokenizer_dir: {args.tokenizer_dir}")
    print(f"output_dir: {output_dir}")
    print(f"train_indices_shape: {summary['train_token_shape']}")
    print(f"eval_indices_shape: {summary['eval_token_shape']}")
    print(f"codebook_size: {summary['codebook_size']}")
    print(f"sequence_length: {summary['sequence_length']}")
    print(
        "combined_codebook: "
        f"active={summary['combined']['active_code_count']}/{summary['codebook_size']} "
        f"ratio={summary['combined']['active_code_ratio']:.8f} "
        f"perplexity={summary['combined']['codebook_perplexity']:.8f} "
        f"entropy={summary['combined']['index_entropy']:.8f}"
    )
    print("files: train_tokens.pt, eval_tokens.pt, token_dataset_summary.json")


def validate_output_dir(output_dir: str) -> Path:
    """Validate that token-prior artifacts stay below ignored outputs/."""
    path = Path(output_dir)
    resolved = path.resolve()
    outputs_root = (Path.cwd() / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise SystemExit(
            f"--output-dir must be under ignored outputs/. Received: {output_dir}"
        ) from exc
    return path


def select_device(device_name: str | None) -> torch.device:
    """Select the requested device, or prefer CUDA when available."""
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
