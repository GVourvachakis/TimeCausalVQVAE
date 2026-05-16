"""Build paired low/high token datasets for separate frequency-token prior training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

EXPECTED_TOKEN_SHAPE = (2457, 60)
EXPECTED_COMPONENT_SHAPE = (2457, 60, 1)


def build_parser() -> argparse.ArgumentParser:
    """Build the paired-token dataset CLI parser."""
    parser = argparse.ArgumentParser(
        description="Assemble matched low/high token streams from separate tokenizer exports.",
    )
    parser.add_argument("--low-token-dir", required=True, help="Low-token extraction directory.")
    parser.add_argument("--high-token-dir", required=True, help="High-token extraction directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory under outputs/.")
    parser.add_argument(
        "--base-data-dir",
        required=True,
        help="Base data directory used for the source scalar dataset.",
    )
    parser.add_argument("--alpha", required=True, type=float, help="Causal EMA alpha value.")
    return parser


def main() -> None:
    """Assemble, validate, and save paired low/high token datasets."""
    args = build_parser().parse_args()
    output_dir = validate_output_dir(args.output_dir)
    base_data_dir = Path(args.base_data_dir)

    low_dir = Path(args.low_token_dir)
    high_dir = Path(args.high_token_dir)
    low_train = load_payload(low_dir / "train_tokens.pt")
    low_eval = load_payload(low_dir / "eval_tokens.pt")
    high_train = load_payload(high_dir / "train_tokens.pt")
    high_eval = load_payload(high_dir / "eval_tokens.pt")
    low_summary = load_summary(low_dir / "token_dataset_summary.json")
    high_summary = load_summary(high_dir / "token_dataset_summary.json")

    train_pair = validate_pair_payloads(
        split="train", low_payload=low_train, high_payload=high_train
    )
    eval_pair = validate_pair_payloads(split="eval", low_payload=low_eval, high_payload=high_eval)
    low_codebook_size = int(low_summary["codebook_size"])
    high_codebook_size = int(high_summary["codebook_size"])

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_pair.low_tokens, output_dir / "train_low_tokens.pt")
    torch.save(train_pair.high_tokens, output_dir / "train_high_tokens.pt")
    torch.save(eval_pair.low_tokens, output_dir / "eval_low_tokens.pt")
    torch.save(eval_pair.high_tokens, output_dir / "eval_high_tokens.pt")
    torch.save(train_pair.labels, output_dir / "train_labels.pt")
    torch.save(eval_pair.labels, output_dir / "eval_labels.pt")
    torch.save(train_pair.recomposed_data, output_dir / "train_data.pt")
    torch.save(eval_pair.recomposed_data, output_dir / "eval_data.pt")

    combined_low = torch.cat([train_pair.low_tokens, eval_pair.low_tokens], dim=0)
    combined_high = torch.cat([train_pair.high_tokens, eval_pair.high_tokens], dim=0)
    summary = build_dataset_summary(
        low_dir=low_dir,
        high_dir=high_dir,
        output_dir=output_dir,
        base_data_dir=base_data_dir,
        alpha=args.alpha,
        train_pair=train_pair,
        eval_pair=eval_pair,
        combined_low=combined_low,
        combined_high=combined_high,
        low_codebook_size=low_codebook_size,
        high_codebook_size=high_codebook_size,
        low_summary=low_summary,
        high_summary=high_summary,
    )
    write_json(output_dir / "paired_token_dataset_summary.json", summary)
    (output_dir / "paired_token_dataset_summary.md").write_text(
        render_markdown_summary(summary),
        encoding="utf-8",
    )

    print("Separate frequency token dataset complete.")
    print(f"output_dir: {output_dir}")
    print(f"train_low_tokens_shape: {summary['shapes']['train_low_tokens']}")
    print(f"train_high_tokens_shape: {summary['shapes']['train_high_tokens']}")
    print(
        "combined_low: "
        f"active={summary['low_usage']['combined']['active_code_count']}/"
        f"{low_codebook_size} "
        f"perplexity={summary['low_usage']['combined']['codebook_perplexity']:.8f}"
    )
    print(
        "combined_high: "
        f"active={summary['high_usage']['combined']['active_code_count']}/"
        f"{high_codebook_size} "
        f"perplexity={summary['high_usage']['combined']['codebook_perplexity']:.8f}"
    )
    print(
        "combined_pairs: "
        f"active={summary['same_time_pairs']['combined']['active_pair_count']}/"
        f"{low_codebook_size * high_codebook_size} "
        f"perplexity={summary['same_time_pairs']['combined']['pair_perplexity']:.8f}"
    )


class PairedPayload:
    """Validated tensors for one paired low/high token split."""

    def __init__(
        self,
        *,
        low_tokens: Tensor,
        high_tokens: Tensor,
        labels: Tensor,
        low_data: Tensor,
        high_data: Tensor,
    ) -> None:
        self.low_tokens = low_tokens
        self.high_tokens = high_tokens
        self.labels = labels
        self.low_data = low_data
        self.high_data = high_data
        self.recomposed_data = low_data + high_data


def validate_output_dir(output_dir: str) -> Path:
    """Validate that paired-token artifacts stay below ignored outputs/."""
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


def load_payload(path: Path) -> dict[str, Tensor]:
    """Load a token payload and validate its basic structure."""
    if not path.exists():
        raise SystemExit(f"Missing token payload: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"Token payload must be a dictionary: {path}")
    required = {"indices", "data", "labels"}
    missing = required.difference(payload)
    if missing:
        raise SystemExit(f"Token payload {path} is missing keys: {sorted(missing)}")
    return cast(dict[str, Tensor], payload)


def load_summary(path: Path) -> dict[str, Any]:
    """Load a JSON token extraction summary."""
    if not path.exists():
        raise SystemExit(f"Missing token summary: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Token summary must be a dictionary: {path}")
    return cast(dict[str, Any], loaded)


def validate_pair_payloads(
    *,
    split: str,
    low_payload: Mapping[str, Tensor],
    high_payload: Mapping[str, Tensor],
) -> PairedPayload:
    """Validate and return matched tensors for one split."""
    low_tokens = low_payload["indices"].long()
    high_tokens = high_payload["indices"].long()
    low_data = low_payload["data"].float()
    high_data = high_payload["data"].float()
    low_labels = low_payload["labels"]
    high_labels = high_payload["labels"]

    require_shape(f"{split} low tokens", low_tokens, EXPECTED_TOKEN_SHAPE)
    require_shape(f"{split} high tokens", high_tokens, EXPECTED_TOKEN_SHAPE)
    require_shape(f"{split} low component data", low_data, EXPECTED_COMPONENT_SHAPE)
    require_shape(f"{split} high component data", high_data, EXPECTED_COMPONENT_SHAPE)
    if low_tokens.shape != high_tokens.shape:
        raise SystemExit(
            f"{split} low/high token shapes differ: "
            f"{tuple(low_tokens.shape)} vs {tuple(high_tokens.shape)}"
        )
    if low_data.shape != high_data.shape:
        raise SystemExit(
            f"{split} low/high component shapes differ: "
            f"{tuple(low_data.shape)} vs {tuple(high_data.shape)}"
        )
    if low_tokens.shape[0] != low_labels.shape[0]:
        raise SystemExit(
            f"{split} low token sample count {low_tokens.shape[0]} does not match "
            f"labels {low_labels.shape[0]}"
        )
    if high_tokens.shape[0] != high_labels.shape[0]:
        raise SystemExit(
            f"{split} high token sample count {high_tokens.shape[0]} does not match "
            f"labels {high_labels.shape[0]}"
        )
    if low_labels.shape != high_labels.shape or not torch.equal(low_labels, high_labels):
        raise SystemExit(f"{split} low/high labels do not match exactly.")
    return PairedPayload(
        low_tokens=low_tokens,
        high_tokens=high_tokens,
        labels=low_labels.detach().cpu(),
        low_data=low_data,
        high_data=high_data,
    )


def require_shape(name: str, tensor: Tensor, expected: tuple[int, ...]) -> None:
    """Raise if a tensor does not have the expected shape."""
    if tuple(tensor.shape) != expected:
        raise SystemExit(f"Expected {name} shape {expected}; got {tuple(tensor.shape)}.")


def build_dataset_summary(
    *,
    low_dir: Path,
    high_dir: Path,
    output_dir: Path,
    base_data_dir: Path,
    alpha: float,
    train_pair: PairedPayload,
    eval_pair: PairedPayload,
    combined_low: Tensor,
    combined_high: Tensor,
    low_codebook_size: int,
    high_codebook_size: int,
    low_summary: Mapping[str, Any],
    high_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build JSON-safe summary statistics for the paired-token dataset."""
    return {
        "paths": {
            "low_token_dir": str(low_dir),
            "high_token_dir": str(high_dir),
            "output_dir": str(output_dir),
            "base_data_dir": str(base_data_dir),
        },
        "alpha": alpha,
        "codebook_sizes": {
            "low": low_codebook_size,
            "high": high_codebook_size,
            "pair_vocab_size": low_codebook_size * high_codebook_size,
        },
        "shapes": {
            "train_low_tokens": list(train_pair.low_tokens.shape),
            "train_high_tokens": list(train_pair.high_tokens.shape),
            "eval_low_tokens": list(eval_pair.low_tokens.shape),
            "eval_high_tokens": list(eval_pair.high_tokens.shape),
            "train_labels": list(train_pair.labels.shape),
            "eval_labels": list(eval_pair.labels.shape),
            "train_low_component_data": list(train_pair.low_data.shape),
            "train_high_component_data": list(train_pair.high_data.shape),
            "eval_low_component_data": list(eval_pair.low_data.shape),
            "eval_high_component_data": list(eval_pair.high_data.shape),
            "train_data": list(train_pair.recomposed_data.shape),
            "eval_data": list(eval_pair.recomposed_data.shape),
        },
        "validations": {
            "expected_token_shape": list(EXPECTED_TOKEN_SHAPE),
            "expected_component_shape": list(EXPECTED_COMPONENT_SHAPE),
            "train_labels_match": True,
            "eval_labels_match": True,
            "train_sample_count_match": True,
            "eval_sample_count_match": True,
            "recomposed_data_note": (
                "train_data.pt and eval_data.pt store low_component + high_component, "
                "which reconstructs the original scalar EMA input path."
            ),
        },
        "low_usage": {
            "train": code_usage(train_pair.low_tokens, low_codebook_size),
            "eval": code_usage(eval_pair.low_tokens, low_codebook_size),
            "combined": code_usage(combined_low, low_codebook_size),
        },
        "high_usage": {
            "train": code_usage(train_pair.high_tokens, high_codebook_size),
            "eval": code_usage(eval_pair.high_tokens, high_codebook_size),
            "combined": code_usage(combined_high, high_codebook_size),
        },
        "same_time_pairs": {
            "train": pair_usage(
                train_pair.low_tokens,
                train_pair.high_tokens,
                low_codebook_size=low_codebook_size,
                high_codebook_size=high_codebook_size,
            ),
            "eval": pair_usage(
                eval_pair.low_tokens,
                eval_pair.high_tokens,
                low_codebook_size=low_codebook_size,
                high_codebook_size=high_codebook_size,
            ),
            "combined": pair_usage(
                combined_low,
                combined_high,
                low_codebook_size=low_codebook_size,
                high_codebook_size=high_codebook_size,
            ),
        },
        "vix_bucket_usage": {
            "low_train": low_summary.get("train_condition_buckets"),
            "low_eval": low_summary.get("eval_condition_buckets"),
            "low_combined": low_summary.get("combined_condition_buckets"),
            "high_train": high_summary.get("train_condition_buckets"),
            "high_eval": high_summary.get("eval_condition_buckets"),
            "high_combined": high_summary.get("combined_condition_buckets"),
        },
        "source_summaries": {
            "low": compact_source_summary(low_summary),
            "high": compact_source_summary(high_summary),
        },
    }


def compact_source_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the source token summary fields needed for provenance."""
    keys = (
        "tokenizer_dir",
        "config_path",
        "seed",
        "quantizer_type",
        "codebook_size",
        "sequence_length",
        "train_token_shape",
        "eval_token_shape",
        "combined",
    )
    return {key: summary.get(key) for key in keys}


def code_usage(indices: Tensor, codebook_size: int) -> dict[str, Any]:
    """Return active-code and perplexity statistics for one token tensor."""
    counts = torch.bincount(indices.reshape(-1), minlength=codebook_size)[:codebook_size]
    return count_usage(counts, codebook_size=codebook_size, active_key="active_code_count")


def pair_usage(
    low_tokens: Tensor,
    high_tokens: Tensor,
    *,
    low_codebook_size: int,
    high_codebook_size: int,
) -> dict[str, Any]:
    """Return same-time low/high pair support and perplexity statistics."""
    if low_tokens.shape != high_tokens.shape:
        raise SystemExit(
            f"Pair usage requires matched token shapes; got {low_tokens.shape} and "
            f"{high_tokens.shape}."
        )
    pair_indices = low_tokens.reshape(-1) * high_codebook_size + high_tokens.reshape(-1)
    pair_vocab_size = low_codebook_size * high_codebook_size
    counts = torch.bincount(pair_indices, minlength=pair_vocab_size)
    stats = count_usage(counts, codebook_size=pair_vocab_size, active_key="active_pair_count")
    stats["pair_vocab_size"] = pair_vocab_size
    stats["pair_perplexity"] = stats.pop("codebook_perplexity")
    stats["pair_entropy"] = stats.pop("index_entropy")
    stats["active_pair_ratio"] = stats.pop("active_code_ratio")
    stats["active_pair_indices"] = stats.pop("active_code_indices")
    return stats


def count_usage(counts: Tensor, *, codebook_size: int, active_key: str) -> dict[str, Any]:
    """Return JSON-safe entropy and active-support statistics from counts."""
    counts = counts.detach().cpu().long()
    total = int(counts.sum().item())
    active = counts > 0
    active_count = int(active.sum().item())
    if total == 0:
        entropy = 0.0
        perplexity = 0.0
    else:
        probabilities = counts[active].float() / float(total)
        entropy = float(-(probabilities * probabilities.log()).sum().item())
        perplexity = float(torch.exp(torch.tensor(entropy)).item())
    return {
        active_key: active_count,
        "active_code_ratio": float(active_count / codebook_size) if codebook_size else 0.0,
        "codebook_perplexity": perplexity,
        "index_entropy": entropy,
        "total_tokens": total,
        "active_code_indices": active.nonzero(as_tuple=False).reshape(-1).tolist(),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def render_markdown_summary(summary: Mapping[str, Any]) -> str:
    """Render a concise Markdown companion summary."""
    paths = cast(Mapping[str, str], summary["paths"])
    shapes = cast(Mapping[str, list[int]], summary["shapes"])
    low_usage = cast(Mapping[str, Mapping[str, Any]], summary["low_usage"])
    high_usage = cast(Mapping[str, Mapping[str, Any]], summary["high_usage"])
    pair_usage_summary = cast(Mapping[str, Mapping[str, Any]], summary["same_time_pairs"])
    codebook_sizes = cast(Mapping[str, int], summary["codebook_sizes"])
    return (
        "# Separate Frequency Paired Token Dataset\n\n"
        f"- Low token dir: `{paths['low_token_dir']}`\n"
        f"- High token dir: `{paths['high_token_dir']}`\n"
        f"- Output dir: `{paths['output_dir']}`\n"
        f"- Base data dir: `{paths['base_data_dir']}`\n"
        f"- EMA alpha: `{summary['alpha']}`\n\n"
        "## Shapes\n\n"
        "| Tensor | Shape |\n"
        "|---|---:|\n"
        f"| train_low_tokens | `{shapes['train_low_tokens']}` |\n"
        f"| train_high_tokens | `{shapes['train_high_tokens']}` |\n"
        f"| eval_low_tokens | `{shapes['eval_low_tokens']}` |\n"
        f"| eval_high_tokens | `{shapes['eval_high_tokens']}` |\n"
        f"| train_data | `{shapes['train_data']}` |\n"
        f"| eval_data | `{shapes['eval_data']}` |\n\n"
        "## Code Usage\n\n"
        "| Stream | Split | Active | Perplexity |\n"
        "|---|---|---:|---:|\n"
        f"{usage_row('low', 'train', low_usage['train'], codebook_sizes['low'])}\n"
        f"{usage_row('low', 'eval', low_usage['eval'], codebook_sizes['low'])}\n"
        f"{usage_row('low', 'combined', low_usage['combined'], codebook_sizes['low'])}\n"
        f"{usage_row('high', 'train', high_usage['train'], codebook_sizes['high'])}\n"
        f"{usage_row('high', 'eval', high_usage['eval'], codebook_sizes['high'])}\n"
        f"{usage_row('high', 'combined', high_usage['combined'], codebook_sizes['high'])}\n\n"
        "## Same-Time Pair Usage\n\n"
        "| Split | Active pairs | Pair perplexity |\n"
        "|---|---:|---:|\n"
        f"{pair_row('train', pair_usage_summary['train'], codebook_sizes['pair_vocab_size'])}\n"
        f"{pair_row('eval', pair_usage_summary['eval'], codebook_sizes['pair_vocab_size'])}\n"
        f"{pair_row('combined', pair_usage_summary['combined'], codebook_sizes['pair_vocab_size'])}"
        "\n"
    )


def usage_row(stream: str, split: str, usage: Mapping[str, Any], codebook_size: int) -> str:
    """Render one code-usage Markdown table row."""
    return (
        f"| {stream} | {split} | {usage['active_code_count']}/{codebook_size} | "
        f"{float(usage['codebook_perplexity']):.8f} |"
    )


def pair_row(split: str, usage: Mapping[str, Any], pair_vocab_size: int) -> str:
    """Render one same-time pair-usage Markdown table row."""
    return (
        f"| {split} | {usage['active_pair_count']}/{pair_vocab_size} | "
        f"{float(usage['pair_perplexity']):.8f} |"
    )


if __name__ == "__main__":
    main()
