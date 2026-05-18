"""Evaluate Hawkes-jump continuous models with log-return-to-price diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import Tensor

from time_causal_vae.evaluation.checkpoints import TargetModelEvaluator
from time_causal_vae.evaluation.jump_diagnostics import (
    detected_jump_sizes,
    jump_count_wasserstein,
    jump_diagnostic_summary,
    log_returns_to_normalized_prices,
)
from time_causal_vae.evaluation.market_diagnostics import (
    compare_market_summaries,
    compute_log_returns,
    terminal_returns,
    volatility_per_path,
    wasserstein_1d,
)
from time_causal_vae.evaluation.metrics import SWD
from time_causal_vae.experiments.config import (
    adapt_selected_config,
    load_selected_config,
)
from time_causal_vae.models.continuous.distances import GaussianMMD
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the Hawkes continuous-evaluation parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Hawkes-jump continuous models with jump-aware diagnostics.",
    )
    parser.add_argument("--config", required=True, help="Continuous experiment YAML config.")
    parser.add_argument(
        "--model-dir",
        help="Continuous final_model directory containing model.pt. Required unless --dry-run.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory under outputs/.")
    parser.add_argument("--n-sample", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-data-dir", default="data/processed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and write a planned-evaluation summary without loading a model.",
    )
    return parser


def main() -> None:
    """Run Hawkes continuous evaluation."""
    args = build_parser().parse_args()
    if args.n_sample <= 0:
        raise SystemExit("--n-sample must be positive.")

    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config = load_yaml(args.config)
    data_output = read_data_output(raw_config)
    if data_output not in {"price", "log_return"}:
        raise SystemExit(f"Unsupported data_output: {data_output}")

    if args.dry_run:
        summary = build_dry_run_summary(
            args=vars(args),
            raw_config=raw_config,
            data_output=data_output,
        )
        write_json(output_dir / "evaluation_summary.json", summary)
        write_markdown_summary(output_dir / "evaluation_summary.md", summary)
        print("Hawkes-jump continuous evaluator dry run complete.")
        print(f"output_dir: {output_dir}")
        print(f"data_output: {data_output}")
        print(f"log_return_to_price_conversion: {data_output == 'log_return'}")
        return

    if args.model_dir is None:
        raise SystemExit("--model-dir is required unless --dry-run is set.")

    set_seed(args.seed)
    model_dir = validate_model_dir(args.model_dir)
    exp_config_path = ensure_legacy_exp_config(
        config_path=args.config,
        model_dir=model_dir,
        base_data_dir=args.base_data_dir,
    )
    evaluator = TargetModelEvaluator(str(model_dir), base_data_dir=args.base_data_dir)
    real_decoder_space, generated_decoder_space, reconstructed_decoder_space = evaluator.load_data(
        n_sample_test=args.n_sample,
        seed=args.seed,
    )

    generated_prices = decoder_output_to_prices(generated_decoder_space, data_output=data_output)
    real_prices = decoder_output_to_prices(real_decoder_space, data_output=data_output)
    reconstructed_prices = decoder_output_to_prices(
        reconstructed_decoder_space,
        data_output=data_output,
    )
    jump_thresholds = fit_common_jump_thresholds(real_prices)
    generated_jumps = detect_jumps_with_threshold(generated_prices, thresholds=jump_thresholds)
    real_jumps = detect_jumps_with_threshold(real_prices, thresholds=jump_thresholds)
    generated_jump_sizes = detected_jump_sizes(generated_prices, generated_jumps)
    real_jump_sizes = detected_jump_sizes(real_prices, real_jumps)

    summary = build_summary(
        args=vars(args),
        raw_config=raw_config,
        data_output=data_output,
        model_dir=model_dir,
        exp_config_path=exp_config_path,
        real_decoder_space=real_decoder_space.detach().cpu(),
        generated_decoder_space=generated_decoder_space.detach().cpu(),
        reconstructed_decoder_space=reconstructed_decoder_space.detach().cpu(),
        real_prices=real_prices.detach().cpu(),
        generated_prices=generated_prices.detach().cpu(),
        reconstructed_prices=reconstructed_prices.detach().cpu(),
        real_jumps=real_jumps.detach().cpu(),
        generated_jumps=generated_jumps.detach().cpu(),
        real_jump_sizes=real_jump_sizes.detach().cpu(),
        generated_jump_sizes=generated_jump_sizes.detach().cpu(),
        jump_thresholds=jump_thresholds,
    )
    write_json(output_dir / "evaluation_summary.json", summary)
    write_markdown_summary(output_dir / "evaluation_summary.md", summary)
    torch.save(
        {
            "real_decoder_space": real_decoder_space.detach().cpu(),
            "generated_decoder_space": generated_decoder_space.detach().cpu(),
            "reconstructed_decoder_space": reconstructed_decoder_space.detach().cpu(),
            "real_prices": real_prices.detach().cpu(),
            "generated_prices": generated_prices.detach().cpu(),
            "reconstructed_prices": reconstructed_prices.detach().cpu(),
            "real_jumps": real_jumps.detach().cpu(),
            "generated_jumps": generated_jumps.detach().cpu(),
        },
        output_dir / "evaluation_batch.pt",
    )

    print("Hawkes-jump continuous evaluation complete.")
    print(f"output_dir: {output_dir}")
    print(f"data_output: {data_output}")
    print(f"generated_prices_shape: {list(generated_prices.shape)}")
    print(f"mmd: {summary['smooth_metrics']['mmd']:.8f}")
    print(f"swd: {summary['smooth_metrics']['swd']:.8f}")
    print(
        "detected jump count W1: "
        f"{summary['jump_comparison']['detected_jump_count_wasserstein']:.8f}"
    )


def validate_output_dir(output_dir: str) -> Path:
    """Validate that generated outputs stay below local outputs/."""
    path = Path(output_dir)
    resolved = path.resolve()
    outputs_root = (Path.cwd() / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise SystemExit(
            f"--output-dir must be under local outputs/. Received: {output_dir}"
        ) from exc
    return path


def validate_model_dir(model_dir: str) -> Path:
    """Validate a continuous final_model checkpoint directory."""
    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"--model-dir must be an existing directory: {model_dir}")
    if not (path / "model.pt").exists():
        raise SystemExit(f"--model-dir must contain model.pt: {model_dir}")
    return path


def ensure_legacy_exp_config(
    *,
    config_path: str,
    model_dir: Path,
    base_data_dir: str,
) -> Path:
    """Ensure ``TargetModelEvaluator`` can find an adjacent legacy config."""
    exp_config_path = model_dir.parent / "exp_config.yaml"
    if exp_config_path.exists():
        return exp_config_path

    selected_config = load_selected_config(config_path)
    legacy_config = adapt_selected_config(
        selected_config,
        output_dir=model_dir.parent,
        base_data_dir=base_data_dir,
        wandb=False,
    )
    with exp_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(legacy_config.to_dict(), handle, sort_keys=False)
    return exp_config_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def read_data_output(raw_config: Mapping[str, Any]) -> str:
    """Read the configured decoder-space convention."""
    data = raw_config.get("data")
    if not isinstance(data, Mapping):
        return "price"
    if "data_output" in data:
        return str(data["data_output"])
    params = data.get("params")
    if isinstance(params, Mapping):
        return str(params.get("data_output", "price"))
    data_params = data.get("data_params")
    if isinstance(data_params, Mapping):
        return str(data_params.get("data_output", "price"))
    return "price"


def decoder_output_to_prices(decoded: Tensor, *, data_output: str) -> Tensor:
    """Convert continuous decoder-space outputs to price paths for diagnostics."""
    if data_output == "log_return":
        return log_returns_to_normalized_prices(decoded)
    if data_output != "price":
        raise ValueError(f"Unsupported data_output: {data_output}")
    if decoded.ndim != 3 or decoded.shape[-1] != 1:
        raise ValueError(f"Expected price paths with shape [batch, time, 1]; got {decoded.shape}.")
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("Decoded price paths must be finite.")
    if not bool((decoded > 0.0).all()):
        raise ValueError("Decoded price paths must be positive.")
    return decoded.float()


def fit_common_jump_thresholds(
    real_prices: Tensor,
    *,
    threshold_multiplier: float = 4.0,
    min_abs_return: float = 0.0,
) -> dict[str, float]:
    """Fit a robust jump threshold on real Ogata evaluation paths."""
    returns = compute_log_returns(real_prices)
    median = returns.median()
    mad = (returns - median).abs().median()
    robust_scale = (1.4826 * mad).clamp_min(1e-8)
    threshold = max(float((threshold_multiplier * robust_scale).item()), float(min_abs_return))
    return {
        "median": float(median.item()),
        "mad": float(mad.item()),
        "robust_scale": float(robust_scale.item()),
        "threshold_multiplier": float(threshold_multiplier),
        "min_abs_return": float(min_abs_return),
        "absolute_threshold": threshold,
    }


def detect_jumps_with_threshold(paths: Tensor, *, thresholds: Mapping[str, float]) -> Tensor:
    """Detect jumps with a fixed threshold fitted on reference paths."""
    returns = compute_log_returns(paths)
    median = float(thresholds["median"])
    threshold = float(thresholds["absolute_threshold"])
    jump_returns = (returns - median).abs() >= threshold
    leading = torch.zeros(
        (jump_returns.shape[0], 1),
        dtype=torch.bool,
        device=jump_returns.device,
    )
    return torch.cat([leading, jump_returns], dim=1).unsqueeze(-1)


def compute_path_distribution_metrics(*, generated: Tensor, real: Tensor) -> dict[str, Any]:
    """Compute smooth path distribution metrics in price space."""
    generated_float = generated.float()
    real_float = real.float()
    generated_terminal = terminal_returns(generated_float)
    real_terminal = terminal_returns(real_float)
    generated_volatility = volatility_per_path(generated_float)
    real_volatility = volatility_per_path(real_float)
    return {
        "decoded_path_shape": list(generated.shape),
        "real_path_shape": list(real.shape),
        "mmd": float(GaussianMMD()(generated_float, real_float).detach().cpu()),
        "swd": float(SWD()(generated_float, real_float).detach().cpu()),
        "terminal_return_generated_mean": float(generated_terminal.mean().detach().cpu()),
        "terminal_return_real_mean": float(real_terminal.mean().detach().cpu()),
        "terminal_return_mean_error": float(
            (generated_terminal.mean() - real_terminal.mean()).abs().detach().cpu()
        ),
        "terminal_return_wasserstein": wasserstein_1d(generated_terminal, real_terminal),
        "volatility_generated_mean": float(generated_volatility.mean().detach().cpu()),
        "volatility_real_mean": float(real_volatility.mean().detach().cpu()),
        "volatility_mean_error": float(
            (generated_volatility.mean() - real_volatility.mean()).abs().detach().cpu()
        ),
        "volatility_wasserstein": wasserstein_1d(generated_volatility, real_volatility),
    }


def build_dry_run_summary(
    *,
    args: Mapping[str, Any],
    raw_config: Mapping[str, Any],
    data_output: str,
) -> dict[str, Any]:
    """Build a JSON-safe dry-run summary."""
    return {
        "manifest": {
            "status": "dry_run",
            "config_path": str(args["config"]),
            "model_dir": args.get("model_dir"),
            "n_sample": int(args["n_sample"]),
            "seed": int(args["seed"]),
            "data_output": data_output,
            "base_data_dir": str(args["base_data_dir"]),
            "config_data": dict(cast(Mapping[str, Any], raw_config.get("data", {}))),
        },
        "log_return_to_price_conversion": data_output == "log_return",
        "planned_outputs": [
            "evaluation_summary.json",
            "evaluation_summary.md",
            "evaluation_batch.pt for real evaluations",
        ],
        "planned_metrics": [
            "MMD",
            "SWD",
            "terminal W1",
            "volatility W1",
            "drawdown W1",
            "jump-count W1",
            "jump-size W1",
            "VaR/ES",
        ],
    }


def build_summary(
    *,
    args: Mapping[str, Any],
    raw_config: Mapping[str, Any],
    data_output: str,
    model_dir: Path,
    exp_config_path: Path,
    real_decoder_space: Tensor,
    generated_decoder_space: Tensor,
    reconstructed_decoder_space: Tensor,
    real_prices: Tensor,
    generated_prices: Tensor,
    reconstructed_prices: Tensor,
    real_jumps: Tensor,
    generated_jumps: Tensor,
    real_jump_sizes: Tensor,
    generated_jump_sizes: Tensor,
    jump_thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Build a JSON-safe Hawkes continuous evaluation summary."""
    generated_jump_summary = jump_diagnostic_summary(
        generated_prices,
        jump_indicators=generated_jumps,
        jump_sizes=generated_jump_sizes,
        tail_reference_paths=real_prices,
    )
    real_jump_summary = jump_diagnostic_summary(
        real_prices,
        jump_indicators=real_jumps,
        jump_sizes=real_jump_sizes,
        tail_reference_paths=real_prices,
    )
    jump_comparison = {
        "detected_jump_count_wasserstein": jump_count_wasserstein(
            generated_jumps,
            real_jumps,
        ),
        "detected_jump_size_wasserstein": wasserstein_1d(
            generated_jump_sizes[generated_jump_sizes.abs() > 0.0],
            real_jump_sizes[real_jump_sizes.abs() > 0.0],
        ),
    }
    return {
        "manifest": {
            "status": "complete",
            "config_path": str(args["config"]),
            "model_dir": str(model_dir),
            "exp_config_path": str(exp_config_path),
            "n_sample": int(args["n_sample"]),
            "seed": int(args["seed"]),
            "data_output": data_output,
            "base_data_dir": str(args["base_data_dir"]),
            "condition_convention": "continuous_eval_labels_from_hawkes_eval_dataset",
            "config_data": dict(cast(Mapping[str, Any], raw_config.get("data", {}))),
        },
        "tensor_shapes": {
            "real_decoder_space": list(real_decoder_space.shape),
            "generated_decoder_space": list(generated_decoder_space.shape),
            "reconstructed_decoder_space": list(reconstructed_decoder_space.shape),
            "real_prices": list(real_prices.shape),
            "generated_prices": list(generated_prices.shape),
            "reconstructed_prices": list(reconstructed_prices.shape),
        },
        "log_return_to_price_conversion": data_output == "log_return",
        "jump_detection_thresholds_from_real": dict(jump_thresholds),
        "smooth_metrics": compute_path_distribution_metrics(
            generated=generated_prices,
            real=real_prices,
        ),
        "reconstruction_smooth_metrics": compute_path_distribution_metrics(
            generated=reconstructed_prices,
            real=real_prices,
        ),
        "market_comparison": compare_market_summaries(
            real_paths=real_prices,
            generated_paths=generated_prices,
        ),
        "jump_diagnostics": {
            "generated": generated_jump_summary,
            "real": real_jump_summary,
        },
        "jump_comparison": jump_comparison,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a compact Markdown summary."""
    manifest = cast(Mapping[str, Any], summary["manifest"])
    lines = [
        "# Hawkes-Jump Continuous Evaluation",
        "",
        "## Manifest",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Config: `{manifest['config_path']}`",
        f"- Model dir: `{manifest.get('model_dir')}`",
        f"- Samples: `{manifest['n_sample']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Data output: `{manifest['data_output']}`",
        f"- Log-return conversion: `{summary['log_return_to_price_conversion']}`",
        "",
    ]
    if manifest["status"] == "dry_run":
        planned = cast(list[str], summary["planned_metrics"])
        lines.extend(
            [
                "## Planned Metrics",
                "",
                *[f"- {metric}" for metric in planned],
                "",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    smooth = cast(Mapping[str, Any], summary["smooth_metrics"])
    jump = cast(Mapping[str, Any], summary["jump_comparison"])
    generated_jump = cast(Mapping[str, Any], summary["jump_diagnostics"])["generated"]
    generated_counts = cast(Mapping[str, Any], generated_jump)["jump_counts"]
    generated_count_dist = cast(Mapping[str, Any], generated_counts)["per_path"]
    lines.extend(
        [
            "## Smooth Metrics",
            "",
            f"- MMD: `{float(smooth['mmd']):.8f}`",
            f"- SWD: `{float(smooth['swd']):.8f}`",
            f"- Terminal W1: `{float(smooth['terminal_return_wasserstein']):.8f}`",
            f"- Volatility W1: `{float(smooth['volatility_wasserstein']):.8f}`",
            "",
            "## Jump Metrics",
            "",
            f"- Detected jump count W1: `{float(jump['detected_jump_count_wasserstein']):.8f}`",
            f"- Detected jump-size W1: `{float(jump['detected_jump_size_wasserstein']):.8f}`",
            f"- Generated mean jumps per path: `{float(generated_count_dist['mean']):.8f}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    """Convert tensors and paths into JSON-safe values."""
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
