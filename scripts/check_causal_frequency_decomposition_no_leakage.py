"""Smoke-check causal EMA frequency decomposition for future leakage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor

from time_causal_vae.data.frequency import (
    causal_ema_decompose,
    compose_low_high,
)


@dataclass(frozen=True)
class CheckResult:
    """Prefix-invariance and reconstruction errors for one tensor layout."""

    shape: tuple[int, ...]
    max_low_prefix_diff: float
    max_high_prefix_diff: float
    max_reconstruction_diff: float


def build_parser() -> argparse.ArgumentParser:
    """Build the script parser."""
    parser = argparse.ArgumentParser(
        description="Check deterministic causal EMA decomposition no-future-leakage.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Number of synthetic paths.")
    parser.add_argument("--length", type=int, default=60, help="Synthetic path length.")
    parser.add_argument("--alpha", type=float, default=0.2, help="EMA smoothing parameter.")
    parser.add_argument("--cutoff", type=int, default=29, help="Inclusive zero-indexed cutoff.")
    parser.add_argument("--seed", type=int, default=99, help="Random seed.")
    parser.add_argument("--atol", type=float, default=1e-7, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance.")
    return parser


def main() -> int:
    """Run deterministic no-leakage and reconstruction checks."""
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.length <= 1:
        raise SystemExit("--length must be greater than one.")
    if args.cutoff < 0 or args.cutoff >= args.length - 1:
        raise SystemExit("--cutoff must leave at least one future step to perturb.")

    torch.manual_seed(args.seed)
    paths_2d = make_positive_paths(batch_size=args.batch_size, length=args.length)
    paths_3d = paths_2d.unsqueeze(-1)

    try:
        results = [
            check_layout(paths_2d, alpha=args.alpha, cutoff=args.cutoff),
            check_layout(paths_3d, alpha=args.alpha, cutoff=args.cutoff),
        ]
        for result in results:
            if result.max_low_prefix_diff > args.atol:
                raise AssertionError(
                    "Future perturbation changed low-frequency prefix for "
                    f"shape {result.shape}: {result.max_low_prefix_diff:.8e}."
                )
            if result.max_high_prefix_diff > args.atol:
                raise AssertionError(
                    "Future perturbation changed high-frequency prefix for "
                    f"shape {result.shape}: {result.max_high_prefix_diff:.8e}."
                )
            if result.max_reconstruction_diff > args.atol + args.rtol:
                raise AssertionError(
                    "Low/high recomposition differs from the original path for "
                    f"shape {result.shape}: {result.max_reconstruction_diff:.8e}."
                )
    except Exception as exc:
        print(f"FAIL causal frequency decomposition no-leakage check: {exc}")
        return 1

    print("PASS causal frequency decomposition no-leakage check")
    print(f"batch_size={args.batch_size}")
    print(f"length={args.length}")
    print(f"alpha={args.alpha}")
    print(f"cutoff={args.cutoff}")
    print(f"seed={args.seed}")
    for result in results:
        print(f"shape={result.shape}")
        print(f"max_low_prefix_diff={result.max_low_prefix_diff:.8e}")
        print(f"max_high_prefix_diff={result.max_high_prefix_diff:.8e}")
        print(f"max_reconstruction_diff={result.max_reconstruction_diff:.8e}")
    return 0


def make_positive_paths(*, batch_size: int, length: int) -> Tensor:
    """Create synthetic positive paths by exponentiating small random increments."""
    increments = 0.02 * torch.randn(batch_size, length)
    increments[:, 0] = 0.0
    return torch.exp(torch.cumsum(increments, dim=1))


def check_layout(path: Tensor, *, alpha: float, cutoff: int) -> CheckResult:
    """Check one supported tensor layout."""
    low, high = causal_ema_decompose(path, alpha)
    changed_future_path = perturb_future(path, cutoff=cutoff)
    changed_low, changed_high = causal_ema_decompose(changed_future_path, alpha)
    prefix = slice(0, cutoff + 1)
    max_low_prefix_diff = max_abs_difference(low[:, prefix], changed_low[:, prefix])
    max_high_prefix_diff = max_abs_difference(high[:, prefix], changed_high[:, prefix])
    reconstructed = compose_low_high(low, high)
    max_reconstruction_diff = max_abs_difference(reconstructed, path)
    return CheckResult(
        shape=tuple(path.shape),
        max_low_prefix_diff=max_low_prefix_diff,
        max_high_prefix_diff=max_high_prefix_diff,
        max_reconstruction_diff=max_reconstruction_diff,
    )


def perturb_future(path: Tensor, *, cutoff: int) -> Tensor:
    """Perturb only future values after the inclusive cutoff while staying positive."""
    changed = path.clone()
    future = changed[:, cutoff + 1 :]
    multipliers = torch.exp(0.5 + 0.1 * torch.randn_like(future))
    changed[:, cutoff + 1 :] = future * multipliers
    return changed


def max_abs_difference(first: Tensor, second: Tensor) -> float:
    """Return the maximum absolute tensor difference as a Python float."""
    return float((first - second).abs().max().item())


if __name__ == "__main__":
    raise SystemExit(main())
