"""Smoke-check causal convolution layers for future leakage."""

from __future__ import annotations

import torch

from time_causal_vae.models.layers import CausalConvStack, assert_no_future_leakage


def main() -> int:
    """Run a deterministic prefix-invariance check."""
    torch.manual_seed(17)

    batch_size = 4
    length = 24
    in_channels = 3
    hidden_channels = 8
    out_channels = 5
    cutoff = 11

    stack = CausalConvStack(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
    )

    reference_inputs = torch.randn(batch_size, length, in_channels)
    changed_future_inputs = reference_inputs.clone()
    changed_future_inputs[:, cutoff + 1 :] += 10.0 * torch.randn_like(
        changed_future_inputs[:, cutoff + 1 :]
    )

    expected_shape = (batch_size, length, out_channels)

    try:
        reference_outputs, changed_future_outputs = assert_no_future_leakage(
            stack,
            reference_inputs,
            changed_future_inputs,
            cutoff,
            atol=1e-6,
            rtol=1e-5,
        )
        if tuple(reference_outputs.shape) != expected_shape:
            raise AssertionError(
                f"Expected output shape {expected_shape}; got {tuple(reference_outputs.shape)}."
            )
        if tuple(changed_future_outputs.shape) != expected_shape:
            raise AssertionError(
                "Changed-future output shape mismatch: "
                f"expected {expected_shape}; got {tuple(changed_future_outputs.shape)}."
            )
    except Exception as exc:
        print(f"FAIL causal-conv no-leakage check: {exc}")
        return 1

    print("PASS causal-conv no-leakage check")
    print(f"input_shape={tuple(reference_inputs.shape)}")
    print(f"output_shape={tuple(reference_outputs.shape)}")
    print(f"cutoff={cutoff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
