"""Smoke-check causal VQ tokenizer shapes and prefix causality."""

from __future__ import annotations

import torch
from ml_collections import ConfigDict

from time_causal_vae.data.pipeline import DataPipeline
from time_causal_vae.models.discrete.tokenizers import (
    CausalVQTokenizer,
    VQTokenizerConfig,
)
from time_causal_vae.models.discrete.tokenizers.causal_vq_tokenizer import (
    assert_tokenizer_no_future_leakage,
)


def main() -> int:
    """Run a deterministic forward and no-future-leakage smoke check."""
    torch.manual_seed(23)

    batch_size = 8
    cutoff = 29
    data_config = ConfigDict({
        "dataset": "black_scholes",
        "n_sample": 64,
        "n_timestep": 60,
        "data_params": {},
    })
    train_dataset, _ = DataPipeline()(data_config)
    inputs = train_dataset.data[:batch_size].to(dtype=torch.float32)

    tokenizer_config = VQTokenizerConfig(
        data_dim=1,
        data_length=inputs.shape[1],
        embedding_dim=8,
        codebook_size=16,
        commitment_weight=0.25,
        encoder_hidden_dim=16,
        decoder_hidden_dim=16,
        num_layers=4,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
    )
    tokenizer = CausalVQTokenizer(tokenizer_config)
    tokenizer.eval()

    try:
        with torch.no_grad():
            output = tokenizer(inputs)
        expected_indices_shape = (batch_size, inputs.shape[1])
        if tuple(output.indices.shape) != expected_indices_shape:
            actual_indices_shape = tuple(output.indices.shape)
            raise AssertionError(
                f"Expected indices shape {expected_indices_shape}; got {actual_indices_shape}."
            )
        if not torch.isfinite(output.loss):
            raise AssertionError(f"Expected finite loss; got {output.loss.item()}.")

        changed_future_inputs = inputs.clone()
        changed_future_inputs[:, cutoff + 1 :] += 5.0 * torch.randn_like(
            changed_future_inputs[:, cutoff + 1 :]
        )
        assert_tokenizer_no_future_leakage(
            tokenizer,
            inputs,
            changed_future_inputs,
            cutoff,
            atol=1e-6,
            rtol=1e-5,
        )
    except Exception as exc:
        print(f"FAIL causal VQ tokenizer shape check: {exc}")
        return 1

    print("PASS causal VQ tokenizer shape check")
    print(f"output_keys={list(output.keys())}")
    print(f"x={tuple(inputs.shape)}")
    print(f"z_e={tuple(output.z_e.shape)}")
    print(f"z_q={tuple(output.z_q.shape)}")
    print(f"indices={tuple(output.indices.shape)}")
    print(f"recon_x={tuple(output.recon_x.shape)}")
    print(f"loss={output.loss.item():.8f}")
    print(f"recon_loss={output.recon_loss.item():.8f}")
    print(f"commitment_loss={output.commitment_loss.item():.8f}")
    print(f"codebook_loss={output.codebook_loss.item():.8f}")
    print(f"leakage_check=cutoff_{cutoff}_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
