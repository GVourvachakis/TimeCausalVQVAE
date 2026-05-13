"""Smoke-check standard, ResidualVQ, and GroupedResidualVQ tokenizer shapes."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import torch

from time_causal_vae.tokenization import CausalVQTokenizer, VQTokenizerConfig


def main() -> int:
    """Run deterministic VQ-family tokenizer shape checks."""
    torch.manual_seed(99)
    batch_size = 4
    sequence_length = 60
    data_dim = 1
    condition_dim = 1
    embedding_dim = 64
    inputs = torch.randn(batch_size, sequence_length, data_dim)
    conditions = torch.randn(batch_size, condition_dim)

    smoke_cases = [
        (
            "vector",
            VQTokenizerConfig(
                data_dim=data_dim,
                data_length=sequence_length,
                embedding_dim=embedding_dim,
                codebook_size=64,
                commitment_weight=0.1,
                encoder_hidden_dim=32,
                decoder_hidden_dim=32,
                num_layers=4,
                dilations=(1, 2, 4, 8),
                dropout=0.0,
                condition_dim=condition_dim,
                quantizer_type="vector",
                kmeans_init=False,
                codebook_dim=16,
            ),
            (batch_size, sequence_length),
        ),
        (
            "residual_vq",
            VQTokenizerConfig(
                data_dim=data_dim,
                data_length=sequence_length,
                embedding_dim=embedding_dim,
                codebook_size=64,
                commitment_weight=0.1,
                encoder_hidden_dim=32,
                decoder_hidden_dim=32,
                num_layers=4,
                dilations=(1, 2, 4, 8),
                dropout=0.0,
                condition_dim=condition_dim,
                quantizer_type="residual_vq",
                num_quantizers=2,
                kmeans_init=False,
                codebook_dim=16,
                shared_codebook=False,
            ),
            (batch_size, sequence_length, 2),
        ),
        (
            "grouped_residual_vq",
            VQTokenizerConfig(
                data_dim=data_dim,
                data_length=sequence_length,
                embedding_dim=embedding_dim,
                codebook_size=64,
                commitment_weight=0.1,
                encoder_hidden_dim=32,
                decoder_hidden_dim=32,
                num_layers=4,
                dilations=(1, 2, 4, 8),
                dropout=0.0,
                condition_dim=condition_dim,
                quantizer_type="grouped_residual_vq",
                num_quantizers=2,
                groups=4,
                kmeans_init=False,
                codebook_dim=16,
                shared_codebook=False,
            ),
            (batch_size, sequence_length, 4, 2),
        ),
    ]

    results: list[dict[str, Any]] = []
    for case_name, config, expected_indices_shape in smoke_cases:
        tokenizer = CausalVQTokenizer(config)
        tokenizer.eval()
        with torch.no_grad():
            output = tokenizer(inputs, conditions)
            decoded_from_indices = tokenizer.decode_indices(output.indices, conditions)

        observed_indices_shape = tuple(output.indices.shape)
        if observed_indices_shape != expected_indices_shape:
            raise AssertionError(
                f"{case_name}: expected indices {expected_indices_shape}; "
                f"got {observed_indices_shape}."
            )
        if tuple(output.z_q.shape) != (batch_size, sequence_length, embedding_dim):
            raise AssertionError(f"{case_name}: unexpected z_q shape {tuple(output.z_q.shape)}.")
        if tuple(output.recon_x.shape) != tuple(inputs.shape):
            raise AssertionError(
                f"{case_name}: unexpected recon_x shape {tuple(output.recon_x.shape)}."
            )
        if tuple(decoded_from_indices.shape) != tuple(inputs.shape):
            raise AssertionError(
                f"{case_name}: decode_indices returned {tuple(decoded_from_indices.shape)}."
            )
        for loss_name in ("loss", "recon_loss", "commitment_loss", "codebook_loss"):
            loss_value = getattr(output, loss_name)
            if not torch.isfinite(loss_value):
                raise AssertionError(f"{case_name}: non-finite {loss_name}.")

        results.append(
            {
                "case": case_name,
                "config": asdict(config),
                "input_shape": list(inputs.shape),
                "condition_shape": list(conditions.shape),
                "z_e_shape": list(output.z_e.shape),
                "z_q_shape": list(output.z_q.shape),
                "indices_shape": list(output.indices.shape),
                "index_shape_metadata": list(output.index_shape),
                "recon_x_shape": list(output.recon_x.shape),
                "decoded_from_indices_shape": list(decoded_from_indices.shape),
                "loss": float(output.loss.detach().cpu()),
                "reconstruction_loss": float(output.recon_loss.detach().cpu()),
                "commitment_loss": float(output.commitment_loss.detach().cpu()),
                "codebook_loss": float(output.codebook_loss.detach().cpu()),
            }
        )

    print(json.dumps({"status": "passed", "cases": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
