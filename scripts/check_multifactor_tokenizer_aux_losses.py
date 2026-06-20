"""Smoke-check optional multifactor tokenizer auxiliary losses."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from time_causal_vae.cli.train_tokenizer import build_auxiliary_loss_context
from time_causal_vae.data.factor_projected_market import FactorProjectedMultifactorMarketDataset
from time_causal_vae.models.discrete.tokenizers import CausalVQTokenizer, VQTokenizerConfig
from time_causal_vae.utils.random import set_seed

AUXILIARY_WEIGHT_TO_OUTPUT = {
    "factor_reconstruction_loss_weight": "factor_reconstruction_aux_loss",
    "factor_covariance_loss_weight": "factor_covariance_loss",
    "factor_correlation_loss_weight": "factor_correlation_loss",
    "inverse_projected_covariance_loss_weight": "inverse_projected_covariance_loss",
    "inverse_projected_correlation_loss_weight": "inverse_projected_correlation_loss",
    "sector_block_loss_weight": "sector_block_loss",
    "equal_weight_portfolio_vol_loss_weight": "equal_weight_portfolio_vol_loss",
}


def main() -> int:
    """Run deterministic auxiliary-loss smoke checks."""
    set_seed(123)
    device = torch.device("cpu")
    dataset = FactorProjectedMultifactorMarketDataset(
        16,
        60,
        n_assets=50,
        n_factors=5,
        n_sectors=5,
        structure_seed=0,
        path_seed=0,
        condition_mode="constant",
        with_jumps=False,
        standardize_returns=True,
        projection_mode="train_pca",
        projection_n_factors=5,
    )
    inputs = dataset.data.to(device)
    conditions = dataset.labels.to(device)
    base_config = VQTokenizerConfig(
        data_dim=5,
        data_length=60,
        embedding_dim=16,
        codebook_size=16,
        commitment_weight=0.1,
        encoder_hidden_dim=16,
        decoder_hidden_dim=16,
        num_layers=2,
        dilations=(1, 2),
        dropout=0.0,
        condition_dim=1,
        kmeans_init=False,
        codebook_dim=8,
    )
    base_tokenizer = CausalVQTokenizer(base_config).to(device)
    context_config = replace(base_config, inverse_projected_correlation_loss_weight=0.1)
    context = build_auxiliary_loss_context(context_config, dataset, device=device)
    if context is None:
        raise AssertionError("Expected factor-projected auxiliary loss context.")

    zero_output = base_tokenizer(inputs, conditions, context)
    legacy_total = zero_output.recon_loss + zero_output.commitment_loss + zero_output.codebook_loss
    assert_close(zero_output.loss, legacy_total, name="zero_weight_legacy_total")
    assert_close(zero_output.auxiliary_loss, zero_output.loss.new_zeros(()), name="zero_aux_loss")
    check_gradients(base_tokenizer, zero_output.loss, name="zero_weight")

    summaries: list[str] = []
    for weight_name, output_name in AUXILIARY_WEIGHT_TO_OUTPUT.items():
        config = replace(base_config, **{weight_name: 0.1})
        tokenizer = CausalVQTokenizer(config).to(device)
        tokenizer.load_state_dict(base_tokenizer.state_dict())
        tokenizer.zero_grad(set_to_none=True)
        output = tokenizer(inputs, conditions, context)
        loss = output.loss
        component_loss = output[output_name]
        if not torch.isfinite(loss):
            raise AssertionError(f"{weight_name} produced non-finite total loss.")
        if not torch.isfinite(component_loss):
            raise AssertionError(f"{weight_name} produced non-finite component loss.")
        if float(component_loss.detach().cpu()) <= 0.0:
            raise AssertionError(f"{weight_name} did not activate {output_name}.")
        old_total = output.recon_loss + output.commitment_loss + output.codebook_loss
        if float((loss - old_total).detach().cpu()) <= 0.0:
            raise AssertionError(f"{weight_name} did not increase total loss.")
        check_gradients(tokenizer, loss, name=weight_name)
        summaries.append(
            f"{weight_name}: component={float(component_loss.detach().cpu()):.8f} "
            f"aux={float(output.auxiliary_loss.detach().cpu()):.8f}"
        )

    print("PASS multifactor tokenizer auxiliary loss smoke")
    print(f"inputs={tuple(inputs.shape)}")
    print(f"conditions={tuple(conditions.shape)}")
    print(f"projection_basis={tuple(context.projection_basis.shape)}")
    print(f"sector_labels={tuple(context.sector_labels.shape)}")
    print(f"zero_loss={float(zero_output.loss.detach().cpu()):.8f}")
    for summary in summaries:
        print(summary)
    return 0


def assert_close(left: Tensor, right: Tensor, *, name: str, atol: float = 1e-7) -> None:
    """Assert two scalar tensors are close."""
    if not torch.allclose(left.detach(), right.detach(), atol=atol, rtol=0.0):
        raise AssertionError(
            f"{name} mismatch: left={float(left.detach().cpu()):.10f}, "
            f"right={float(right.detach().cpu()):.10f}."
        )


def check_gradients(tokenizer: CausalVQTokenizer, loss: Tensor, *, name: str) -> None:
    """Backpropagate and assert finite non-zero parameter gradients."""
    tokenizer.zero_grad(set_to_none=True)
    loss.backward()
    total_abs_grad = 0.0
    for parameter in tokenizer.parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError(f"{name} produced a non-finite gradient.")
            total_abs_grad += float(parameter.grad.detach().abs().sum().cpu())
    if total_abs_grad <= 0.0:
        raise AssertionError(f"{name} produced zero total gradient.")


if __name__ == "__main__":
    raise SystemExit(main())
