"""Cross-sectional diagnostics for multi-asset return paths."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import sqrt
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "compare_cross_sectional_diagnostics",
    "compare_factor_coordinate_reconstruction",
    "correlation_frobenius_error",
    "covariance_frobenius_error",
    "eigenvalue_spectrum_distance",
    "empirical_correlation_matrix",
    "empirical_covariance_matrix",
    "equal_weight_portfolio_returns",
    "factor_beta_loading_diagnostic",
    "factor_coordinate_distribution_diagnostics",
    "factor_projection_subspace_consistency",
    "inverse_standardize_returns",
    "inverse_standardize_returns_from_metadata",
    "multifactor_detected_jump_tail_diagnostics",
    "multifactor_jump_stress_diagnostics",
    "portfolio_var_es",
    "random_portfolio_return_diagnostics",
    "random_portfolio_returns",
    "reconstruct_returns_from_factor_coordinates",
    "reconstructed_factor_cross_sectional_diagnostics",
    "sector_block_correlation_summary",
]


def inverse_standardize_returns(
    standardized_returns: Tensor,
    *,
    mean: Tensor | list[float],
    std: Tensor | list[float],
) -> Tensor:
    """Map standardised multi-asset returns back to raw log-return scale."""
    returns = _validate_returns(standardized_returns)
    mean_tensor = _as_asset_vector(mean, n_assets=returns.shape[-1], name="mean")
    std_tensor = _as_asset_vector(std, n_assets=returns.shape[-1], name="std")
    if not bool((std_tensor > 0.0).all()):
        raise ValueError("std entries must be positive.")
    mean_tensor = mean_tensor.to(device=returns.device, dtype=returns.dtype)
    std_tensor = std_tensor.to(device=returns.device, dtype=returns.dtype)
    return returns * std_tensor.view(1, 1, -1) + mean_tensor.view(1, 1, -1)


def inverse_standardize_returns_from_metadata(
    standardized_returns: Tensor,
    metadata: Mapping[str, Any],
) -> Tensor:
    """Inverse standardisation using dataset metadata if enabled."""
    standardization = metadata.get("standardization")
    if not isinstance(standardization, Mapping) or not standardization.get("enabled", False):
        return _validate_returns(standardized_returns)
    if "mean" not in standardization or "std" not in standardization:
        raise ValueError("metadata standardization requires 'mean' and 'std'.")
    return inverse_standardize_returns(
        standardized_returns,
        mean=standardization["mean"],
        std=standardization["std"],
    )


def empirical_covariance_matrix(returns: Tensor, *, demean: bool = True) -> Tensor:
    """Return the empirical asset covariance matrix.

    ``returns`` must have shape ``[batch, time, assets]``. The batch and time
    axes are pooled before estimating the cross-sectional covariance.
    """
    flat_returns = _flatten_cross_sectional_returns(returns)
    if demean:
        flat_returns = flat_returns - flat_returns.mean(dim=0, keepdim=True)
    denominator = max(flat_returns.shape[0] - 1, 1)
    return flat_returns.T.matmul(flat_returns) / float(denominator)


def empirical_correlation_matrix(returns: Tensor) -> Tensor:
    """Return the empirical asset correlation matrix."""
    return _covariance_to_correlation(empirical_covariance_matrix(returns))


def covariance_frobenius_error(
    reference: Tensor,
    candidate: Tensor,
    *,
    relative: bool = False,
) -> float:
    """Return Frobenius error between covariance matrices or return samples."""
    reference_covariance = _as_matrix_or_covariance(reference)
    candidate_covariance = _as_matrix_or_covariance(candidate)
    error = torch.linalg.matrix_norm(reference_covariance - candidate_covariance, ord="fro")
    if relative:
        denominator = torch.linalg.matrix_norm(reference_covariance, ord="fro").clamp_min(1e-12)
        error = error / denominator
    return float(error.item())


def correlation_frobenius_error(
    reference: Tensor,
    candidate: Tensor,
    *,
    relative: bool = False,
) -> float:
    """Return Frobenius error between correlation matrices or return samples."""
    reference_correlation = _as_matrix_or_correlation(reference)
    candidate_correlation = _as_matrix_or_correlation(candidate)
    error = torch.linalg.matrix_norm(reference_correlation - candidate_correlation, ord="fro")
    if relative:
        denominator = torch.linalg.matrix_norm(reference_correlation, ord="fro").clamp_min(1e-12)
        error = error / denominator
    return float(error.item())


def eigenvalue_spectrum_distance(
    reference: Tensor,
    candidate: Tensor,
    *,
    matrix: str = "correlation",
    normalize: bool = True,
) -> float:
    """Return L2 distance between sorted covariance or correlation spectra."""
    reference_matrix = _as_named_matrix(reference, matrix=matrix)
    candidate_matrix = _as_named_matrix(candidate, matrix=matrix)
    reference_values = torch.linalg.eigvalsh(reference_matrix).flip(0)
    candidate_values = torch.linalg.eigvalsh(candidate_matrix).flip(0)
    if normalize:
        reference_values = reference_values / reference_values.abs().sum().clamp_min(1e-12)
        candidate_values = candidate_values / candidate_values.abs().sum().clamp_min(1e-12)
    return float(torch.linalg.vector_norm(reference_values - candidate_values).item())


def sector_block_correlation_summary(
    returns_or_correlation: Tensor,
    sector_labels: Tensor,
) -> dict[str, Any]:
    """Summarise within- and between-sector correlation blocks."""
    correlation = _as_matrix_or_correlation(returns_or_correlation)
    labels = sector_labels.detach().cpu().long().reshape(-1)
    if labels.numel() != correlation.shape[0]:
        raise ValueError(
            "sector_labels length must match the asset dimension of the correlation matrix."
        )

    unique_labels = torch.unique(labels, sorted=True)
    blocks: list[dict[str, Any]] = []
    within_values: list[float] = []
    between_values: list[float] = []
    for row_sector in unique_labels.tolist():
        row_mask = labels == int(row_sector)
        for col_sector in unique_labels.tolist():
            col_mask = labels == int(col_sector)
            block = correlation[row_mask][:, col_mask]
            if row_sector == col_sector:
                block_values = block[~torch.eye(block.shape[0], dtype=torch.bool)]
                target = within_values
            else:
                block_values = block.reshape(-1)
                target = between_values
            block_mean = _safe_mean(block_values)
            if block_values.numel() > 0:
                target.append(block_mean)
            blocks.append({
                "row_sector": int(row_sector),
                "col_sector": int(col_sector),
                "asset_count": int(block.numel()),
                "mean_correlation": block_mean,
            })

    within_mean = _safe_float_mean(within_values)
    between_mean = _safe_float_mean(between_values)
    return {
        "sector_count": int(unique_labels.numel()),
        "within_sector_mean_correlation": within_mean,
        "between_sector_mean_correlation": between_mean,
        "within_minus_between": within_mean - between_mean,
        "blocks": blocks,
    }


def equal_weight_portfolio_returns(returns: Tensor) -> Tensor:
    """Return equal-weight portfolio log returns with shape ``[batch, time]``."""
    return _validate_returns(returns).mean(dim=-1)


def random_portfolio_returns(
    returns: Tensor,
    *,
    n_portfolios: int = 128,
    seed: int | None = 0,
    long_only: bool = True,
) -> Tensor:
    """Return random portfolio log returns with shape ``[portfolio, batch, time]``."""
    if n_portfolios <= 0:
        raise ValueError("n_portfolios must be positive.")
    validated_returns = _validate_returns(returns)
    n_assets = validated_returns.shape[-1]
    weights = _random_portfolio_weights(
        n_assets=n_assets,
        n_portfolios=n_portfolios,
        seed=seed,
        long_only=long_only,
        dtype=validated_returns.dtype,
    ).to(validated_returns.device)
    return torch.einsum("bta,pa->pbt", validated_returns, weights)


def portfolio_var_es(
    portfolio_returns: Tensor,
    *,
    levels: Iterable[float] = (0.01, 0.05),
) -> dict[str, float]:
    """Return lower-tail empirical VaR and expected shortfall for portfolio returns."""
    flattened = portfolio_returns.detach().float().reshape(-1)
    summary: dict[str, float] = {}
    for level in levels:
        _validate_probability(level, "level")
        key = _threshold_key(level)
        var = _value_at_risk(flattened, level=level)
        summary[f"lower_tail_var_{key}"] = var
        summary[f"lower_tail_es_{key}"] = _expected_shortfall(flattened, var=var)
    return summary


def random_portfolio_return_diagnostics(
    returns: Tensor,
    *,
    n_portfolios: int = 128,
    seed: int | None = 0,
    long_only: bool = True,
) -> dict[str, Any]:
    """Summarise random portfolio one-step, terminal, tail, and drawdown behaviour."""
    portfolio_returns = random_portfolio_returns(
        returns,
        n_portfolios=n_portfolios,
        seed=seed,
        long_only=long_only,
    )
    terminal_log_returns = portfolio_returns.sum(dim=-1)
    realised_volatilities = portfolio_returns.std(dim=-1, unbiased=False)
    drawdowns = _portfolio_max_drawdowns(portfolio_returns)
    return {
        "n_portfolios": int(n_portfolios),
        "long_only": bool(long_only),
        "one_step_mean": float(portfolio_returns.mean().item()),
        "one_step_std": float(portfolio_returns.std(unbiased=False).item()),
        "terminal_log_return_mean": float(terminal_log_returns.mean().item()),
        "terminal_log_return_std": float(terminal_log_returns.std(unbiased=False).item()),
        "realised_volatility_mean": float(realised_volatilities.mean().item()),
        "realised_volatility_std": float(realised_volatilities.std(unbiased=False).item()),
        "max_drawdown_mean": float(drawdowns.mean().item()),
        "max_drawdown_worst": float(drawdowns.min().item()),
        "var_es": portfolio_var_es(portfolio_returns),
    }


def factor_beta_loading_diagnostic(
    returns: Tensor,
    true_loadings: Tensor,
    *,
    n_factors: int | None = None,
) -> dict[str, float | int]:
    """Compare empirical principal subspace with true simulator loadings."""
    validated_returns = _validate_returns(returns)
    loadings = true_loadings.detach().float()
    if loadings.ndim != 2:
        raise ValueError("true_loadings must have shape [assets, factors].")
    if loadings.shape[0] != validated_returns.shape[-1]:
        raise ValueError("true_loadings asset dimension must match returns.")
    factor_count = min(
        int(n_factors if n_factors is not None else loadings.shape[1]),
        loadings.shape[1],
        validated_returns.shape[-1],
    )
    if factor_count <= 0:
        raise ValueError("n_factors must be positive.")

    covariance = empirical_covariance_matrix(validated_returns)
    eigenvectors = torch.linalg.eigh(covariance).eigenvectors[:, -factor_count:]
    true_basis = torch.linalg.qr(loadings[:, :factor_count], mode="reduced").Q
    empirical_basis = torch.linalg.qr(eigenvectors, mode="reduced").Q
    true_projection = true_basis.matmul(true_basis.T)
    empirical_projection = empirical_basis.matmul(empirical_basis.T)
    projection_error = torch.linalg.matrix_norm(
        true_projection - empirical_projection,
        ord="fro",
    )
    singular_values = torch.linalg.svdvals(true_basis.T.matmul(empirical_basis)).clamp(0.0, 1.0)
    principal_angles = torch.acos(singular_values)
    return {
        "factor_count": int(factor_count),
        "subspace_frobenius_error": float((projection_error / sqrt(float(factor_count))).item()),
        "mean_principal_cosine": float(singular_values.mean().item()),
        "max_principal_angle_rad": float(principal_angles.max().item()),
    }


def compare_factor_coordinate_reconstruction(
    reference_factors: Tensor,
    reconstructed_factors: Tensor,
) -> dict[str, float | int]:
    """Compare reconstructed factor coordinates with reference coordinates."""
    reference = _validate_factor_coordinates(reference_factors)
    reconstructed = _validate_factor_coordinates(reconstructed_factors)
    if reference.shape != reconstructed.shape:
        raise ValueError(
            "reference_factors and reconstructed_factors must have the same shape; "
            f"got {reference.shape} and {reconstructed.shape}."
        )
    error = reconstructed - reference
    return {
        "n_factors": int(reference.shape[-1]),
        "mae": float(error.abs().mean().item()),
        "mse": float(error.square().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "max_abs_error": float(error.abs().max().item()),
        "mean_coordinate_correlation": _mean_coordinate_correlation(reference, reconstructed),
    }


def factor_coordinate_distribution_diagnostics(
    reference_factors: Tensor,
    candidate_factors: Tensor,
) -> dict[str, float | int]:
    """Compare generated or reconstructed factor-coordinate distributions."""
    reference = _validate_factor_coordinates(reference_factors)
    candidate = _validate_factor_coordinates(candidate_factors)
    if reference.shape[-1] != candidate.shape[-1]:
        raise ValueError("reference_factors and candidate_factors must have the same factor count.")
    reference_flat = reference.reshape(-1, reference.shape[-1])
    candidate_flat = candidate.reshape(-1, candidate.shape[-1])
    reference_std = reference_flat.std(dim=0, unbiased=False).clamp_min(1e-12)
    candidate_std = candidate_flat.std(dim=0, unbiased=False)
    return {
        "n_factors": int(reference.shape[-1]),
        "mean_abs_error": float(
            (candidate_flat.mean(dim=0) - reference_flat.mean(dim=0)).abs().mean().item()
        ),
        "std_relative_mae": float(
            ((candidate_std - reference_std).abs() / reference_std).mean().item()
        ),
        "covariance_frobenius_error": covariance_frobenius_error(reference, candidate),
        "covariance_relative_frobenius_error": covariance_frobenius_error(
            reference,
            candidate,
            relative=True,
        ),
        "correlation_frobenius_error": correlation_frobenius_error(reference, candidate),
        "correlation_relative_frobenius_error": correlation_frobenius_error(
            reference,
            candidate,
            relative=True,
        ),
    }


def reconstruct_returns_from_factor_coordinates(
    factor_coordinates: Tensor,
    projection_basis: Tensor,
    *,
    projection_mean: Tensor | list[float] | None = None,
    standardization_mean: Tensor | list[float] | None = None,
    standardization_std: Tensor | list[float] | None = None,
) -> Tensor:
    """Map factor coordinates back to 50D returns for diagnostics."""
    factors = _validate_factor_coordinates(factor_coordinates)
    basis = _validate_projection_basis(projection_basis, n_factors=factors.shape[-1])
    basis = basis.to(device=factors.device, dtype=factors.dtype)
    if projection_mean is None:
        mean = factors.new_zeros(basis.shape[0])
    else:
        mean = _as_asset_vector(
            projection_mean,
            n_assets=basis.shape[0],
            name="projection_mean",
        ).to(device=factors.device, dtype=factors.dtype)
    reconstructed = torch.matmul(factors, basis.T) + mean.view(1, 1, -1)
    if standardization_mean is None and standardization_std is None:
        return reconstructed
    if standardization_mean is None or standardization_std is None:
        raise ValueError("standardization_mean and standardization_std must be provided together.")
    return inverse_standardize_returns(
        reconstructed,
        mean=standardization_mean,
        std=standardization_std,
    )


def reconstructed_factor_cross_sectional_diagnostics(
    reference_returns: Tensor,
    factor_coordinates: Tensor,
    projection_basis: Tensor,
    *,
    projection_mean: Tensor | list[float] | None = None,
    standardization_mean: Tensor | list[float] | None = None,
    standardization_std: Tensor | list[float] | None = None,
    sector_labels: Tensor | None = None,
    true_loadings: Tensor | None = None,
    n_random_portfolios: int = 128,
    random_seed: int | None = 0,
) -> dict[str, Any]:
    """Compare reference returns with returns reconstructed from factor coordinates."""
    reconstructed = reconstruct_returns_from_factor_coordinates(
        factor_coordinates,
        projection_basis,
        projection_mean=projection_mean,
        standardization_mean=standardization_mean,
        standardization_std=standardization_std,
    )
    return compare_cross_sectional_diagnostics(
        reference_returns,
        reconstructed,
        sector_labels=sector_labels,
        true_loadings=true_loadings,
        n_random_portfolios=n_random_portfolios,
        random_seed=random_seed,
    )


def factor_projection_subspace_consistency(
    projection_basis: Tensor,
    true_loadings: Tensor,
    *,
    n_factors: int | None = None,
) -> dict[str, float | int]:
    """Compare a fitted projection basis with simulator loadings."""
    basis = projection_basis.detach().float()
    loadings = true_loadings.detach().float()
    if basis.ndim != 2:
        raise ValueError("projection_basis must have shape [assets, factors].")
    if loadings.ndim != 2:
        raise ValueError("true_loadings must have shape [assets, factors].")
    if basis.shape[0] != loadings.shape[0]:
        raise ValueError("projection_basis and true_loadings must share the asset dimension.")
    factor_count = min(
        int(n_factors if n_factors is not None else basis.shape[1]),
        basis.shape[1],
        loadings.shape[1],
    )
    if factor_count <= 0:
        raise ValueError("n_factors must be positive.")

    projection_basis_q = torch.linalg.qr(basis[:, :factor_count], mode="reduced").Q
    true_basis_q = torch.linalg.qr(loadings[:, :factor_count], mode="reduced").Q
    projection = projection_basis_q.matmul(projection_basis_q.T)
    true_projection = true_basis_q.matmul(true_basis_q.T)
    projection_error = torch.linalg.matrix_norm(projection - true_projection, ord="fro")
    singular_values = torch.linalg.svdvals(true_basis_q.T.matmul(projection_basis_q)).clamp(
        0.0,
        1.0,
    )
    principal_angles = torch.acos(singular_values)
    return {
        "factor_count": int(factor_count),
        "subspace_frobenius_error": float((projection_error / sqrt(float(factor_count))).item()),
        "mean_principal_cosine": float(singular_values.mean().item()),
        "max_principal_angle_rad": float(principal_angles.max().item()),
    }


def compare_cross_sectional_diagnostics(
    reference_returns: Tensor,
    candidate_returns: Tensor,
    *,
    sector_labels: Tensor | None = None,
    true_loadings: Tensor | None = None,
    n_random_portfolios: int = 128,
    random_seed: int | None = 0,
) -> dict[str, Any]:
    """Compare two multi-asset return samples with cross-sectional diagnostics."""
    reference = _validate_returns(reference_returns)
    candidate = _validate_returns(candidate_returns)
    if reference.shape[-1] != candidate.shape[-1]:
        raise ValueError("reference and candidate must have the same asset dimension.")

    reference_correlation = empirical_correlation_matrix(reference)
    candidate_correlation = empirical_correlation_matrix(candidate)
    summary: dict[str, Any] = {
        "covariance_frobenius_error": covariance_frobenius_error(reference, candidate),
        "covariance_relative_frobenius_error": covariance_frobenius_error(
            reference,
            candidate,
            relative=True,
        ),
        "correlation_frobenius_error": correlation_frobenius_error(reference, candidate),
        "correlation_relative_frobenius_error": correlation_frobenius_error(
            reference,
            candidate,
            relative=True,
        ),
        "correlation_eigenvalue_spectrum_distance": eigenvalue_spectrum_distance(
            reference,
            candidate,
            matrix="correlation",
        ),
        "covariance_eigenvalue_spectrum_distance": eigenvalue_spectrum_distance(
            reference,
            candidate,
            matrix="covariance",
        ),
        "equal_weight": {
            "reference": _portfolio_return_summary(equal_weight_portfolio_returns(reference)),
            "candidate": _portfolio_return_summary(equal_weight_portfolio_returns(candidate)),
        },
        "random_portfolios": {
            "reference": random_portfolio_return_diagnostics(
                reference,
                n_portfolios=n_random_portfolios,
                seed=random_seed,
            ),
            "candidate": random_portfolio_return_diagnostics(
                candidate,
                n_portfolios=n_random_portfolios,
                seed=random_seed,
            ),
        },
    }
    if sector_labels is not None:
        summary["sector_blocks"] = {
            "reference": sector_block_correlation_summary(reference_correlation, sector_labels),
            "candidate": sector_block_correlation_summary(candidate_correlation, sector_labels),
        }
    if true_loadings is not None:
        summary["factor_loadings"] = {
            "reference": factor_beta_loading_diagnostic(reference, true_loadings),
            "candidate": factor_beta_loading_diagnostic(candidate, true_loadings),
        }
    return summary


def multifactor_jump_stress_diagnostics(
    returns: Tensor,
    *,
    common_jump_indicators: Tensor,
    sector_jump_indicators: Tensor,
    sector_labels: Tensor,
    jump_sizes: Tensor | None = None,
    n_random_portfolios: int = 128,
    random_seed: int | None = 0,
    tail_level: float = 0.05,
) -> dict[str, Any]:
    """Summarise oracle common/sector jump stress diagnostics.

    The jump masks are simulator oracle metadata and must remain diagnostic-only.
    They are not model-visible conditions.
    """
    validated_returns = _validate_returns(returns)
    _validate_probability(tail_level, "tail_level")
    common_mask = _common_jump_mask(
        common_jump_indicators,
        batch_size=validated_returns.shape[0],
        n_timesteps=validated_returns.shape[1],
    )
    sector_mask = _sector_jump_mask(
        sector_jump_indicators,
        batch_size=validated_returns.shape[0],
        n_timesteps=validated_returns.shape[1],
    )
    if sector_mask.shape[-1] == 0:
        raise ValueError("sector_jump_indicators must contain at least one sector.")
    sector_any_mask = sector_mask.any(dim=-1)
    jump_mask = common_mask | sector_any_mask
    non_jump_mask = ~jump_mask
    sector_counts = sector_mask.float().sum(dim=-1)
    equal_weight_returns_tensor = equal_weight_portfolio_returns(validated_returns)
    random_portfolios = random_portfolio_returns(
        validated_returns,
        n_portfolios=n_random_portfolios,
        seed=random_seed,
    )
    tail_threshold = _value_at_risk(equal_weight_returns_tensor, level=tail_level)
    tail_mask = equal_weight_returns_tensor <= tail_threshold

    return {
        "oracle_metadata_model_visible": False,
        "counts": {
            "window_count": int(jump_mask.numel()),
            "jump_window_count": int(jump_mask.sum().item()),
            "jump_window_fraction": _mask_fraction(jump_mask),
            "non_jump_window_count": int(non_jump_mask.sum().item()),
            "common_jump_count": int(common_mask.sum().item()),
            "common_jump_fraction": _mask_fraction(common_mask),
            "sector_jump_count": int(sector_mask.sum().item()),
            "sector_jump_step_count": int(sector_any_mask.sum().item()),
            "sector_jump_step_fraction": _mask_fraction(sector_any_mask),
        },
        "sector_synchronization": _sector_jump_synchronization(
            common_mask=common_mask,
            sector_any_mask=sector_any_mask,
            sector_counts=sector_counts,
        ),
        "portfolio_tail_co_movement": _portfolio_tail_co_movement(
            equal_weight_returns_tensor=equal_weight_returns_tensor,
            random_portfolios=random_portfolios,
            jump_mask=jump_mask,
            tail_mask=tail_mask,
            tail_level=tail_level,
        ),
        "cross_sectional_tail_correlation": _masked_cross_sectional_summary(
            validated_returns,
            tail_mask,
            sector_labels=sector_labels,
        ),
        "conditional_sector_blocks": {
            "jump_windows": _masked_cross_sectional_summary(
                validated_returns,
                jump_mask,
                sector_labels=sector_labels,
            ),
            "non_jump_windows": _masked_cross_sectional_summary(
                validated_returns,
                non_jump_mask,
                sector_labels=sector_labels,
            ),
        },
        "jump_size_summary": _jump_size_summary(jump_sizes),
    }


def multifactor_detected_jump_tail_diagnostics(
    reference_returns: Tensor,
    generated_returns: Tensor,
    *,
    sector_labels: Tensor,
    jump_level: float = 0.05,
    tail_level: float = 0.05,
    n_random_portfolios: int = 128,
    random_seed: int | None = 0,
) -> dict[str, Any]:
    """Compare jump/tail structure using thresholds fitted on reference returns.

    This diagnostic is intended for unconditional generated samples. It does
    not use reference oracle jump masks to label generated windows.
    """
    reference = _validate_returns(reference_returns)
    generated = _validate_returns(generated_returns)
    if reference.shape[-1] != generated.shape[-1]:
        raise ValueError("reference_returns and generated_returns must share asset dimension.")
    _validate_probability(jump_level, "jump_level")
    _validate_probability(tail_level, "tail_level")

    thresholds = _fit_detected_jump_thresholds(
        reference,
        sector_labels=sector_labels,
        jump_level=jump_level,
        tail_level=tail_level,
    )
    reference_summary = _detected_jump_tail_summary(
        reference,
        thresholds=thresholds,
        sector_labels=sector_labels,
        n_random_portfolios=n_random_portfolios,
        random_seed=random_seed,
        role="reference_detected",
    )
    generated_summary = _detected_jump_tail_summary(
        generated,
        thresholds=thresholds,
        sector_labels=sector_labels,
        n_random_portfolios=n_random_portfolios,
        random_seed=random_seed,
        role="generated_detected",
    )
    return {
        "diagnostic_type": "unconditional_detected_jump_tail",
        "generated_uses_reference_oracle_masks": False,
        "threshold_convention": (
            "Thresholds are fitted on held-out reference returns and then applied "
            "independently to reference and generated samples."
        ),
        "thresholds": _json_thresholds(thresholds),
        "reference_detected_jump_diagnostics": reference_summary,
        "detected_generated_jump_diagnostics": generated_summary,
        "distributional_comparison": _detected_jump_distributional_comparison(
            reference_summary,
            generated_summary,
        ),
    }


def _fit_detected_jump_thresholds(
    returns: Tensor,
    *,
    sector_labels: Tensor,
    jump_level: float,
    tail_level: float,
) -> dict[str, Tensor | float]:
    equal_weight = equal_weight_portfolio_returns(returns)
    sector_returns = _sector_portfolio_returns(returns, sector_labels)
    return {
        "jump_level": float(jump_level),
        "tail_level": float(tail_level),
        "equal_weight_abs_jump": torch.quantile(equal_weight.abs().reshape(-1), 1.0 - jump_level),
        "equal_weight_lower_tail": torch.quantile(equal_weight.reshape(-1), tail_level),
        "sector_abs_jump": torch.quantile(
            sector_returns.abs().reshape(-1, sector_returns.shape[-1]),
            1.0 - jump_level,
            dim=0,
        ),
    }


def _detected_jump_tail_summary(
    returns: Tensor,
    *,
    thresholds: Mapping[str, Tensor | float],
    sector_labels: Tensor,
    n_random_portfolios: int,
    random_seed: int | None,
    role: str,
) -> dict[str, Any]:
    equal_weight = equal_weight_portfolio_returns(returns)
    sector_returns = _sector_portfolio_returns(returns, sector_labels)
    common_mask = equal_weight.abs() >= _threshold_tensor(
        thresholds["equal_weight_abs_jump"],
        device=returns.device,
        dtype=returns.dtype,
    )
    sector_thresholds = _threshold_tensor(
        thresholds["sector_abs_jump"],
        device=returns.device,
        dtype=returns.dtype,
    ).view(1, 1, -1)
    sector_mask = sector_returns.abs() >= sector_thresholds
    sector_any_mask = sector_mask.any(dim=-1)
    jump_mask = common_mask | sector_any_mask
    non_jump_mask = ~jump_mask
    tail_mask = equal_weight <= _threshold_tensor(
        thresholds["equal_weight_lower_tail"],
        device=returns.device,
        dtype=returns.dtype,
    )
    sector_counts = sector_mask.float().sum(dim=-1)
    random_portfolios = random_portfolio_returns(
        returns,
        n_portfolios=n_random_portfolios,
        seed=random_seed,
    )
    common_sector_steps = common_mask & sector_any_mask
    return {
        "role": role,
        "uses_oracle_masks": False,
        "counts": {
            "window_count": int(jump_mask.numel()),
            "detected_jump_window_count": int(jump_mask.sum().item()),
            "detected_jump_window_fraction": _mask_fraction(jump_mask),
            "detected_non_jump_window_count": int(non_jump_mask.sum().item()),
            "detected_common_jump_count": int(common_mask.sum().item()),
            "detected_common_jump_fraction": _mask_fraction(common_mask),
            "detected_sector_jump_count": int(sector_mask.sum().item()),
            "detected_sector_jump_step_count": int(sector_any_mask.sum().item()),
            "detected_sector_jump_step_fraction": _mask_fraction(sector_any_mask),
        },
        "jump_count_distribution": {
            "detected_jump_windows_per_path": _count_distribution(jump_mask.sum(dim=1).float()),
            "detected_common_windows_per_path": _count_distribution(common_mask.sum(dim=1).float()),
            "detected_sector_windows_per_path": _count_distribution(
                sector_any_mask.sum(dim=1).float(),
            ),
        },
        "sector_cojump_distribution": {
            "active_sector_count_per_window": _count_distribution(sector_counts.reshape(-1)),
            "multi_sector_jump_step_count": int((sector_counts > 1.0).sum().item()),
            "multi_sector_jump_step_fraction": _conditional_fraction(
                sector_counts > 1.0,
                sector_any_mask,
            ),
            "mean_active_sectors_per_sector_jump_step": _conditional_mean(
                sector_counts,
                sector_any_mask,
            ),
            "max_active_sectors_per_step": int(sector_counts.max().item()),
            "common_sector_cojump_step_count": int(common_sector_steps.sum().item()),
            "common_sector_cojump_fraction_of_common": _conditional_fraction(
                common_sector_steps,
                common_mask,
            ),
            "common_sector_cojump_fraction_of_sector": _conditional_fraction(
                common_sector_steps,
                sector_any_mask,
            ),
        },
        "portfolio_tail_co_movement": _portfolio_tail_co_movement(
            equal_weight_returns_tensor=equal_weight,
            random_portfolios=random_portfolios,
            jump_mask=jump_mask,
            tail_mask=tail_mask,
            tail_level=float(thresholds["tail_level"]),
        ),
        "tail_window_correlation": _masked_cross_sectional_summary(
            returns,
            tail_mask,
            sector_labels=sector_labels,
        ),
        "conditional_cross_sectional": {
            "detected_jump_windows": _masked_cross_sectional_summary(
                returns,
                jump_mask,
                sector_labels=sector_labels,
            ),
            "detected_non_jump_windows": _masked_cross_sectional_summary(
                returns,
                non_jump_mask,
                sector_labels=sector_labels,
            ),
        },
    }


def _detected_jump_distributional_comparison(
    reference: Mapping[str, Any],
    generated: Mapping[str, Any],
) -> dict[str, float]:
    ref_counts = _nested_mapping(reference, "counts")
    gen_counts = _nested_mapping(generated, "counts")
    ref_tail = _nested_mapping(
        _nested_mapping(reference, "portfolio_tail_co_movement"), "tail_jump_overlap"
    )
    gen_tail = _nested_mapping(
        _nested_mapping(generated, "portfolio_tail_co_movement"), "tail_jump_overlap"
    )
    ref_portfolio = _nested_mapping(
        _nested_mapping(reference, "portfolio_tail_co_movement"), "equal_weight"
    )
    gen_portfolio = _nested_mapping(
        _nested_mapping(generated, "portfolio_tail_co_movement"), "equal_weight"
    )
    ref_cross = _nested_mapping(reference, "conditional_cross_sectional")
    gen_cross = _nested_mapping(generated, "conditional_cross_sectional")
    ref_jump = _nested_mapping(ref_cross, "detected_jump_windows")
    gen_jump = _nested_mapping(gen_cross, "detected_jump_windows")
    ref_non_jump = _nested_mapping(ref_cross, "detected_non_jump_windows")
    gen_non_jump = _nested_mapping(gen_cross, "detected_non_jump_windows")
    ref_tail_corr = _nested_mapping(reference, "tail_window_correlation")
    gen_tail_corr = _nested_mapping(generated, "tail_window_correlation")
    ref_counts_dist = _nested_mapping(
        _nested_mapping(reference, "jump_count_distribution"), "detected_jump_windows_per_path"
    )
    gen_counts_dist = _nested_mapping(
        _nested_mapping(generated, "jump_count_distribution"), "detected_jump_windows_per_path"
    )
    ref_sector_dist = _nested_mapping(
        _nested_mapping(reference, "sector_cojump_distribution"), "active_sector_count_per_window"
    )
    gen_sector_dist = _nested_mapping(
        _nested_mapping(generated, "sector_cojump_distribution"), "active_sector_count_per_window"
    )
    ref_sector = _nested_mapping(reference, "sector_cojump_distribution")
    gen_sector = _nested_mapping(generated, "sector_cojump_distribution")
    ref_ew_all = _nested_mapping(ref_portfolio, "all_windows")
    gen_ew_all = _nested_mapping(gen_portfolio, "all_windows")
    ref_ew_jump = _nested_mapping(ref_portfolio, "jump_windows")
    gen_ew_jump = _nested_mapping(gen_portfolio, "jump_windows")
    ref_ew_non_jump = _nested_mapping(ref_portfolio, "non_jump_windows")
    gen_ew_non_jump = _nested_mapping(gen_portfolio, "non_jump_windows")
    return {
        "detected_jump_window_fraction_abs_error": abs(
            float(gen_counts["detected_jump_window_fraction"])
            - float(ref_counts["detected_jump_window_fraction"])
        ),
        "detected_common_jump_fraction_abs_error": abs(
            float(gen_counts["detected_common_jump_fraction"])
            - float(ref_counts["detected_common_jump_fraction"])
        ),
        "detected_sector_jump_step_fraction_abs_error": abs(
            float(gen_counts["detected_sector_jump_step_fraction"])
            - float(ref_counts["detected_sector_jump_step_fraction"])
        ),
        "multi_sector_jump_step_fraction_abs_error": abs(
            float(gen_sector["multi_sector_jump_step_fraction"])
            - float(ref_sector["multi_sector_jump_step_fraction"])
        ),
        "tail_fraction_within_detected_jump_abs_error": abs(
            float(gen_tail["tail_fraction_within_jump"])
            - float(ref_tail["tail_fraction_within_jump"])
        ),
        "detected_jump_fraction_within_tail_abs_error": abs(
            float(gen_tail["jump_fraction_within_tail"])
            - float(ref_tail["jump_fraction_within_tail"])
        ),
        "detected_jump_count_per_path_mean_abs_error": abs(
            float(gen_counts_dist["mean"]) - float(ref_counts_dist["mean"])
        ),
        "sector_cojump_active_count_mean_abs_error": abs(
            float(gen_sector_dist["mean"]) - float(ref_sector_dist["mean"])
        ),
        "equal_weight_all_std_abs_error": abs(float(gen_ew_all["std"]) - float(ref_ew_all["std"])),
        "equal_weight_jump_std_abs_error": abs(
            float(gen_ew_jump["std"]) - float(ref_ew_jump["std"])
        ),
        "equal_weight_non_jump_std_abs_error": abs(
            float(gen_ew_non_jump["std"]) - float(ref_ew_non_jump["std"])
        ),
        "jump_window_mean_abs_correlation_abs_error": abs(
            float(gen_jump["mean_abs_offdiagonal_correlation"])
            - float(ref_jump["mean_abs_offdiagonal_correlation"])
        ),
        "non_jump_window_mean_abs_correlation_abs_error": abs(
            float(gen_non_jump["mean_abs_offdiagonal_correlation"])
            - float(ref_non_jump["mean_abs_offdiagonal_correlation"])
        ),
        "tail_window_mean_abs_correlation_abs_error": abs(
            float(gen_tail_corr["mean_abs_offdiagonal_correlation"])
            - float(ref_tail_corr["mean_abs_offdiagonal_correlation"])
        ),
        "jump_window_sector_block_mae": _sector_summary_mae(ref_jump, gen_jump),
        "non_jump_window_sector_block_mae": _sector_summary_mae(ref_non_jump, gen_non_jump),
        "tail_window_sector_block_mae": _sector_summary_mae(ref_tail_corr, gen_tail_corr),
    }


def _sector_portfolio_returns(returns: Tensor, sector_labels: Tensor) -> Tensor:
    labels = sector_labels.detach().cpu().long().reshape(-1)
    if labels.numel() != returns.shape[-1]:
        raise ValueError("sector_labels length must match asset dimension.")
    sector_returns: list[Tensor] = []
    for sector in torch.unique(labels, sorted=True).tolist():
        mask = labels == int(sector)
        sector_returns.append(returns[..., mask.to(device=returns.device)].mean(dim=-1))
    if not sector_returns:
        raise ValueError("sector_labels must contain at least one sector.")
    return torch.stack(sector_returns, dim=-1)


def _threshold_tensor(value: Tensor | float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _json_thresholds(thresholds: Mapping[str, Tensor | float]) -> dict[str, Any]:
    return {
        key: value.detach().cpu().tolist() if isinstance(value, Tensor) else float(value)
        for key, value in thresholds.items()
    }


def _count_distribution(values: Tensor) -> dict[str, float | int]:
    flattened = values.detach().float().reshape(-1)
    if flattened.numel() == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "q05": 0.0, "q50": 0.0, "q95": 0.0}
    quantiles = torch.quantile(flattened, torch.tensor([0.05, 0.50, 0.95]))
    return {
        "count": int(flattened.numel()),
        "mean": float(flattened.mean().item()),
        "std": float(flattened.std(unbiased=False).item()),
        "q05": float(quantiles[0].item()),
        "q50": float(quantiles[1].item()),
        "q95": float(quantiles[2].item()),
    }


def _nested_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping at key {key!r}.")
    return value


def _sector_summary_mae(reference: Mapping[str, Any], generated: Mapping[str, Any]) -> float:
    reference_blocks = reference.get("sector_blocks")
    generated_blocks = generated.get("sector_blocks")
    if not isinstance(reference_blocks, Mapping) or not isinstance(generated_blocks, Mapping):
        return 0.0
    reference_entries = reference_blocks.get("blocks")
    generated_entries = generated_blocks.get("blocks")
    if not isinstance(reference_entries, list) or not isinstance(generated_entries, list):
        return 0.0
    if len(reference_entries) != len(generated_entries):
        raise ValueError("sector-block summaries must have matching lengths.")
    errors: list[float] = []
    for reference_entry, generated_entry in zip(reference_entries, generated_entries, strict=True):
        if not isinstance(reference_entry, Mapping) or not isinstance(generated_entry, Mapping):
            raise ValueError("sector-block entries must be mappings.")
        reference_key = (reference_entry["row_sector"], reference_entry["col_sector"])
        generated_key = (generated_entry["row_sector"], generated_entry["col_sector"])
        if reference_key != generated_key:
            raise ValueError("sector-block summaries must be aligned.")
        errors.append(
            abs(
                float(generated_entry["mean_correlation"])
                - float(reference_entry["mean_correlation"])
            )
        )
    return _safe_float_mean(errors)


def _common_jump_mask(indicators: Tensor, *, batch_size: int, n_timesteps: int) -> Tensor:
    mask = indicators.detach().bool()
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2 or mask.shape != (batch_size, n_timesteps):
        raise ValueError(
            "common_jump_indicators must have shape [batch, time] or [batch, time, 1]; "
            f"got {tuple(indicators.shape)}."
        )
    return mask


def _sector_jump_mask(indicators: Tensor, *, batch_size: int, n_timesteps: int) -> Tensor:
    mask = indicators.detach().bool()
    if mask.ndim != 3 or mask.shape[0] != batch_size or mask.shape[1] != n_timesteps:
        raise ValueError(
            "sector_jump_indicators must have shape [batch, time, sectors]; "
            f"got {tuple(indicators.shape)}."
        )
    return mask


def _mask_fraction(mask: Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def _sector_jump_synchronization(
    *,
    common_mask: Tensor,
    sector_any_mask: Tensor,
    sector_counts: Tensor,
) -> dict[str, float | int]:
    active_sector_steps = sector_counts > 0.0
    multi_sector_steps = sector_counts > 1.0
    common_sector_steps = common_mask & sector_any_mask
    return {
        "active_sector_jump_step_count": int(active_sector_steps.sum().item()),
        "multi_sector_jump_step_count": int(multi_sector_steps.sum().item()),
        "multi_sector_jump_step_fraction": _conditional_fraction(
            multi_sector_steps,
            active_sector_steps,
        ),
        "mean_active_sectors_per_sector_jump_step": _conditional_mean(
            sector_counts,
            active_sector_steps,
        ),
        "max_active_sectors_per_step": int(sector_counts.max().item()),
        "common_sector_cojump_step_count": int(common_sector_steps.sum().item()),
        "common_sector_cojump_fraction_of_common": _conditional_fraction(
            common_sector_steps,
            common_mask,
        ),
        "common_sector_cojump_fraction_of_sector": _conditional_fraction(
            common_sector_steps,
            sector_any_mask,
        ),
    }


def _portfolio_tail_co_movement(
    *,
    equal_weight_returns_tensor: Tensor,
    random_portfolios: Tensor,
    jump_mask: Tensor,
    tail_mask: Tensor,
    tail_level: float,
) -> dict[str, Any]:
    random_jump_returns = random_portfolios[:, jump_mask]
    random_non_jump_returns = random_portfolios[:, ~jump_mask]
    return {
        "tail_level": float(tail_level),
        "equal_weight": {
            "all_windows": _flat_return_summary(equal_weight_returns_tensor),
            "jump_windows": _flat_return_summary(equal_weight_returns_tensor[jump_mask]),
            "non_jump_windows": _flat_return_summary(equal_weight_returns_tensor[~jump_mask]),
        },
        "random_portfolios": {
            "all_windows": _flat_return_summary(random_portfolios),
            "jump_windows": _flat_return_summary(random_jump_returns),
            "non_jump_windows": _flat_return_summary(random_non_jump_returns),
        },
        "tail_jump_overlap": {
            "tail_window_count": int(tail_mask.sum().item()),
            "tail_window_fraction": _mask_fraction(tail_mask),
            "jump_windows_in_tail_count": int((tail_mask & jump_mask).sum().item()),
            "jump_fraction_within_tail": _conditional_fraction(jump_mask, tail_mask),
            "tail_fraction_within_jump": _conditional_fraction(tail_mask, jump_mask),
        },
    }


def _flat_return_summary(values: Tensor) -> dict[str, Any]:
    flattened = values.detach().float().reshape(-1)
    if flattened.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "var_es": portfolio_var_es(flattened),
        }
    return {
        "count": int(flattened.numel()),
        "mean": float(flattened.mean().item()),
        "std": float(flattened.std(unbiased=False).item()),
        "var_es": portfolio_var_es(flattened),
    }


def _masked_cross_sectional_summary(
    returns: Tensor,
    mask: Tensor,
    *,
    sector_labels: Tensor,
) -> dict[str, Any]:
    window_count = int(mask.sum().item())
    if window_count < 2:
        return {
            "available": False,
            "window_count": window_count,
            "mean_abs_offdiagonal_correlation": 0.0,
            "sector_blocks": None,
        }
    masked_returns = returns[mask].view(1, window_count, returns.shape[-1])
    correlation = empirical_correlation_matrix(masked_returns)
    return {
        "available": True,
        "window_count": window_count,
        "mean_abs_offdiagonal_correlation": _mean_abs_offdiagonal(correlation),
        "sector_blocks": sector_block_correlation_summary(correlation, sector_labels),
    }


def _jump_size_summary(jump_sizes: Tensor | None) -> dict[str, float | int | bool]:
    if jump_sizes is None:
        return {"available": False}
    sizes = jump_sizes.detach().float()
    if sizes.ndim != 3:
        raise ValueError(f"jump_sizes must have shape [batch, time, assets]; got {sizes.shape}.")
    nonzero = sizes[sizes.abs() > 0.0]
    if nonzero.numel() == 0:
        return {
            "available": True,
            "nonzero_asset_jump_count": 0,
            "negative_jump_fraction": 0.0,
            "mean_jump_size": 0.0,
            "mean_abs_jump_size": 0.0,
            "max_abs_jump_size": 0.0,
        }
    return {
        "available": True,
        "nonzero_asset_jump_count": int(nonzero.numel()),
        "negative_jump_fraction": float((nonzero < 0.0).float().mean().item()),
        "mean_jump_size": float(nonzero.mean().item()),
        "mean_abs_jump_size": float(nonzero.abs().mean().item()),
        "max_abs_jump_size": float(nonzero.abs().max().item()),
    }


def _conditional_fraction(numerator_mask: Tensor, denominator_mask: Tensor) -> float:
    denominator = int(denominator_mask.sum().item())
    if denominator == 0:
        return 0.0
    numerator = int((numerator_mask & denominator_mask).sum().item())
    return float(numerator / denominator)


def _conditional_mean(values: Tensor, mask: Tensor) -> float:
    if not bool(mask.any().item()):
        return 0.0
    return float(values[mask].float().mean().item())


def _mean_abs_offdiagonal(matrix: Tensor) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return float(matrix[mask].abs().mean().item())


def _validate_returns(returns: Tensor) -> Tensor:
    if returns.ndim != 3:
        raise ValueError(f"Expected returns with shape [batch, time, assets]; got {returns.shape}.")
    if returns.shape[-1] < 2:
        raise ValueError("Expected at least two assets for cross-sectional diagnostics.")
    if not bool(torch.isfinite(returns).all()):
        raise ValueError("returns must be finite.")
    return returns.detach().float()


def _validate_factor_coordinates(factor_coordinates: Tensor) -> Tensor:
    if factor_coordinates.ndim != 3:
        raise ValueError(
            "Expected factor coordinates with shape [batch, time, factors]; "
            f"got {factor_coordinates.shape}."
        )
    if factor_coordinates.shape[-1] <= 0:
        raise ValueError("Expected at least one factor coordinate.")
    if not bool(torch.isfinite(factor_coordinates).all()):
        raise ValueError("factor coordinates must be finite.")
    return factor_coordinates.detach().float()


def _validate_projection_basis(projection_basis: Tensor, *, n_factors: int) -> Tensor:
    basis = projection_basis.detach().float()
    if basis.ndim != 2:
        raise ValueError(f"projection_basis must have shape [assets, factors]; got {basis.shape}.")
    if basis.shape[-1] != n_factors:
        raise ValueError(f"projection_basis must have {n_factors} factors; got {basis.shape[-1]}.")
    if basis.shape[0] < 2:
        raise ValueError("projection_basis must contain at least two assets.")
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("projection_basis must be finite.")
    return basis


def _as_asset_vector(values: Tensor | list[float], *, n_assets: int, name: str) -> Tensor:
    vector = torch.as_tensor(values).detach().float().reshape(-1)
    if vector.numel() != n_assets:
        raise ValueError(f"{name} must have length {n_assets}; got {vector.numel()}.")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{name} must be finite.")
    return vector


def _flatten_cross_sectional_returns(returns: Tensor) -> Tensor:
    validated_returns = _validate_returns(returns)
    return validated_returns.reshape(-1, validated_returns.shape[-1])


def _as_matrix_or_covariance(values: Tensor) -> Tensor:
    if values.ndim == 2 and values.shape[0] == values.shape[1]:
        return values.detach().float()
    return empirical_covariance_matrix(values)


def _as_matrix_or_correlation(values: Tensor) -> Tensor:
    if values.ndim == 2 and values.shape[0] == values.shape[1]:
        return values.detach().float()
    return empirical_correlation_matrix(values)


def _as_named_matrix(values: Tensor, *, matrix: str) -> Tensor:
    if matrix == "covariance":
        return _as_matrix_or_covariance(values)
    if matrix == "correlation":
        return _as_matrix_or_correlation(values)
    raise ValueError("matrix must be 'covariance' or 'correlation'.")


def _covariance_to_correlation(covariance: Tensor) -> Tensor:
    std = covariance.diag().clamp_min(1e-12).sqrt()
    correlation = covariance / (std.view(-1, 1) * std.view(1, -1))
    correlation = correlation.clamp(-1.0, 1.0)
    correlation.fill_diagonal_(1.0)
    return correlation


def _random_portfolio_weights(
    *,
    n_assets: int,
    n_portfolios: int,
    seed: int | None,
    long_only: bool,
    dtype: torch.dtype,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    if long_only:
        weights = torch.rand((n_portfolios, n_assets), generator=generator, dtype=dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    weights = torch.randn((n_portfolios, n_assets), generator=generator, dtype=dtype)
    return weights / weights.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _portfolio_max_drawdowns(portfolio_returns: Tensor) -> Tensor:
    cumulative = portfolio_returns.cumsum(dim=-1)
    running_peak = cumulative.cummax(dim=-1).values
    drawdowns = cumulative - running_peak
    return drawdowns.min(dim=-1).values


def _portfolio_return_summary(portfolio_returns: Tensor) -> dict[str, Any]:
    terminal = portfolio_returns.sum(dim=-1)
    drawdowns = _portfolio_max_drawdowns(portfolio_returns)
    return {
        "one_step_mean": float(portfolio_returns.mean().item()),
        "one_step_std": float(portfolio_returns.std(unbiased=False).item()),
        "terminal_log_return_mean": float(terminal.mean().item()),
        "terminal_log_return_std": float(terminal.std(unbiased=False).item()),
        "max_drawdown_mean": float(drawdowns.mean().item()),
        "max_drawdown_worst": float(drawdowns.min().item()),
        "var_es": portfolio_var_es(portfolio_returns),
    }


def _value_at_risk(values: Tensor, *, level: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, level).item())


def _expected_shortfall(values: Tensor, *, var: float) -> float:
    if values.numel() == 0:
        return 0.0
    tail_values = values[values <= var]
    if tail_values.numel() == 0:
        return var
    return float(tail_values.mean().item())


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1).")


def _threshold_key(level: float) -> str:
    return f"q{round(level * 100):02d}"


def _safe_mean(values: Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def _safe_float_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _mean_coordinate_correlation(reference: Tensor, candidate: Tensor) -> float:
    reference_flat = reference.reshape(-1, reference.shape[-1])
    candidate_flat = candidate.reshape(-1, candidate.shape[-1])
    reference_centered = reference_flat - reference_flat.mean(dim=0, keepdim=True)
    candidate_centered = candidate_flat - candidate_flat.mean(dim=0, keepdim=True)
    numerator = (reference_centered * candidate_centered).mean(dim=0)
    denominator = reference_centered.std(dim=0, unbiased=False) * candidate_centered.std(
        dim=0,
        unbiased=False,
    )
    correlations = numerator / denominator.clamp_min(1e-12)
    return float(correlations.clamp(-1.0, 1.0).mean().item())
