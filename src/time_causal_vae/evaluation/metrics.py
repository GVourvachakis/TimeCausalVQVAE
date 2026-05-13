# mypy: ignore-errors
# ruff: noqa
"""Target evaluation metrics with legacy formulas."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from time_causal_vae.models.distances import GaussianMMD, GaussianMMD2

try:
    from ot import sliced_wasserstein_distance
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    sliced_wasserstein_distance = None


def _raise_missing_optional_dependency(package_name: str) -> None:
    msg = (
        f"Optional dependency '{package_name}' is required for this evaluation metric. "
        "Install the corresponding optional evaluation dependency before calling it."
    )
    raise ModuleNotFoundError(msg)


class SWD(nn.Module):
    """Sliced Wasserstein distance with the legacy projection count and seed."""

    def __init__(self, n_projections: int = 100, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_projections = n_projections

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """Compute sliced Wasserstein distance after flattening paths."""
        if sliced_wasserstein_distance is None:
            _raise_missing_optional_dependency("pot")
        X = X.flatten(start_dim=1)
        Y = Y.flatten(start_dim=1)
        return sliced_wasserstein_distance(X, Y, n_projections=self.n_projections, seed=0)


def l2_dist(x: Tensor, y: Tensor) -> Tensor:
    """Return the Euclidean distance used by legacy signature metrics."""
    return (x - y).pow(2).sum().sqrt()


class SignatureMMD(nn.Module):
    """Expected-signature MMD wrapper retained as optional evaluation behaviour."""

    def __init__(
        self, trunc: int = 3, augmented: bool = True, normalise: bool = True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trunc = trunc
        self.augmented = augmented
        self.normalise = normalise

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """Compute signature MMD through the optional external implementation."""
        from time_causal_vae.evaluation.external.signatures import compute_exp_sig

        esig_X = compute_exp_sig(
            x=X,
            trunc=self.trunc,
            augmented=self.augmented,
            normalise=self.normalise,
        )
        esig_Y = compute_exp_sig(
            x=Y,
            trunc=self.trunc,
            augmented=self.augmented,
            normalise=self.normalise,
        )
        return l2_dist(esig_X, esig_Y)


class MomentMMD(nn.Module):
    """Moment MMD copied from the legacy evaluator."""

    def __init__(self, p: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """Compute moment distance for path tensors."""
        moment_X = X.abs().pow(self.p).mean(axis=0)
        moment_Y = Y.abs().pow(self.p).mean(axis=0)
        return l2_dist(moment_X, moment_Y)


class SAWD(nn.Module):
    """Sliced adapted Wasserstein distance retained as optional borrowed behaviour."""

    def __init__(self, n_compute_awd: int, n_slices: int, len_slices: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_compute_awd = n_compute_awd
        self.n_slices = n_slices
        self.len_slices = len_slices

    def forward(self, X, Y):
        """Delegate adapted Wasserstein computation to optional borrowed code."""
        try:
            from time_causal_vae.evaluation.external.awd.pathstodist import paths_to_dist_parallel
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
            if exc.name == "ot":
                _raise_missing_optional_dependency("pot")
            raise

        data1 = np.array(X[..., 0])
        data2 = np.array(Y[..., 0])

        n_data = self.n_compute_awd
        t_max = data1.shape[1]

        path1 = data1[:n_data, :t_max]
        path2 = data2[:n_data, :t_max]
        k_list = [1] + [int(np.round(n_data ** (1 / 2))) for _ in range(1, t_max)]

        return paths_to_dist_parallel(
            path1,
            path2,
            n_slices=self.n_slices,
            len_slices=self.len_slices,
            use_klist=1,
            k_list=k_list,
            markov=0,
            verbose=0,
            max_workers=4,
        )
