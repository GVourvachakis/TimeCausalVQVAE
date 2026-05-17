# mypy: ignore-errors
# ruff: noqa
import torch
from torch import Tensor, nn


class RBFKernel(nn.Module):
    r"""
    Gaussian kernel
    """

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)
        return self.bandwidth

    def forward(self, X: Tensor) -> Tensor:
        L2_distances = torch.cdist(X, X) ** 2
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(
            L2_distances.device
        )  # put the bandwidth to the same device as X
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]
        ).sum(dim=0)


class MMD2(nn.Module):
    r"""
    MMD_square!
    """

    def __init__(self, kernel=RBFKernel(), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kernel = kernel

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        r"""
        Inputs:
            X: torch.Tensor
            Y: torch.Tensor
        """
        X = X.flatten(start_dim=1)
        Y = Y.flatten(start_dim=1)
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class GaussianMMD2(MMD2):
    r"""
    Gaussian MMD2
    """

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, *args, **kwargs) -> None:
        kernel = RBFKernel(n_kernels=n_kernels, mul_factor=mul_factor, bandwidth=bandwidth)
        super().__init__(kernel, *args, **kwargs)


class GaussianMMD(GaussianMMD2):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, *args, **kwargs) -> None:
        super().__init__(n_kernels, mul_factor, bandwidth, *args, **kwargs)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return super().forward(X, Y).sqrt()
