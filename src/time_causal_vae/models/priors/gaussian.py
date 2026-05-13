# mypy: ignore-errors
# ruff: noqa
import numpy as np
import torch
from torch import Tensor

from time_causal_vae.models.priors.base import BasePrior

PI = torch.tensor(np.pi)


def log_standard_normal(x):
    """Return log density under a standard normal distribution."""
    log_p = -0.5 * torch.log(2.0 * PI) - 0.5 * x**2.0
    log_p = torch.sum(log_p, dim=-1)
    return log_p


def entropy_normal(log_var):
    """Return entropy of a diagonal Gaussian from log-variance."""
    entropy = -0.5 * (1 + torch.log(2.0 * PI) + log_var)
    entropy = torch.sum(entropy, dim=-1)
    return entropy


class GaussianPrior(BasePrior):
    """Standard Gaussian prior over flattened latent paths."""

    def __init__(self, dim=2):
        """Initialise the prior dimension."""
        super().__init__()
        self.dim = dim
        # params weights

    def sample(self, n_sample: int, device) -> Tensor:
        """Sample standard Gaussian latent vectors."""
        return torch.randn(n_sample, self.dim, device=device)

    def log_prob(self, x: Tensor) -> Tensor:
        """Return standard Gaussian log probability."""
        return log_standard_normal(x)
