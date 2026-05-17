# mypy: ignore-errors
# ruff: noqa
from torch import Tensor, nn


class BasePrior(nn.Module):
    """Base class for tractable latent priors."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the prior module."""
        super().__init__(*args, **kwargs)

    def sample(self, n_sample: int, device) -> Tensor:
        """Sample latent vectors."""
        raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        """Return log probability for latent vectors."""
        raise NotImplementedError
