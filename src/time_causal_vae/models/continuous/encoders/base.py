# mypy: ignore-errors
# ruff: noqa
import torch.nn as nn


class BaseEncoder(nn.Module):
    """Base class for encoder neural networks."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the encoder module."""
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """Encode observations into latent distribution parameters."""
        raise NotImplementedError()
