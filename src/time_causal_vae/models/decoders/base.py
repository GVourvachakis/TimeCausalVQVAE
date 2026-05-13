# mypy: ignore-errors
# ruff: noqa
import torch.nn as nn


class BaseDecoder(nn.Module):
    """Base class for decoder neural networks."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the decoder module."""
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """Decode latent values into reconstructed observations."""
        raise NotImplementedError()
