# mypy: ignore-errors
# ruff: noqa
import torch.nn as nn


class BaseConditioner(nn.Module):
    """Base class for condition preprocessing modules."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the conditioner module."""
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """Transform condition values."""
        raise NotImplementedError()
