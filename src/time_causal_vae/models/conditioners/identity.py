# mypy: ignore-errors
# ruff: noqa
import torch

from time_causal_vae.models.conditioners.base import BaseConditioner


class IdentityConditioner(BaseConditioner):
    """No-op conditioner used by the selected paper configurations."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise an identity condition transform."""
        super().__init__(*args, **kwargs)
        self.net = torch.nn.Identity()

    def forward(self, x):
        """Return condition values unchanged."""
        return self.net(x)
