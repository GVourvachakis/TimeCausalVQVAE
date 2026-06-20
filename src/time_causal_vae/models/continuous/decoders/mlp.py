# mypy: ignore-errors
# ruff: noqa
import torch
import torch.nn as nn

from time_causal_vae.models.continuous.decoders.base import BaseDecoder
from time_causal_vae.utils.output import ModelOutput


class MLPDecoder(BaseDecoder):
    r"""
    MLP(Z)
    """

    def __init__(self, data_dim, data_length, latent_dim, latent_length, hidden_dim):
        BaseDecoder.__init__(self)
        self.data_dim = data_dim
        self.data_length = data_length
        self.latent_length = latent_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_dim = self.latent_dim * self.latent_length
        self.hidden_dim = self.hidden_dim
        self.output_dim = self.data_dim * self.latent_length

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, z: torch.Tensor):
        z = z.flatten(start_dim=1)
        h1 = self.net(z)
        x = h1.view(-1, self.data_length, self.data_dim)
        output = ModelOutput(reconstruction=x)
        return output


class CMLPDecoder(MLPDecoder):
    r"""
    MLP(X,Z)
    """

    def __init__(
        self,
        data_dim: int,
        data_length: int,
        latent_dim: int,
        latent_length: int,
        hidden_dim: int,
        condition_dim: int,
        conditioner,
    ):
        super().__init__(
            data_dim,
            data_length,
            latent_dim,
            latent_length,
            hidden_dim,
        )
        self.condition_dim = condition_dim
        self.conditioner = conditioner
        self.latent_dim0 = self.latent_dim - self.condition_dim
        # now the latent_dim = original_latent_dim + condition_dim

    def forward(self, z, c):
        c = self.conditioner(c)
        c = c.view(-1, 1, self.condition_dim).repeat(1, self.latent_length, 1)
        z = z.view(-1, self.latent_length, self.latent_dim0)
        x = torch.cat([z, c], dim=-1)
        output = super().forward(x)
        return output


class CAddMLPDecoder(MLPDecoder):
    r"""
    MLP(X+Z)
    """

    def __init__(
        self,
        data_dim: int,
        data_length: int,
        latent_dim: int,
        latent_length: int,
        hidden_dim: int,
        condition_dim: int,
        conditioner,
    ):
        super().__init__(
            data_dim,
            data_length,
            latent_dim,
            latent_length,
            hidden_dim,
        )
        self.condition_dim = condition_dim
        self.conditioner = conditioner
        # now the latent_dim = original_latent_dim + condition_dim

    def forward(self, z, c):
        c = self.conditioner(c)
        c = c.view(-1, 1, self.condition_dim).repeat(1, self.latent_length, 1)
        z = z.view(-1, self.latent_length, self.latent_dim)
        x = z + c
        output = super().forward(x)
        return output


class IdDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, z, c):
        output = ModelOutput(reconstruction=z)
        return output
