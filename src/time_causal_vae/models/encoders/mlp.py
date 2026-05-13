# mypy: ignore-errors
# ruff: noqa
import torch
import torch.nn as nn

from time_causal_vae.models.encoders.base import BaseEncoder
from time_causal_vae.utils.output import ModelOutput


class MLPEncoder(BaseEncoder):
    def __init__(self, data_dim, data_length, latent_dim, latent_length, hidden_dim):
        BaseEncoder.__init__(self)
        self.data_dim = data_dim
        self.data_length = data_length
        self.latent_length = latent_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_dim = self.data_dim * self.data_length
        self.hidden_dim = self.hidden_dim
        self.output_dim = self.latent_dim * self.latent_length

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.mean_net = nn.Linear(self.hidden_dim, self.output_dim)
        self.log_var_net = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        h1 = self.net(x)
        mean = self.mean_net(h1)
        log_var = self.log_var_net(h1)
        # mean = mean.view(-1, self.latent_length, self.latent_dim)
        # log_var = log_var.view(-1, self.latent_length, self.latent_dim)
        output = ModelOutput(embedding=mean, log_covariance=log_var)
        return output


class CMLPEncoder(MLPEncoder):
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
        super().__init__(data_dim, data_length, latent_dim, latent_length, hidden_dim)
        self.condition_dim = condition_dim
        self.conditioner = conditioner

    def forward(self, x, c):
        c = self.conditioner(c)
        c = c.view(-1, 1, self.condition_dim).repeat(1, self.data_length, 1)
        y = torch.cat([x, c], axis=-1)
        output = super().forward(y)
        return output


class IdEncoder(BaseEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x, c):
        output = ModelOutput(embedding=x, log_covariance=torch.zeros_like(x))
        return output
