# mypy: ignore-errors
# ruff: noqa
import torch
import torch.nn as nn

from time_causal_vae.models.continuous.encoders.base import BaseEncoder
from time_causal_vae.models.continuous.initialisation import init_weights
from time_causal_vae.utils.output import ModelOutput


class LSTMEncoder(BaseEncoder):
    """LSTM encoder that outputs flattened latent mean and log-variance."""

    def __init__(
        self,
        data_dim: int,
        data_length: int,
        latent_dim: int,
        latent_length: int,
        hidden_dim: int,
        n_layers: int,
    ):
        """Initialise the legacy-compatible LSTM encoder."""
        super().__init__()
        self.data_dim = data_dim
        self.data_length = data_length
        self.latent_dim = latent_dim
        self.latent_length = latent_length
        self.activation = nn.Tanh()
        self.hidden_dim = hidden_dim

        self.linear_pre = nn.Linear(data_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.linear_post = nn.Linear(hidden_dim, hidden_dim)

        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, latent_dim),
        )

        self.log_var_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, latent_dim),
        )

        self.lstm.apply(init_weights)
        self.linear_pre.apply(init_weights)
        self.linear_post.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x = nn.ReLU()(self.linear_pre(x))
        x, _ = self.lstm(x)
        x = self.linear_post(self.activation(x))

        mean = self.mean_net(x)
        log_var = self.log_var_net(x)

        mean = mean.flatten(start_dim=1)
        log_var = log_var.flatten(start_dim=1)
        output = ModelOutput(embedding=mean, log_covariance=log_var)

        return output


class CLSTMEncoder(LSTMEncoder):
    """Conditional LSTM encoder that appends a repeated condition path."""

    def __init__(
        self,
        data_dim: int,
        data_length: int,
        latent_dim: int,
        latent_length: int,
        hidden_dim: int,
        n_layers: int,
        condition_dim: int,
        conditioner,
    ):
        super().__init__(data_dim, data_length, latent_dim, latent_length, hidden_dim, n_layers)
        self.condition_dim = condition_dim
        self.conditioner = conditioner

    def forward(self, x, c):
        c = self.conditioner(c)
        c = c.view(-1, 1, self.condition_dim).repeat(1, self.data_length, 1)
        y = torch.cat([x, c], axis=-1)
        output = super().forward(y)
        return output


class LSTMResEncoder(LSTMEncoder):
    """Residual LSTM encoder preserving the legacy residual projections."""

    def __init__(
        self,
        data_dim: int,
        data_length: int,
        latent_dim: int,
        latent_length: int,
        hidden_dim: int,
        n_layers: int,
    ):
        super().__init__(data_dim, data_length, latent_dim, latent_length, hidden_dim, n_layers)

        self.linear_post0 = nn.Linear(hidden_dim + self.data_dim, hidden_dim)
        self.linear_post0.apply(init_weights)

        self.linear_post = nn.Linear(hidden_dim + hidden_dim + self.data_dim, hidden_dim)
        self.linear_post.apply(init_weights)

    def forward(self, x0: torch.Tensor):
        x = x0
        x = nn.ReLU()(self.linear_pre(x))
        x, _ = self.lstm(x)
        x = self.activation(x)

        y0 = torch.cat([x, x0], dim=-1)
        y = self.activation(self.linear_post0(y0))

        z0 = torch.cat([y, y0], dim=-1)
        z = self.activation(self.linear_post(z0))

        mean = self.mean_net(z)
        log_var = self.log_var_net(z)

        mean = mean.flatten(start_dim=1)
        log_var = log_var.flatten(start_dim=1)
        output = ModelOutput(embedding=mean, log_covariance=log_var)

        return output


class ConditionalResidualLSTMEncoder(LSTMResEncoder):
    """Conditional residual LSTM encoder used by selected paper configs."""

    def __init__(
        self,
        data_dim: int,
        data_length: int,
        latent_dim: int,
        latent_length: int,
        hidden_dim: int,
        n_layers: int,
        condition_dim: int,
        conditioner,
    ):
        super().__init__(data_dim, data_length, latent_dim, latent_length, hidden_dim, n_layers)
        self.condition_dim = condition_dim
        self.conditioner = conditioner

    def forward(self, x, c):
        """Concatenate repeated condition values to the observed path."""
        c = self.conditioner(c)
        c = c.view(-1, 1, self.condition_dim).repeat(1, self.data_length, 1)
        y = torch.cat([x, c], axis=-1)
        output = super().forward(y)
        return output


CLSTMResEncoder = ConditionalResidualLSTMEncoder
