# mypy: ignore-errors
# ruff: noqa
import torch
import torch.nn as nn

from time_causal_vae.models.continuous.decoders.base import BaseDecoder
from time_causal_vae.models.continuous.initialisation import init_weights
from time_causal_vae.utils.output import ModelOutput


class LSTMDecoder(BaseDecoder):
    """LSTM decoder that maps flattened latent paths to data paths."""

    def __init__(
        self,
        data_dim: int,
        data_length: int,
        latent_dim: int,
        latent_length: int,
        hidden_dim: int,
        n_layers: int,
        activation=nn.Tanh(),
    ):
        """Initialise the legacy-compatible LSTM decoder."""
        super().__init__()
        self.data_dim = data_dim
        self.data_length = data_length
        self.latent_dim = latent_dim
        self.latent_length = latent_length
        self.activation = activation

        # LSTM
        self.linear_pre = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.linear_post = nn.Linear(hidden_dim, data_dim)

        self.lstm.apply(init_weights)
        self.linear_post.apply(init_weights)
        self.linear_pre.apply(init_weights)

    def forward(self, noise) -> torch.Tensor:
        noise = noise.view(-1, self.latent_length, self.latent_dim)
        x = self.activation(self.linear_pre(noise))
        x, _ = self.lstm(x)
        x = self.linear_post(self.activation(x))
        output = ModelOutput(reconstruction=x)
        return output


class LSTMResDecoder(LSTMDecoder):
    """Residual LSTM decoder preserving legacy projection attributes."""

    def __init__(
        self,
        data_dim: int,
        data_length: int,
        latent_dim: int,
        latent_length: int,
        hidden_dim: int,
        n_layers: int,
        activation=nn.Tanh(),
    ):
        super().__init__(
            data_dim,
            data_length,
            latent_dim,
            latent_length,
            hidden_dim,
            n_layers,
            activation,
        )
        self.linear_post0 = nn.Linear(hidden_dim + self.latent_dim, hidden_dim)
        self.linear_post0.apply(init_weights)

    def forward(self, noise):
        noise = noise.view(-1, self.latent_length, self.latent_dim)
        x = self.activation(self.linear_pre(noise))
        x, _ = self.lstm(x)
        y = torch.cat([x, noise], dim=-1)
        y = self.linear_post0(self.activation(y))
        y = self.linear_post(self.activation(y))
        output = ModelOutput(reconstruction=y)
        return output


class ConditionalResidualLSTMDecoder(LSTMResDecoder):
    """Conditional residual LSTM decoder used by selected paper configs."""

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
        activation=nn.ReLU(),
    ):
        super().__init__(
            data_dim,
            data_length,
            latent_dim,
            latent_length,
            hidden_dim,
            n_layers,
            activation,
        )
        self.condition_dim = condition_dim
        self.conditioner = conditioner
        self.latent_dim0 = self.latent_dim - self.condition_dim
        # now the latent_dim = original_latent_dim + condition_dim

    def forward(self, z, c):
        """Concatenate repeated condition values to the latent path."""
        c = self.conditioner(c)
        c = c.view(-1, 1, self.condition_dim).repeat(1, self.latent_length, 1)
        z = z.view(-1, self.latent_length, self.latent_dim0)
        x = torch.cat([z, c], dim=-1)
        output = super().forward(x)
        return output


CLSTMResDecoder = ConditionalResidualLSTMDecoder
