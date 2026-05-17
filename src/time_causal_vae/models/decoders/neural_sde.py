# mypy: ignore-errors
# ruff: noqa

import torch
import torch.nn as nn

from time_causal_vae.models.decoders.base import BaseDecoder
from time_causal_vae.utils.output import ModelOutput


class NeuralSDEDecoder(BaseDecoder):
    """
    Class implementing the NeuralSDE generator model
    Code from
    https://github.com/niklaswalter/Randomised-Signature-TimeSeries-Generation
    """

    def __init__(
        self,
        n_lag: int,
        input_dim: int,
        output_dim: int,
        reservoir_dim: int,
        brownian_dim: int,
        activation,
        hidden_dim: int = 32,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reservoir_dim = reservoir_dim
        self.brownian_dim = brownian_dim
        self.activation = activation
        self.device = device

        """
        Linear layers for initial condition NN
        """
        self.hidden_dim = hidden_dim
        self.init_layer1 = nn.Linear(self.input_dim, self.hidden_dim, device=self.device)
        self.init_layer2 = nn.Linear(self.hidden_dim, self.reservoir_dim, device=self.device)

        """
        Sample random matrices and biases for reservoir
        """
        self.rho1, self.rho2, self.rho3, self.rho4 = (
            nn.Parameter(torch.randn(1, 1).to(self.device)),
            nn.Parameter(torch.randn(1, 1).to(self.device)),
            nn.Parameter(torch.randn(1, 1).to(self.device)),
            nn.Parameter(torch.randn(1, 1).to(self.device)),
        )

        self.B1, self.B2 = (
            torch.randn(self.reservoir_dim, self.reservoir_dim, device=device),
            torch.randn(self.brownian_dim, self.reservoir_dim, self.reservoir_dim, device=device),
        )

        self.lambda1, self.lambda2 = (
            torch.randn(self.reservoir_dim, 1, device=device),
            torch.randn(self.brownian_dim, self.reservoir_dim, 1, device=device),
        )

        self.activation = activation

        """
        Linear readout layers for the reservoir
        """

        self.readouts = nn.ModuleList([
            nn.Linear(self.reservoir_dim, self.output_dim, device=device) for i in range(n_lag)
        ])

    def solve_neural_sde(self, V: torch.tensor, W: torch.tensor) -> torch.tensor:
        """
        Methods solving the NeuralSDE numerically using an EM scheme
        """
        device = W.device
        R = torch.empty(W.shape[0], W.shape[1], self.B1.shape[0], 1, device=device).clone()
        R[:, 0, :] = V.clone()

        for t in range(1, W.shape[1]):
            R[:, t, :] = (
                R[:, t - 1, :].clone()
                + self.activation(
                    self.rho1 * self.B1 @ R[:, t - 1, :].clone() + self.rho2 * self.lambda1
                )
                + torch.sum(
                    self.activation(
                        self.rho3 * self.B2 @ R[:, t - 1, :].unsqueeze(-3).clone()
                        + self.rho4 * self.lambda2
                    )
                    @ (W[:, t, :, None, None] - W[:, t - 1, :, None, None]),
                    axis=1,
                )
            )

        return R

    def forward(self, W) -> torch.Tensor:
        """
        Methods implementing forward call
        """
        device = W.device
        batch_size = W.shape[0]
        n_lags = W.shape[1]

        V = torch.randn(batch_size, self.input_dim, device=device)
        V = self.init_layer1(V)
        V = self.activation(V)
        V = self.init_layer2(V)
        V = torch.reshape(V, (batch_size, self.reservoir_dim, 1))

        print(V.shape)

        R = self.solve_neural_sde(V, W)

        print(R.shape)

        for n in range(n_lags):
            if n == 0:
                x = self.readouts[n](R[:, n].reshape(R[:, n].shape[0], -1))
                x = x.unsqueeze(1)
                print(x.shape)
            else:
                y = self.readouts[n](R[:, n].reshape(R[:, n].shape[0], -1))
                y = y.unsqueeze(1)
                x = torch.cat((x, y), 1)

        output = ModelOutput(reconstruction=x.view(x.shape[0], x.shape[1], -1))
        return output


class CRSigDecoder(NeuralSDEDecoder):
    def __init__(
        self,
        n_lag: int,
        input_dim: int,
        output_dim: int,
        reservoir_dim: int,
        brownian_dim: int,
        activation,
        conditioner,
        condition_dim,
        hidden_dim: int = 32,
        device: str = "cpu",
    ):
        super().__init__(
            n_lag,
            input_dim,
            output_dim,
            reservoir_dim,
            brownian_dim,
            activation,
            hidden_dim,
            device,
        )

        self.condition_dim = condition_dim
        self.conditioner = conditioner
        self.latent_dim = brownian_dim
        self.latent_dim0 = self.latent_dim - self.condition_dim
        self.latent_length = n_lag

    def forward(self, z, c) -> torch.Tensor:
        c = self.conditioner(c)
        c = c.view(-1, 1, self.condition_dim).repeat(1, self.latent_length, 1)
        z = z.view(-1, self.latent_length, self.latent_dim0)
        x = torch.cat([z, c], dim=-1)
        return super().forward(x)
