# mypy: ignore-errors
# ruff: noqa
import torch
from torch import Tensor, nn

from time_causal_vae.models.continuous.priors.base import BasePrior
from time_causal_vae.models.continuous.priors.gaussian import log_standard_normal


class RealNVPPrior(BasePrior):
    """Unconditional RealNVP prior over flattened latent paths."""

    def __init__(self, num_flows, latent_dim, hidden_dim):
        """Initialise the flow prior with legacy coupling networks."""
        super().__init__()

        L = latent_dim  # number of latents
        M = hidden_dim  # the number of neurons in scale (s) and translation (t) nets

        # scale (s) network
        nets = lambda: nn.Sequential(
            nn.Linear(L // 2, M),
            nn.LeakyReLU(),
            nn.Linear(M, M),
            nn.LeakyReLU(),
            nn.Linear(M, L // 2),
            nn.Tanh(),
        )

        # translation (t) network
        nett = lambda: nn.Sequential(
            nn.Linear(L // 2, M),
            nn.LeakyReLU(),
            nn.Linear(M, M),
            nn.LeakyReLU(),
            nn.Linear(M, L // 2),
        )

        self.D = latent_dim
        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
        self.num_flows = num_flows

    def coupling(self, x, index, forward=True):
        """Apply one affine coupling layer."""
        (xa, xb) = torch.chunk(x, 2, 1)

        s = self.s[index](xa)
        t = self.t[index](xa)

        if forward:
            # yb = f^{-1}(x)
            yb = (xb - t) * torch.exp(-s)
        else:
            # xb = f(y)
            yb = torch.exp(s) * xb + t

        return torch.cat((xa, yb), 1), s

    def permute(self, x):
        """Reverse latent coordinates between coupling layers."""
        return x.flip(1)

    def f(self, x):
        """Map data-space latent vectors to base Gaussian space."""
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in range(self.num_flows):
            z, s = self.coupling(z, i, forward=True)
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=1)

        return z, log_det_J

    def f_inv(self, z):
        """Map base Gaussian samples through the inverse flow."""
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)

        return x

    def sample(self, n_sample: int, device) -> Tensor:
        """Sample latent vectors from the flow prior."""
        z = torch.randn(n_sample, self.D, device=device)
        x = self.f_inv(z)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        """Return log probability under the flow prior."""
        z, log_det_J = self.f(x)
        log_p = log_standard_normal(z) + log_det_J
        return log_p


RealNVP = RealNVPPrior
FlowPrior = RealNVPPrior
