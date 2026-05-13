# mypy: ignore-errors
# ruff: noqa
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from time_causal_vae.models.base import BaseConfig, BaseModel
from time_causal_vae.models.decoders.base import BaseDecoder
from time_causal_vae.models.encoders.base import BaseEncoder
from time_causal_vae.models.losses import get_loss
from time_causal_vae.models.priors.base import BasePrior
from time_causal_vae.models.priors.gaussian import entropy_normal
from time_causal_vae.models.transforms import get_transform
from time_causal_vae.utils.output import ModelOutput


@dataclass
class VAEConfig(BaseConfig):
    """Configuration for unconditional VAE objectives."""

    data_dim: int = 1
    data_length: int = 1
    latent_length: int = 1
    latent_dim: int = 1
    reconstruction_loss: str = "l1"

    transform: str = ""
    inv_transform: str = ""

    uses_default_encoder: bool = False
    uses_default_decoder: bool = False


class VAE(BaseModel):
    """Variational autoencoder with legacy-compatible tensor semantics."""

    model_config: VAEConfig

    def __init__(
        self,
        model_config: VAEConfig,
        encoder: BaseEncoder | None = None,
        decoder: BaseDecoder | None = None,
        prior: BasePrior | None = None,
    ):
        """Initialise the VAE objective."""
        super().__init__(model_config)
        self.model_config
        self.model_name = "VAE"
        self.recon_loss_func = get_loss(model_config.reconstruction_loss)

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        self.transform = get_transform(self.model_config.transform)
        self.inv_transform = get_transform(self.model_config.inv_transform)

    def forward(self, inputs: Any, **kwargs):
        """Return losses, reconstruction, and latent sample for a batch."""
        x0 = inputs["data"]
        x = self.transform(x0)
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]
        recon_x0 = self.inv_transform(recon_x)

        loss_dict = self._loss_function(recon_x0, x0, mu, log_var, z)
        data_dict = {"recon_x": recon_x0, "z": z}
        output = ModelOutput({**loss_dict, **data_dict})
        return output

    def _loss_function(
        self, recon_x0: Tensor, x0: Tensor, mu: Tensor, log_var: Tensor, z: Tensor
    ) -> dict:
        r"""
        Recon + DKL
        DKL = E[log(posterior)] - E[log(prior] (expectation under posterior)
        """
        recon_loss = self.recon_loss_func(x0, recon_x0)
        posterior_term = entropy_normal(log_var)
        prior_term = self.prior.log_prob(z)
        kld_loss = posterior_term.mean() - prior_term.mean()
        total_loss = recon_loss + kld_loss

        loss_dict = {"recon_loss": recon_loss, "reg_loss": kld_loss, "loss": total_loss}
        return loss_dict

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def generation(self, n_sample: int, **kwargs):
        z = self.prior.sample(n_sample, device=self.device)
        recon_x = self.decoder(z)["reconstruction"]
        recon_x0 = self.inv_transform(recon_x)
        return recon_x0


@dataclass
class CVAEConfig(VAEConfig):
    """Configuration for conditional VAE objectives."""

    pass


class CVAE(VAE):
    """Conditional VAE preserving legacy condition handling."""

    model_config: CVAEConfig

    def __init__(
        self,
        model_config: CVAEConfig,
        encoder: BaseEncoder | None = None,
        decoder: BaseDecoder | None = None,
        prior: BasePrior | None = None,
    ):
        """Initialise the conditional VAE objective."""
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "CVAE"

    def forward(self, inputs: Any, **kwargs):
        """Pass labels into encoder and decoder exactly as in the legacy CVAE."""
        x0 = inputs["data"]
        x = self.transform(x0)
        c = inputs["labels"]
        encoder_output = self.encoder(x, c)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z, c)["reconstruction"]
        recon_x0 = self.inv_transform(recon_x)
        loss_dict = self._loss_function(recon_x0, x0, mu, log_var, z)
        data_dict = {"recon_x": recon_x0, "z": z}
        output = ModelOutput({**loss_dict, **data_dict})
        return output

    def generation(self, n_sample: int, **kwargs):
        c = kwargs.pop("c")
        z = self.prior.sample(n_sample, device=self.device)
        recon_x = self.decoder(z, c)["reconstruction"]
        recon_x0 = self.inv_transform(recon_x)
        return recon_x0

    def _loss_function(
        self, recon_x0: Tensor, x0: Tensor, mu: Tensor, log_var: Tensor, z: Tensor
    ) -> dict:
        return super()._loss_function(recon_x0, x0, mu, log_var, z)
