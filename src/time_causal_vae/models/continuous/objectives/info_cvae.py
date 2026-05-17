# mypy: ignore-errors
# ruff: noqa
from dataclasses import dataclass

from torch import Tensor

from time_causal_vae.models.continuous.decoders.base import BaseDecoder
from time_causal_vae.models.continuous.distances import GaussianMMD2
from time_causal_vae.models.continuous.encoders.base import BaseEncoder
from time_causal_vae.models.continuous.objectives.vae import CVAE, VAE, CVAEConfig, VAEConfig
from time_causal_vae.models.continuous.priors.base import BasePrior
from time_causal_vae.models.continuous.priors.gaussian import entropy_normal


@dataclass
class InfoVAEConfig(VAEConfig):
    """Configuration for InfoVAE objectives."""

    beta: float = 1.0
    alpha: float = 1.5


class InfoVAE(VAE):
    """InfoVAE objective with legacy beta and MMD weighting."""

    model_config: InfoVAEConfig

    def __init__(
        self,
        model_config: InfoVAEConfig,
        encoder: BaseEncoder | None = None,
        decoder: BaseDecoder | None = None,
        prior: BasePrior | None = None,
    ):
        """Initialise the InfoVAE objective."""
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "InfoVAE"
        self.beta = model_config.beta
        self.alpha = model_config.alpha
        self.mmd2 = GaussianMMD2()

    def _loss_function(self, recon_x: Tensor, x: Tensor, mu: Tensor, log_var: Tensor, z: Tensor):
        recon_loss = self.recon_loss_func(x, recon_x)
        posterior_term = entropy_normal(log_var)
        prior_term = self.prior.log_prob(z.flatten(start_dim=1))
        kld_loss = (posterior_term - prior_term).mean()

        z_prior = self.prior.sample(len(z), device=z.device)
        mmd_loss = self.mmd2(z.flatten(start_dim=1), z_prior.flatten(start_dim=1))
        # alpha >= beta
        total_loss = recon_loss + self.beta * kld_loss + (self.alpha - self.beta) * mmd_loss

        loss_dict = {
            "recon_loss": recon_loss,
            "reg_loss": kld_loss,
            "loss": total_loss,
            "mmd_loss": mmd_loss,
        }
        return loss_dict


@dataclass
class InfoCVAEConfig(CVAEConfig):
    """Configuration for conditional InfoVAE objectives."""

    beta: float = 1.0
    alpha: float = 1.5


class InfoConditionalVAE(CVAE):
    """Conditional InfoVAE used by selected paper configurations."""

    model_config: InfoCVAEConfig

    def __init__(
        self,
        model_config: InfoCVAEConfig,
        encoder: BaseEncoder | None = None,
        decoder: BaseDecoder | None = None,
        prior: BasePrior | None = None,
    ):
        """Initialise the conditional InfoVAE objective."""
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "InfoCVAE"
        self.beta = model_config.beta
        self.alpha = model_config.alpha
        self.mmd2 = GaussianMMD2()

    def _loss_function(self, recon_x: Tensor, x: Tensor, mu: Tensor, log_var: Tensor, z: Tensor):
        recon_loss = self.recon_loss_func(x, recon_x)
        posterior_term = entropy_normal(log_var)
        prior_term = self.prior.log_prob(z.flatten(start_dim=1))
        kld_loss = (posterior_term - prior_term).mean()

        z_prior = self.prior.sample(len(z), device=z.device)
        mmd_loss = self.mmd2(z.flatten(start_dim=1), z_prior.flatten(start_dim=1))
        # alpha >= beta
        total_loss = recon_loss + self.beta * kld_loss + (self.alpha - self.beta) * mmd_loss

        loss_dict = {
            "recon_loss": recon_loss,
            "reg_loss": kld_loss,
            "loss": total_loss,
            "mmd_loss": mmd_loss,
        }
        return loss_dict


InfoCVAE = InfoConditionalVAE
