# mypy: ignore-errors
# ruff: noqa
from dataclasses import dataclass

from torch import Tensor

from time_causal_vae.models.decoders.base import BaseDecoder
from time_causal_vae.models.encoders.base import BaseEncoder
from time_causal_vae.models.objectives.vae import CVAE, VAE, CVAEConfig, VAEConfig
from time_causal_vae.models.priors.base import BasePrior
from time_causal_vae.models.priors.gaussian import entropy_normal


@dataclass
class BetaVAEConfig(VAEConfig):
    """Configuration for beta-weighted VAE objectives."""

    beta: float = 1.0


class BetaVAE(VAE):
    """Unconditional beta-VAE preserving the legacy KL weighting."""

    model_config: BetaVAEConfig

    def __init__(
        self,
        model_config: BetaVAEConfig,
        encoder: BaseEncoder | None = None,
        decoder: BaseDecoder | None = None,
        prior: BasePrior | None = None,
    ):
        """Initialise the beta-VAE objective."""
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "BetaVAE"
        self.beta = model_config.beta

    def _loss_function(
        self, recon_x0: Tensor, x0: Tensor, mu: Tensor, log_var: Tensor, z: Tensor
    ) -> dict:
        recon_loss = self.recon_loss_func(x0, recon_x0)
        posterior_term = entropy_normal(log_var)
        prior_term = self.prior.log_prob(z.flatten(start_dim=1))
        kld_loss = (posterior_term - prior_term).mean()

        total_loss = recon_loss + self.beta * kld_loss  # The only line different from VAE

        loss_dict = {"recon_loss": recon_loss, "reg_loss": kld_loss, "loss": total_loss}
        return loss_dict


@dataclass
class BetaCVAEConfig(CVAEConfig):
    """Configuration for beta-weighted conditional VAE objectives."""

    beta: float = 1.0


class BetaConditionalVAE(CVAE):
    """Conditional beta-VAE used by selected paper configurations."""

    model_config: BetaCVAEConfig

    def __init__(
        self,
        model_config: BetaCVAEConfig,
        encoder: BaseEncoder | None = None,
        decoder: BaseDecoder | None = None,
        prior: BasePrior | None = None,
    ):
        """Initialise the conditional beta-VAE objective."""
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "BetaCVAE"
        self.beta = model_config.beta

    def _loss_function(
        self, recon_x0: Tensor, x0: Tensor, mu: Tensor, log_var: Tensor, z: Tensor
    ) -> dict:
        recon_loss = self.recon_loss_func(x0, recon_x0)
        posterior_term = entropy_normal(log_var)
        prior_term = self.prior.log_prob(z.flatten(start_dim=1))
        kld_loss = (posterior_term - prior_term).mean()

        total_loss = recon_loss + self.beta * kld_loss  # The only line different from VAE

        loss_dict = {"recon_loss": recon_loss, "reg_loss": kld_loss, "loss": total_loss}
        return loss_dict


BetaCVAE = BetaConditionalVAE
