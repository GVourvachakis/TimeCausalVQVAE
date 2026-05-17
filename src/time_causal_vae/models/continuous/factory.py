# mypy: ignore-errors
# ruff: noqa
from torch import relu

from time_causal_vae.models.continuous.conditioners.identity import IdentityConditioner
from time_causal_vae.models.continuous.config import BasePipeline
from time_causal_vae.models.continuous.decoders.mlp import IdDecoder
from time_causal_vae.models.continuous.decoders.neural_sde import CRSigDecoder
from time_causal_vae.models.continuous.encoders.mlp import IdEncoder
from time_causal_vae.models.continuous.priors.gaussian import GaussianPrior
from time_causal_vae.models.continuous.priors.realnvp import RealNVPPrior


class ModelFactory(BasePipeline):
    """Build target models from the legacy flat experiment config.

    Notes
    -----
    The factory accepts the same legacy names as ``NetworkPipeline`` so parity
    checks can compare target models against the oracle implementation without
    changing selected YAML values.
    """

    def __init__(
        self,
    ):
        """Initialise the stateless model factory."""
        pass

    def __call__(self, exp_config, **kwargs):
        """Build conditioner, encoder, decoder, prior, and objective."""
        conditioner = self._get_conditioner(exp_config, **kwargs)
        encoder = self._get_encoder(exp_config, conditioner, **kwargs)
        decoder = self._get_decoder(exp_config, conditioner, **kwargs)
        prior = self._get_prior(exp_config, **kwargs)
        model = self._get_model(encoder, decoder, prior, exp_config, **kwargs)
        return model

    def _get_conditioner(self, exp_config, **kwargs):
        """Build a conditioner from the legacy config name."""
        if exp_config.conditioner == "Id":
            conditioner = IdentityConditioner()
        else:
            raise Exception("No such conditioner")
        return conditioner

    def _get_encoder(self, exp_config, conditioner=None, **kwargs):
        """Build an encoder from the legacy config name."""
        if exp_config.encoder == "MLP":
            from time_causal_vae.models.continuous.encoders.mlp import MLPEncoder

            encoder = MLPEncoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.E_hidden_dim,
            )

        elif exp_config.encoder == "CMLP":
            from time_causal_vae.models.continuous.encoders.mlp import CMLPEncoder

            encoder = CMLPEncoder(
                exp_config.data_dim + exp_config.condition_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.E_hidden_dim,
                exp_config.condition_dim,
                conditioner,
            )

        elif exp_config.encoder == "LSTM":
            from time_causal_vae.models.continuous.encoders.lstm import LSTMEncoder

            encoder = LSTMEncoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.E_hidden_dim,
                exp_config.E_num_layers,
            )
        elif exp_config.encoder == "CLSTM":
            from time_causal_vae.models.continuous.encoders.lstm import CLSTMEncoder

            encoder = CLSTMEncoder(
                exp_config.data_dim + exp_config.condition_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.E_hidden_dim,
                exp_config.E_num_layers,
                exp_config.condition_dim,
                conditioner,
            )
        elif exp_config.encoder == "CLSTMRes":
            from time_causal_vae.models.continuous.encoders.lstm import (
                ConditionalResidualLSTMEncoder,
            )

            encoder = ConditionalResidualLSTMEncoder(
                exp_config.data_dim + exp_config.condition_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.E_hidden_dim,
                exp_config.E_num_layers,
                exp_config.condition_dim,
                conditioner,
            )

        elif exp_config.encoder == "IdEncoder":
            encoder = IdEncoder()
        else:
            raise Exception("No such encoder")
        return encoder

    def _get_decoder(self, exp_config, conditioner=None, **kwargs):
        """Build a decoder from the legacy config name."""
        # Decoder
        if exp_config.decoder == "MLP":
            from time_causal_vae.models.continuous.decoders.mlp import MLPDecoder

            decoder = MLPDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
            )
        elif exp_config.decoder == "CMLP":
            from time_causal_vae.models.continuous.decoders.mlp import CMLPDecoder

            decoder = CMLPDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim + exp_config.condition_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
                exp_config.condition_dim,
                conditioner,
            )
        elif exp_config.decoder == "CAddMLP":
            from time_causal_vae.models.continuous.decoders.mlp import CAddMLPDecoder

            decoder = CAddMLPDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
                exp_config.condition_dim,
                conditioner,
            )

        elif exp_config.decoder == "LSTM":
            from time_causal_vae.models.continuous.decoders.lstm import LSTMDecoder

            decoder = LSTMDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
                exp_config.D_num_layers,
            )
        elif exp_config.decoder == "LSTMRes":
            from time_causal_vae.models.continuous.decoders.lstm import LSTMResDecoder

            decoder = LSTMResDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
                exp_config.D_num_layers,
            )
        elif exp_config.decoder == "CLSTMRes":
            from time_causal_vae.models.continuous.decoders.lstm import (
                ConditionalResidualLSTMDecoder,
            )

            decoder = ConditionalResidualLSTMDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim + exp_config.condition_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
                exp_config.D_num_layers,
                exp_config.condition_dim,
                conditioner,
            )
        elif exp_config.decoder == "CRSigDecoder":
            decoder = CRSigDecoder(
                n_lag=exp_config.latent_length,
                input_dim=5,
                output_dim=exp_config.data_dim,
                reservoir_dim=50,
                brownian_dim=exp_config.latent_dim + exp_config.condition_dim,
                activation=relu,
                conditioner=conditioner,
                condition_dim=exp_config.condition_dim,
            )
        elif exp_config.decoder == "IdDecoder":
            decoder = IdDecoder()
        else:
            raise Exception("No such decoder")
        return decoder

    def _get_prior(self, exp_config, **kwargs):
        """Build a prior from the legacy config name."""
        # Prior
        if exp_config.prior == "RealNVP":
            prior = RealNVPPrior(
                num_flows=exp_config.P_num_flows,
                latent_dim=exp_config.latent_dim * exp_config.latent_length,
                hidden_dim=exp_config.P_hidden_dim,
            )
        elif exp_config.prior == "Gaussian":
            prior = GaussianPrior(dim=exp_config.latent_dim * exp_config.latent_length)
        else:
            raise Exception("No such prior")

        return prior

    def _get_model(self, encoder, decoder, prior, exp_config, **kwargs):
        """Build the objective model from the legacy config name."""
        if exp_config.model == "VAE":
            from time_causal_vae.models.continuous.objectives.vae import VAEConfig

            model_config = VAEConfig(
                data_dim=exp_config.data_dim,
                data_length=exp_config.data_length,
                latent_length=exp_config.latent_length,
                latent_dim=exp_config.latent_dim,
                reconstruction_loss="l1",
                transform=exp_config.transform,
                inv_transform=exp_config.inv_transform,
            )
            from time_causal_vae.models.continuous.objectives.vae import VAE

            model = VAE(model_config=model_config, encoder=encoder, decoder=decoder, prior=prior)
        elif exp_config.model == "BetaCVAE":
            from time_causal_vae.models.continuous.objectives.beta_cvae import BetaVAEConfig

            model_config = BetaVAEConfig(
                data_dim=exp_config.data_dim,
                data_length=exp_config.data_length,
                latent_length=exp_config.latent_length,
                latent_dim=exp_config.latent_dim,
                reconstruction_loss="l1",
                beta=exp_config.beta,
                transform=exp_config.transform,
                inv_transform=exp_config.inv_transform,
            )
            from time_causal_vae.models.continuous.objectives.beta_cvae import BetaConditionalVAE

            model = BetaConditionalVAE(
                model_config=model_config, encoder=encoder, decoder=decoder, prior=prior
            )
        elif exp_config.model == "InfoCVAE":
            from time_causal_vae.models.continuous.objectives.info_cvae import (
                InfoConditionalVAE,
                InfoCVAEConfig,
            )

            model_config = InfoCVAEConfig(
                data_dim=exp_config.data_dim,
                data_length=exp_config.data_length,
                latent_length=exp_config.latent_length,
                latent_dim=exp_config.latent_dim,
                reconstruction_loss="l1",
                beta=exp_config.beta,
                alpha=exp_config.alpha,
                transform=exp_config.transform,
                inv_transform=exp_config.inv_transform,
            )
            model = InfoConditionalVAE(
                model_config=model_config, encoder=encoder, decoder=decoder, prior=prior
            )
        else:
            raise Exception("No such model")

        return model


NetworkPipeline = ModelFactory
