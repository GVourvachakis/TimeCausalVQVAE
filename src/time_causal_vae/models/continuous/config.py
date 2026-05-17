# mypy: ignore-errors
# ruff: noqa
"""Model configuration helpers."""

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class BaseConfig:
    """Serializable dataclass base used by model config objects.

    Notes
    -----
    This mirrors the legacy ``tsvae.base.BaseConfig`` save format so target
    checkpoints can keep the same JSON layout.
    """

    name: str = field(init=False)

    def __post_init__(self) -> None:
        """Store the concrete dataclass name for checkpoint metadata."""
        self.name = self.__class__.__name__

    def to_dict(self) -> dict:
        """Return the config as a plain Python dictionary."""
        return asdict(self)

    def to_json_string(self) -> str:
        """Return the config as a JSON string."""
        return json.dumps(self.to_dict())

    def save_json(self, dir_path: str, filename: str) -> None:
        """Save the config to ``<dir_path>/<filename>.json``."""
        with open(os.path.join(dir_path, f"{filename}.json"), "w", encoding="utf-8") as fp:
            fp.write(self.to_json_string())


class BasePipeline:
    """Minimal pipeline protocol retained for factory compatibility."""

    def __call__(self, *args, **kwargs):
        """Run the pipeline.

        Subclasses implement the concrete construction behaviour.
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a Time-Causal VAE model."""

    objective: str
    encoder: str
    decoder: str
    conditioner: str
    prior: str
    data_dim: int
    data_length: int
    latent_dim: int
    latent_length: int
    condition_dim: int = 0
    beta: float = 1.0
    alpha: float | None = None
    encoder_hidden_dim: int = 16
    encoder_num_layers: int = 2
    decoder_hidden_dim: int = 16
    decoder_num_layers: int = 2
    prior_num_flows: int = 3
    prior_hidden_dim: int = 250
