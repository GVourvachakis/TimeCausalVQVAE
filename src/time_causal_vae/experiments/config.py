"""Experiment configuration loading utilities."""

from pathlib import Path
from typing import Any, cast

import yaml

from time_causal_vae.typing import PathLike


def load_experiment_config(path: PathLike) -> dict[str, Any]:
    """Load an experiment configuration from YAML.

    Parameters
    ----------
    path
        Path to the experiment YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return cast(dict[str, Any], yaml.safe_load(handle))
