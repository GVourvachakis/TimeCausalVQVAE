"""Experimental multidimensional profile metadata utilities.

This module intentionally reads ``trained_models/multidim_profiles.yaml``
instead of ``trained_models/model_registry.yaml``. The multidimensional entries
are profile labels for research notebooks and are not public registry defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from time_causal_vae.typing import PathLike

FamilyName = Literal["continuous", "discrete"]
MultidimProfiles = dict[str, Any]


def _default_multidim_profiles_path() -> Path:
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[3] / "trained_models" / "multidim_profiles.yaml",
        module_path.parents[2] / "trained_models" / "multidim_profiles.yaml",
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


DEFAULT_MULTIDIM_PROFILES_PATH = _default_multidim_profiles_path()


@dataclass(frozen=True)
class MultidimProfileSelection:
    """Selected experimental multidimensional profile metadata."""

    experiment: str
    profile: str
    family: str
    status: str
    public_default: bool
    metadata: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Return a selected metadata value by key."""
        if key in {"experiment", "profile", "family", "status", "public_default", "metadata"}:
            return getattr(self, key)
        return self.metadata[key]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serialisable representation."""
        return asdict(self)


def load_multidim_profiles(path: PathLike | None = None) -> MultidimProfiles:
    """Load experimental multidimensional profile metadata."""
    profiles_path = Path(path) if path is not None else _default_multidim_profiles_path()
    with profiles_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Multidim profile file {profiles_path} must contain a YAML mapping.")
    return cast(MultidimProfiles, loaded)


def select_multidim_profile(
    experiment: str,
    profile: str,
    family: FamilyName | None = None,
    *,
    path: PathLike | None = None,
) -> MultidimProfileSelection:
    """Select one experimental multidimensional profile.

    Parameters
    ----------
    experiment:
        Experiment key, for example ``"multifactor_market"`` or
        ``"sp500_50_panel"``.
    profile:
        Profile key inside the experiment, for example ``"portfolio_tail"``.
    family:
        Optional guard that the selected profile belongs to the requested model
        family.
    path:
        Optional profile YAML path. Defaults to
        ``trained_models/multidim_profiles.yaml``.
    """
    profiles = load_multidim_profiles(path)
    experiment_section = _experiment_section(profiles, experiment)
    profile_mapping = _profile_mapping(experiment_section, experiment)
    if profile not in profile_mapping:
        known = ", ".join(sorted(profile_mapping)) or "<none>"
        raise KeyError(
            f"Unknown multidim profile {profile!r} for experiment {experiment!r}. "
            f"Known profiles: {known}."
        )

    profile_section = dict(profile_mapping[profile])
    selected_family = _required_string(profile_section, "family")
    if family is not None and selected_family != family:
        raise ValueError(
            f"Profile {experiment}.{profile} has family {selected_family!r}; expected {family!r}."
        )

    return MultidimProfileSelection(
        experiment=experiment,
        profile=profile,
        family=selected_family,
        status=_required_string(experiment_section, "status"),
        public_default=_required_bool(experiment_section, "public_default"),
        metadata=profile_section,
    )


def list_multidim_profiles(path: PathLike | None = None) -> dict[str, list[str]]:
    """Return available experimental multidimensional profiles by experiment."""
    profiles = load_multidim_profiles(path)
    return {
        experiment: sorted(_profile_mapping(section, experiment))
        for experiment, section in profiles.items()
        if isinstance(section, Mapping)
    }


def _experiment_section(
    profiles: Mapping[str, Any],
    experiment: str,
) -> Mapping[str, Any]:
    if experiment not in profiles:
        known = ", ".join(sorted(str(name) for name in profiles)) or "<none>"
        raise KeyError(f"Unknown multidim experiment {experiment!r}. Known experiments: {known}.")
    section = profiles[experiment]
    if not isinstance(section, Mapping):
        raise ValueError(f"Multidim experiment {experiment!r} must be a mapping.")
    return section


def _profile_mapping(
    experiment_section: Mapping[str, Any],
    experiment: str,
) -> Mapping[str, Mapping[str, Any]]:
    profiles = experiment_section.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError(f"Multidim experiment {experiment!r} has no profiles.")
    return {
        str(profile_name): cast(Mapping[str, Any], profile_section)
        for profile_name, profile_section in profiles.items()
        if isinstance(profile_section, Mapping)
    }


def _required_string(section: Mapping[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string key {key!r} in multidim profile metadata.")
    return value


def _required_bool(section: Mapping[str, Any], key: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean key {key!r} in multidim profile metadata.")
    return value


__all__ = [
    "DEFAULT_MULTIDIM_PROFILES_PATH",
    "MultidimProfileSelection",
    "list_multidim_profiles",
    "load_multidim_profiles",
    "select_multidim_profile",
]
