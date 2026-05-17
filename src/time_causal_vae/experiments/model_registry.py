"""Metadata registry utilities for selected continuous and discrete models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from time_causal_vae.experiments.selection_profiles import ProfileName, rank_candidates
from time_causal_vae.typing import PathLike

FamilyName = Literal["continuous", "discrete"]
RegistryData = dict[str, Any]

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3] / "trained_models" / "model_registry.yaml"
)
FAMILIES: tuple[FamilyName, ...] = ("continuous", "discrete")
LOCAL_PATH_KEYS = (
    "checkpoint_path",
    "tokenizer_checkpoint_path",
    "prior_checkpoint_path",
    "token_data_path",
)


@dataclass(frozen=True)
class RegisteredModel:
    """Selected registry entry with config paths, metrics, and local path status."""

    experiment: str
    family: FamilyName
    candidate_id: str
    selected_by: str
    selection_profile: str | None
    config: str | None
    tokenizer_config: str | None
    prior_config: str | None
    checkpoint_paths: dict[str, str]
    local_checkpoint_status: dict[str, bool | None]
    sampling: dict[str, Any]
    metrics: dict[str, float]
    missing_metrics: list[str]
    warnings: list[str]
    candidate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def load_registry(path: PathLike | None = None) -> RegistryData:
    """Load the trained-model metadata registry."""
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Registry {registry_path} must contain a YAML mapping.")
    return cast(RegistryData, data)


def list_experiments(registry: Mapping[str, Any]) -> list[str]:
    """List experiments registered in metadata."""
    experiments = registry.get("experiments", {})
    if not isinstance(experiments, Mapping):
        return []
    return sorted(str(name) for name in experiments)


def list_candidates(
    registry: Mapping[str, Any],
    experiment: str,
    family: FamilyName,
) -> list[str]:
    """List candidate identifiers for one experiment and model family."""
    candidates = _candidate_mapping(registry, experiment, family)
    return sorted(candidates)


def select_registered_model(
    registry: Mapping[str, Any],
    experiment: str,
    family: FamilyName,
    *,
    metric: str | None = None,
    profile: ProfileName = "balanced_market",
    allow_mmd_only: bool = False,
) -> RegisteredModel:
    """Select a registered candidate by metric, explicit selection, or profile score.

    Selection order is:

    1. user-specified lower-is-better metric, when provided;
    2. explicit `selected` field from the registry;
    3. lower-is-better profile ranking across available path metrics.
    """
    family_section = _family_section(registry, experiment, family)
    candidates = _candidate_mapping(registry, experiment, family)
    warnings: list[str] = []

    if metric is not None:
        candidate_id, metric_warnings = _select_by_metric(candidates, metric)
        warnings.extend(metric_warnings)
        selected_by = f"metric:{metric}"
    else:
        explicit_selected = family_section.get("selected")
        if isinstance(explicit_selected, str) and explicit_selected in candidates:
            candidate_id = explicit_selected
            selected_by = "explicit_selected"
        else:
            candidate_id, profile_warnings = _select_by_profile(
                candidates,
                profile,
                allow_mmd_only=allow_mmd_only,
            )
            warnings.extend(profile_warnings)
            selected_by = f"profile:{profile}"

    candidate = dict(candidates[candidate_id])
    candidate_warnings = candidate.get("warnings", [])
    if isinstance(candidate_warnings, list):
        warnings.extend(str(item) for item in candidate_warnings)

    return RegisteredModel(
        experiment=experiment,
        family=family,
        candidate_id=candidate_id,
        selected_by=selected_by,
        selection_profile=_optional_string(candidate.get("selection_profile")),
        config=_optional_string(candidate.get("config")),
        tokenizer_config=_optional_string(candidate.get("tokenizer_config")),
        prior_config=_optional_string(candidate.get("prior_config")),
        checkpoint_paths=_checkpoint_paths(candidate),
        local_checkpoint_status=_local_checkpoint_status(candidate),
        sampling=_mapping_to_dict(candidate.get("sampling")),
        metrics=_numeric_metrics(candidate.get("metrics")),
        missing_metrics=_string_list(candidate.get("missing_metrics")),
        warnings=warnings,
        candidate=candidate,
    )


def registry_summary(path: PathLike | None = None) -> dict[str, Any]:
    """Return a compact summary for notebooks and scripts."""
    registry = load_registry(path)
    return {
        "schema_version": registry.get("schema_version"),
        "experiments": {
            experiment: {
                family: list_candidates(registry, experiment, family)
                for family in FAMILIES
                if _has_family(registry, experiment, family)
            }
            for experiment in list_experiments(registry)
        },
    }


def _family_section(
    registry: Mapping[str, Any],
    experiment: str,
    family: FamilyName,
) -> Mapping[str, Any]:
    experiments = registry.get("experiments", {})
    if not isinstance(experiments, Mapping) or experiment not in experiments:
        known = ", ".join(sorted(str(name) for name in experiments)) or "<none>"
        raise KeyError(f"Unknown experiment {experiment!r}. Known experiments: {known}.")
    experiment_section = experiments[experiment]
    if not isinstance(experiment_section, Mapping) or family not in experiment_section:
        raise KeyError(f"Experiment {experiment!r} has no {family!r} model family.")
    family_section = experiment_section[family]
    if not isinstance(family_section, Mapping):
        raise ValueError(f"Registry section experiments.{experiment}.{family} must be a mapping.")
    return cast(Mapping[str, Any], family_section)


def _candidate_mapping(
    registry: Mapping[str, Any],
    experiment: str,
    family: FamilyName,
) -> Mapping[str, Mapping[str, Any]]:
    family_section = _family_section(registry, experiment, family)
    candidates = family_section.get("candidates", {})
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError(f"Experiment {experiment!r} family {family!r} has no candidates.")
    return {
        str(candidate_id): cast(Mapping[str, Any], candidate)
        for candidate_id, candidate in candidates.items()
        if isinstance(candidate, Mapping)
    }


def _select_by_metric(
    candidates: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> tuple[str, list[str]]:
    metric_values = {
        candidate_id: value
        for candidate_id, candidate in candidates.items()
        if (value := _metric_value(candidate, metric)) is not None
    }
    if not metric_values:
        raise ValueError(f"Metric {metric!r} is unavailable for all registered candidates.")
    candidate_id = min(metric_values, key=metric_values.__getitem__)
    missing = sorted(set(candidates) - set(metric_values))
    warnings = [f"metric {metric!r} missing for candidate {candidate!r}." for candidate in missing]
    return candidate_id, warnings


def _select_by_profile(
    candidates: Mapping[str, Mapping[str, Any]],
    profile: ProfileName,
    *,
    allow_mmd_only: bool,
) -> tuple[str, list[str]]:
    metric_sets = {
        candidate_id: _numeric_metrics(candidate.get("metrics"))
        for candidate_id, candidate in candidates.items()
    }
    ranked = rank_candidates(metric_sets, profile, allow_mmd_only=allow_mmd_only)
    if not ranked or ranked[0].score is None:
        raise ValueError(f"Profile {profile!r} cannot score any registered candidate.")
    warnings: list[str] = []
    for scored in ranked:
        warnings.extend(scored.warnings)
    return ranked[0].candidate, warnings


def _metric_value(candidate: Mapping[str, Any], metric: str) -> float | None:
    metrics = candidate.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(metric)
    if isinstance(value, int | float):
        numeric = float(value)
        if isfinite(numeric):
            return numeric
    return None


def _numeric_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, metric_value in value.items():
        if isinstance(metric_value, int | float):
            numeric = float(metric_value)
            if isfinite(numeric):
                metrics[str(key)] = numeric
    return metrics


def _checkpoint_paths(candidate: Mapping[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key in (*LOCAL_PATH_KEYS, "checkpoint_convention"):
        value = candidate.get(key)
        if isinstance(value, str):
            paths[key] = value
    return paths


def _local_checkpoint_status(candidate: Mapping[str, Any]) -> dict[str, bool | None]:
    status: dict[str, bool | None] = {}
    for key in LOCAL_PATH_KEYS:
        value = candidate.get(key)
        if not isinstance(value, str):
            continue
        if value == "local_outputs_only":
            status[key] = None
        elif value.startswith("outputs/"):
            status[key] = Path(value).exists()
        else:
            status[key] = None
    return status


def _mapping_to_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _has_family(registry: Mapping[str, Any], experiment: str, family: FamilyName) -> bool:
    try:
        _family_section(registry, experiment, family)
    except (KeyError, ValueError):
        return False
    return True
