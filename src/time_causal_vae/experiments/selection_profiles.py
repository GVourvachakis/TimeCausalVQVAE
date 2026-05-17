"""Selection-profile scoring for per-experiment model comparison."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

ProfileName = Literal[
    "distributional",
    "tail_risk",
    "sequential_dependence",
    "balanced_market",
]

MetricValue = int | float
MetricMapping = Mapping[str, MetricValue]

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "mmd": ("mmd",),
    "swd": ("swd",),
    "returns_wasserstein": (
        "returns_wasserstein",
        "return_wasserstein",
        "returns_w1",
        "return_w1",
    ),
    "terminal_wasserstein": (
        "terminal_return_wasserstein",
        "terminal_wasserstein",
        "terminal_w1",
    ),
    "volatility_wasserstein": (
        "volatility_wasserstein",
        "volatility_w1",
    ),
    "drawdown_wasserstein": (
        "maximum_drawdown_wasserstein",
        "drawdown_wasserstein",
        "drawdown_w1",
    ),
    "return_ac_l1": (
        "return_autocorrelation_within_path_l1",
        "return_autocorrelation_l1",
        "return_ac_l1",
    ),
    "squared_return_ac_l1": (
        "squared_return_autocorrelation_within_path_l1",
        "squared_return_autocorrelation_l1",
        "squared_return_ac_l1",
    ),
}

PROFILE_COMPONENTS: dict[ProfileName, tuple[str, ...]] = {
    "distributional": ("mmd", "swd", "returns_wasserstein"),
    "tail_risk": (
        "terminal_wasserstein",
        "volatility_wasserstein",
        "drawdown_wasserstein",
    ),
    "sequential_dependence": ("return_ac_l1", "squared_return_ac_l1"),
    "balanced_market": (
        "mmd",
        "swd",
        "returns_wasserstein",
        "terminal_wasserstein",
        "volatility_wasserstein",
        "drawdown_wasserstein",
        "return_ac_l1",
        "squared_return_ac_l1",
    ),
}


@dataclass(frozen=True)
class ProfileScore:
    """Visible component metrics and lower-is-better profile score."""

    profile: ProfileName
    score: float | None
    components: dict[str, float]
    missing_metrics: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class CandidateProfileScore:
    """Rank-based profile score for one named candidate."""

    candidate: str
    profile: ProfileName
    score: float | None
    rank_components: dict[str, float]
    metric_components: dict[str, float]
    missing_metrics: list[str]
    warnings: list[str]


def distributional_profile_score(
    metrics: MetricMapping,
    *,
    allow_mmd_only: bool = False,
) -> ProfileScore:
    """Score MMD, SWD, and returns W1 when available."""
    return score_profile(metrics, "distributional", allow_mmd_only=allow_mmd_only)


def tail_risk_profile_score(
    metrics: MetricMapping,
    *,
    allow_mmd_only: bool = False,
) -> ProfileScore:
    """Score terminal, volatility, and drawdown W1 when available."""
    return score_profile(metrics, "tail_risk", allow_mmd_only=allow_mmd_only)


def sequential_dependence_profile_score(
    metrics: MetricMapping,
    *,
    allow_mmd_only: bool = False,
) -> ProfileScore:
    """Score return and squared-return autocorrelation L1 metrics."""
    return score_profile(metrics, "sequential_dependence", allow_mmd_only=allow_mmd_only)


def balanced_market_profile_score(
    metrics: MetricMapping,
    *,
    allow_mmd_only: bool = False,
) -> ProfileScore:
    """Score all available market-profile path metrics."""
    return score_profile(metrics, "balanced_market", allow_mmd_only=allow_mmd_only)


def score_profile(
    metrics: MetricMapping,
    profile: ProfileName,
    *,
    allow_mmd_only: bool = False,
) -> ProfileScore:
    """Return a lower-is-better profile score while keeping components visible."""
    components, missing = visible_components(metrics, PROFILE_COMPONENTS[profile])
    warnings = missing_metric_warnings(profile, missing)
    if not components:
        return ProfileScore(
            profile=profile,
            score=None,
            components=components,
            missing_metrics=missing,
            warnings=[*warnings, f"profile {profile!r} has no available metrics."],
        )
    if set(components) == {"mmd"} and not allow_mmd_only:
        return ProfileScore(
            profile=profile,
            score=None,
            components=components,
            missing_metrics=missing,
            warnings=[
                *warnings,
                "MMD is the only available component; refusing profile selection from MMD alone.",
            ],
        )
    return ProfileScore(
        profile=profile,
        score=sum(components.values()) / len(components),
        components=components,
        missing_metrics=missing,
        warnings=warnings,
    )


def rank_candidates(
    candidates: Mapping[str, MetricMapping],
    profile: ProfileName,
    *,
    allow_mmd_only: bool = False,
) -> list[CandidateProfileScore]:
    """Rank candidates by average lower-is-better metric ranks."""
    component_names = PROFILE_COMPONENTS[profile]
    candidate_components = {
        name: visible_components(metrics, component_names)[0]
        for name, metrics in candidates.items()
    }
    usable_components = sorted(
        {component for components in candidate_components.values() for component in components}
    )
    if usable_components == ["mmd"] and not allow_mmd_only:
        return [
            CandidateProfileScore(
                candidate=name,
                profile=profile,
                score=None,
                rank_components={},
                metric_components=dict(candidate_components[name]),
                missing_metrics=visible_components(candidates[name], component_names)[1],
                warnings=[
                    "MMD is the only available component; refusing profile "
                    "selection from MMD alone."
                ],
            )
            for name in candidates
        ]

    ranks_by_component = {
        component: lower_is_better_ranks(
            {
                candidate: components[component]
                for candidate, components in candidate_components.items()
                if component in components
            }
        )
        for component in usable_components
    }
    scored: list[CandidateProfileScore] = []
    for candidate, metrics in candidates.items():
        components, missing = visible_components(metrics, component_names)
        rank_components = {
            component: ranks[candidate]
            for component, ranks in ranks_by_component.items()
            if candidate in ranks
        }
        warnings = missing_metric_warnings(profile, missing)
        score = None
        if rank_components:
            score = sum(rank_components.values()) / len(rank_components)
        else:
            warnings.append(f"profile {profile!r} has no available metrics.")
        scored.append(
            CandidateProfileScore(
                candidate=candidate,
                profile=profile,
                score=score,
                rank_components=rank_components,
                metric_components=components,
                missing_metrics=missing,
                warnings=warnings,
            )
        )
    return sorted(scored, key=lambda item: float("inf") if item.score is None else item.score)


def visible_components(
    metrics: MetricMapping,
    component_names: Iterable[str],
) -> tuple[dict[str, float], list[str]]:
    """Extract canonical metric components and report missing components."""
    components: dict[str, float] = {}
    missing: list[str] = []
    for component in component_names:
        value = first_available_metric(metrics, METRIC_ALIASES[component])
        if value is None:
            missing.append(component)
        else:
            components[component] = value
    return components, missing


def first_available_metric(metrics: MetricMapping, names: Sequence[str]) -> float | None:
    """Return the first finite metric matching one of the aliases."""
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        numeric_value = float(value)
        if isfinite(numeric_value):
            return numeric_value
    return None


def lower_is_better_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Return average ranks for lower-is-better values, with tie handling."""
    sorted_items = sorted(values.items(), key=lambda item: item[1])
    ranks: dict[str, float] = {}
    index = 0
    while index < len(sorted_items):
        tie_end = index + 1
        while tie_end < len(sorted_items) and sorted_items[tie_end][1] == sorted_items[index][1]:
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        for candidate, _value in sorted_items[index:tie_end]:
            ranks[candidate] = average_rank
        index = tie_end
    return ranks


def missing_metric_warnings(profile: ProfileName, missing: Sequence[str]) -> list[str]:
    """Return explicit warnings for missing profile metrics."""
    return [f"profile {profile!r} missing metric component {metric!r}." for metric in missing]
