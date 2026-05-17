"""Automatic model selection for evaluated Time-Causal VAE checkpoints."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

SelectionCriterion = Literal["mmd", "swd", "weighted_sum", "pareto"]


@dataclass(frozen=True)
class CandidateMetric:
    """Metrics and metadata for one candidate ``final_model`` directory."""

    model_dir: Path
    metric_path: Path
    metrics: dict[str, float]


def find_final_model_dirs(experiment_dir: Path) -> list[Path]:
    """Return candidate ``final_model`` directories below an experiment directory."""
    return sorted(
        (path for path in experiment_dir.rglob("final_model") if path.is_dir()),
        key=lambda path: str(path),
    )


def select_model(
    *,
    experiment_dir: Path,
    criterion: SelectionCriterion = "mmd",
    mmd_weight: float = 1.0,
    swd_weight: float = 1.0,
    compute_missing: bool = False,
    base_data_dir: str = "data",
    n_sample_test: int = 1000,
    seed: int = 0,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Select a checkpoint from evaluated candidate ``final_model`` folders.

    Parameters
    ----------
    experiment_dir:
        Directory containing one or more training runs.
    criterion:
        Selection criterion. Lower values are better for `mmd`, `swd`, and
        `weighted_sum`. `pareto` currently computes a non-dominated front and
        chooses the lowest weighted sum within that front.
    mmd_weight:
        Weight used by `weighted_sum` and the Pareto tie-break.
    swd_weight:
        Weight used by `weighted_sum` and the Pareto tie-break.
    compute_missing:
        If true, compute missing `hyper_metric.pkl` files through the target
        evaluator and save them inside each candidate `final_model` directory.
    base_data_dir:
        Base data directory passed to the evaluator when computing missing metrics.
    n_sample_test:
        Number of generated/test samples used when computing missing metrics.
    seed:
        Evaluation seed used when computing missing metrics.
    output_path:
        Metadata path. Defaults to `<experiment_dir>/selected_model.json`.
    """
    experiment_dir = experiment_dir.resolve()
    output_path = output_path or experiment_dir / "selected_model.json"
    candidates = find_final_model_dirs(experiment_dir)
    if not candidates:
        raise FileNotFoundError(f"No final_model directories found below {experiment_dir}")

    candidate_metrics = [
        _load_or_compute_candidate(
            model_dir,
            experiment_dir=experiment_dir,
            compute_missing=compute_missing,
            base_data_dir=base_data_dir,
            n_sample_test=n_sample_test,
            seed=seed,
            candidate_count=len(candidates),
        )
        for model_dir in candidates
    ]
    available = [candidate for candidate in candidate_metrics if candidate is not None]
    if not available:
        raise FileNotFoundError(
            "No candidate hyper_metric.pkl files found. Re-run with --compute-missing "
            "or evaluate checkpoints before selecting a model."
        )

    ranked = rank_candidates(
        available,
        criterion=criterion,
        mmd_weight=mmd_weight,
        swd_weight=swd_weight,
    )
    selected = ranked[0]
    metadata = _selection_metadata(
        experiment_dir=experiment_dir,
        criterion=criterion,
        selected=selected,
        ranked=ranked,
        scanned_count=len(candidates),
        mmd_weight=mmd_weight,
        swd_weight=swd_weight,
        compute_missing=compute_missing,
        base_data_dir=base_data_dir,
        n_sample_test=n_sample_test,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def rank_candidates(
    candidates: list[CandidateMetric],
    *,
    criterion: SelectionCriterion,
    mmd_weight: float,
    swd_weight: float,
) -> list[CandidateMetric]:
    """Rank candidates by the requested criterion."""
    if criterion == "mmd":
        return sorted(
            candidates, key=lambda candidate: (candidate.metrics["mmd"], str(candidate.model_dir))
        )
    if criterion == "swd":
        return sorted(
            candidates, key=lambda candidate: (candidate.metrics["swd"], str(candidate.model_dir))
        )
    if criterion == "weighted_sum":
        return sorted(
            candidates,
            key=lambda candidate: (
                _weighted_sum(candidate.metrics, mmd_weight=mmd_weight, swd_weight=swd_weight),
                str(candidate.model_dir),
            ),
        )
    if criterion == "pareto":
        front = _pareto_front(candidates)
        return sorted(
            front,
            key=lambda candidate: (
                _weighted_sum(candidate.metrics, mmd_weight=mmd_weight, swd_weight=swd_weight),
                str(candidate.model_dir),
            ),
        )
    raise ValueError(f"Unsupported selection criterion: {criterion}")


def _load_or_compute_candidate(
    model_dir: Path,
    *,
    experiment_dir: Path,
    compute_missing: bool,
    base_data_dir: str,
    n_sample_test: int,
    seed: int,
    candidate_count: int,
) -> CandidateMetric | None:
    metric_path = _find_metric_path(
        model_dir,
        experiment_dir=experiment_dir,
        candidate_count=candidate_count,
    )
    if metric_path is None and compute_missing:
        metric_path = _compute_metric(
            model_dir,
            base_data_dir=base_data_dir,
            n_sample_test=n_sample_test,
            seed=seed,
        )
    if metric_path is None:
        return None

    metrics = _load_metric(metric_path)
    _require_metric(metrics, "mmd", metric_path)
    _require_metric(metrics, "swd", metric_path)
    return CandidateMetric(model_dir=model_dir, metric_path=metric_path, metrics=metrics)


def _find_metric_path(
    model_dir: Path,
    *,
    experiment_dir: Path,
    candidate_count: int,
) -> Path | None:
    candidates = [
        model_dir / "hyper_metric.pkl",
        model_dir.parent / "evaluation" / "hyper_metric.pkl",
    ]
    if candidate_count == 1:
        candidates.append(experiment_dir / "evaluation" / "hyper_metric.pkl")
    for path in candidates:
        if path.exists():
            return path
    return None


def _compute_metric(
    model_dir: Path,
    *,
    base_data_dir: str,
    n_sample_test: int,
    seed: int,
) -> Path:
    from time_causal_vae.evaluation.checkpoints import TargetModelEvaluator

    evaluator = TargetModelEvaluator(str(model_dir), base_data_dir=base_data_dir)
    real_data, fake_data, _ = evaluator.load_data(n_sample_test=n_sample_test, seed=seed)
    hyper_metric = evaluator.compute_hyper_metric(real_data, fake_data)
    metric_path = model_dir / "hyper_metric.pkl"
    with metric_path.open("wb") as handle:
        pickle.dump(hyper_metric, handle)
    return metric_path


def _load_metric(metric_path: Path) -> dict[str, float]:
    with metric_path.open("rb") as handle:
        raw_metric = pickle.load(handle)
    if not isinstance(raw_metric, dict):
        raise TypeError(f"{metric_path} must contain a metric mapping")
    return {str(key): _metric_scalar(value) for key, value in raw_metric.items()}


def _metric_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Metric tensors must contain a single scalar")
        return float(value.detach().cpu())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _require_metric(metrics: dict[str, float], key: str, metric_path: Path) -> None:
    if key not in metrics:
        raise KeyError(f"{metric_path} is missing required metric '{key}'")


def _weighted_sum(metrics: dict[str, float], *, mmd_weight: float, swd_weight: float) -> float:
    return mmd_weight * metrics["mmd"] + swd_weight * metrics["swd"]


def _pareto_front(candidates: list[CandidateMetric]) -> list[CandidateMetric]:
    front = []
    for candidate in candidates:
        dominated = any(_dominates(other, candidate) for other in candidates if other != candidate)
        if not dominated:
            front.append(candidate)
    return front


def _dominates(left: CandidateMetric, right: CandidateMetric) -> bool:
    left_mmd = left.metrics["mmd"]
    left_swd = left.metrics["swd"]
    right_mmd = right.metrics["mmd"]
    right_swd = right.metrics["swd"]
    return (left_mmd <= right_mmd and left_swd <= right_swd) and (
        left_mmd < right_mmd or left_swd < right_swd
    )


def _selection_metadata(
    *,
    experiment_dir: Path,
    criterion: SelectionCriterion,
    selected: CandidateMetric,
    ranked: list[CandidateMetric],
    scanned_count: int,
    mmd_weight: float,
    swd_weight: float,
    compute_missing: bool,
    base_data_dir: str,
    n_sample_test: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "experiment_dir": str(experiment_dir),
        "criterion": criterion,
        "criterion_note": _criterion_note(criterion),
        "selected_model_dir": str(selected.model_dir),
        "selected_metric_path": str(selected.metric_path),
        "selected_metrics": selected.metrics,
        "scanned_final_model_count": scanned_count,
        "ranked_candidate_count": len(ranked),
        "weights": {
            "mmd": mmd_weight,
            "swd": swd_weight,
        },
        "compute_missing": compute_missing,
        "base_data_dir": base_data_dir,
        "n_sample_test": n_sample_test,
        "seed": seed,
        "ranked_candidates": [
            {
                "rank": rank,
                "model_dir": str(candidate.model_dir),
                "metric_path": str(candidate.metric_path),
                "metrics": candidate.metrics,
                "weighted_sum": _weighted_sum(
                    candidate.metrics,
                    mmd_weight=mmd_weight,
                    swd_weight=swd_weight,
                ),
            }
            for rank, candidate in enumerate(ranked, start=1)
        ],
    }


def _criterion_note(criterion: SelectionCriterion) -> str:
    if criterion == "pareto":
        return (
            "Pareto selection currently uses the non-dominated MMD/SWD front "
            "with weighted-sum tie-breaking."
        )
    return "Lower metric values are ranked first."
