"""Target evaluation package for Time-Causal VAE."""

from typing import Any

__all__ = [
    "SWD",
    "GaussianMMD",
    "GaussianMMD2",
    "ModelEvaluator",
    "TargetModelEvaluator",
]


def __getattr__(name: str) -> Any:
    """Lazily expose evaluation helpers without importing heavy backends."""
    if name in {"ModelEvaluator", "TargetModelEvaluator"}:
        from time_causal_vae.evaluation.checkpoints import ModelEvaluator, TargetModelEvaluator

        return {"ModelEvaluator": ModelEvaluator, "TargetModelEvaluator": TargetModelEvaluator}[
            name
        ]
    if name in {"SWD", "GaussianMMD", "GaussianMMD2"}:
        from time_causal_vae.evaluation.metrics import SWD, GaussianMMD, GaussianMMD2

        return {"SWD": SWD, "GaussianMMD": GaussianMMD, "GaussianMMD2": GaussianMMD2}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
