"""Check Hawkes-jump dataset exposure for oracle-label leakage."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import ml_collections
import torch
import yaml
from torch import Tensor

from time_causal_vae.data.hawkes_jump import HawkesJumpDataset
from time_causal_vae.data.pipeline import DataPipeline

ORACLE_FIELD_NAMES = (
    "jump_indicators",
    "jump_counts",
    "jump_sizes",
    "intensities",
    "volatilities",
    "metadata",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the script parser."""
    parser = argparse.ArgumentParser(
        description="Check Hawkes-jump dataset model-visible tensors for oracle leakage.",
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/hawkes_jump_causal_vq_tokenizer.yaml",
        help="Hawkes-jump experiment config to inspect.",
    )
    parser.add_argument("--n-samples", type=int, help="Optional sample-count override.")
    parser.add_argument("--n-timesteps", type=int, help="Optional path-length override.")
    parser.add_argument("--seed", type=int, help="Optional simulation seed override.")
    parser.add_argument(
        "--simulation-scheme",
        choices=("fixed_grid", "ogata"),
        help="Optional simulation scheme override.",
    )
    return parser


def main() -> int:
    """Run the Hawkes-jump dataset no-leakage check."""
    args = build_parser().parse_args()
    raw_config = load_yaml(args.config)
    dataset_name = str(require_mapping(raw_config, "experiment")["dataset"])
    if dataset_name != "hawkes_jump":
        raise SystemExit(f"Expected experiment.dataset='hawkes_jump'; got {dataset_name!r}.")

    data_config = require_mapping(raw_config, "data")
    data_params = extract_data_params(data_config)
    if args.simulation_scheme is not None:
        data_params["simulation_scheme"] = args.simulation_scheme
    if args.seed is not None:
        data_params["seed"] = args.seed

    n_samples = int(args.n_samples or data_config["n_samples"])
    n_timesteps = int(args.n_timesteps or data_config["n_timesteps"])
    direct_dataset = HawkesJumpDataset(n_samples, n_timesteps, **data_params)

    pipeline_config = ml_collections.ConfigDict()
    pipeline_config.dataset = "hawkes_jump"
    pipeline_config.n_sample = n_samples
    pipeline_config.n_timestep = n_timesteps
    pipeline_config.data_params = dict(data_params)
    pipeline_config.base_data_dir = "data/processed"
    train_dataset, eval_dataset = DataPipeline()(pipeline_config)

    try:
        checks = run_checks(
            direct_dataset=direct_dataset,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_data=train_dataset.data,
            train_labels=train_dataset.labels,
            eval_data=eval_dataset.data,
            eval_labels=eval_dataset.labels,
            data_output=cast(str, data_params.get("data_output", "price")),
        )
    except Exception as exc:
        print(f"FAIL Hawkes-jump dataset no-leakage check: {exc}")
        return 1

    print("PASS Hawkes-jump dataset no-leakage check")
    print(f"config={args.config}")
    print(f"simulation_scheme={data_params.get('simulation_scheme', 'fixed_grid')}")
    print(f"data_output={data_params.get('data_output', 'price')}")
    print(f"direct_data={tuple(direct_dataset.data.shape)}")
    print(f"direct_labels={tuple(direct_dataset.labels.shape)}")
    print(f"pipeline_train_data={tuple(train_dataset.data.shape)}")
    print(f"pipeline_train_labels={tuple(train_dataset.labels.shape)}")
    print(f"pipeline_eval_data={tuple(eval_dataset.data.shape)}")
    print(f"pipeline_eval_labels={tuple(eval_dataset.labels.shape)}")
    for name, value in checks.items():
        print(f"{name}={value}")
    return 0


def run_checks(
    *,
    direct_dataset: HawkesJumpDataset,
    train_dataset: object,
    eval_dataset: object,
    train_data: Tensor,
    train_labels: Tensor,
    eval_data: Tensor,
    eval_labels: Tensor,
    data_output: str,
) -> dict[str, bool | str]:
    """Run leakage checks and raise on failure."""
    checks: dict[str, bool | str] = {}
    visible_reference = (
        direct_dataset.log_returns if data_output == "log_return" else direct_dataset.prices
    )
    assert_true(torch.allclose(direct_dataset.data, visible_reference), "visible data mismatch")
    checks["visible_data_matches_configured_output"] = True

    assert_true(torch.allclose(train_data, direct_dataset.data), "pipeline changed visible data")
    checks["pipeline_exposes_only_configured_data_tensor"] = True

    oracle_names_present_on_wrapped_dataset = [
        name for name in ORACLE_FIELD_NAMES if hasattr(train_dataset, name)
    ]
    oracle_names_present_on_eval_dataset = [
        name for name in ORACLE_FIELD_NAMES if hasattr(eval_dataset, name)
    ]
    assert_true(
        not oracle_names_present_on_wrapped_dataset and not oracle_names_present_on_eval_dataset,
        "oracle fields unexpectedly attached to wrapped pipeline datasets: "
        f"train={oracle_names_present_on_wrapped_dataset}, "
        f"eval={oracle_names_present_on_eval_dataset}",
    )
    checks["oracle_fields_are_dataset_attributes_only"] = True

    for oracle_name in (
        "jump_indicators",
        "jump_counts",
        "jump_sizes",
        "intensities",
        "volatilities",
    ):
        oracle_tensor = cast(Tensor, getattr(direct_dataset, oracle_name))
        assert_true(
            not torch.allclose(train_data, oracle_tensor.float()),
            f"model-visible data is identical to oracle field {oracle_name}",
        )
    checks["visible_data_not_oracle_tensor"] = True

    assert_true(train_labels.ndim == 2, "labels must be scalar per path for current config")
    assert_true(eval_labels.ndim == 2, "eval labels must be scalar per path for current config")
    assert_true(train_labels.shape[1] == 1, "expected scalar condition labels")
    assert_true(eval_labels.shape[1] == 1, "expected scalar eval condition labels")
    assert_true(train_labels.unique().numel() == 1, "train labels are informative")
    assert_true(eval_labels.unique().numel() == 1, "eval labels are informative")
    checks["labels_are_non_informative_constant_scalars"] = True

    assert_true(not torch.allclose(train_data, eval_data), "train/eval data unexpectedly identical")
    checks["eval_split_uses_distinct_seed_when_config_seed_present"] = True
    checks["condition_prefix_safety"] = "passed_scalar_constant_condition"
    return checks


def extract_data_params(data_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return Hawkes-jump data params from either config schema."""
    raw_params = data_config.get("data_params", data_config.get("params", {}))
    if raw_params is None:
        return {}
    if not isinstance(raw_params, Mapping):
        raise SystemExit("Expected data params to be a mapping.")
    return dict(raw_params)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cast(dict[str, Any], loaded)


def require_mapping(raw_config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a required config section."""
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected config section {key!r}.")
    return cast(dict[str, Any], value)


def assert_true(value: bool, message: str) -> None:
    """Raise an assertion if value is false."""
    if not value:
        raise AssertionError(message)


if __name__ == "__main__":
    raise SystemExit(main())
