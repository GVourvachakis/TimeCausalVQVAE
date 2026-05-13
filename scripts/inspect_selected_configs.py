"""Inspect selected paper reproduction configs without importing model code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"

HEADERS = [
    "experiment",
    "dataset",
    "objective",
    "encoder",
    "decoder",
    "conditioner",
    "prior",
    "beta",
    "alpha",
    "epochs",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one selected experiment YAML file."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def require_mapping(data: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    """Return a nested mapping or raise a compact validation error."""
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a '{key}' mapping")
    return value


def value_text(value: Any) -> str:
    """Format YAML scalar values for the compact table."""
    if value is None:
        return "-"
    return str(value)


def row_for(path: Path) -> list[str]:
    """Build one display row from a selected experiment config."""
    data = load_yaml(path)
    experiment = require_mapping(data, "experiment", path)
    model = require_mapping(data, "model", path)
    training = require_mapping(data, "training", path)

    return [
        value_text(experiment.get("name")),
        value_text(experiment.get("dataset")),
        value_text(model.get("objective")),
        value_text(model.get("encoder")),
        value_text(model.get("decoder")),
        value_text(model.get("conditioner")),
        value_text(model.get("prior")),
        value_text(model.get("beta")),
        value_text(model.get("alpha")),
        value_text(training.get("epochs")),
    ]


def print_table(rows: list[list[str]]) -> None:
    """Print a simple fixed-width table."""
    widths = [len(header) for header in HEADERS]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))

    print(format_row(HEADERS))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def main() -> None:
    """Inspect all selected configs under configs/experiments."""
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    if not paths:
        raise SystemExit(f"No YAML configs found under {CONFIG_DIR}")
    rows = [row_for(path) for path in paths]
    print_table(rows)


if __name__ == "__main__":
    main()
