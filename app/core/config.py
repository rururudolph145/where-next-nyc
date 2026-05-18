"""Centralized configuration loading for the NYC Taxi Demand Forecasting project."""

from typing import Any

import yaml


def load_config(path: str = "config.yml") -> dict[str, Any]:
    """Load and return the YAML configuration file.

    Args:
        path: Path to the YAML configuration file. Defaults to
            ``"config.yml"`` relative to the current working directory.

    Returns:
        Parsed configuration as a nested dictionary.
    """
    with open(path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config
