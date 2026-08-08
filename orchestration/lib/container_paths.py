"""container_paths — join the container mount prefix onto config-relative data paths.

Airflow-free so host pytest can import it without a DAG environment.
Each task that needs absolute in-container paths calls this helper after its
own load_config() rather than reading module-level hardcoded strings.

Usage (in a DAG task callable)::

    config = load_config()
    paths = container_paths(config)
    ingest_config = build_ingest_config(
        config, raw_root=paths["raw"], bronze_root=paths["bronze"]
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_DEFAULT_PREFIX = "/opt/airflow"

_RELATIVE_GETTERS: list[tuple[str, list[str]]] = [
    ("raw", ["data", "raw_path"]),
    ("bronze", ["data", "bronze_path"]),
    ("silver", ["data", "silver_path"]),
    ("gold", ["data", "gold_path"]),
    ("shapefile", ["feature", "zone_shapefile_path"]),
    ("iqr_root", ["model", "dag_iqr_root"]),
    ("trained", ["model", "trained_path"]),
    ("predictions", ["serving", "predictions_path"]),
    ("watermark", ["serving", "watermark_path"]),
    ("monitoring", ["monitoring", "predictions_path"]),
]


def container_paths(
    config: dict[str, Any],
    prefix: str = _DEFAULT_PREFIX,
) -> dict[str, str]:
    """Return a dict of absolute in-container paths derived from *config*.

    Reproduces the exact /opt/airflow/... strings that were previously
    hardcoded as module-level DAG constants. Each value is ``prefix / relative``
    where ``relative`` is the config value and must NOT start with ``/``.

    Args:
        config: Parsed configuration dictionary (from load_config()).
        prefix: Container mount prefix. Defaults to ``/opt/airflow``.

    Returns:
        Dict with keys: raw, bronze, silver, gold, shapefile, iqr_root,
        trained, predictions, watermark, monitoring.

    Raises:
        ValueError: If any config-derived relative path starts with ``/``,
            which would silently drop the prefix in pathlib.
    """
    p = Path(prefix)
    result: dict[str, str] = {}
    for key, key_path in _RELATIVE_GETTERS:
        section, *rest = key_path
        node: Any = config[section]
        for part in rest:
            node = node[part]
        relative: str = str(node)
        if relative.startswith("/"):
            raise ValueError(
                f"container_paths: config key {key!r} has a leading slash "
                f"({relative!r}), which would silently drop the prefix {prefix!r}. "
                "Remove the leading slash from config.yml."
            )
        result[key] = str(p / relative)
    return result
