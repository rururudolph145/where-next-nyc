"""validate_helpers.py — bronze schema validation helpers for 15_minute_refresh_inference_dag.

Airflow-free by design: the experiment container has Airflow but the host
test env does not, so the DAG's logic lives here where pytest can import it.
The DAG file (``exp03a_validate_dag.py``) is a thin wrapper that pulls
``bronze_path`` from XCom, builds the override config, and calls
``run_validate``.

Two seams keep this unit-testable:

  * :func:`build_validate_config` deep-overrides ``data.silver_path`` so
    ``validation_log_dir(config)`` resolves under the experiment root —
    never production ``data/``.
  * :func:`run_validate` reads bronze, calls ``validate_bronze_schema``,
    and writes a log inside ``finally`` (so failures are logged too). On
    ``DataValidationError`` it re-raises as ``RuntimeError`` so Airflow
    marks the task failed → ``advance``/``write`` skip under
    ``all_success`` → watermark held.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from app.core.clock import Clock
from app.core.paths import validation_log_dir
from app.ingestion.schema import (
    DataValidationError,
    validate_bronze_schema,
    write_validation_log,
)


def build_validate_config(base: dict[str, Any], *, silver_root: str) -> dict[str, Any]:
    """Return a copy of *base* with ``data.silver_path`` pointed at *silver_root*.

    The override is the isolation seam: ``validation_log_dir(config)`` derives
    its path from ``data.silver_path``, so redirecting that key here keeps logs
    under the experiment root and never the production ``data/`` tree.

    Args:
        base: The parsed config to override (not mutated).
        silver_root: Absolute path for the experiment silver parquet.

    Returns:
        A deep copy of *base* with ``data.silver_path`` overridden.
    """
    config = copy.deepcopy(base)
    config["data"]["silver_path"] = silver_root
    return config


def run_validate(
    bronze_path: str,
    *,
    config: dict[str, Any],
    clock: Clock | None = None,
) -> str:
    """Validate bronze schema; write a log; re-raise on failure.

    Reads *bronze_path*, runs ``validate_bronze_schema``, and writes a
    ``validation_<timestamp>.json`` log via ``write_validation_log`` in a
    ``finally`` block — so the log is written on BOTH the happy path and the
    failure path before any exception propagates.

    On ``DataValidationError`` the exception is re-raised as ``RuntimeError``
    so Airflow marks the task failed; ``advance_watermark`` and
    ``write_watermark`` then skip under ``trigger_rule="all_success"``,
    leaving the watermark unchanged.

    Args:
        bronze_path: Absolute path to the bronze parquet to validate.
        config: Override config with ``data.silver_path`` pointing at the
            experiment root (see :func:`build_validate_config`).
        clock: Optional ``Clock`` for deterministic timestamps in tests.
            Defaults to ``WallClock()`` inside ``write_validation_log``.

    Returns:
        *bronze_path* unchanged — a stable XCom value for downstream tasks.

    Raises:
        RuntimeError: If bronze fails schema validation.
    """
    df = pd.read_parquet(bronze_path)
    log_dir = validation_log_dir(config)
    failure_reasons: list[str] = []
    try:
        validate_bronze_schema(df)
    except DataValidationError as exc:
        failure_reasons = exc.failure_reasons
        raise RuntimeError(f"bronze validation failed: {failure_reasons}") from exc
    finally:
        write_validation_log(log_dir, df, failure_reasons, clock=clock)
    return bronze_path
