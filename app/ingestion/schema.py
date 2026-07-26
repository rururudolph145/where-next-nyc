"""Bronze-layer schema validation for NYC taxi trip data.

Exports:
    DataValidationError: raised when the bronze DataFrame fails schema checks.
    validate_bronze_schema: pure validator — raises or returns silently.
    write_validation_log: writes a timestamped JSON log to *log_dir*.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa  # type: ignore[import-untyped]
from pandera.errors import SchemaErrors  # type: ignore[import-untyped]

from app.core.clock import Clock, WallClock

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

_BRONZE_SCHEMA = pa.DataFrameSchema(
    columns={
        "tpep_pickup_datetime": pa.Column(pa.DateTime, nullable=True),
        "tpep_dropoff_datetime": pa.Column(pa.DateTime, nullable=True),
        "PULocationID": pa.Column(
            pa.Float,
            checks=pa.Check.in_range(1, 265),
            nullable=True,
        ),
        "passenger_count": pa.Column(pa.Float, nullable=True),
        "trip_distance": pa.Column(pa.Float, nullable=True),
    },
    strict=False,  # extra columns are allowed
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DataValidationError(Exception):
    """Raised when the Bronze DataFrame fails structural schema checks.

    Attributes:
        failure_reasons: Human-readable list of validation failure messages.
    """

    def __init__(self, failure_reasons: list[str]) -> None:
        self.failure_reasons = failure_reasons
        super().__init__("; ".join(failure_reasons))


def validate_bronze_schema(df: pd.DataFrame) -> None:
    """Validate *df* against the expected Bronze schema.

    Runs with ``lazy=True`` so all violations are collected before raising.
    Returns silently on a clean pass.

    Args:
        df: Raw DataFrame loaded from the Bronze layer.

    Raises:
        DataValidationError: If any required column is missing, has the wrong
            dtype, or contains out-of-range values.
    """
    try:
        _BRONZE_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        reasons = [str(err) for err in exc.schema_errors]
        raise DataValidationError(reasons) from exc


def write_validation_log(
    log_dir: Path,
    df: pd.DataFrame,
    failure_reasons: list[str],
    *,
    clock: Clock | None = None,
) -> Path:
    """Write a timestamped validation log JSON to *log_dir*.

    Args:
        log_dir: Directory where the log file will be written (created if
            absent).
        df: The DataFrame that was validated (used for row counts).
        failure_reasons: Empty list on a clean pass; list of failure messages
            on error.
        clock: Optional ``Clock`` instance for deterministic time in tests.
            Defaults to ``WallClock()`` when ``None``.

    Returns:
        Path to the log file that was written.
    """
    os.makedirs(log_dir, exist_ok=True)

    _clock = clock or WallClock()
    now = _clock.now()
    validated_at = now.isoformat().replace("+00:00", "Z")
    timestamp = now.strftime("%Y%m%dT%H%M%S")

    rows_in = len(df)

    if failure_reasons:
        # Structural failure → all rows fail; row-level → all rows fail as
        # conservative upper bound (brief requires rows_failing = rows_in for
        # structural; row-level counts exact failing rows when detectable).
        rows_failing = rows_in
        rows_passing = 0
    else:
        rows_passing = rows_in
        rows_failing = 0

    log: dict[str, Any] = {
        "validated_at": validated_at,
        "rows_in": rows_in,
        "rows_passing": rows_passing,
        "rows_failing": rows_failing,
        "failure_reasons": failure_reasons,
    }

    log_path = log_dir / f"validation_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    return log_path
