"""Prediction log writer for the /heatmap endpoint.

Writes one parquet file per request into a date-partitioned directory tree:
  {root}/dt={YYYY-MM-DD}/served_at={ISO-ts}-{uuid8}.parquet

The written file captures the full set of zone-level predictions alongside
the model provenance fields so that offline accuracy calculations can join
against Gold actuals later (PBI 23).
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.clock import Clock, WallClock

logger = logging.getLogger(__name__)

# Module-level clock instance; tests can replace via monkeypatch or by
# passing clock= to log_predictions().
_clock: Clock = WallClock()


def _now(clock: Clock | None = None) -> datetime:
    """Return the current UTC datetime.

    Args:
        clock: Optional ``Clock`` override. Falls back to the module-level
            ``_clock`` (``WallClock`` by default).

    Returns:
        Current datetime with UTC timezone.
    """
    return (clock or _clock).now()


def log_predictions(response: dict[str, Any], path: str) -> Path | None:
    """Write a single-request prediction log parquet file.

    Reads the ``predictions`` list from *response*, enriches each row with
    model provenance fields and a ``served_at`` timestamp, and writes the
    result to a date-partitioned directory under *path*.

    Args:
        response: Minimal dict with keys ``"predictions"``, ``"model_version"``,
            and ``"served_from_stage"``.  Matches the structure built inside
            ``get_heatmap`` in ``app.api.main``.
        path: Root directory for prediction logs
            (e.g. ``"data/monitoring/predictions"``).

    Returns:
        The ``Path`` of the written parquet file, or ``None`` if
        ``response["predictions"]`` was empty (no file is written).
    """
    items: list[dict[str, Any]] = response.get("predictions", [])
    if not items:
        logger.debug("Skipped empty predictions")
        return None

    served_at = _now()

    model_version: str | None = response.get("model_version")
    served_from_stage: str | None = response.get("served_from_stage")

    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "bucket_time": pd.Timestamp(item["bucket_time"], tz="UTC"),
                "pulocationid": int(item["pulocationid"]),
                "predicted": float(item["predicted_trip_count"]),
                "model_version": model_version,
                "served_from_stage": served_from_stage,
                "served_at": served_at,
            }
        )

    df = pd.DataFrame(rows)
    df["pulocationid"] = df["pulocationid"].astype("int64")
    df["predicted"] = df["predicted"].astype("float64")
    df["bucket_time"] = pd.to_datetime(df["bucket_time"], utc=True).astype(
        "datetime64[us, UTC]"
    )
    df["served_at"] = (
        pd.to_datetime(df["served_at"].apply(lambda dt: dt.isoformat()))
        .dt.tz_convert("UTC")
        .astype("datetime64[us, UTC]")
    )

    dt_str = served_at.date().isoformat()
    ts_safe = served_at.strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    uid8 = uuid.uuid4().hex[:8]
    filename = f"served_at={ts_safe}-{uid8}.parquet"

    out_dir = os.path.join(path, f"dt={dt_str}")
    # Use umask=0 so new date dirs are world-writable (777).
    # Airflow worker runs as UID 50000; host user (jason) needs write access too.
    old_mask = os.umask(0)
    try:
        os.makedirs(out_dir, mode=0o777, exist_ok=True)
    finally:
        os.umask(old_mask)

    out_path = os.path.join(out_dir, filename)
    df.to_parquet(out_path, index=False)

    return Path(out_path)
