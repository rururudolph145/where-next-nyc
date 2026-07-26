"""log_predict_helpers.py — prediction logging helpers for 15_minute_refresh_inference_dag.

Airflow-free by design (the host test env has no Airflow). The DAG file
(``exp04_predict_log_dag.py``) is a thin wrapper that calls ``run_log_predict``.

The only real logic here is the reshape: reading the predictions parquet and
building the ``/heatmap``-shaped response dict that ``log_predictions`` expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.monitoring.prediction_log import log_predictions

logger = logging.getLogger(__name__)


def run_log_predict(
    parquet_path: str,
    *,
    monitoring_root: str,
) -> Path | None:
    """Read predictions parquet, reshape to heatmap response, call log_predictions.

    Returns None (and skips logging) when the parquet file is missing or empty —
    this covers the EXP-03b pre-cutoff empty-gold reality.
    """
    path = Path(parquet_path)
    if not path.exists():
        logger.warning(
            "log_predict: predictions parquet not found at %s — skipping", path
        )
        return None

    df = pd.read_parquet(path)
    if df.empty:
        logger.warning(
            "log_predict: predictions parquet is empty at %s — skipping", path
        )
        return None

    predictions: list[dict[str, Any]] = [
        {
            "pulocationid": int(row["pulocationid"]),
            "predicted_trip_count": float(row["predicted_trip_count"]),
            "bucket_time": pd.Timestamp(row["hour"]).isoformat(),
        }
        for _, row in df.iterrows()
    ]

    response: dict[str, Any] = {
        "predictions": predictions,
        "model_version": None,
        "served_from_stage": "exp04_predict_log_dag",
    }

    return log_predictions(response, path=monitoring_root)
