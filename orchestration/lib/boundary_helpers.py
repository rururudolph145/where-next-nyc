"""boundary_helpers.py — dataset-exhaustion check for 15_minute_refresh_inference_dag.

Pure function. Airflow-free — the DAG file wraps this and raises
AirflowSkipException when the watermark reaches DATASET_MAX.
"""

from __future__ import annotations

from datetime import datetime


def is_dataset_exhausted(window_start: datetime, *, dataset_max: datetime) -> bool:
    """Return True when window_start is at or past the dataset boundary.

    Uses ``>=`` to match production ``refresh_15min_dag._dataset_max()`` semantics:
    the dataset is exhausted when there is no new data starting at or after
    ``dataset_max``.
    """
    return window_start >= dataset_max
