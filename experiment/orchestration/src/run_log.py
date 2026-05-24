"""Per-run audit log for the inference simulation.

Each DAG run appends one row describing what just happened: the
simulated bucket, how many rows were predicted, how long the step took,
and whether the API smoke test succeeded.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_DIR = os.environ.get("INFERENCE_STATE_DIR", "/opt/airflow/state")
LOG_FILENAME = "run_log.csv"

_FIELDS = [
    "run_started_at",
    "simulated_bucket",
    "n_predictions",
    "predict_duration_s",
    "api_status_code",
    "api_zone_count",
    "ok",
]


@dataclass
class RunRecord:
    run_started_at: str
    simulated_bucket: str
    n_predictions: int
    predict_duration_s: float
    api_status_code: int | None
    api_zone_count: int | None
    ok: bool


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(record: RunRecord, log_dir: str = DEFAULT_LOG_DIR) -> Path:
    """Append one record to the run log CSV. Creates the file with a header on first call."""
    path = Path(log_dir) / LOG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(asdict(record))
    return path
