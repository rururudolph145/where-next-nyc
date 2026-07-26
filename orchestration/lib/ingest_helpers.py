"""ingest_helpers.py — bronze ingestion helpers for 15_minute_refresh_inference_dag.

Airflow-free by design: the experiment container has Airflow but the host test
env does not, so the DAG's logic lives here where pytest can import it. The DAG
file (``exp02_ingest_bronze_dag.py``) is a thin wrapper that pulls ``window_end``
from XCom, freezes ``seed_start`` from config, and calls these functions.

Two seams keep this unit-testable without a 340MB raw read (testing rule: no
external data in pytest):

  * :func:`build_ingest_config` deep-overrides ``data.raw_path`` /
    ``data.bronze_path`` so ``Ingestion`` reads raw from the read-only mount and
    writes bronze under the isolated experiment root — never the production
    ``data/`` tree. It preserves ``ingest.slice_duration == feature.bucket`` (an
    ``Ingestion.__init__`` assertion) by leaving those keys untouched.
  * :func:`run_ingest` constructs ``Ingestion(config=...)`` and calls
    ``run(window_start=seed_start, window_end=window_end)`` with the seed FROZEN
    as the lower bound. Bronze is therefore cumulative ``[seed, window_end)`` and
    grows monotonically as the watermark advances (ADR-001, matches the old prod
    DAG).
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from app.ingestion.ingest import Ingestion


def build_ingest_config(
    base: dict[str, Any], *, raw_root: str, bronze_root: str
) -> dict[str, Any]:
    """Return a copy of *base* with the data paths pointed at experiment roots.

    The override is the isolation seam: ``Ingestion.__init__`` reads
    ``data.raw_path`` / ``data.bronze_path`` directly, so redirecting them here
    keeps the container off the CWD-relative production paths. Everything else —
    including ``ingest.slice_duration`` and ``feature.bucket`` (which must stay
    equal or ``Ingestion.__init__`` asserts) — passes through unchanged.

    Args:
        base: The parsed config to override (not mutated).
        raw_root: Absolute path to the read-only raw ``yellow_tripdata`` root.
        bronze_root: Absolute path for the bronze parquet under the experiment
            data root.

    Returns:
        A deep copy of *base* with the two data paths overridden.
    """
    config = copy.deepcopy(base)
    config["data"]["raw_path"] = raw_root
    config["data"]["bronze_path"] = bronze_root
    return config


def run_ingest(
    window_end: datetime, *, seed_start: datetime, config: dict[str, Any]
) -> str:
    """Ingest cumulative bronze ``[seed_start, window_end)`` and return its path.

    ``window_start`` is FROZEN at *seed_start* every run; only *window_end*
    advances with the watermark, so bronze grows monotonically. Both bounds are
    naive (the raw ``tpep_pickup_datetime`` is tz-naive NYC wall-clock).

    Args:
        window_end: Exclusive upper bound — the advancing watermark position.
        seed_start: Inclusive lower bound — the fixed seed, never advances.
        config: The injected override config (see :func:`build_ingest_config`).

    Returns:
        Path to the written bronze parquet.
    """
    ingestion = Ingestion(config=config)
    return ingestion.run(window_start=seed_start, window_end=window_end)
