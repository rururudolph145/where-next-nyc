"""ingest_helpers.py — bronze ingestion helpers for 15_minute_refresh_inference_dag.

Airflow-free by design: the experiment container has Airflow but the host test
env does not, so the DAG's logic lives here where pytest can import it.

Two seams keep this unit-testable without a 340MB raw read (testing rule: no
external data in pytest):

  * :func:`build_ingest_config` deep-overrides ``data.raw_path`` /
    ``data.bronze_path`` so ``Ingestion`` reads raw from the read-only mount and
    writes bronze under the isolated experiment root — never the production
    ``data/`` tree.
  * :func:`run_ingest` implements an incremental append strategy: on first run
    (no bronze yet) it seeds the full ``[seed_start, window_end)`` range; on
    subsequent runs it reads only the month(s) at-or-after ``window_start`` and
    appends idempotently via ``Ingestion.append_slice``. Bronze stays cumulative
    on disk (required by DataCleaner.load_bronze); only the *read* path changes.
"""

from __future__ import annotations

from datetime import datetime

from app.core.config import AppConfig
from app.ingestion.ingest import Ingestion


def build_ingest_config(
    base: AppConfig, *, raw_root: str, bronze_root: str
) -> AppConfig:
    """Return a new AppConfig with the data paths pointed at experiment roots.

    The override is the isolation seam: ``Ingestion.__init__`` reads
    ``data.raw_path`` / ``data.bronze_path`` directly, so redirecting them here
    keeps the container off the CWD-relative production paths. Everything else —
    including ``ingest.slice_duration`` and ``feature.bucket`` (which must stay
    equal or ``Ingestion.__init__`` asserts) — passes through unchanged.

    Args:
        base: The parsed AppConfig to override (not mutated — frozen model).
        raw_root: Absolute path to the read-only raw ``yellow_tripdata`` root.
        bronze_root: Absolute path for the bronze parquet under the experiment
            data root.

    Returns:
        A new AppConfig with the two data paths overridden.
    """
    return base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"raw_path": raw_root, "bronze_path": bronze_root}
            )
        }
    )


def run_ingest(
    window_end: datetime,
    *,
    window_start: datetime,
    seed_start: datetime,
    config: AppConfig,
) -> str:
    """Incrementally append to cumulative bronze and return its path.

    First run (no bronze on disk): seeds the full ``[seed_start, window_end)``
    range via a full raw read — a one-time acceptable cost.

    Subsequent runs (bronze exists): reads only the month(s) at-or-after
    ``window_start``, clips to ``[window_start, window_end)``, then appends to
    the existing bronze via ``Ingestion.append_slice`` (idempotent retry-safe).

    Both approaches write bronze once via ``save_bronze``; the caller never sees
    partial writes.

    Args:
        window_end: Exclusive upper bound — the advancing watermark position.
        window_start: Inclusive lower bound of the *new* slice (= current watermark).
        seed_start: Fixed seed for the first-run full ingest.
        config: Injected override config (see :func:`build_ingest_config`).

    Returns:
        Path to the written bronze parquet.
    """
    ingestion = Ingestion(config=config)
    if not ingestion.bronze_exists():
        raw = ingestion.load_raw(since=None)
        seeded = ingestion._filter_window(raw, seed_start, window_end)
        return ingestion.save_bronze(seeded)

    existing = ingestion.load_bronze()
    raw = ingestion.load_raw(since=window_start)
    new_slice = ingestion._filter_window(raw, window_start, window_end)
    merged = ingestion.append_slice(existing, new_slice, cutoff=window_start)
    return ingestion.save_bronze(merged)
