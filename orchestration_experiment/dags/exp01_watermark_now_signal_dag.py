"""exp01_watermark_now_signal_dag -- EXP-01 "now-signal" DAG.

Topology (linear, 3 tasks):
    read_watermark -> advance_watermark -> write_watermark

The DAG proves the simulation "now-signal" (the data watermark) advances by
exactly one ``SLICE`` per run, with NO data processing. Later EXP PBIs insert
ingest / validate / feature / predict BETWEEN read_watermark and
advance_watermark without restructuring — which is why advance and write are
kept as separate tasks.

All real logic lives in the Airflow-free ``exp01_watermark`` module so it is
unit-testable without Airflow (see tests/test_exp01_watermark.py). This file is
a thin wrapper: it injects the concrete path / seed / slice and wires XCom.

Crash safety: ``advance_watermark`` and ``write_watermark`` are the last tasks
and run ``all_success``; a failure upstream leaves the control file unchanged,
so the next run reprocesses the same slice.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG  # type: ignore[import-not-found]
from airflow.operators.python import PythonOperator  # type: ignore[import-not-found]

try:
    # Host / pytest: repo root on sys.path, imported as a package.
    from orchestration_experiment.dags.exp01_watermark import (
        next_window,
        read_position,
        write_position,
    )
except ModuleNotFoundError:
    # Container: /opt/airflow/dags on sys.path, imported flat.
    from exp01_watermark import (  # type: ignore[no-redef]
        next_window,
        read_position,
        write_position,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (injected here; the pure module stays config-free)
# ---------------------------------------------------------------------------

# Mounted into the container via docker-compose (- ./data:/opt/airflow/data).
WATERMARK_FILE = Path("/opt/airflow/data/watermark.json")
SEED_START = datetime(2024, 6, 3, 0, 0, 0)  # naive NYC wall-clock
SLICE = timedelta(minutes=15)


# ---------------------------------------------------------------------------
# Task callables (thin wrappers over the pure module)
# ---------------------------------------------------------------------------


def _read_watermark(**context: object) -> dict[str, str]:
    """Resolve the window ``[W, W + SLICE)`` and push both bounds via XCom."""
    position = read_position(WATERMARK_FILE, seed=SEED_START)
    start, end = next_window(position, slice_=SLICE)
    window = {"window_start": start.isoformat(), "window_end": end.isoformat()}
    logger.info("read_watermark: %s", window)
    return window


def _advance_watermark(**context: object) -> str:
    """Compute the advanced position (kept separate for EXP-02+ insertion)."""
    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_end = window["window_end"]
    logger.info("advance_watermark: %s", window_end)
    return window_end


def _write_watermark(**context: object) -> None:
    """Atomically persist the advanced watermark."""
    ti = context["ti"]  # type: ignore[index]
    window_end: str = ti.xcom_pull(task_ids="advance_watermark")  # type: ignore[attr-defined]
    write_position(WATERMARK_FILE, datetime.fromisoformat(window_end))
    logger.info("write_watermark: %s", window_end)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="exp01_watermark_now_signal_dag",
    schedule="*/15 * * * *",
    catchup=False,
    max_active_runs=1,
    start_date=datetime(2024, 6, 3, tzinfo=timezone.utc),
    default_args={"retries": 1},
    tags=["exp", "watermark"],
) as dag:
    read_watermark = PythonOperator(
        task_id="read_watermark",
        python_callable=_read_watermark,
    )
    advance_watermark = PythonOperator(
        task_id="advance_watermark",
        python_callable=_advance_watermark,
        trigger_rule="all_success",
    )
    write_watermark = PythonOperator(
        task_id="write_watermark",
        python_callable=_write_watermark,
        trigger_rule="all_success",
    )

    read_watermark >> advance_watermark >> write_watermark
