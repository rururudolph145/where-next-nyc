"""exp02_ingest_bronze_dag -- EXP-02 cumulative-bronze ingest DAG.

Topology (linear, 4 tasks):
    read_watermark -> ingest -> advance_watermark -> write_watermark

Extends the EXP-01 now-signal DAG by inserting a single ``ingest`` task between
``read_watermark`` and ``advance_watermark``. A run reads the window, writes
CUMULATIVE bronze ``[seed_start, window_end)`` via the real
``app.ingestion.Ingestion``, then advances the watermark ONLY if ingest
succeeded (``advance``/``write`` run ``all_success``). So a failed ingest leaves
the watermark and prior bronze untouched, and the next run reprocesses the same
slice — the crash-safety contract from EXP-01, now guarding real data.

All real logic lives in Airflow-free modules so it is unit-testable without
Airflow (see tests/test_exp02_ingest.py and tests/test_exp01_watermark.py). This
file is a thin wrapper: it injects concrete paths / seed / slice and wires XCom.

Path isolation: ``build_ingest_config`` redirects raw/bronze to the mounted
experiment roots so ``Ingestion`` never touches the production ``data/`` tree.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG  # type: ignore[import-not-found]
from airflow.operators.python import PythonOperator  # type: ignore[import-not-found]

from app.core.config import load_config

try:
    # Host / pytest: repo root on sys.path, imported as a package.
    from orchestration_experiment.dags.exp01_watermark import (
        next_window,
        read_position,
        write_position,
    )
    from orchestration_experiment.dags.exp02_ingest import (
        build_ingest_config,
        run_ingest,
    )
except ModuleNotFoundError:
    # Container: /opt/airflow/dags on sys.path, imported flat.
    from exp01_watermark import (  # type: ignore[no-redef,import-not-found]
        next_window,
        read_position,
        write_position,
    )
    from exp02_ingest import (  # type: ignore[no-redef,import-not-found]
        build_ingest_config,
        run_ingest,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (injected here; the pure modules stay config-free)
# ---------------------------------------------------------------------------

# Mounted into the container via docker-compose.
WATERMARK_FILE = Path("/opt/airflow/data/watermark.json")
SEED_START = datetime(2021, 1, 18, 0, 0, 0)  # naive NYC wall-clock; from config.yml
SLICE = timedelta(minutes=15)

# Absolute experiment roots (the isolation seam). Raw is a read-only mount;
# bronze is written under the writable experiment data root, NOT production.
RAW_ROOT = "/opt/airflow/data/raw/yellow_tripdata"
BRONZE_ROOT = "/opt/airflow/data_exp/bronze/yellow_tripdata/bronze.parquet"


# ---------------------------------------------------------------------------
# Task callables (thin wrappers over the pure modules)
# ---------------------------------------------------------------------------


def _read_watermark(**context: object) -> dict[str, str]:
    """Resolve the window ``[W, W + SLICE)`` and push both bounds via XCom."""
    position = read_position(WATERMARK_FILE, seed=SEED_START)
    start, end = next_window(position, slice_=SLICE)
    window = {"window_start": start.isoformat(), "window_end": end.isoformat()}
    logger.info("read_watermark: %s", window)
    return window


def _ingest(**context: object) -> str:
    """Write cumulative bronze ``[SEED_START, window_end)`` via Ingestion.

    Pulls the advancing ``window_end`` from the read_watermark XCom, freezes the
    lower bound at ``SEED_START``, and injects the path-isolated override config
    so the real ``Ingestion`` writes only under the experiment root.
    """
    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_end = datetime.fromisoformat(window["window_end"])

    config = build_ingest_config(
        load_config(), raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT
    )
    bronze_path = run_ingest(window_end, seed_start=SEED_START, config=config)
    logger.info(
        "ingest: wrote bronze [%s, %s) -> %s", SEED_START, window_end, bronze_path
    )
    return bronze_path


def _advance_watermark(**context: object) -> str:
    """Compute the advanced position (kept separate for topology stability)."""
    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_end = window["window_end"]
    logger.info("advance_watermark: %s", window_end)
    return window_end


def _write_watermark(**context: object) -> None:
    """Atomically persist the advanced watermark (only after ingest succeeded)."""
    ti = context["ti"]  # type: ignore[index]
    window_end: str = ti.xcom_pull(task_ids="advance_watermark")  # type: ignore[attr-defined]
    write_position(WATERMARK_FILE, datetime.fromisoformat(window_end))
    logger.info("write_watermark: %s", window_end)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="exp02_ingest_bronze_dag",
    schedule="*/15 * * * *",
    catchup=False,
    max_active_runs=1,
    start_date=datetime(2021, 1, 18, tzinfo=timezone.utc),
    default_args={"retries": 1},
    tags=["exp", "ingest", "bronze"],
) as dag:
    read_watermark = PythonOperator(
        task_id="read_watermark",
        python_callable=_read_watermark,
    )
    ingest = PythonOperator(
        task_id="ingest",
        python_callable=_ingest,
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

    read_watermark >> ingest >> advance_watermark >> write_watermark
