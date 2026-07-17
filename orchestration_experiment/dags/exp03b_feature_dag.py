"""exp03b_feature_dag -- EXP-03b bronze→silver→gold feature DAG.

Topology (linear, 6 tasks):
    read_watermark -> ingest -> validate -> feature -> advance_watermark -> write_watermark

Extends the EXP-03a validate DAG by inserting a single ``feature`` task between
``validate`` and ``advance_watermark``. A run reads the window, writes cumulative
bronze, validates it, then runs DataCleaner (bronze→silver) and FeatureEngineer
(silver→gold) under the ``feature`` label. If any upstream task fails,
``advance``/``write`` skip under ``all_success`` → the watermark is held.

All real logic lives in Airflow-free modules so it is unit-testable without
Airflow (see tests/test_exp03b_feature.py). This file is a thin wrapper: it
pulls ``bronze_path`` from the ``ingest`` XCom and ``window_end`` from the
``read_watermark`` XCom (needed for the OQ4 train/inference cutoff), builds
the override config via ``build_feature_config``, and calls ``run_feature``.

Path isolation: ``build_feature_config`` redirects silver/gold/shapefile/model
paths to the experiment root so artefacts land under ``data_exp/`` and
``data/reference/`` (read-only shapefile mount) — never touching production
``data/`` or ``models/``.
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
    from orchestration_experiment.dags.exp03a_validate import (
        build_validate_config,
        run_validate,
    )
    from orchestration_experiment.dags.exp03b_feature import (
        build_feature_config,
        run_feature,
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
    from exp03a_validate import (  # type: ignore[no-redef,import-not-found]
        build_validate_config,
        run_validate,
    )
    from exp03b_feature import (  # type: ignore[no-redef,import-not-found]
        build_feature_config,
        run_feature,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (injected here; the pure modules stay config-free)
# ---------------------------------------------------------------------------

WATERMARK_FILE = Path("/opt/airflow/data/watermark.json")
SEED_START = datetime(2021, 1, 18, 0, 0, 0)  # naive NYC wall-clock; from config.yml
SLICE = timedelta(minutes=15)

RAW_ROOT = "/opt/airflow/data/raw/yellow_tripdata"
BRONZE_ROOT = "/opt/airflow/data_exp/bronze/yellow_tripdata/bronze.parquet"
SILVER_ROOT = "/opt/airflow/data_exp/silver/yellow_tripdata_cleaned.parquet"
GOLD_ROOT = "/opt/airflow/data_exp/gold/yellow_tripdata"
SHAPEFILE_PATH = "/opt/airflow/data/reference/taxi_zones.shp"
MODEL_ROOT = "/opt/airflow/data_exp/models"


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
    """Write cumulative bronze ``[SEED_START, window_end)`` via Ingestion."""
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


def _validate(**context: object) -> None:
    """Validate bronze schema; hold the watermark if it fails."""
    ti = context["ti"]  # type: ignore[index]
    bronze_path: str = ti.xcom_pull(task_ids="ingest")  # type: ignore[attr-defined]
    config = build_validate_config(load_config(), silver_root=SILVER_ROOT)
    run_validate(bronze_path, config=config)
    logger.info("validate: OK for %s", bronze_path)


def _feature(**context: object) -> None:
    """Run DataCleaner (bronze→silver) then FeatureEngineer (silver→gold)."""
    ti = context["ti"]  # type: ignore[index]
    bronze_path: str = ti.xcom_pull(task_ids="ingest")  # type: ignore[attr-defined]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_end = datetime.fromisoformat(window["window_end"])
    config = build_feature_config(
        load_config(),
        silver_root=SILVER_ROOT,
        gold_root=GOLD_ROOT,
        shapefile_path=SHAPEFILE_PATH,
        model_root=MODEL_ROOT,
        window_end=window_end,
    )
    run_feature(bronze_path, config=config)
    logger.info("feature: silver+gold materialized for %s", bronze_path)


def _advance_watermark(**context: object) -> str:
    """Compute the advanced position (kept separate for topology stability)."""
    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_end = window["window_end"]
    logger.info("advance_watermark: %s", window_end)
    return window_end


def _write_watermark(**context: object) -> None:
    """Atomically persist the advanced watermark (only after feature succeeded)."""
    ti = context["ti"]  # type: ignore[index]
    window_end: str = ti.xcom_pull(task_ids="advance_watermark")  # type: ignore[attr-defined]
    write_position(WATERMARK_FILE, datetime.fromisoformat(window_end))
    logger.info("write_watermark: %s", window_end)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="exp03b_feature_dag",
    schedule="*/15 * * * *",
    catchup=False,
    max_active_runs=1,
    start_date=datetime(2021, 1, 18, tzinfo=timezone.utc),
    default_args={"retries": 1},
    tags=["exp", "validate", "feature", "gold"],
) as dag:
    read_watermark = PythonOperator(
        task_id="read_watermark",
        python_callable=_read_watermark,
    )
    ingest = PythonOperator(
        task_id="ingest",
        python_callable=_ingest,
    )
    validate = PythonOperator(
        task_id="validate",
        python_callable=_validate,
    )
    feature = PythonOperator(
        task_id="feature",
        python_callable=_feature,
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

    (
        read_watermark
        >> ingest
        >> validate
        >> feature
        >> advance_watermark
        >> write_watermark
    )
