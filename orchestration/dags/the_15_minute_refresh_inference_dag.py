"""the_15_minute_refresh_inference_dag -- production inference refresh cycle.

Topology (linear, 8 tasks):
    read_watermark -> check_no_new_data -> ingest -> validate -> feature
        -> predict -> log_predict -> write_watermark

Incremental model
-----------------
The simulation position is a DATA watermark, not a clock. The fixed 2021 raw
data is walked one ``ingest.slice_duration`` slice at a time:

  * ``read_watermark`` loads the current position W (or ``ingest.seed_start``
    on the first run) and pushes the window ``[W, W + duration)``.
  * ``ingest`` filters the raw data to that window. The lower bound of the
    cumulative bronze is always the seed; the upper bound advances each run,
    so bronze is a growing superset (single-file full rewrite, ADR 001).
  * ``check_no_new_data`` raises AirflowSkipException when the watermark has
    reached ``DATASET_MAX`` -- the ``all_success`` trigger rule on
    ``write_watermark`` propagates the skip and leaves the watermark unchanged.
  * ``write_watermark`` writes ``window_end`` only after every upstream task
    succeeds. A mid-run failure leaves the watermark unchanged, so the next
    run reprocesses the same slice.

Timezone: the watermark / seed are tz-NAIVE datetimes (the raw
``tpep_pickup_datetime`` is tz-naive NYC wall-clock); the window filter is a
naive-vs-naive comparison.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from airflow import DAG  # type: ignore[import-not-found]
from airflow.exceptions import AirflowSkipException  # type: ignore[import-not-found]
from airflow.operators.python import PythonOperator  # type: ignore[import-not-found]

from app.core.config import load_config
from app.core.watermark import next_window, read_watermark, write_watermark

try:
    from orchestration.lib.boundary_helpers import is_dataset_exhausted
    from orchestration.lib.container_paths import container_paths
    from orchestration.lib.feature_helpers import build_feature_config, run_feature
    from orchestration.lib.ingest_helpers import build_ingest_config, run_ingest
    from orchestration.lib.log_predict_helpers import run_log_predict
    from orchestration.lib.predict_helpers import build_predict_config, run_predict
    from orchestration.lib.validate_helpers import build_validate_config, run_validate
except ImportError:
    from boundary_helpers import is_dataset_exhausted  # type: ignore[import-not-found,no-redef]
    from container_paths import container_paths  # type: ignore[import-not-found,no-redef]
    from feature_helpers import build_feature_config, run_feature  # type: ignore[import-not-found,no-redef]
    from ingest_helpers import build_ingest_config, run_ingest  # type: ignore[import-not-found,no-redef]
    from log_predict_helpers import run_log_predict  # type: ignore[import-not-found,no-redef]
    from predict_helpers import build_predict_config, run_predict  # type: ignore[import-not-found,no-redef]
    from validate_helpers import build_validate_config, run_validate  # type: ignore[import-not-found,no-redef]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# _WATERMARK_ROOT: pure module-level literal — used by _read_watermark /
# _write_watermark as a Path argument. Tasks derive all other paths at
# execution time via container_paths(config).
_WATERMARK_ROOT = Path("/opt/airflow/data/watermarks/refresh_15min")


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _read_watermark(**context: object) -> dict[str, str]:
    """Resolve the window ``[W, W + slice_duration)`` and push it via XCom.

    ``seed_start`` and ``slice_duration`` are pulled from ``config.yml`` at
    task-execution time (not module-import time -- avoids DAG-parse latency).
    """
    config = load_config()
    paths = container_paths(config)
    seed_start = datetime.fromisoformat(config.ingest.seed_start)
    duration = pd.Timedelta(config.ingest.slice_duration).to_pytimedelta()

    position = read_watermark(Path(paths["watermark"])) or seed_start
    start, end = next_window(position, duration)
    window = {"window_start": start.isoformat(), "window_end": end.isoformat()}
    logger.info("read_watermark: %s", window)
    return window


def _check_no_new_data(**context: object) -> None:
    """Skip the run when the dataset boundary has been reached."""
    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_start = datetime.fromisoformat(window["window_start"])
    config = load_config()
    dataset_max = datetime.fromisoformat(config.ingest.dataset_max)
    if is_dataset_exhausted(window_start, dataset_max=dataset_max):
        logger.info(
            "check_no_new_data: window_start=%s >= dataset_max=%s -- skipping run",
            window_start,
            dataset_max,
        )
        raise AirflowSkipException("no new data / dataset boundary reached")
    logger.info("check_no_new_data: window_start=%s -- continuing", window_start)


def _ingest(**context: object) -> str:
    """Write cumulative bronze ``[seed_start, window_end)``."""
    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_end = datetime.fromisoformat(window["window_end"])
    config = load_config()
    paths = container_paths(config)
    seed_start = datetime.fromisoformat(config.ingest.seed_start)

    ingest_config = build_ingest_config(
        config,
        raw_root=paths["raw"],
        bronze_root=paths["bronze"],
    )
    bronze_path = run_ingest(window_end, seed_start=seed_start, config=ingest_config)
    logger.info(
        "ingest: wrote bronze [%s, %s) -> %s", seed_start, window_end, bronze_path
    )
    return bronze_path


def _validate(**context: object) -> None:
    """Validate bronze schema; hold the watermark if it fails."""
    ti = context["ti"]  # type: ignore[index]
    bronze_path: str = ti.xcom_pull(task_ids="ingest")  # type: ignore[attr-defined]
    config = load_config()
    paths = container_paths(config)
    validate_config = build_validate_config(config, silver_root=paths["silver"])
    run_validate(bronze_path, config=validate_config)
    logger.info("validate: OK for %s", bronze_path)


def _feature(**context: object) -> None:
    """Run DataCleaner (bronze->silver) then FeatureEngineer (silver->gold)."""
    ti = context["ti"]  # type: ignore[index]
    bronze_path: str = ti.xcom_pull(task_ids="ingest")  # type: ignore[attr-defined]
    config = load_config()
    paths = container_paths(config)
    feature_config = build_feature_config(
        config,
        silver_root=paths["silver"],
        gold_root=paths["gold"],
        shapefile_path=paths["shapefile"],
        model_root=paths["models"],
    )
    run_feature(bronze_path, config=feature_config)
    logger.info("feature: silver+gold materialized for %s", bronze_path)


def _predict(**context: object) -> str | None:
    """Run Predictor on gold -> write demand_predictions.parquet.

    Pulls window_start from the read_watermark XCom and passes it as ``since``
    so only the new slice is scored (not the full cumulative gold table).
    """
    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    since = datetime.fromisoformat(window["window_start"])
    config = load_config()
    paths = container_paths(config)
    predict_config = build_predict_config(
        config,
        gold_root=paths["gold"],
        model_root=paths["models"],
        predictions_path=paths["predictions"],
    )
    output_path = run_predict(config=predict_config, since=since)
    logger.info("predict: wrote predictions -> %s", output_path)
    return output_path


def _log_predict(**context: object) -> None:
    """Reshape predictions parquet and write monitoring log."""
    config = load_config()
    paths = container_paths(config)
    result = run_log_predict(paths["predictions"], monitoring_root=paths["monitoring"])
    logger.info("log_predict: %s", result or "skipped (empty predictions)")


def _write_watermark(**context: object) -> None:
    """Atomically persist the advanced watermark (only after all tasks succeeded).

    Absorbs the former EXP-05 ``advance_watermark`` XCom-passthrough -- the
    single terminal task both computes the advance and persists it.
    """
    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_end = datetime.fromisoformat(window["window_end"])
    config = load_config()
    paths = container_paths(config)
    write_watermark(Path(paths["watermark"]), window_end)
    logger.info("write_watermark: %s", window_end)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="15_minute_refresh_inference_dag",
    schedule="*/15 * * * *",
    catchup=False,
    max_active_runs=1,
    start_date=datetime(2021, 1, 18, tzinfo=timezone.utc),
    default_args={"retries": 1},
    tags=["inference", "refresh"],
) as dag:
    read_watermark_task = PythonOperator(
        task_id="read_watermark",
        python_callable=_read_watermark,
    )
    check_no_new_data = PythonOperator(
        task_id="check_no_new_data",
        python_callable=_check_no_new_data,
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
    predict = PythonOperator(
        task_id="predict",
        python_callable=_predict,
    )
    log_predict = PythonOperator(
        task_id="log_predict",
        python_callable=_log_predict,
    )
    write_watermark_task = PythonOperator(
        task_id="write_watermark",
        python_callable=_write_watermark,
        trigger_rule="all_success",
    )

    (
        read_watermark_task
        >> check_no_new_data
        >> ingest
        >> validate
        >> feature
        >> predict
        >> log_predict
        >> write_watermark_task
    )
