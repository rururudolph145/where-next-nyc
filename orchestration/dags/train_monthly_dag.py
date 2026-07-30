"""train_monthly_dag -- manually-triggered monthly training cycle.

Topology (linear, 5 tasks):
    read_watermarks -> check_window_ready -> compute_window -> train
        -> write_training_watermark

Training model
--------------
The DAG is manually triggered (``schedule=None``). It reads two watermarks:

  * ``refresh_15min/.current_watermark`` -- the inference position (must
    exist; training-before-inference is invalid).
  * ``last_training/.current_watermark`` -- the last training position
    (absent on first run → bootstrap fallback to ``ingest.seed_start``).

``check_window_ready`` skips the run via AirflowSkipException when the
inference watermark has not advanced ≥30 days ahead of the last training
watermark (ADR-003 §Decision.6).  A crash in ``train`` holds
``last_training`` watermark unchanged (crash→hold semantics).  A gate
rejection advances the watermark but leaves the production model unchanged
(reject→advance semantics).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from airflow import DAG  # type: ignore[import-not-found]
from airflow.operators.python import PythonOperator  # type: ignore[import-not-found]

from app.core.config import load_config
from app.core.watermark import write_watermark

try:
    from orchestration.lib.training_helpers import (
        check_window_ready,
        compute_window,
        resolve_watermarks,
        run_train,
    )
except ImportError:
    from training_helpers import (  # type: ignore[import-not-found,no-redef]
        check_window_ready,
        compute_window,
        resolve_watermarks,
        run_train,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REFRESH_WM_ROOT = Path("/opt/airflow/data/watermarks/refresh_15min")
_LAST_TRAINING_ROOT = Path("/opt/airflow/data/watermarks/last_training")
GOLD_FILE = "/opt/airflow/data/gold/yellow_tripdata/hourly_demand_featured.parquet"
MODEL_NAME = "ny_taxi_demand_model"


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _read_watermarks(**context: object) -> dict[str, object]:
    """Resolve both watermarks and push them via XCom."""
    config = load_config()
    result = resolve_watermarks(config, _REFRESH_WM_ROOT, _LAST_TRAINING_ROOT)
    logger.info("read_watermarks: %s", result)
    return result


def _check_window_ready(**context: object) -> None:
    """Skip when inference hasn't advanced ≥30d ahead of last training."""
    ti = context["ti"]  # type: ignore[index]
    wm: dict[str, object] = ti.xcom_pull(task_ids="read_watermarks")  # type: ignore[attr-defined]
    refresh_wm = datetime.fromisoformat(str(wm["refresh_wm"]))
    last_training_wm = datetime.fromisoformat(str(wm["last_training_wm"]))
    check_window_ready(refresh_wm, last_training_wm)
    logger.info(
        "check_window_ready: refresh=%s last_training=%s — continuing",
        refresh_wm,
        last_training_wm,
    )


def _compute_window(**context: object) -> dict[str, str]:
    """Compute (window_start, window_end) and push via XCom."""
    ti = context["ti"]  # type: ignore[index]
    wm: dict[str, object] = ti.xcom_pull(task_ids="read_watermarks")  # type: ignore[attr-defined]
    config = load_config()
    seed_start = datetime.fromisoformat(config["ingest"]["seed_start"])
    last_training_wm = datetime.fromisoformat(str(wm["last_training_wm"]))
    start, end = compute_window(seed_start, last_training_wm)
    window = {"window_start": start.isoformat(), "window_end": end.isoformat()}
    logger.info("compute_window: %s", window)
    return window


def _train(**context: object) -> dict[str, object]:
    """Call run_train; return gate_result, model_version, wape_lift_pct via XCom."""
    from mlflow.tracking import MlflowClient  # type: ignore[import-not-found]

    from app.training.trainer import Trainer

    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="compute_window")  # type: ignore[attr-defined]
    window_start = datetime.fromisoformat(window["window_start"])
    window_end = datetime.fromisoformat(window["window_end"])

    result = run_train(
        gold_file=Path(GOLD_FILE),
        window_start=window_start,
        window_end=window_end,
        model_name=MODEL_NAME,
        trainer_factory=Trainer,
        client_factory=MlflowClient,
    )
    logger.info("train: %s", result)
    return result


def _write_training_watermark(**context: object) -> None:
    """Atomically persist the advanced last_training watermark."""
    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="compute_window")  # type: ignore[attr-defined]
    window_end = datetime.fromisoformat(window["window_end"])
    write_watermark(_LAST_TRAINING_ROOT, window_end)
    logger.info("write_training_watermark: %s", window_end)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="train_monthly_dag",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    start_date=datetime(2021, 1, 18, tzinfo=timezone.utc),
    default_args={"retries": 1},
    tags=["training", "monthly"],
) as dag:
    read_watermarks = PythonOperator(
        task_id="read_watermarks",
        python_callable=_read_watermarks,
    )
    check_window_ready_task = PythonOperator(
        task_id="check_window_ready",
        python_callable=_check_window_ready,
    )
    compute_window_task = PythonOperator(
        task_id="compute_window",
        python_callable=_compute_window,
    )
    train = PythonOperator(
        task_id="train",
        python_callable=_train,
    )
    write_training_watermark = PythonOperator(
        task_id="write_training_watermark",
        python_callable=_write_training_watermark,
        trigger_rule="all_success",
    )

    (
        read_watermarks
        >> check_window_ready_task
        >> compute_window_task
        >> train
        >> write_training_watermark
    )
