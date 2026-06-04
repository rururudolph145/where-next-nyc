"""refresh_15min_dag -- 15-minute ingestion-to-inference refresh cycle.

Topology (linear chain):
    decide_bucket -> check_watermark -> ingest -> validate
        -> feature -> predict -> log_predict -> write_done

Idempotency: ``check_watermark`` inspects the watermark directory for a
``.done`` file keyed by ``(date, bucket)``.  If the file exists the task
raises ``AirflowSkipException`` and all downstream tasks skip (the
default ``trigger_rule="all_success"`` propagates the skip).  On a clean
run, ``write_done`` writes the ``.done`` file after all upstream tasks
succeed.

Clock discipline (AC #11): all time access goes through
``get_active_clock()``; ``datetime.now()`` and ``time.sleep()`` are
forbidden in this file.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

from app.core.clock import get_active_clock
from app.core.watermark import is_complete, mark_complete

# Side-effect: register the active clock for this process.
from orchestration.dags._clock_factory import make_clock  # noqa: F401

# Run the clock factory once per process.  Airflow re-parses DAG files
# periodically; without this guard, DEMO_MODE=1 would start a new
# _SpeedupClock background thread on each parse (the prior threads would
# leak, since their .stop() is never called).
_CLOCK_INITIALIZED_KEY = "_REFRESH_15MIN_CLOCK_INITIALIZED"
if not os.environ.get(_CLOCK_INITIALIZED_KEY):
    make_clock()
    os.environ[_CLOCK_INITIALIZED_KEY] = "1"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WATERMARK_ROOT = Path("data/watermarks/refresh_15min")

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _decide_bucket(**context: object) -> dict[str, object]:
    """Compute the ``(day, bucket)`` pair from the active clock.

    ``bucket = hour * 4 + minute // 15`` gives 96 quarter-hour slots per
    UTC day (0 = 00:00-00:15, 95 = 23:45-24:00).
    """
    now = get_active_clock().now()
    day_str = now.date().isoformat()
    bucket = now.hour * 4 + now.minute // 15
    result = {"day": day_str, "bucket": bucket}
    logger.info("decide_bucket: %s", result)
    return result


def _check_watermark(**context: object) -> None:
    """Skip downstream tasks if this ``(day, bucket)`` is already done."""
    ti = context["ti"]  # type: ignore[index]
    info: dict[str, object] = ti.xcom_pull(task_ids="decide_bucket")  # type: ignore[attr-defined]
    from datetime import date as _date

    day = _date.fromisoformat(str(info["day"]))
    bucket = int(str(info["bucket"]))

    if is_complete(root=_WATERMARK_ROOT, day=day, bucket=bucket):
        logger.info("Bucket %s/%02d already complete -- skipping.", day, bucket)
        raise AirflowSkipException(f"Watermark exists for {day}/{bucket:02d}")


def _ingest(**context: object) -> None:
    """Run the ingestion pipeline (raw -> bronze)."""
    from app.ingestion.ingest import Ingestion

    ingestion = Ingestion()
    ingestion.run()


def _validate(**context: object) -> None:
    """Validate the bronze parquet against the Pandera schema."""
    import pandas as pd

    from app.core.config import load_config
    from app.ingestion.schema import validate_bronze_schema, write_validation_log

    config = load_config()
    bronze_path = config["data"]["bronze_path"]
    df = pd.read_parquet(bronze_path)
    failure_reasons: list[str] = []
    try:
        validate_bronze_schema(df)
    except Exception as exc:
        failure_reasons = [str(exc)]

    write_validation_log(
        log_dir=Path("data/logs/validation"),
        df=df,
        failure_reasons=failure_reasons,
        clock=get_active_clock(),
    )

    if failure_reasons:
        raise RuntimeError(f"Bronze validation failed: {failure_reasons}")


def _feature(**context: object) -> None:
    """Run the feature engineering pipeline (silver -> gold)."""
    from app.ingestion.feature import FeatureEngineer

    fe = FeatureEngineer()
    fe.run()


def _predict(**context: object) -> None:
    """Run the inference pipeline (gold -> predictions)."""
    from app.inference.predictor import Predictor

    predictor = Predictor()
    predictor.run()


def _log_predict(**context: object) -> None:
    """Log prediction results for monitoring (PBI 22)."""
    import pandas as pd

    from app.core.config import load_config
    from app.monitoring.prediction_log import log_predictions

    config = load_config()
    inference_cfg = config.get("inference", {})
    output_path = inference_cfg.get(
        "output_path", "data/predictions/demand_predictions.parquet"
    )

    if not Path(output_path).exists():
        logger.warning("No predictions file at %s -- skipping log.", output_path)
        return

    preds_df = pd.read_parquet(output_path)
    if preds_df.empty:
        logger.warning("Empty predictions file -- skipping log.")
        return

    # Build a response dict matching the shape log_predictions expects
    predictions_list = []
    for _, row in preds_df.iterrows():
        predictions_list.append(
            {
                "bucket_time": str(row.get("hour", row.get("bucket_time", ""))),
                "pulocationid": int(row.get("pulocationid", 0)),
                "predicted_trip_count": float(row.get("predicted_trip_count", 0.0)),
            }
        )

    response = {
        "predictions": predictions_list,
        "model_version": "latest",
        "served_from_stage": "batch",
    }
    log_predictions(
        response=response,
        path="data/monitoring/predictions",
    )


def _write_done(**context: object) -> None:
    """Write the watermark ``.done`` file for this ``(day, bucket)``."""
    ti = context["ti"]  # type: ignore[index]
    info: dict[str, object] = ti.xcom_pull(task_ids="decide_bucket")  # type: ignore[attr-defined]
    from datetime import date as _date

    day = _date.fromisoformat(str(info["day"]))
    bucket = int(str(info["bucket"]))

    mark_complete(root=_WATERMARK_ROOT, day=day, bucket=bucket)
    logger.info("Watermark written: %s/%02d", day, bucket)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="refresh_15min_dag",
    schedule="*/15 * * * *",
    catchup=False,
    max_active_runs=1,
    start_date=datetime(2024, 6, 3, tzinfo=timezone.utc),
    default_args={"retries": 1},
    tags=["pbi-25b", "refresh"],
) as dag:
    decide_bucket = PythonOperator(
        task_id="decide_bucket",
        python_callable=_decide_bucket,
    )
    check_watermark = PythonOperator(
        task_id="check_watermark",
        python_callable=_check_watermark,
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
    write_done = PythonOperator(
        task_id="write_done",
        python_callable=_write_done,
        trigger_rule="all_success",
    )

    (
        decide_bucket
        >> check_watermark
        >> ingest
        >> validate
        >> feature
        >> predict
        >> log_predict
        >> write_done
    )
