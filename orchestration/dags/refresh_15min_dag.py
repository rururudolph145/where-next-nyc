"""refresh_15min_dag -- watermark-advancing incremental refresh cycle.

Topology (linear chain, 11 tasks):
    decide_bucket -> read_watermark -> check_no_new_data -> check_watermark
        -> ingest -> validate -> feature -> predict -> log_predict
        -> advance_watermark -> write_done

Incremental model (PBI 25b FIX)
-------------------------------
The "simulation position" is a DATA watermark, not a clock. The fixed 2021
raw data is walked one ``slice_duration`` slice at a time:

  * ``read_watermark`` loads the current position W (or ``ingest.seed_start``
    on the first run) and pushes the window ``[W, W + slice_duration)``.
  * ``ingest`` filters the raw data to that window. The LOWER bound of the
    cumulative bronze is always the seed; the UPPER bound advances each run,
    so bronze is a growing superset (single-file full rewrite, ADR 001).
  * ``advance_watermark`` writes ``window_end`` only after every upstream
    task succeeds (``trigger_rule="all_success"``). A mid-run failure leaves
    the watermark unchanged, so the next run reprocesses the same slice.

Timezone: the watermark / seed are tz-NAIVE datetimes (the raw
``tpep_pickup_datetime`` is tz-naive NYC wall-clock); the window filter is a
naive-vs-naive comparison.

Idempotency: ``decide_bucket`` / ``check_watermark`` / ``write_done`` keep the
per-``(date, bucket)`` ``.done`` sentinel so re-triggering the same wall-clock
bucket is a no-op. The validation-log timestamp uses a real wall clock
(``datetime.now(tz=timezone.utc)``) — it is an audit field, not simulation
state.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from airflow import DAG  # type: ignore[import-not-found]
from airflow.exceptions import AirflowSkipException  # type: ignore[import-not-found]
from airflow.operators.python import PythonOperator  # type: ignore[import-not-found]

from app.core.watermark import (
    is_complete,
    mark_complete,
    read_watermark,
    write_watermark,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WATERMARK_ROOT = Path("data/watermarks/refresh_15min")

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _decide_bucket(**context: object) -> dict[str, object]:
    """Compute the ``(day, bucket)`` pair from the real wall clock.

    ``bucket = hour * 4 + minute // 15`` gives 96 quarter-hour slots per
    UTC day (0 = 00:00-00:15, 95 = 23:45-24:00). This drives the
    per-(date, bucket) ``.done`` idempotency sentinel only — the simulation
    position is the DATA watermark, computed in ``_read_watermark``.
    """
    now = datetime.now(tz=timezone.utc)
    day_str = now.date().isoformat()
    bucket = now.hour * 4 + now.minute // 15
    result = {"day": day_str, "bucket": bucket}
    logger.info("decide_bucket: %s", result)
    return result


def _read_watermark(**context: object) -> dict[str, str]:
    """Resolve the current ingest window ``[W, W + slice_duration)``.

    W is the stored watermark, or ``ingest.seed_start`` on the first run.
    Both bounds are NAIVE ISO strings pushed via XCom for downstream tasks.
    """
    from app.core.config import load_config

    config = load_config()
    slice_duration = config["ingest"]["slice_duration"]

    watermark = read_watermark(_WATERMARK_ROOT)
    if watermark is None:
        watermark = datetime.fromisoformat(config["ingest"]["seed_start"])

    window_end = watermark + pd.Timedelta(slice_duration)
    window = {
        "window_start": watermark.isoformat(),
        "window_end": window_end.isoformat(),
    }
    logger.info("read_watermark: %s", window)
    return window


def _dataset_max(config: dict[str, object]) -> datetime:
    """Return the inclusive-end NAIVE timestamp of the fixed dataset.

    Hardcoded per PBI 25b FIX brief: the fixed 2021 set's last pickup is
    ``2021-03-31 23:59:57``. Deriving this from ``end_year_month`` produced
    a 2-second gap (``23:59:59``) that caused two extra DAG runs past the
    real data end before stopping. ``config`` is retained for signature
    compatibility with callers and future flexibility.
    """
    del config  # signature kept for compatibility; value is hardcoded
    return datetime(2021, 3, 31, 23, 59, 57)


def _check_no_new_data(**context: object) -> None:
    """Skip the run when the watermark has reached the dataset boundary.

    Boundary semantics: stop when the NEXT slice's start (``window_start``)
    is at or after the dataset's last timestamp — there is no further data
    to ingest, so the whole run is skipped.
    """
    from app.core.config import load_config

    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_start = datetime.fromisoformat(window["window_start"])

    dataset_max = _dataset_max(load_config())
    if window_start >= dataset_max:
        logger.info(
            "Watermark %s reached dataset boundary %s -- skipping.",
            window_start,
            dataset_max,
        )
        raise AirflowSkipException("no new data / dataset boundary reached")


def _check_watermark(**context: object) -> None:
    """Skip downstream tasks if this ``(day, bucket)`` is already done."""
    ti = context["ti"]  # type: ignore[index]
    info: dict[str, object] = ti.xcom_pull(task_ids="decide_bucket")  # type: ignore[attr-defined]

    day = date.fromisoformat(str(info["day"]))
    bucket = int(str(info["bucket"]))

    if is_complete(root=_WATERMARK_ROOT, day=day, bucket=bucket):
        logger.info("Bucket %s/%02d already complete -- skipping.", day, bucket)
        raise AirflowSkipException(f"Watermark exists for {day}/{bucket:02d}")


def _ingest(**context: object) -> None:
    """Ingest into bronze with a CUMULATIVE window.

    The lower bound is FROZEN at ``ingest.seed_start`` (read from config, not
    from XCom), so bronze grows as a superset on every run -- single-file
    full rewrite per ADR-001. Only the upper bound (``window_end`` from the
    advancing watermark) moves. Reading the XCom's ``window_start`` would
    produce a sliding 15-min slice and erase prior data on every refresh.
    """
    from app.core.config import load_config
    from app.ingestion.ingest import Ingestion

    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_end = datetime.fromisoformat(window["window_end"])

    config = load_config()
    seed_start = datetime.fromisoformat(config["ingest"]["seed_start"])

    Ingestion().run(window_start=seed_start, window_end=window_end)


def _validate(**context: object) -> None:
    """Validate the bronze parquet against the Pandera schema."""
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


def _advance_watermark(**context: object) -> None:
    """Advance the data watermark to ``window_end``.

    Runs only when every upstream task succeeded (default
    ``trigger_rule="all_success"``), so a failure leaves the watermark
    unchanged and the next run reprocesses the same slice.
    """
    ti = context["ti"]  # type: ignore[index]
    window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
    window_end = datetime.fromisoformat(window["window_end"])

    write_watermark(_WATERMARK_ROOT, window_end)
    logger.info("Watermark advanced to %s", window_end)


def _write_done(**context: object) -> None:
    """Write the watermark ``.done`` file for this ``(day, bucket)``."""
    ti = context["ti"]  # type: ignore[index]
    info: dict[str, object] = ti.xcom_pull(task_ids="decide_bucket")  # type: ignore[attr-defined]

    day = date.fromisoformat(str(info["day"]))
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
    read_watermark_task = PythonOperator(
        task_id="read_watermark",
        python_callable=_read_watermark,
    )
    check_no_new_data = PythonOperator(
        task_id="check_no_new_data",
        python_callable=_check_no_new_data,
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
    advance_watermark = PythonOperator(
        task_id="advance_watermark",
        python_callable=_advance_watermark,
        trigger_rule="all_success",
    )
    write_done = PythonOperator(
        task_id="write_done",
        python_callable=_write_done,
        trigger_rule="all_success",
    )

    (
        decide_bucket
        >> read_watermark_task
        >> check_no_new_data
        >> check_watermark
        >> ingest
        >> validate
        >> feature
        >> predict
        >> log_predict
        >> advance_watermark
        >> write_done
    )
