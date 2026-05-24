"""Airflow DAG: inference at a simulated 15-min cadence.

This DAG is a **consumer** of the ingestion DAG. It waits for a new gold
bucket marker (``state/last_gold_bucket.json``) to appear, predicts that
bucket, refreshes ``data/predictions/demand_predictions.parquet``, calls
``GET /heatmap`` as a smoke test, and appends one row to the run log.

The BucketClock is owned exclusively by ``ingestion_15min_simulation``.
This DAG never advances time; it only consumes whatever ingestion has
published.

If the same bucket is published twice (e.g., ingestion ran but the
inference scheduler hadn't yet ticked), the predict step is idempotent
(de-dups on ``(pulocationid, hour)``) and we additionally short-circuit
via ``state/last_predicted_bucket.json``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pendulum
import requests
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

SRC_DIR = "/opt/airflow/src"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference_step import predict_bucket  # noqa: E402
from ingestion_step import read_gold_marker  # noqa: E402
from run_log import RunRecord, append, now_iso  # noqa: E402

logger = logging.getLogger(__name__)

SLEEP_SECONDS = int(os.environ.get("INFERENCE_REAL_SLEEP_SECONDS", "5"))
API_BASE_URL = os.environ.get(
    "INFERENCE_API_BASE_URL", "http://host.docker.internal:8000"
)
STATE_DIR = os.environ.get("INFERENCE_STATE_DIR", "/opt/airflow/state")
LAST_PREDICTED_FILE = Path(STATE_DIR) / "last_predicted_bucket.json"


def _last_predicted() -> str | None:
    if not LAST_PREDICTED_FILE.exists():
        return None
    return json.loads(LAST_PREDICTED_FILE.read_text()).get("bucket")


def _record_predicted(bucket_iso: str) -> None:
    LAST_PREDICTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_PREDICTED_FILE.write_text(
        json.dumps(
            {"bucket": bucket_iso, "predicted_at": now_iso()},
            indent=2,
        )
    )


@dag(
    dag_id="inference_15min_simulation",
    description=(
        "Consumes gold buckets published by ingestion_15min_simulation. "
        "Predicts each new bucket once, refreshes /heatmap data, smoke-tests the API."
    ),
    schedule="* * * * *",  # every real minute; will skip if no new bucket
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "experiment",
        "retries": 0,
        "execution_timeout": timedelta(minutes=10),
    },
    tags=["experiment", "inference", "15min"],
)
def inference_15min_simulation() -> None:
    @task
    def wait_for_gold() -> str:
        """Read the gold marker; skip the run if it hasn't advanced.

        Returns the bucket ISO string to process.
        """
        marker = read_gold_marker()
        if marker is None:
            raise AirflowSkipException(
                "No gold marker yet. Is the ingestion DAG running?"
            )

        last = _last_predicted()
        if last == marker["bucket"]:
            raise AirflowSkipException(
                f"Bucket {marker['bucket']} already predicted; waiting for ingestion to advance."
            )

        logger.info(
            "Gold bucket ready: %s (rows=%s); last predicted: %s",
            marker["bucket"],
            marker.get("n_rows"),
            last,
        )
        return marker["bucket"]

    @task
    def predict(bucket_iso: str) -> dict[str, Any]:
        """Predict demand for the bucket published by ingestion."""
        bucket = pd.Timestamp(bucket_iso)
        started = time.perf_counter()
        result = predict_bucket(bucket)
        duration = time.perf_counter() - started

        _record_predicted(bucket_iso)

        if SLEEP_SECONDS > 0:
            logger.info("Sleeping %d s to slow the visible cadence", SLEEP_SECONDS)
            time.sleep(SLEEP_SECONDS)

        return {
            "bucket": result.bucket.isoformat(),
            "n_predictions": result.n_predictions,
            "duration_s": duration,
            "output_path": result.output_path,
        }

    @task
    def smoke_test_api(prediction: dict[str, Any]) -> dict[str, Any]:
        """Call /heatmap?hour=<bucket> to confirm the API serves the new bucket.

        Failure is logged but does NOT fail the DAG — we want the
        simulation to continue even if uvicorn isn't running.
        """
        url = f"{API_BASE_URL}/heatmap"
        try:
            resp = requests.get(
                url, params={"hour": prediction["bucket"]}, timeout=10
            )
            body = (
                resp.json()
                if resp.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            return {
                "status_code": resp.status_code,
                "zone_count": body.get("zone_count"),
                "ok": resp.status_code == 200,
            }
        except requests.RequestException as exc:
            logger.warning("API smoke test failed: %s", exc)
            return {"status_code": None, "zone_count": None, "ok": False}

    @task
    def log_run(
        prediction: dict[str, Any], api_result: dict[str, Any]
    ) -> str:
        record = RunRecord(
            run_started_at=now_iso(),
            simulated_bucket=prediction["bucket"],
            n_predictions=prediction["n_predictions"],
            predict_duration_s=round(prediction["duration_s"], 3),
            api_status_code=api_result["status_code"],
            api_zone_count=api_result["zone_count"],
            ok=api_result["ok"],
        )
        path = append(record)
        logger.info("Run logged → %s", path)
        return str(path)

    bucket_iso = wait_for_gold()
    prediction = predict(bucket_iso)
    api_result = smoke_test_api(prediction)
    log_run(prediction, api_result)


inference_15min_simulation()
