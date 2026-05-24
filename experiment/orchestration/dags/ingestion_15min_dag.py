"""Airflow DAG: simulated 15-minute ingestion cadence.

Each scheduled run advances the simulated BucketClock by one 15-min
bucket and reveals that bucket from the pre-built gold parquet. The DAG
preserves the bronze→silver→gold visual structure in the Airflow UI;
each task body is thin and idempotent.

Downstream DAGs (inference, monitoring) consume the marker file
``state/last_gold_bucket.json`` rather than the BucketClock directly.
This makes ingestion the single source of "what's ready now".
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta
from typing import Any

import pandas as pd
import pendulum
from airflow.decorators import dag, task

SRC_DIR = "/opt/airflow/src"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from bucket_clock import advance, initialise  # noqa: E402
from ingestion_step import (  # noqa: E402
    confirm_bucket_in_gold,
    publish_gold_marker,
)

from app.core.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# Optional: when True, the DAG occasionally fires off a full-rebuild as a
# smoke test of the real Ingestion/DataCleaner/FeatureEngineer classes.
# Off by default — would be wasteful per bucket.
STRICT_MODE = os.environ.get("INGESTION_STRICT_MODE", "false").lower() == "true"


@dag(
    dag_id="ingestion_15min_simulation",
    description=(
        "Reveals one new gold bucket per simulated 15-min tick. "
        "Drives the inference DAG via state/last_gold_bucket.json."
    ),
    schedule="* * * * *",  # every real minute = one simulated 15-min bucket
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "experiment",
        "retries": 0,
        "execution_timeout": timedelta(minutes=10),
    },
    tags=["experiment", "ingestion", "15min"],
)
def ingestion_15min_simulation() -> None:
    @task
    def tick() -> str:
        """Advance the bucket clock by one 15-min step; return ISO string."""
        cfg = load_config()
        start = pd.Timestamp(cfg["inference"]["simulation"]["start_hour"])
        initialise(start)
        new_state = advance()
        logger.info("Clock advanced → %s", new_state.current)
        return new_state.current.isoformat()

    @task
    def ingest_bucket(bucket_iso: str) -> str:
        """Simulated bronze-layer ingestion for one bucket.

        In a real streaming system this would read newly-arrived raw
        trip records, append to bronze, and return the row count. Here
        we just announce we'd be ready to do so.
        """
        logger.info("[bronze] Would ingest raw trips for bucket %s", bucket_iso)
        return bucket_iso

    @task
    def clean_bucket(bucket_iso: str) -> str:
        """Simulated silver-layer cleaning for one bucket."""
        logger.info("[silver] Would clean bronze→silver for bucket %s", bucket_iso)
        return bucket_iso

    @task
    def featurize_bucket(bucket_iso: str) -> dict[str, Any]:
        """Simulated gold-layer feature build; confirm bucket exists in pre-built gold."""
        bucket = pd.Timestamp(bucket_iso)
        n_rows = confirm_bucket_in_gold(bucket)
        logger.info(
            "[gold] Bucket %s confirmed in pre-built gold (%d zone-rows)",
            bucket,
            n_rows,
        )
        return {"bucket_iso": bucket_iso, "n_rows": n_rows}

    @task
    def publish_marker(featurized: dict[str, Any]) -> str:
        """Publish the gold-ready marker file consumed by downstream DAGs."""
        bucket = pd.Timestamp(featurized["bucket_iso"])
        result = publish_gold_marker(bucket=bucket, n_rows=featurized["n_rows"])
        return result.marker_path

    bucket_iso = tick()
    bronze = ingest_bucket(bucket_iso)
    silver = clean_bucket(bronze)
    gold = featurize_bucket(silver)
    publish_marker(gold)


ingestion_15min_simulation()
