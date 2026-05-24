"""Per-bucket ingestion simulation.

The real pipeline (``app/ingestion/{ingest,clean,feature}.py``) operates
on the full dataset, not bucket-by-bucket. For the 15-min simulation we
do not want to re-run that whole pipeline every minute — that would be
both wasteful and inconsistent with the experiment's intent ("pretend
data arrived 15 minutes at a time").

Instead this module *reveals* one bucket at a time from a gold parquet
that was already built end-to-end. Each ingestion step:

1. Reads the simulated bucket clock.
2. Confirms that bucket exists in the gold parquet.
3. Writes a marker file ``state/last_gold_bucket.json`` so downstream
   DAGs (inference, monitoring) know what is "ready" to consume.

The four tasks in the ingestion DAG (``ingest_bucket``,
``clean_bucket``, ``featurize_bucket``, ``publish_gold_marker``) each
call into this module and log progress. They preserve the
bronze→silver→gold visual structure in the Airflow UI without
re-executing the heavy full-rebuild classes.

A ``strict`` mode is available for smoke-testing the real pipeline
classes end-to-end; see :func:`run_strict_pipeline`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import load_config

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = os.environ.get("INFERENCE_STATE_DIR", "/opt/airflow/state")
MARKER_FILENAME = "last_gold_bucket.json"

# Cache the gold DataFrame across calls in the same worker process.
_GOLD_DF: pd.DataFrame | None = None


@dataclass(frozen=True)
class BucketRevealResult:
    """Result of revealing one bucket from the pre-built gold layer."""

    bucket: pd.Timestamp
    n_rows: int
    marker_path: str


def _gold(config: dict[str, Any]) -> pd.DataFrame:
    global _GOLD_DF
    if _GOLD_DF is None:
        gold_file = os.path.join(
            config["data"]["gold_path"], "hourly_demand_featured.parquet"
        )
        logger.info("Loading Gold layer (one-time per worker) from %s", gold_file)
        _GOLD_DF = pd.read_parquet(gold_file)
    return _GOLD_DF


def confirm_bucket_in_gold(
    bucket: pd.Timestamp, config: dict[str, Any] | None = None
) -> int:
    """Confirm the bucket exists in the pre-built gold parquet; return row count."""
    cfg = config or load_config()
    gold_df = _gold(cfg)
    mask = gold_df["hour"] == bucket
    n_rows = int(mask.sum())
    if n_rows == 0:
        raise ValueError(
            f"Bucket {bucket} not present in gold parquet. The simulation "
            f"window may have run past the available data."
        )
    return n_rows


def publish_gold_marker(
    bucket: pd.Timestamp,
    n_rows: int,
    state_dir: str = DEFAULT_STATE_DIR,
) -> BucketRevealResult:
    """Write the marker file that announces a new gold bucket is ready."""
    path = Path(state_dir) / MARKER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "bucket": bucket.isoformat(),
                "n_rows": n_rows,
                "published_at": pd.Timestamp.utcnow().isoformat(),
            },
            indent=2,
        )
    )
    logger.info("Marker published → %s (bucket=%s, rows=%d)", path, bucket, n_rows)
    return BucketRevealResult(bucket=bucket, n_rows=n_rows, marker_path=str(path))


def read_gold_marker(state_dir: str = DEFAULT_STATE_DIR) -> dict[str, Any] | None:
    """Return the last published marker or ``None`` if no marker exists yet."""
    path = Path(state_dir) / MARKER_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------
# Strict mode: actually re-run the full pipeline classes.
# Only useful as an occasional smoke test; do NOT schedule per-bucket.
# ---------------------------------------------------------------------
def run_strict_pipeline(config: dict[str, Any] | None = None) -> None:
    """Re-run Ingestion → DataCleaner → FeatureEngineer on the full dataset.

    Expensive. Provided so the Airflow stack can occasionally validate
    that the real pipeline classes still execute end-to-end. Not part of
    the per-bucket loop.
    """
    from app.ingestion.clean import DataCleaner
    from app.ingestion.feature import FeatureEngineer
    from app.ingestion.ingest import Ingestion

    cfg = config or load_config()
    logger.info("STRICT: ingest bronze")
    Ingestion(cfg).run()
    logger.info("STRICT: clean silver")
    DataCleaner(cfg).run()
    logger.info("STRICT: featurize gold")
    FeatureEngineer(cfg).run()
    logger.info("STRICT: pipeline rebuild complete")
