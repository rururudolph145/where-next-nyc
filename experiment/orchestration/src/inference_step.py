"""Single-bucket inference step.

Loads the trained model and the Gold layer, predicts demand for a single
15-minute bucket (the current simulated timestamp), and refreshes the
served predictions parquet so the API can pick up the new bucket.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import joblib  # type: ignore[import-untyped]
import pandas as pd

from app.core.config import load_config

logger = logging.getLogger(__name__)

# Cache the model and Gold frame across DAG invocations within the same
# worker process. Airflow LocalExecutor reuses the worker, so this gives
# us a "load once" effect without any IPC.
_MODEL: Any = None
_GOLD_DF: pd.DataFrame | None = None


@dataclass(frozen=True)
class StepResult:
    """Outcome of one inference step."""

    bucket: pd.Timestamp
    n_predictions: int
    output_path: str


def _model(config: dict[str, Any]) -> Any:
    global _MODEL
    if _MODEL is None:
        model_path = os.path.join(config["model"]["store_path"], "model.pkl")
        logger.info("Loading model from %s", model_path)
        _MODEL = joblib.load(model_path)
    return _MODEL


def _gold(config: dict[str, Any]) -> pd.DataFrame:
    global _GOLD_DF
    if _GOLD_DF is None:
        gold_file = os.path.join(
            config["data"]["gold_path"], "hourly_demand_featured.parquet"
        )
        logger.info("Loading Gold layer from %s", gold_file)
        _GOLD_DF = pd.read_parquet(gold_file)
    return _GOLD_DF


def predict_bucket(
    bucket: pd.Timestamp,
    config: dict[str, Any] | None = None,
) -> StepResult:
    """Predict demand for every zone in a single 15-min bucket.

    The predicted rows are appended to the served predictions parquet at
    ``config["serving"]["predictions_path"]`` (de-duplicated on
    ``(pulocationid, hour)`` so re-runs are idempotent).
    """
    cfg = config or load_config()
    model = _model(cfg)
    gold_df = _gold(cfg)

    features: list[str] = cfg["train"]["features"]
    target: str = cfg["train"]["target"]

    slice_df = gold_df.loc[gold_df["hour"] == bucket]
    if slice_df.empty:
        raise ValueError(
            f"No Gold rows for bucket {bucket}. The simulation window may "
            f"have run past the end of the available data."
        )

    preds = model.predict(slice_df[features])
    new_rows = pd.DataFrame(
        {
            "pulocationid": slice_df["pulocationid"].to_numpy(),
            "hour": slice_df["hour"].to_numpy(),
            "predicted_trip_count": preds,
            "actual_trip_count": slice_df[target].to_numpy(),
        }
    )

    output_path = cfg["serving"]["predictions_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        existing = pd.read_parquet(output_path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["pulocationid", "hour"], keep="last"
        )
    else:
        combined = new_rows

    combined = combined.sort_values(["hour", "pulocationid"])
    combined.to_parquet(output_path, index=False)

    logger.info(
        "Bucket %s → %d predictions written to %s (total rows: %d)",
        bucket,
        len(new_rows),
        output_path,
        len(combined),
    )

    return StepResult(
        bucket=bucket, n_predictions=len(new_rows), output_path=output_path
    )
