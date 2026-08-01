"""predict_helpers.py — inference helpers for 15_minute_refresh_inference_dag.

Airflow-free by design (the host test env has no Airflow). The DAG file
is a thin wrapper that builds the override config and calls ``run_predict``.

Two seams keep this unit-testable:

  * :func:`build_predict_config` redirects gold, model, and predictions
    output paths to the experiment root. The base config is deep-copied, never
    mutated.
  * :func:`run_predict` constructs ``Predictor(config=config)``, calls
    ``.run()``, and returns the ``output_path`` string.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

import pandas as pd

from app.inference.predictor import Predictor

logger = logging.getLogger(__name__)


def build_predict_config(
    base: dict[str, Any],
    *,
    gold_root: str,
    model_root: str,
    predictions_path: str,
) -> dict[str, Any]:
    """Return a copy of *base* redirected to the experiment roots.

    Overrides three paths. The base is deep-copied, never mutated.
    """
    config = copy.deepcopy(base)
    config["data"]["gold_path"] = gold_root
    config["model"]["store_path"] = model_root
    config["inference"]["output_path"] = predictions_path
    return config


def run_predict(*, config: dict[str, Any], since: Any | None = None) -> str | None:
    """Run Predictor with the override config and return the output path.

    Args:
        config: Inference configuration dict (built by build_predict_config).
        since: Optional lower-bound timestamp passed to Predictor.run(since=).
            When supplied, only rows with ``hour >= since`` are scored.

    Returns None when gold has 0 rows — the lag window exceeds the available
    history on early pipeline runs, so FeatureEngineer drops every row.
    The DAG logs the skip and lets write_watermark advance normally.
    """
    gold_file = os.path.join(
        config["data"]["gold_path"], "hourly_demand_featured.parquet"
    )
    gold = pd.read_parquet(gold_file)
    if gold.empty:
        logger.info(
            "run_predict: gold frame is empty — skipping inference for this window"
        )
        return None
    response = Predictor(config=config).run(since=since)
    return response["output_path"]
