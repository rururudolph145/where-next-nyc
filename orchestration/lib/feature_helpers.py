"""feature_helpers.py — silver+gold feature engineering helpers for 15_minute_refresh_inference_dag.

Airflow-free by design (the host test env has no Airflow). The DAG file
(``exp03b_feature_dag.py``) is a thin wrapper that pulls ``bronze_path`` from
XCom, builds the override config, and calls ``run_feature``.

Two seams keep this unit-testable:

  * :func:`build_feature_config` redirects silver/gold/shapefile/model paths
    to the experiment root. The base is deep-copied, never mutated.
  * :func:`run_feature` cleans bronze→silver using pre-fitted IQR bounds
    (transform-only — no re-split, no re-fit), then runs FeatureEngineer
    (silver→gold), returning the bronze path unchanged for a stable XCom flow.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

from app.ingestion.clean import DataCleaner
from app.ingestion.feature import FeatureEngineer


def build_feature_config(
    base: dict[str, Any],
    *,
    silver_root: str,
    gold_root: str,
    shapefile_path: str,
    model_root: str,
) -> dict[str, Any]:
    """Return a copy of *base* redirected to the experiment roots.

    The base is deep-copied, never mutated.
    """
    config = copy.deepcopy(base)
    config["data"]["silver_path"] = silver_root
    config["data"]["gold_path"] = gold_root
    config["feature"]["zone_shapefile_path"] = shapefile_path
    config["model"]["store_path"] = model_root
    return config


def run_feature(bronze_path: str, *, config: dict[str, Any]) -> str:
    """Clean bronze→silver (transform-only), then engineer silver→gold.

    Uses pre-fitted IQR bounds from model_root/iqr_bounds.json so the
    cleaning step never re-splits or re-fits on the cumulative bronze.
    Returns bronze_path unchanged for a stable XCom flow.
    """
    cleaner = DataCleaner(config)
    raw = cleaner.load_bronze()

    iqr_bounds_path = os.path.join(config["model"]["store_path"], "iqr_bounds.json")
    with open(iqr_bounds_path) as f:
        iqr_bounds = json.load(f)

    cleaned, _ = cleaner.clean(raw, iqr_bounds=iqr_bounds)
    cleaner.save_silver(cleaned)

    FeatureEngineer(config=config).run(for_training=False)
    return bronze_path
