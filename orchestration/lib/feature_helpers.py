"""feature_helpers.py — silver+gold feature engineering helpers for 15_minute_refresh_inference_dag.

Airflow-free by design (the host test env has no Airflow). The DAG file
(``exp03b_feature_dag.py``) is a thin wrapper that pulls ``bronze_path`` from
XCom, builds the override config, and calls ``run_feature``.

Two seams keep this unit-testable:

  * :func:`build_feature_config` redirects silver/gold/shapefile/model paths
    to the experiment root. The base is never mutated (frozen model).
  * :func:`run_feature` cleans bronze→silver using pre-fitted IQR bounds
    (transform-only — no re-split, no re-fit), then runs FeatureEngineer
    (silver→gold), returning the bronze path unchanged for a stable XCom flow.
"""

from __future__ import annotations

import json
import os

from app.core.config import AppConfig
from app.ingestion.clean import DataCleaner
from app.ingestion.feature import FeatureEngineer


def build_feature_config(
    base: AppConfig,
    *,
    silver_root: str,
    gold_root: str,
    shapefile_path: str,
    model_root: str,
) -> AppConfig:
    """Return a new AppConfig redirected to the experiment roots.

    The base is never mutated (frozen Pydantic model).
    """
    return base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"silver_path": silver_root, "gold_path": gold_root}
            ),
            "feature": base.feature.model_copy(
                update={"zone_shapefile_path": shapefile_path}
            ),
            "model": base.model.model_copy(update={"store_path": model_root}),
        }
    )


def run_feature_train(
    bronze_path: str,
    *,
    config: AppConfig,
) -> str:
    """Fit IQR bounds from bronze, save them, then delegate to run_feature.

    Training-only path: DataCleaner.clean() is called without pre-fitted bounds
    so bounds are inferred from the training window. The fitted bounds are
    persisted to model.store_path/iqr_bounds.json so subsequent inference runs
    can load them via run_feature (transform-only).
    """
    cleaner = DataCleaner(config)
    raw = cleaner.load_bronze()
    cleaned, bounds = cleaner.clean(raw)

    store_path: str = config.model.store_path
    os.makedirs(store_path, exist_ok=True)
    bounds_path = os.path.join(store_path, "iqr_bounds.json")
    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=2)

    cleaner.save_silver(cleaned)
    FeatureEngineer(config=config).run(for_training=True)
    return bronze_path


def run_feature(
    bronze_path: str,
    *,
    config: AppConfig,
    for_training: bool = False,
) -> str:
    """Clean bronze→silver (transform-only), then engineer silver→gold.

    Uses pre-fitted IQR bounds from model_root/iqr_bounds.json so the
    cleaning step never re-splits or re-fits on the cumulative bronze.
    Returns bronze_path unchanged for a stable XCom flow.

    Args:
        for_training: When True, FeatureEngineer computes the target column
            (target_sum_t1_t2). Use False for inference gold (no label needed).
    """
    cleaner = DataCleaner(config)
    raw = cleaner.load_bronze()

    iqr_bounds_path = os.path.join(config.model.store_path, "iqr_bounds.json")
    with open(iqr_bounds_path) as f:
        iqr_bounds = json.load(f)

    cleaned, _ = cleaner.clean(raw, iqr_bounds=iqr_bounds)
    cleaner.save_silver(cleaned)

    FeatureEngineer(config=config).run(for_training=for_training)
    return bronze_path
