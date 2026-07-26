"""Pure feature helpers for the EXP-03b silver→gold DAG.

Airflow-free by design (the host test env has no Airflow). The DAG file
(``exp03b_feature_dag.py``) is a thin wrapper that pulls ``bronze_path`` from
XCom, builds the override config, and calls ``run_feature``.

Two seams keep this unit-testable:

  * :func:`build_feature_config` redirects silver/gold/shapefile/model paths
    to the experiment root and pins the train/inference cutoff to
    ``window_end`` (OQ4 — the pre-cutoff experiment window would otherwise
    leave DataCleaner's inference split empty).
  * :func:`run_feature` runs ``DataCleaner(config).run()`` (bronze→silver)
    then ``FeatureEngineer(config).run()`` (silver→gold), returning the
    bronze path unchanged for a stable XCom flow.
"""

from __future__ import annotations

import copy
from datetime import datetime
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
    window_end: datetime,
) -> dict[str, Any]:
    """Return a copy of *base* redirected to the experiment roots.

    Overrides four output paths plus the train/inference cutoff (OQ4). The
    base is deep-copied, never mutated.
    """
    config = copy.deepcopy(base)
    config["data"]["silver_path"] = silver_root
    config["data"]["gold_path"] = gold_root
    config["feature"]["zone_shapefile_path"] = shapefile_path
    config["model"]["store_path"] = model_root
    config["inference"]["simulation"]["start_hour"] = window_end.isoformat()
    return config


def run_feature(bronze_path: str, *, config: dict[str, Any]) -> str:
    """Clean bronze→silver, then engineer silver→gold. Return bronze_path.

    DataCleaner takes config positionally; FeatureEngineer takes it as a
    keyword. The cleaner MUST run first — the engineer reads the silver it
    writes. bronze_path is returned unchanged so the DAG's XCom flow
    (ingest → validate → feature → advance) stays predictable.
    """
    DataCleaner(config).run()
    FeatureEngineer(config=config).run()
    return bronze_path
