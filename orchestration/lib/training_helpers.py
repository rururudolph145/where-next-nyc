"""training_helpers.py — training helpers for train_monthly_dag.

Airflow-free by design (the host test env has no Airflow). The DAG file is a
thin wrapper that delegates all logic here.

Two seams keep this unit-testable:

  * :func:`run_train` accepts injectable ``trainer_factory`` and
    ``client_factory`` callables so tests can substitute fakes without a live
    MLflow or real parquet data.
  * :func:`check_window_ready` imports ``AirflowSkipException`` LAZILY to
    keep this module host-importable when Airflow is not installed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.core.watermark import read_watermark

logger = logging.getLogger(__name__)

# ADR-003 §Decision.6 — fixed 30-day training window
_WINDOW_DAYS = 30


class WatermarkNotFoundError(Exception):
    pass


class WatermarkInvariantError(Exception):
    pass


class EmptyTrainingWindowError(Exception):
    pass


def resolve_watermarks(
    config: dict[str, Any],
    refresh_root: Path,
    last_training_root: Path,
) -> dict[str, Any]:
    """Return resolved watermark positions and a bootstrap flag.

    Raises WatermarkNotFoundError when the refresh watermark is absent —
    training before inference has run is invalid.
    """
    refresh_wm = read_watermark(refresh_root)
    if refresh_wm is None:
        raise WatermarkNotFoundError(
            f"Refresh watermark absent at {refresh_root}; "
            "inference must run before training."
        )

    last_wm_raw = read_watermark(last_training_root)
    is_bootstrap = last_wm_raw is None
    last_wm: datetime = (
        last_wm_raw
        if last_wm_raw is not None
        else datetime.fromisoformat(config["ingest"]["seed_start"])
    )

    return {
        "refresh_wm": refresh_wm.isoformat(),
        "last_training_wm": last_wm.isoformat(),
        "is_bootstrap": is_bootstrap,
    }


def check_window_ready(
    refresh_wm: datetime,
    last_training_wm: datetime,
    window_days: int = _WINDOW_DAYS,
) -> None:
    """Raise WatermarkInvariantError or AirflowSkipException as appropriate.

    Invariant check runs FIRST (host-testable without Airflow). The skip
    branch imports AirflowSkipException lazily so the module stays
    host-importable.
    """
    if refresh_wm < last_training_wm:
        raise WatermarkInvariantError(
            f"refresh_wm {refresh_wm} < last_training_wm {last_training_wm}; "
            "impossible under normal operation."
        )

    delta = refresh_wm - last_training_wm
    if delta < timedelta(days=window_days):
        from airflow.exceptions import AirflowSkipException  # type: ignore[import-not-found]

        raise AirflowSkipException(
            f"Inference only {delta.days}d ahead of last training "
            f"(need >= {window_days}d); skipping."
        )


def compute_window(
    seed_start: datetime,
    last_training_wm: datetime,
    window_days: int = _WINDOW_DAYS,  # ADR-003 §Decision.6
) -> tuple[datetime, datetime]:
    """Return (window_start, window_end) for the training read.

    window_start is always seed_start (full history).
    window_end advances last_training_wm by window_days.
    """
    return seed_start, last_training_wm + timedelta(days=window_days)


def read_gold_windowed(
    gold_file: Path,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Read parquet and filter `hour` to [start, end) (upper-exclusive).

    Raises EmptyTrainingWindowError when the resulting slice is empty.
    """
    df = pd.read_parquet(gold_file)
    mask = (df["hour"] >= start) & (df["hour"] < end)
    filtered = df[mask].copy()
    if filtered.empty:
        raise EmptyTrainingWindowError(
            f"No gold rows in window [{start}, {end}); cannot train."
        )
    return filtered


def tag_promoted_version(
    client: Any,
    model_name: str,
    version: str,
    window_end: datetime,
) -> None:
    """Write the trained_at_watermark tag on the promoted model version."""
    client.set_model_version_tag(
        model_name, version, "trained_at_watermark", window_end.isoformat()
    )


def run_train(
    *,
    gold_file: Path,
    window_start: datetime,
    window_end: datetime,
    model_name: str,
    trainer_factory: Callable[[], Any],
    client_factory: Callable[[], Any],
) -> dict[str, Any]:
    """Read gold, train, derive gate result, tag on promote, return XCom dict.

    Fault model:
      * crash (Trainer.run raises) → re-raise; watermark held (crash→hold)
      * reject (model_path None) → return gate_result=rejected, no tag
      * promote (model_path set) → tag version, return gate_result=promoted
    """
    filtered_df = read_gold_windowed(gold_file, window_start, window_end)

    trainer = trainer_factory()
    result = trainer.run(data=filtered_df)

    gate_result = "promoted" if result["model_path"] else "rejected"

    model_version = None
    if gate_result == "promoted":
        client = client_factory()
        versions = client.get_latest_versions(model_name, stages=["Production"])
        model_version = versions[0].version
        tag_promoted_version(client, model_name, model_version, window_end)

    return {
        "gate_result": gate_result,
        "model_version": model_version,
        "wape_lift_pct": result["wape_lift_pct"],
    }
