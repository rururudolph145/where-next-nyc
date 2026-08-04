"""FastAPI serving endpoints for NYC Taxi demand predictions.

Provides:
- ``GET /`` — health check
- ``GET /health`` — service health + data freshness (PBI 27)
- ``GET /model-info`` — Production model provenance (PBI 27)
- ``GET /heatmap`` — zone-level demand predictions (PBI 3)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from mlflow.tracking import MlflowClient

from app.core.clock import Clock, WallClock
from app.core.config import load_config
from app.core.watermark import read_watermark
from app.ingestion.feature import build_centroid_lookup

# Module-level import: a broken monitoring module should fail at startup,
# not be silently swallowed on the first request.
from app.monitoring import prediction_log

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup: load predictions + zone lookup into memory for fast responses
# ---------------------------------------------------------------------------

_predictions_df: pd.DataFrame | None = None
_zone_lookup: dict[int, str] = {}
_centroid_lookup: dict[int, tuple[float, float]] = {}
_generated_at: str = ""
_model_name: str = "ny_taxi_demand_model"
_model_version: str | None = None
_served_from_stage: str | None = None
_config: dict[str, Any] = {}
_clock: Clock = WallClock()
# Source of truth for the simulation position (PBI 25b FIX): the incremental
# pipeline's data watermark, NOT a clock. /health reads this for simulated_now
# and data_publication_lag_days. Mirrors refresh_15min_dag's _WATERMARK_ROOT.
_watermark_root: Path = Path("data/watermarks/refresh_15min")


def _load_predictions(config: dict[str, Any]) -> pd.DataFrame:
    """Load the predictions parquet from the configured path.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        Predictions DataFrame with columns pulocationid, hour,
        predicted_trip_count, actual_trip_count.

    Raises:
        FileNotFoundError: If predictions file does not exist.
    """
    path: str = config["serving"]["predictions_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")
    df = pd.read_parquet(path)
    df["pulocationid"] = df["pulocationid"].astype(int)
    df["hour"] = pd.to_datetime(df["hour"])
    return df


def _load_zone_lookup(config: dict[str, Any]) -> dict[int, str]:
    """Build a LocationID → zone_name mapping from the taxi zones shapefile.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        Dictionary mapping integer zone IDs to zone name strings.
    """
    path: str = config["serving"]["zone_shapefile_path"]
    gdf = gpd.read_file(path)
    lookup: dict[int, str] = dict(
        zip(
            gdf["LocationID"].astype(int),
            gdf["zone"].astype(str),
            strict=True,
        )
    )
    return lookup


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load predictions and zone lookup into memory at server start."""
    global _predictions_df, _zone_lookup, _centroid_lookup, _generated_at, _model_name, _model_version, _served_from_stage, _config  # noqa: PLW0603

    config = load_config()
    _config = config
    _model_name = config.get("mlflow", {}).get("model_name", "ny_taxi_demand_model")
    try:
        _predictions_df = _load_predictions(config)
        _zone_lookup = _load_zone_lookup(config)
        _centroid_lookup = build_centroid_lookup(
            config["serving"]["zone_shapefile_path"]
        )
        _generated_at = _clock.now().isoformat()
        df = _predictions_df
        if not df.empty and "model_version" in df.columns:
            _model_version = str(df["model_version"].iloc[0])
        if not df.empty and "served_from_stage" in df.columns:
            stage_val = str(df["served_from_stage"].iloc[0])
            _served_from_stage = (
                stage_val if stage_val in ("Production", "Archived") else None
            )
        logger.info(
            "Loaded %d predictions, %d zones",
            len(_predictions_df),
            len(_zone_lookup),
        )
    except FileNotFoundError:
        logger.warning(
            "Predictions file not found — /heatmap will return 503 "
            "until the simulation pipeline runs."
        )
    yield


def _current_sim_position() -> datetime:
    """Return the current simulation position as a NAIVE datetime.

    The simulation position is the incremental pipeline's data watermark. When
    the watermark control file is absent (before the first DAG run) it falls
    back to the configured ``ingest.seed_start``. Both are tz-naive NYC
    wall-clock timestamps — never attach tzinfo (the watermark/feature window
    comparisons are naive-vs-naive).

    Returns:
        The watermark position, or the seed_start fallback, as a naive datetime.
    """
    watermark = read_watermark(_watermark_root)
    if watermark is not None:
        return watermark
    seed_start: str = _config["ingest"]["seed_start"]
    return datetime.fromisoformat(seed_start)


app = FastAPI(title="NYC Taxi Demand Forecasting API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def read_root() -> dict[str, object]:
    """Health check endpoint.

    Returns:
        Status information.
    """
    return {"status": "alive"}


@app.get("/heatmap")
async def get_heatmap(
    hour: str | None = Query(
        default=None,
        description="Filter by hour (ISO format, e.g. 2021-03-01T14:00:00). "
        "Defaults to the latest available hour.",
    ),
) -> dict[str, Any]:
    """Return zone-level demand predictions for a given hour.

    If no ``hour`` query parameter is provided, returns predictions for
    the latest hour available in the predictions store.

    Args:
        hour: Optional ISO-format hour string to filter predictions.

    Returns:
        JSON with ``generated_at``, ``hour``, and ``predictions`` list
        containing ``pulocationid``, ``zone_name``, and
        ``predicted_trip_count`` per zone.

    Raises:
        HTTPException: 503 if predictions are not loaded.
        HTTPException: 404 if the requested hour has no data.
    """
    if _predictions_df is None or _predictions_df.empty:
        raise HTTPException(
            status_code=503,
            detail="Predictions not available. Run the simulation pipeline first.",
        )

    df = _predictions_df

    if hour is not None:
        try:
            target_hour = pd.Timestamp(hour)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid hour format: {hour!r}. Use ISO format.",
            )
        df = df[df["hour"] == target_hour]
        if df.empty:
            available = sorted(_predictions_df["hour"].unique())
            raise HTTPException(
                status_code=404,
                detail=f"No predictions for hour {hour}. "
                f"Available range: {available[0]} → {available[-1]}.",
            )
    else:
        watermark = read_watermark(_watermark_root)
        if watermark is not None:
            df = df[df["hour"] <= pd.Timestamp(watermark)]
            if df.empty:
                raise HTTPException(
                    status_code=503,
                    detail="No predictions available at or before the current watermark position.",
                )
        latest = df["hour"].max()
        df = df[df["hour"] == latest]
        target_hour = latest

    predictions: list[dict[str, object]] = []
    for _, row in df.iterrows():
        zone_id = int(row["pulocationid"])
        centroid = _centroid_lookup.get(zone_id)
        predictions.append(
            {
                "pulocationid": zone_id,
                "zone_name": _zone_lookup.get(zone_id, "Unknown"),
                "predicted_trip_count": round(float(row["predicted_trip_count"]), 2),
                "latitude": centroid[0] if centroid is not None else None,
                "longitude": centroid[1] if centroid is not None else None,
                "bucket_time": target_hour.isoformat(),
            }
        )

    predictions.sort(
        key=lambda x: float(x["predicted_trip_count"]),  # type: ignore[arg-type]
        reverse=True,
    )

    try:
        prediction_log.log_predictions(
            {
                "predictions": predictions,
                "model_version": _model_version,
                "served_from_stage": _served_from_stage,
            },
            _config.get("monitoring", {}).get(
                "predictions_path", "data/monitoring/predictions"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write prediction log: %s", exc)

    return {
        "generated_at": _generated_at,
        "simulated": os.environ.get("DEMO_MODE") == "1",
        "model_version": _model_version,
        "served_from_stage": _served_from_stage,
        "hour": target_hour.isoformat(),
        "zone_count": len(predictions),
        "predictions": predictions,
    }


@app.get("/health")
async def get_health() -> dict[str, Any]:
    """Return service health, data freshness, and model version.

    Returns HTTP 200 in all non-503 cases:
    - ``status="ok"`` when both predictions and a Production model are available.
    - ``status="degraded"`` when predictions are loaded but no Production version
      exists in the registry, or the registry is unreachable.

    Returns:
        JSON with ``status``, ``simulated``, ``data_publication_lag_days``,
        ``last_prediction_bucket``, and ``model_version``.

    Raises:
        HTTPException: 503 if predictions are not loaded.
    """
    if _predictions_df is None:
        raise HTTPException(
            status_code=503,
            detail="Predictions not available. Run the simulation pipeline first.",
        )

    # Compute freshness from loaded predictions.
    last_prediction_bucket: str | None
    data_publication_lag_days: int | None

    if _predictions_df.empty:
        last_prediction_bucket = None
        data_publication_lag_days = None
    else:
        last_bucket = _predictions_df["hour"].max()
        last_prediction_bucket = last_bucket.isoformat()
        # Lag is measured against the simulation position (watermark), not the
        # wall clock: in a simulation "now" is how far ingestion has advanced,
        # so lag ≈ 0 when ingestion is current. Both sides are naive.
        try:
            today = _current_sim_position().date()
            lag = (today - last_bucket.date()).days
            data_publication_lag_days = max(0, lag)
        except (KeyError, ValueError, TypeError):
            # /health must never raise: degrade lag to unknown on a malformed
            # watermark or missing seed_start rather than 500.
            data_publication_lag_days = None

    # Look up Production model version from registry; degrade gracefully on error.
    model_version: str | None = None
    status = "degraded"
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(_model_name, stages=["Production"])
        if versions:
            model_version = versions[0].version
            status = "ok"
    except Exception:  # noqa: BLE001
        pass

    demo_mode = os.environ.get("DEMO_MODE") == "1"
    payload: dict[str, Any] = {
        "status": status,
        "simulated": demo_mode,
        "data_publication_lag_days": data_publication_lag_days,
        "last_prediction_bucket": last_prediction_bucket,
        "model_version": model_version,
    }
    if demo_mode:
        try:
            payload["simulated_now"] = _current_sim_position().isoformat()
        except (KeyError, ValueError, TypeError):
            payload["simulated_now"] = None
        payload["wall_clock_now"] = datetime.now(tz=timezone.utc).isoformat()
    return payload


@app.get("/model-info")
async def get_model_info() -> dict[str, Any]:
    """Return Production model provenance from the MLflow Model Registry.

    Never raises 5xx. Returns ``status="no_production_model"`` with null fields
    when no Production version exists or the registry is unreachable.

    Returns:
        JSON with ``status``, ``model_version``, ``stage``, ``training_date``,
        ``model_test_wape``, and ``wape_lift_pct``.
    """
    _null_response: dict[str, Any] = {
        "status": "no_production_model",
        "model_version": None,
        "stage": None,
        "training_date": None,
        "model_test_wape": None,
        "wape_lift_pct": None,
    }

    try:
        client = MlflowClient()
        versions = client.get_latest_versions(_model_name, stages=["Production"])
        if not versions:
            return _null_response
        v = versions[0]
    except Exception:  # noqa: BLE001
        return _null_response

    training_date = datetime.fromtimestamp(
        v.creation_timestamp / 1000, tz=timezone.utc
    ).isoformat()

    model_test_wape: float | None
    wape_lift_pct: float | None
    try:
        model_test_wape = float(v.tags["model_test_wape"])
    except (KeyError, ValueError, TypeError):
        model_test_wape = None
    try:
        wape_lift_pct = float(v.tags["wape_lift_pct"])
    except (KeyError, ValueError, TypeError):
        wape_lift_pct = None

    return {
        "status": "ok",
        "model_version": v.version,
        "stage": v.current_stage,
        "training_date": training_date,
        "model_test_wape": model_test_wape,
        "wape_lift_pct": wape_lift_pct,
    }
