"""FastAPI serving endpoints for NYC Taxi demand predictions.

Provides:
- ``GET /`` — health check
- ``GET /heatmap`` — zone-level demand predictions (PBI 3)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from app.core.config import load_config
from app.ingestion.feature import build_centroid_lookup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup: load predictions + zone lookup into memory for fast responses
# ---------------------------------------------------------------------------

_predictions_df: pd.DataFrame | None = None
_zone_lookup: dict[int, str] = {}
_centroid_lookup: dict[int, tuple[float, float]] = {}
_generated_at: str = ""


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
    global _predictions_df, _zone_lookup, _centroid_lookup, _generated_at  # noqa: PLW0603

    config = load_config()
    try:
        _predictions_df = _load_predictions(config)
        _zone_lookup = _load_zone_lookup(config)
        _centroid_lookup = build_centroid_lookup(
            config["serving"]["zone_shapefile_path"]
        )
        _generated_at = datetime.now(tz=timezone.utc).isoformat()
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
    return {"health_check": "OK"}


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

    return {
        "generated_at": _generated_at,
        "hour": target_hour.isoformat(),
        "zone_count": len(predictions),
        "predictions": predictions,
    }
