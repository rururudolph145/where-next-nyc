"""Acceptance tests for PBI 27 — /health, /model-info, /heatmap simulated flag,
and trainer wape_lift_pct version tag.

Story: As an API consumer, I want dedicated endpoints that report the health of
the prediction service and the current model's provenance, so that I can verify
data freshness, confirm the simulated nature of the feed, and inspect model
quality without reading logs or source code.

Coverage map (every numbered AC in the story):
  AC-1  → test_ac1_health_happy_path_returns_ok_shape
  AC-2  → test_ac2_health_lag_is_zero_when_bucket_in_future
  AC-3  → test_ac3_health_lag_is_null_when_predictions_empty
  AC-4  → test_ac4_health_degraded_when_no_production_version
  AC-5  → test_ac5_health_returns_503_when_predictions_not_loaded
  AC-6  → test_ac6_model_info_happy_path_returns_all_fields
  AC-7  → test_ac7_model_info_reads_production_stage_only
  AC-8  → test_ac8_model_info_no_production_model_returns_null_fields
  AC-9  → test_ac9_heatmap_envelope_contains_simulated_true
  AC-10 → test_ac10_heatmap_existing_criteria_still_pass
  AC-11 → test_ac11_trainer_writes_wape_lift_pct_tag_on_promotion
  AC-12 → test_ac12_trainer_skips_wape_lift_pct_tag_when_none
  AC-13 → test_ac13_wape_lift_pct_tag_value_is_float_parseable_string
  AC-14 → test_ac14_trainer_skips_wape_lift_pct_tag_when_inf

Edge cases from story:
  Edge: future bucket  → test_ac2_health_lag_is_zero_when_bucket_in_future
  Edge: empty parquet  → test_ac3_health_lag_is_null_when_predictions_empty
  Edge: missing tag    → test_ac6_model_info_happy_path_returns_all_fields (null branch)
  Edge: mlflow raises  → test_health_degraded_when_mlflow_client_raises
  Edge: tag absent     → test_model_info_returns_null_wape_when_tag_absent
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Shared fixture helpers — reuse patterns from existing test_serving.py
# ---------------------------------------------------------------------------

_FAKE_SERVING_CONFIG: dict[str, Any] = {
    "serving": {
        "predictions_path": "fake/predictions.parquet",
        "zone_shapefile_path": "fake/taxi_zones.shp",
    },
    "mlflow": {"model_name": "ny_taxi_demand_model"},
}


def _make_mock_version(
    version: str = "3",
    stage: str = "Production",
    creation_timestamp_ms: int = 1_710_058_921_000,
    tags: dict[str, str] | None = None,
) -> MagicMock:
    """Return a ModelVersion-like MagicMock suitable for /health and /model-info tests."""
    mock_v = MagicMock()
    mock_v.version = version
    mock_v.current_stage = stage
    mock_v.creation_timestamp = creation_timestamp_ms
    mock_v.tags = tags if tags is not None else {}
    return mock_v


def _make_predictions_df(hour_str: str = "2021-03-01 00:00:00") -> pd.DataFrame:
    """Return a one-row predictions DataFrame with a given hour."""
    return pd.DataFrame(
        {
            "pulocationid": [1],
            "hour": pd.to_datetime([hour_str]),
            "predicted_trip_count": [10.0],
            "actual_trip_count": [9.0],
        }
    )


def _make_empty_predictions_df() -> pd.DataFrame:
    """Return a zero-row predictions DataFrame (columns present, no rows)."""
    return pd.DataFrame(
        columns=[
            "pulocationid",
            "hour",
            "predicted_trip_count",
            "actual_trip_count",
        ]
    )


# ---------------------------------------------------------------------------
# Trainer-side fixtures — reuse _build_mlflow_mock / _registry_config pattern
# from test_e2e_registry_gate.py
# ---------------------------------------------------------------------------

_TRAINER_FEATURES = [
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "pickup_longitude",
    "pickup_latitude",
    "trip_count_lag_1",
    "trip_count_lag_3",
    "trip_count_lag_6",
    "trip_count_lag_12",
    "trip_count_lag_24",
    "trip_count_lag_168",
]

_BASE_TS = pd.Timestamp("2022-01-03 00:00:00")  # Monday anchor for weekly pattern
_ZONES = [1, 2, 3]


def _make_4week_gold(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """4-week × 24-hour × 3-zone Gold parquet that produces non-None wape_lift_pct.

    trip_count = zone * 10 + hour_of_day → perfectly repeating weekly pattern.
    The last 10 % of the data has week-ago overlap, so aligned_df is non-empty
    and wape_lift_pct is computable.
    """
    records = []
    for week in range(4):
        for hour in range(24):
            ts = _BASE_TS + pd.Timedelta(weeks=week, hours=hour)
            for zone in _ZONES:
                records.append(
                    {
                        "pulocationid": zone,
                        "hour": ts,
                        "trip_count": float(zone * 10 + hour),
                        "hour_of_day_sin": float(np.sin(2 * np.pi * hour / 24)),
                        "hour_of_day_cos": float(np.cos(2 * np.pi * hour / 24)),
                        "day_of_week_sin": float(
                            np.sin(2 * np.pi * ts.dayofweek / 7)
                        ),
                        "day_of_week_cos": float(
                            np.cos(2 * np.pi * ts.dayofweek / 7)
                        ),
                        "month_sin": float(np.sin(2 * np.pi * ts.month / 12)),
                        "month_cos": float(np.cos(2 * np.pi * ts.month / 12)),
                        "pickup_longitude": float(-74.00 - zone * 0.01),
                        "pickup_latitude": float(40.70 + zone * 0.01),
                        "trip_count_lag_1": float(zone * 10 + (hour - 1) % 24),
                        "trip_count_lag_3": float(zone * 10 + (hour - 3) % 24),
                        "trip_count_lag_6": float(zone * 10 + (hour - 6) % 24),
                        "trip_count_lag_12": float(zone * 10 + (hour - 12) % 24),
                        "trip_count_lag_24": float(zone * 10 + (hour - 24) % 24),
                        "trip_count_lag_168": float(zone * 10 + hour),
                    }
                )
    gold_df = pd.DataFrame(records)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)
    gold_df.to_parquet(gold_dir / "hourly_demand_featured.parquet", index=False)
    return gold_dir, gold_df


def _registry_config(tmp_path: Path) -> dict[str, Any]:
    """Full Trainer config with PBI-20 registry keys; all paths inside tmp_path."""
    return {
        "env": "local",
        "data": {"gold_path": str(tmp_path / "gold")},
        "feature": {"bucket": "h"},
        "train": {
            "features": _TRAINER_FEATURES,
            "target": "trip_count",
            "split_ratios": [0.8, 0.1, 0.1],
            "gate": {"min_wape_lift_pct": 2.0},
        },
        "model": {
            "name": "LGBMRegressor",
            "params": {
                "n_estimators": 5,
                "learning_rate": 0.1,
                "num_leaves": 4,
                "random_state": 42,
                "verbosity": -1,
            },
            "store_path": str(tmp_path / "models"),
        },
        "mlflow": {
            "experiment_name": "test_pbi27_acceptance",
            "tracking_uri": "sqlite:///mlflow.db",
            "model_name": "ny_taxi_demand_model",
        },
    }


def _build_mlflow_mock(
    run_id: str = "abc123",
    registered_version: str = "1",
    production_versions: list[Any] | None = None,
) -> MagicMock:
    """Boundary mock for app.training.trainer.mlflow (mirrors test_e2e_registry_gate)."""
    mock_mlflow = MagicMock()

    active_run = MagicMock()
    active_run.info.run_id = run_id
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    mock_version = MagicMock()
    mock_version.version = registered_version
    mock_mlflow.register_model.return_value = mock_version

    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = (
        production_versions if production_versions is not None else []
    )
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    return mock_mlflow


def _champion_mock(version: str = "0", wape_tag: str = "15.0") -> MagicMock:
    """Return a ModelVersion mock representing a Production champion."""
    champion = MagicMock()
    champion.version = version
    champion.tags = {"model_test_wape": wape_tag}
    return champion


# ---------------------------------------------------------------------------
# AC-1 — GET /health happy path: HTTP 200, status=ok, all fields present
# ---------------------------------------------------------------------------


class TestHealthHappyPath:
    """AC-1: /health returns ok shape when predictions loaded + Production model present."""

    def test_ac1_health_happy_path_returns_ok_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-1: 200 response has status=ok, simulated=true, non-negative lag, model_version string.

        Arrange: non-empty predictions df; MlflowClient returns Production version "3".
        Act: GET /health.
        Assert: 200, status=="ok", simulated==True, model_version=="3",
                data_publication_lag_days >= 0 (int), last_prediction_bucket parseable.
        """
        import app.api.main as serving_module

        # Arrange
        predictions = _make_predictions_df("2021-03-01 00:00:00")
        monkeypatch.setattr(serving_module, "_predictions_df", predictions)

        mock_version = _make_mock_version(
            version="3",
            stage="Production",
            creation_timestamp_ms=1_710_058_921_000,
        )
        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = [mock_version]

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_SERVING_CONFIG):
            with patch(
                "app.api.main.MlflowClient", return_value=mock_client_instance
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/health")

        # Assert
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["status"] == "ok"
        assert body["simulated"] is True
        assert body["model_version"] == "3"
        assert isinstance(body["data_publication_lag_days"], int)
        assert body["data_publication_lag_days"] >= 0
        # last_prediction_bucket must be ISO-8601 parseable
        pd.Timestamp(body["last_prediction_bucket"])


# ---------------------------------------------------------------------------
# AC-2 — lag clamps to 0 when max(bucket_time) is in the future
# ---------------------------------------------------------------------------


class TestHealthLagClamping:
    """AC-2: data_publication_lag_days is 0 when predictions are from the future."""

    def test_ac2_health_lag_is_zero_when_bucket_in_future(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-2: lag must never be negative — clamped to 0 for future bucket timestamps.

        Arrange: prediction hour is far in the future (2099).
        Act: GET /health.
        Assert: data_publication_lag_days == 0.
        """
        import app.api.main as serving_module

        # Arrange
        predictions = _make_predictions_df("2099-12-31 23:00:00")
        monkeypatch.setattr(serving_module, "_predictions_df", predictions)

        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_SERVING_CONFIG):
            with patch(
                "app.api.main.MlflowClient", return_value=mock_client_instance
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/health")

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["data_publication_lag_days"] == 0


# ---------------------------------------------------------------------------
# AC-3 — lag and bucket are null when predictions parquet is empty
# ---------------------------------------------------------------------------


class TestHealthEmptyPredictions:
    """AC-3: null freshness fields when predictions DataFrame is empty."""

    def test_ac3_health_lag_is_null_when_predictions_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-3: data_publication_lag_days and last_prediction_bucket are null on empty parquet.

        Arrange: _predictions_df is an empty DataFrame (0 rows, correct columns).
        Act: GET /health.
        Assert: data_publication_lag_days is null, last_prediction_bucket is null.
        """
        import app.api.main as serving_module

        # Arrange
        empty_df = _make_empty_predictions_df()
        monkeypatch.setattr(serving_module, "_predictions_df", empty_df)

        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_SERVING_CONFIG):
            with patch(
                "app.api.main.MlflowClient", return_value=mock_client_instance
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/health")

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["data_publication_lag_days"] is None
        assert body["last_prediction_bucket"] is None


# ---------------------------------------------------------------------------
# AC-4 — GET /health degraded when no Production model exists
# ---------------------------------------------------------------------------


class TestHealthDegraded:
    """AC-4: /health returns degraded shape when predictions loaded but no Production model."""

    def test_ac4_health_degraded_when_no_production_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-4: status=degraded, model_version=null, freshness fields still populated.

        Arrange: non-empty predictions df; MlflowClient returns empty version list.
        Act: GET /health.
        Assert: 200, status=="degraded", model_version is null,
                simulated==True, freshness fields are non-null.
        """
        import app.api.main as serving_module

        # Arrange
        predictions = _make_predictions_df("2021-03-01 00:00:00")
        monkeypatch.setattr(serving_module, "_predictions_df", predictions)

        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_SERVING_CONFIG):
            with patch(
                "app.api.main.MlflowClient", return_value=mock_client_instance
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/health")

        # Assert
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["status"] == "degraded"
        assert body["simulated"] is True
        assert body["model_version"] is None
        # Freshness fields must still be populated even without a model
        assert body["last_prediction_bucket"] is not None
        assert body["data_publication_lag_days"] is not None

    def test_health_degraded_when_mlflow_client_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Edge: degraded when MlflowClient constructor raises (e.g. missing mlflow.db).

        This covers the 'Registry SQLite file does not exist' edge case from the story.
        """
        import app.api.main as serving_module

        # Arrange
        predictions = _make_predictions_df("2021-03-01 00:00:00")
        monkeypatch.setattr(serving_module, "_predictions_df", predictions)

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_SERVING_CONFIG):
            with patch(
                "app.api.main.MlflowClient", side_effect=Exception("db not found")
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/health")

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["model_version"] is None


# ---------------------------------------------------------------------------
# AC-5 — GET /health returns 503 when predictions not loaded
# ---------------------------------------------------------------------------


class TestHealthPredictionsUnavailable:
    """AC-5: /health returns 503 with detail when predictions file was not loaded."""

    def test_ac5_health_returns_503_when_predictions_not_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-5: 503 with a detail message when _predictions_df is None.

        This matches the existing 503 behaviour already in /heatmap (AC from prior PBI).

        Arrange: _predictions_df = None.
        Act: GET /health.
        Assert: 503, 'detail' in body, detail string is non-empty.
        """
        import app.api.main as serving_module

        # Arrange
        monkeypatch.setattr(serving_module, "_predictions_df", None)

        # Act
        local_client = TestClient(serving_module.app)
        resp = local_client.get("/health")

        # Assert
        assert resp.status_code == 503
        body = resp.json()
        assert "detail" in body
        assert body["detail"]  # non-empty string


# ---------------------------------------------------------------------------
# AC-6 — GET /model-info happy path: all fields present and typed correctly
# ---------------------------------------------------------------------------


class TestModelInfoHappyPath:
    """AC-6: /model-info returns complete shape when Production version exists."""

    def test_ac6_model_info_happy_path_returns_all_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-6: 200 with status=ok, typed floats for wape, ISO-8601 training_date, stage=Production.

        Arrange: MlflowClient returns one Production version with both tags set.
        Act: GET /model-info.
        Assert: 200, status=="ok", model_version=="3", stage=="Production",
                model_test_wape and wape_lift_pct are float, training_date ISO-8601 parseable.
        """
        import app.api.main as serving_module

        # Arrange: creation_timestamp_ms = 1_710_058_921_000 → ~2024-03-10 UTC
        mock_version = _make_mock_version(
            version="3",
            stage="Production",
            creation_timestamp_ms=1_710_058_921_000,
            tags={"model_test_wape": "0.1823", "wape_lift_pct": "12.4"},
        )
        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = [mock_version]

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_SERVING_CONFIG):
            with patch(
                "app.api.main.MlflowClient", return_value=mock_client_instance
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/model-info")

        # Assert
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["status"] == "ok"
        assert body["model_version"] == "3"
        assert body["stage"] == "Production"
        assert isinstance(body["model_test_wape"], float)
        assert body["model_test_wape"] == pytest.approx(0.1823, rel=1e-4)
        assert isinstance(body["wape_lift_pct"], float)
        assert body["wape_lift_pct"] == pytest.approx(12.4, rel=1e-4)
        # training_date must be ISO-8601 parseable (creation_timestamp in ms → seconds)
        ts = pd.Timestamp(body["training_date"])
        assert ts.year == 2024  # sanity-check the ms → seconds division

    def test_model_info_returns_null_wape_when_tag_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Edge (AC-6 null branch): model_test_wape and wape_lift_pct are null when tags absent.

        Arrange: Production version exists but has no tags set.
        Act: GET /model-info.
        Assert: 200, status=="ok", model_test_wape is null, wape_lift_pct is null.
        """
        import app.api.main as serving_module

        # Arrange
        mock_version = _make_mock_version(version="2", stage="Production", tags={})
        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = [mock_version]

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_SERVING_CONFIG):
            with patch(
                "app.api.main.MlflowClient", return_value=mock_client_instance
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/model-info")

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_test_wape"] is None
        assert body["wape_lift_pct"] is None


# ---------------------------------------------------------------------------
# AC-7 — /model-info reads Production stage only; other stages ignored
# ---------------------------------------------------------------------------


class TestModelInfoProductionOnly:
    """AC-7: /model-info ignores non-Production versions."""

    def test_ac7_model_info_reads_production_stage_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-7: get_latest_versions is called with stages=['Production']; Staging/None ignored.

        Arrange: MlflowClient.get_latest_versions returns a Production version.
        Act: GET /model-info.
        Assert: the call to get_latest_versions passed stages=['Production'] (not all stages),
                and the response reflects the Production version.
        """
        import app.api.main as serving_module

        # Arrange
        mock_version = _make_mock_version(
            version="5",
            stage="Production",
            tags={"model_test_wape": "0.20"},
        )
        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = [mock_version]

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_SERVING_CONFIG):
            with patch(
                "app.api.main.MlflowClient", return_value=mock_client_instance
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/model-info")

        # Assert: production version surfaced
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_version"] == "5"
        assert body["stage"] == "Production"
        # Assert: the client queried exactly the Production stage
        call_kwargs = mock_client_instance.get_latest_versions.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert "Production" in kwargs.get("stages", call_kwargs.args[1] if len(call_kwargs.args) > 1 else [])


# ---------------------------------------------------------------------------
# AC-8 — GET /model-info when no Production model: all fields null
# ---------------------------------------------------------------------------


class TestModelInfoNoProduction:
    """AC-8: /model-info returns no_production_model when registry is empty."""

    def test_ac8_model_info_no_production_model_returns_null_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-8: 200 with status=no_production_model and every other field null.

        Arrange: MlflowClient.get_latest_versions returns empty list.
        Act: GET /model-info.
        Assert: 200, status=="no_production_model", model_version/stage/training_date/
                model_test_wape/wape_lift_pct all null.
        """
        import app.api.main as serving_module

        # Arrange
        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_SERVING_CONFIG):
            with patch(
                "app.api.main.MlflowClient", return_value=mock_client_instance
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/model-info")

        # Assert
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["status"] == "no_production_model"
        assert body["model_version"] is None
        assert body["stage"] is None
        assert body["training_date"] is None
        assert body["model_test_wape"] is None
        assert body["wape_lift_pct"] is None


# ---------------------------------------------------------------------------
# AC-9 — /heatmap envelope contains "simulated": true
# ---------------------------------------------------------------------------


class TestHeatmapSimulatedFlag:
    """AC-9: /heatmap top-level envelope includes simulated=true."""

    def test_ac9_heatmap_envelope_contains_simulated_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-9: 'simulated' key is present and equals True in the /heatmap response envelope.

        Arrange: non-empty predictions df with centroid and zone lookups.
        Act: GET /heatmap (no hour param → latest hour).
        Assert: body["simulated"] is True.
        """
        import app.api.main as serving_module

        # Arrange
        predictions = pd.DataFrame(
            {
                "pulocationid": [1, 2, 3],
                "hour": pd.to_datetime(["2021-03-01 01:00:00"] * 3),
                "predicted_trip_count": [12.0, 18.7, 6.2],
                "actual_trip_count": [13.0, 17.0, 7.0],
            }
        )
        monkeypatch.setattr(serving_module, "_predictions_df", predictions)
        monkeypatch.setattr(
            serving_module,
            "_zone_lookup",
            {1: "Newark Airport", 2: "Jamaica Bay", 3: "Allerton/Pelham Gardens"},
        )
        monkeypatch.setattr(
            serving_module,
            "_centroid_lookup",
            {
                1: (40.693, -74.174),
                2: (40.618, -73.825),
                3: (40.866, -73.868),
            },
        )
        monkeypatch.setattr(
            serving_module, "_generated_at", "2021-03-01T12:00:00+00:00"
        )

        # Act
        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")

        # Assert
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert "simulated" in body
        assert body["simulated"] is True


# ---------------------------------------------------------------------------
# AC-10 — existing /heatmap acceptance criteria remain satisfied
# ---------------------------------------------------------------------------


class TestHeatmapBackwardsCompatibility:
    """AC-10: all prior /heatmap criteria still hold after PBI-27 changes."""

    @pytest.fixture(autouse=True)
    def _setup_module_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.api.main as serving_module

        monkeypatch.setattr(
            serving_module,
            "_predictions_df",
            pd.DataFrame(
                {
                    "pulocationid": [1, 2, 3, 1, 2, 3],
                    "hour": pd.to_datetime(
                        [
                            "2021-03-01 00:00:00",
                            "2021-03-01 00:00:00",
                            "2021-03-01 00:00:00",
                            "2021-03-01 01:00:00",
                            "2021-03-01 01:00:00",
                            "2021-03-01 01:00:00",
                        ]
                    ),
                    "predicted_trip_count": [10.5, 20.3, 5.1, 12.0, 18.7, 6.2],
                    "actual_trip_count": [11.0, 19.0, 4.0, 13.0, 17.0, 7.0],
                }
            ),
        )
        monkeypatch.setattr(
            serving_module,
            "_zone_lookup",
            {1: "Newark Airport", 2: "Jamaica Bay", 3: "Allerton/Pelham Gardens"},
        )
        monkeypatch.setattr(
            serving_module,
            "_centroid_lookup",
            {1: (40.693, -74.174), 2: (40.618, -73.825), 3: (40.866, -73.868)},
        )
        monkeypatch.setattr(
            serving_module, "_generated_at", "2021-03-01T12:00:00+00:00"
        )

    def test_ac10_heatmap_existing_criteria_still_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-10: zone count, descending sort order, required envelope fields all intact.

        Arrange: standard predictions fixture (3 zones × 2 hours).
        Act: GET /heatmap (default latest hour).
        Assert: zone_count==3, predictions sorted descending, generated_at/hour/predictions present.
        """
        import app.api.main as serving_module

        # Act
        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")

        # Assert
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["zone_count"] == 3
        counts = [p["predicted_trip_count"] for p in body["predictions"]]
        assert counts == sorted(counts, reverse=True)
        assert "generated_at" in body
        assert "hour" in body
        assert "predictions" in body

    def test_ac10_heatmap_returns_503_when_no_predictions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-10 (503 sub-criterion): /heatmap still returns 503 when _predictions_df is None."""
        import app.api.main as serving_module

        # Arrange: override the autouse fixture
        monkeypatch.setattr(serving_module, "_predictions_df", None)

        # Act
        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")

        # Assert
        assert resp.status_code == 503
        assert "detail" in resp.json()

    def test_ac10_heatmap_returns_404_for_unknown_hour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-10 (404 sub-criterion): /heatmap returns 404 for an unknown hour."""
        import app.api.main as serving_module

        # Act
        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap", params={"hour": "2099-01-01T00:00:00"})

        # Assert
        assert resp.status_code == 404

    def test_ac10_heatmap_returns_400_for_malformed_hour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-10 (400 sub-criterion): /heatmap returns 400 for malformed hour parameter."""
        import app.api.main as serving_module

        # Act
        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap", params={"hour": "not-a-date"})

        # Assert
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# AC-11 — trainer writes wape_lift_pct tag on promotion (end-to-end)
# ---------------------------------------------------------------------------


class TestTrainerWapeLiftPctTag:
    """AC-11 to AC-14: Trainer.run() writes / skips wape_lift_pct tag correctly."""

    def test_ac11_trainer_writes_wape_lift_pct_tag_on_promotion(
        self, tmp_path: Path
    ) -> None:
        """AC-11: wape_lift_pct tag is written to the registered version on promotion.

        Arrange: 4-week Gold fixture with repeating weekly pattern so aligned_df is
                 non-empty; champion with high WAPE (15.0) so challenger beats it;
                 compute_wape_lift patched to return 5.0 (> threshold 2.0).
        Act: Trainer.run().
        Assert: set_model_version_tag was called with key 'wape_lift_pct' and the
                value is a non-empty, float-parseable string.
        """
        # Arrange
        gold_dir, gold_df = _make_4week_gold(tmp_path)
        champion = _champion_mock(version="0", wape_tag="15.0")
        trainer = Trainer(config=_registry_config(tmp_path))
        mock_mlflow = _build_mlflow_mock(
            registered_version="1", production_versions=[champion]
        )
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Act: patch both copies of compute_wape_lift so the local wape_lift_pct
        # variable in trainer.py is 5.0 (finite) and apply_gate also sees 5.0.
        # Patching only registry_gate is not enough: the 4-week fixture produces
        # baseline_wape=0, so the unpatched trainer copy returns inf.
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.trainer.compute_wape_lift", return_value=5.0
            ):
                with patch(
                    "app.training.registry_gate.compute_wape_lift", return_value=5.0
                ):
                    with patch.object(
                        trainer, "save_model", return_value="models/model.pkl"
                    ):
                        trainer.run(data=gold_df)

        # Assert: tag written on the registered version
        tag_keys = [c.args[2] for c in mock_client.set_model_version_tag.call_args_list]
        assert "wape_lift_pct" in tag_keys

        # Tag value is a non-empty float-parseable string
        for call in mock_client.set_model_version_tag.call_args_list:
            if call.args[2] == "wape_lift_pct":
                tag_value = call.args[3]
                assert isinstance(tag_value, str)
                assert tag_value != ""
                parsed = float(tag_value)
                assert math.isfinite(parsed)

    def test_ac12_trainer_skips_wape_lift_pct_tag_when_none(
        self, tmp_path: Path
    ) -> None:
        """AC-12: wape_lift_pct tag is NOT written when wape_lift_pct is None.

        Arrange: tiny 100-row dataset (< 1 week) so aligned_df is empty →
                 wape_lift_pct=None from Trainer.run(); no Production incumbent
                 so auto-promote path fires (gate_result="promoted").
        Act: Trainer.run().
        Assert: result["wape_lift_pct"] is None AND set_model_version_tag was
                never called with key "wape_lift_pct". No exception raised.
        """
        # Arrange: short dataset → no week-ago overlap → wape_lift_pct=None
        rng = np.random.RandomState(0)
        n = 100
        hours = pd.date_range("2022-01-01", periods=n, freq="h")
        data: dict[str, Any] = {
            "hour": hours,
            "pulocationid": rng.randint(1, 5, n),
        }
        for feat in _TRAINER_FEATURES:
            data[feat] = rng.rand(n)
        data["trip_count"] = rng.randint(1, 20, n).astype(float)
        tiny_df = pd.DataFrame(data)

        trainer = Trainer(config=_registry_config(tmp_path))
        mock_mlflow = _build_mlflow_mock(
            registered_version="1", production_versions=[]  # no incumbent → auto-promote
        )
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch.object(trainer, "save_model", return_value="models/model.pkl"):
                result = trainer.run(data=tiny_df)

        # Assert: wape_lift_pct is None AND tag not written
        assert result["wape_lift_pct"] is None
        tag_keys = [c.args[2] for c in mock_client.set_model_version_tag.call_args_list]
        assert "wape_lift_pct" not in tag_keys

    def test_ac13_wape_lift_pct_tag_value_is_float_parseable_string(
        self, tmp_path: Path
    ) -> None:
        """AC-13: the tag value stored in the registry is str(float) — no 'None' or 'inf'.

        Arrange: same promotion scenario as AC-11.
        Act: Trainer.run() with patched compute_wape_lift returning a normal float.
        Assert: every wape_lift_pct tag call uses a str-typed value that
                float() can parse without raising ValueError.
        """
        # Arrange
        gold_dir, gold_df = _make_4week_gold(tmp_path)
        champion = _champion_mock(version="0", wape_tag="20.0")
        trainer = Trainer(config=_registry_config(tmp_path))
        mock_mlflow = _build_mlflow_mock(
            registered_version="2", production_versions=[champion]
        )
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Act: patch both compute_wape_lift copies to force finite local value
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.trainer.compute_wape_lift", return_value=7.5
            ):
                with patch(
                    "app.training.registry_gate.compute_wape_lift", return_value=7.5
                ):
                    with patch.object(
                        trainer, "save_model", return_value="models/model.pkl"
                    ):
                        trainer.run(data=gold_df)

        # Assert: every wape_lift_pct tag call stores a float-parseable string
        for call in mock_client.set_model_version_tag.call_args_list:
            if call.args[2] == "wape_lift_pct":
                tag_value = call.args[3]
                assert isinstance(tag_value, str), (
                    f"Expected str, got {type(tag_value).__name__}: {tag_value!r}"
                )
                # Must not be the literal strings "None" or "inf"
                assert tag_value not in ("None", "inf", "Inf", "INF")
                float(tag_value)  # must not raise

    def test_ac14_trainer_skips_wape_lift_pct_tag_when_inf(
        self, tmp_path: Path
    ) -> None:
        """AC-14: wape_lift_pct tag is NOT written when wape_lift_pct is float('inf').

        Arrange: 4-week Gold; both compute_wape_lift copies patched to return inf so
                 that the local wape_lift_pct variable is inf AND apply_gate still
                 promotes (inf >= threshold).
        Act: Trainer.run().
        Assert: result["wape_lift_pct"] == float('inf') AND 'wape_lift_pct' tag not
                written.  The string "inf" is never stored in the registry.
        """
        # Arrange
        gold_dir, gold_df = _make_4week_gold(tmp_path)
        champion = _champion_mock(version="0", wape_tag="10.0")
        trainer = Trainer(config=_registry_config(tmp_path))
        mock_mlflow = _build_mlflow_mock(
            registered_version="3", production_versions=[champion]
        )
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Act: both trainer-local and registry-gate copies return inf
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.trainer.compute_wape_lift",
                return_value=float("inf"),
            ):
                with patch(
                    "app.training.registry_gate.compute_wape_lift",
                    return_value=float("inf"),
                ):
                    with patch.object(
                        trainer, "save_model", return_value="models/model.pkl"
                    ):
                        result = trainer.run(data=gold_df)

        # Assert: inf result and no wape_lift_pct tag written
        assert math.isinf(result["wape_lift_pct"])
        tag_keys = [c.args[2] for c in mock_client.set_model_version_tag.call_args_list]
        assert "wape_lift_pct" not in tag_keys
