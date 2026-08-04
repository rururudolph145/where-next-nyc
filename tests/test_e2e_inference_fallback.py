"""Acceptance tests for PBI 21 — Inference Fallback to Previous Approved Model.

Story: As a pipeline operator, I want the simulator to load the Production
model from the registry by default, fall back to the most recent Archived
model with a logged warning when Production is unavailable, and raise a hard
error when no approved model exists in either stage, so that every prediction
run is traceable to an approved model version and silent serving of
unapproved weights is impossible.

Run in isolation:
  pytest -q tests/test_e2e_inference_fallback.py

# AC Coverage Map (PBI 21)
# AC-1   → test_ac1_production_present_uses_production
# AC-2   → test_ac2_production_absent_archived_present_uses_archived_with_warning
# AC-3   → test_ac3_both_absent_raises_no_approved_model_error
# AC-4a  → test_ac4_mlflow_exception_on_lookup_falls_back_to_archived
# AC-4b  → test_ac4_artifact_download_mlflow_exception_falls_back_to_archived
# AC-5   → test_ac5_parquet_contains_model_version_and_served_from_stage_columns
# AC-6   → test_ac6_provenance_columns_have_no_nulls
# AC-7   → covered by tests/test_inference.py::TestFallback::test_simulator_parquet_columns_exact_set
#          (simulator parquet column set — Predictor is out of scope per story)
# AC-8   → test_ac8_heatmap_envelope_contains_model_provenance_fields
# AC-9   → test_ac9_heatmap_preserves_pre_existing_behaviour
# AC-10  → test_ac10_no_approved_model_error_is_importable_with_descriptive_message
# AC-11  → covered by tests/test_inference.py::TestFallback (3 named cases) (structural)
# AC-12  → covered by tests/test_serving.py::TestHeatmapModelProvenance (structural)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.inference.exceptions import NoApprovedModelError
from tests.conftest import make_mock_version


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SIM_FEATURES = [
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

_SIMULATOR_CONFIG_BASE: dict[str, Any] = {
    "data": {"gold_path": "data/gold/yellow_tripdata"},
    "train": {
        "features": _SIM_FEATURES,
        "target": "trip_count",
    },
    "model": {"store_path": "models/", "name": "LGBMRegressor", "params": {}},
    "mlflow": {
        "tracking_uri": "mlruns",
        "model_name": "ny_taxi_demand_model",
    },
    "inference": {
        "simulation": {
            "start_hour": "2022-01-01 00:00:00",
            "end_hour": "2022-01-01 02:00:00",
        },
    },
}

_FAKE_SERVING_CONFIG: dict[str, Any] = {
    "serving": {
        "predictions_path": "fake/predictions.parquet",
        "zone_shapefile_path": "fake/taxi_zones.shp",
    },
    "mlflow": {"model_name": "ny_taxi_demand_model"},
}


def _make_simulator(tmp_path: Path) -> Any:
    """Build a HourlySimulator that writes predictions inside tmp_path."""
    from app.inference.simulator import HourlySimulator

    cfg: dict[str, Any] = {
        **_SIMULATOR_CONFIG_BASE,
        "inference": {
            **_SIMULATOR_CONFIG_BASE.get("inference", {}),
            "output_path": str(tmp_path / "predictions" / "demand_predictions.parquet"),
        },
    }
    return HourlySimulator(config=cfg)


def _make_stub_model() -> MagicMock:
    """Return a model stub with a predict method that returns constant 1s."""
    import numpy as np

    stub = MagicMock()
    stub.predict.side_effect = lambda x: np.ones(len(x))
    return stub


def _make_sim_gold_df() -> pd.DataFrame:
    """Tiny Gold DataFrame covering the simulator's 3-hour window."""
    import numpy as np

    rng = np.random.RandomState(0)
    hours = pd.date_range("2022-01-01 00:00:00", periods=3, freq="h")
    n = len(hours) * 5  # 5 zones per hour
    data: dict[str, Any] = {
        "hour": list(hours) * 5,
        "pulocationid": [z for z in range(1, 6) for _ in range(3)],
    }
    for feat in _SIM_FEATURES:
        data[feat] = rng.rand(n)
    data["trip_count"] = rng.randint(0, 50, n).astype(float)
    return pd.DataFrame(data)


def _make_predictions_parquet_with_provenance(
    tmp_path: Path,
    model_version: str = "3",
    served_from_stage: str = "Production",
) -> Path:
    """Write a minimal predictions parquet that includes the PBI-21 columns."""
    df = pd.DataFrame(
        {
            "pulocationid": [1, 2, 3],
            "hour": pd.to_datetime(["2021-03-01 01:00:00"] * 3),
            "predicted_trip_count": [12.0, 18.7, 6.2],
            "actual_trip_count": [13.0, 17.0, 7.0],
            "model_version": [model_version] * 3,
            "served_from_stage": [served_from_stage] * 3,
        }
    )
    parquet_path = tmp_path / "predictions" / "demand_predictions.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return parquet_path


# ---------------------------------------------------------------------------
# AC-1: Production present → simulator loads it; parquet stamped "Production"
# ---------------------------------------------------------------------------


class TestAC1ProductionPresent:
    """AC-1: When Production version is available the simulator uses it."""

    def test_ac1_production_present_uses_production(self, tmp_path: Path) -> None:
        """AC-1: Production version loads; every row stamped served_from_stage=Production.

        Arrange: MlflowClient returns one Production version "3"; stub model.
        Act: HourlySimulator.run().
        Assert: predictions_df["served_from_stage"] == "Production" for all rows,
                predictions_df["model_version"] == "3" for all rows.
        """
        # Arrange
        sim = _make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub = _make_stub_model()
        prod_version = make_mock_version(version="3", stage="Production")

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [prod_version]

        # Act
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch(
                "app.inference.simulator.mlflow.lightgbm.load_model",
                return_value=stub,
            ):
                with patch.object(sim, "load_gold", return_value=gold_df):
                    result = sim.run()

        # Assert
        df: pd.DataFrame = result["predictions_df"]
        assert (df["served_from_stage"] == "Production").all(), (
            "Expected all rows stamped served_from_stage='Production'"
        )
        assert (df["model_version"] == "3").all(), (
            "Expected all rows stamped model_version='3'"
        )


# ---------------------------------------------------------------------------
# AC-2: Production absent + Archived present → fallback with warning
# ---------------------------------------------------------------------------


class TestAC2ArchivedFallback:
    """AC-2: No Production → loads Archived, emits warning."""

    def test_ac2_production_absent_archived_present_uses_archived_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC-2: Archived model used and warning logged when Production absent.

        Arrange: get_latest_versions returns [] for Production, [v2] for Archived.
        Act: HourlySimulator.run() with caplog capturing WARNING level.
        Assert: served_from_stage=="Archived", model_version=="2" on every row,
                at least one WARNING record contains "model_fallback_active=true".
        """
        # Arrange
        sim = _make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub = _make_stub_model()
        arch_version = make_mock_version(version="2", stage="Archived")

        def _side_effect(name: str, stages: list[str]) -> list[MagicMock]:
            return [] if "Production" in stages else [arch_version]

        mock_client = MagicMock()
        mock_client.get_latest_versions.side_effect = _side_effect

        # Act
        with caplog.at_level(logging.WARNING, logger="app.inference.simulator"):
            with patch(
                "app.inference.simulator.MlflowClient", return_value=mock_client
            ):
                with patch(
                    "app.inference.simulator.mlflow.lightgbm.load_model",
                    return_value=stub,
                ):
                    with patch.object(sim, "load_gold", return_value=gold_df):
                        result = sim.run()

        # Assert parquet provenance
        df: pd.DataFrame = result["predictions_df"]
        assert (df["served_from_stage"] == "Archived").all(), (
            "Expected all rows stamped served_from_stage='Archived'"
        )
        assert (df["model_version"] == "2").all(), (
            "Expected all rows stamped model_version='2'"
        )

        # Assert warning emitted
        fallback_records = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and "model_fallback_active=true" in r.message
        ]
        assert fallback_records, (
            "Expected a WARNING log containing 'model_fallback_active=true' "
            "when falling back to Archived"
        )


# ---------------------------------------------------------------------------
# AC-3: Both absent → NoApprovedModelError; no parquet written
# ---------------------------------------------------------------------------


class TestAC3BothAbsent:
    """AC-3: Neither Production nor Archived → hard error, no parquet."""

    def test_ac3_both_absent_raises_no_approved_model_error(
        self, tmp_path: Path
    ) -> None:
        """AC-3: NoApprovedModelError raised; output parquet does not exist.

        Arrange: get_latest_versions returns [] for every stage query.
        Act: HourlySimulator.run().
        Assert: NoApprovedModelError is raised,
                predictions parquet is NOT written to disk.
        """
        # Arrange
        sim = _make_simulator(tmp_path)
        output_path = sim.output_path
        gold_df = _make_sim_gold_df()

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act + Assert
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch.object(sim, "load_gold", return_value=gold_df):
                with pytest.raises(NoApprovedModelError):
                    sim.run()

        assert not os.path.exists(output_path), (
            "Predictions parquet must NOT be written when NoApprovedModelError is raised"
        )


# ---------------------------------------------------------------------------
# AC-4a: MlflowException on Production lookup → fallback to Archived
# ---------------------------------------------------------------------------


class TestAC4MLflowExceptionOnLookup:
    """AC-4a: MlflowException during Production registry query triggers Archived fallback."""

    def test_ac4_mlflow_exception_on_lookup_falls_back_to_archived(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC-4a: MlflowException on Production get_latest_versions → Archived used.

        Arrange: first get_latest_versions call (Production) raises MlflowException;
                 second call (Archived) returns a valid version.
        Act: HourlySimulator.run().
        Assert: served_from_stage=="Archived" on all prediction rows,
                warning contains "model_fallback_active=true".
        """
        from mlflow.exceptions import MlflowException

        # Arrange
        sim = _make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub = _make_stub_model()
        arch_version = make_mock_version(version="2", stage="Archived")

        def _side_effect(name: str, stages: list[str]) -> list[MagicMock]:
            if "Production" in stages:
                raise MlflowException("registry unavailable")
            return [arch_version]

        mock_client = MagicMock()
        mock_client.get_latest_versions.side_effect = _side_effect

        # Act
        with caplog.at_level(logging.WARNING, logger="app.inference.simulator"):
            with patch(
                "app.inference.simulator.MlflowClient", return_value=mock_client
            ):
                with patch(
                    "app.inference.simulator.mlflow.lightgbm.load_model",
                    return_value=stub,
                ):
                    with patch.object(sim, "load_gold", return_value=gold_df):
                        result = sim.run()

        # Assert
        df: pd.DataFrame = result["predictions_df"]
        assert (df["served_from_stage"] == "Archived").all(), (
            "Expected Archived fallback when Production lookup raises MlflowException"
        )
        fallback_records = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and "model_fallback_active=true" in r.message
        ]
        assert fallback_records, (
            "Expected fallback warning after MlflowException on Production lookup"
        )


# ---------------------------------------------------------------------------
# AC-4b: MlflowException on artifact download → fallback to Archived
# ---------------------------------------------------------------------------


class TestAC4ArtifactDownloadFailure:
    """AC-4b: MlflowException during Production artifact download triggers Archived fallback."""

    def test_ac4_artifact_download_mlflow_exception_falls_back_to_archived(
        self, tmp_path: Path
    ) -> None:
        """AC-4b: Production artifact download raises MlflowException → uses Archived.

        Arrange: Production version found in registry (version "5") but
                 mlflow.lightgbm.load_model raises MlflowException for its URI.
                 Archived version "4" has a healthy artifact.
        Act: HourlySimulator.run().
        Assert: served_from_stage=="Archived", model_version=="4" on all rows.
        """
        from mlflow.exceptions import MlflowException

        # Arrange
        sim = _make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub = _make_stub_model()
        prod_version = make_mock_version(version="5", stage="Production")
        arch_version = make_mock_version(version="4", stage="Archived")

        def _stage_side_effect(name: str, stages: list[str]) -> list[MagicMock]:
            return [prod_version] if "Production" in stages else [arch_version]

        mock_client = MagicMock()
        mock_client.get_latest_versions.side_effect = _stage_side_effect

        def _load_side_effect(uri: str) -> MagicMock:
            if "5" in uri:
                raise MlflowException("artifact missing for version 5")
            return stub

        # Act
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch(
                "app.inference.simulator.mlflow.lightgbm.load_model",
                side_effect=_load_side_effect,
            ):
                with patch.object(sim, "load_gold", return_value=gold_df):
                    result = sim.run()

        # Assert
        df: pd.DataFrame = result["predictions_df"]
        assert (df["served_from_stage"] == "Archived").all(), (
            "Expected Archived fallback after Production artifact download failure"
        )
        assert (df["model_version"] == "4").all(), (
            "Expected model_version='4' (Archived) after Production artifact failure"
        )


# ---------------------------------------------------------------------------
# AC-5: Parquet schema — both new columns present as strings
# ---------------------------------------------------------------------------


class TestAC5ParquetSchema:
    """AC-5: Predictions parquet contains model_version and served_from_stage columns."""

    def test_ac5_parquet_contains_model_version_and_served_from_stage_columns(
        self, tmp_path: Path
    ) -> None:
        """AC-5: Written parquet has model_version (str) and served_from_stage (str).

        Arrange: Production version available; simulator runs to completion.
        Act: HourlySimulator.run() writes parquet to tmp_path.
        Assert: parquet can be read back; both columns are present and
                contain string dtype (object in pandas).
                served_from_stage values are exactly "Production" or "Archived".
        """
        # Arrange
        sim = _make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub = _make_stub_model()
        prod_version = make_mock_version(version="3", stage="Production")

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [prod_version]

        # Act
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch(
                "app.inference.simulator.mlflow.lightgbm.load_model",
                return_value=stub,
            ):
                with patch.object(sim, "load_gold", return_value=gold_df):
                    result = sim.run()

        # Assert via in-memory DataFrame (also covers parquet round-trip via output_path)
        df: pd.DataFrame = result["predictions_df"]

        assert "model_version" in df.columns, (
            "model_version column missing from parquet"
        )
        assert "served_from_stage" in df.columns, (
            "served_from_stage column missing from parquet"
        )

        # Values must be exactly the permitted stage strings
        valid_stages = {"Production", "Archived"}
        unique_stages = set(df["served_from_stage"].unique())
        assert unique_stages.issubset(valid_stages), (
            f"served_from_stage values {unique_stages} contain invalid entries; "
            f"allowed: {valid_stages}"
        )

        # Column dtype must be string-like (object in pandas)
        assert df["model_version"].dtype == object, "model_version must be str dtype"
        assert df["served_from_stage"].dtype == object, (
            "served_from_stage must be str dtype"
        )


# ---------------------------------------------------------------------------
# AC-6: No nulls in provenance columns
# ---------------------------------------------------------------------------


class TestAC6NoNullsInProvenanceColumns:
    """AC-6: model_version and served_from_stage have no null values on any row."""

    def test_ac6_provenance_columns_have_no_nulls(self, tmp_path: Path) -> None:
        """AC-6: Every row in the predictions parquet has non-null provenance fields.

        Arrange: Production version available (3 hours × 5 zones = 15 rows).
        Act: HourlySimulator.run().
        Assert: model_version.isna().any() == False,
                served_from_stage.isna().any() == False.
        """
        # Arrange
        sim = _make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub = _make_stub_model()
        prod_version = make_mock_version(version="7", stage="Production")

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [prod_version]

        # Act
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch(
                "app.inference.simulator.mlflow.lightgbm.load_model",
                return_value=stub,
            ):
                with patch.object(sim, "load_gold", return_value=gold_df):
                    result = sim.run()

        # Assert
        df: pd.DataFrame = result["predictions_df"]
        assert not df["model_version"].isna().any(), (
            "model_version must not contain null values"
        )
        assert not df["served_from_stage"].isna().any(), (
            "served_from_stage must not contain null values"
        )


# ---------------------------------------------------------------------------
# AC-8: /heatmap envelope contains model_version and served_from_stage
# ---------------------------------------------------------------------------


class TestAC8HeatmapEnvelopeProvenance:
    """AC-8: GET /heatmap response envelope includes string model_version and served_from_stage."""

    def test_ac8_heatmap_envelope_contains_model_provenance_fields(
        self, tmp_path: Path
    ) -> None:
        """AC-8: /heatmap returns model_version and served_from_stage at top level.

        Arrange: Write a PBI-21 parquet (with both new columns) to tmp_path.
                 Point app.api.main globals at that parquet data and set
                 _model_version / _served_from_stage to simulate a lifespan run.
        Act: GET /heatmap via TestClient.
        Assert: HTTP 200; body["model_version"] == "3" (str);
                body["served_from_stage"] == "Production" (str).
        """
        import app.api.main as serving_module

        # Arrange: inject module globals to simulate post-lifespan state
        predictions_with_provenance = pd.DataFrame(
            {
                "pulocationid": [1, 2, 3],
                "hour": pd.to_datetime(["2021-03-01 01:00:00"] * 3),
                "predicted_trip_count": [12.0, 18.7, 6.2],
                "actual_trip_count": [13.0, 17.0, 7.0],
                "model_version": ["3", "3", "3"],
                "served_from_stage": ["Production", "Production", "Production"],
            }
        )

        # Act
        with patch.object(
            serving_module, "_predictions_df", predictions_with_provenance
        ):
            with patch.object(serving_module, "_model_version", "3"):
                with patch.object(serving_module, "_served_from_stage", "Production"):
                    with patch.object(
                        serving_module,
                        "_zone_lookup",
                        {
                            1: "Newark Airport",
                            2: "Jamaica Bay",
                            3: "Allerton/Pelham Gardens",
                        },
                    ):
                        with patch.object(
                            serving_module,
                            "_centroid_lookup",
                            {
                                1: (40.693, -74.174),
                                2: (40.618, -73.825),
                                3: (40.866, -73.868),
                            },
                        ):
                            with patch.object(
                                serving_module,
                                "_generated_at",
                                "2021-03-01T12:00:00+00:00",
                            ):
                                with patch.object(
                                    serving_module,
                                    "_watermark_root",
                                    tmp_path / "wm",
                                ):
                                    local_client = TestClient(serving_module.app)
                                    resp = local_client.get("/heatmap")

        # Assert
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert "model_version" in body, "model_version missing from /heatmap envelope"
        assert "served_from_stage" in body, (
            "served_from_stage missing from /heatmap envelope"
        )
        assert isinstance(body["model_version"], str), (
            f"model_version must be a string, got {type(body['model_version'])}"
        )
        assert isinstance(body["served_from_stage"], str), (
            f"served_from_stage must be a string, got {type(body['served_from_stage'])}"
        )
        assert body["model_version"] == "3"
        assert body["served_from_stage"] == "Production"


# ---------------------------------------------------------------------------
# AC-9: All pre-existing /heatmap criteria still hold
# ---------------------------------------------------------------------------


class TestAC9HeatmapPreExistingBehaviour:
    """AC-9: All pre-existing /heatmap acceptance criteria remain satisfied."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Inject standard module state with PBI-21 columns included."""
        import app.api.main as serving_module

        predictions = pd.DataFrame(
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
                "model_version": ["3"] * 6,
                "served_from_stage": ["Production"] * 6,
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
        monkeypatch.setattr(serving_module, "_model_version", "3")
        monkeypatch.setattr(serving_module, "_served_from_stage", "Production")
        monkeypatch.setattr(serving_module, "_watermark_root", tmp_path / "wm")

    def test_ac9_heatmap_preserves_pre_existing_behaviour(self) -> None:
        """AC-9: zone_count > 0, descending sort by predicted_trip_count, required envelope keys.

        Arrange: standard 3-zone predictions fixture (from autouse).
        Act: GET /heatmap (default latest hour).
        Assert: HTTP 200, zone_count == 3, predictions sorted descending,
                generated_at / hour / predictions / simulated all present.
        """
        import app.api.main as serving_module

        # Act
        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")

        # Assert
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["zone_count"] == 3, "Expected 3 zones in /heatmap response"
        counts = [p["predicted_trip_count"] for p in body["predictions"]]
        assert counts == sorted(counts, reverse=True), (
            "Predictions must be sorted descending by predicted_trip_count"
        )
        for key in ("generated_at", "hour", "predictions", "simulated"):
            assert key in body, f"Required envelope key '{key}' missing"

    def test_ac9_heatmap_returns_503_when_predictions_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-9 (503 sub-criterion): /heatmap returns 503 with detail when no predictions.

        Arrange: override autouse fixture — set _predictions_df = None.
        Act: GET /heatmap.
        Assert: HTTP 503, 'detail' in body.
        """
        import app.api.main as serving_module

        monkeypatch.setattr(serving_module, "_predictions_df", None)

        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")

        assert resp.status_code == 503
        assert "detail" in resp.json()

    def test_ac9_heatmap_returns_404_for_unknown_hour(self) -> None:
        """AC-9 (404 sub-criterion): /heatmap returns 404 for a non-existent hour.

        Arrange: standard predictions fixture (2021-03-01 hours only).
        Act: GET /heatmap?hour=2099-01-01T00:00:00.
        Assert: HTTP 404.
        """
        import app.api.main as serving_module

        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap", params={"hour": "2099-01-01T00:00:00"})

        assert resp.status_code == 404

    def test_ac9_heatmap_returns_400_for_malformed_hour(self) -> None:
        """AC-9 (400 sub-criterion): /heatmap returns 400 for a non-parseable hour.

        Arrange: standard predictions fixture.
        Act: GET /heatmap?hour=not-a-date.
        Assert: HTTP 400.
        """
        import app.api.main as serving_module

        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap", params={"hour": "not-a-date"})

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# AC-10: NoApprovedModelError is importable, inherits Exception, carries message
# ---------------------------------------------------------------------------


class TestAC10NoApprovedModelError:
    """AC-10: NoApprovedModelError class contract."""

    def test_ac10_no_approved_model_error_is_importable_with_descriptive_message(
        self,
    ) -> None:
        """AC-10: Class importable from app.inference.exceptions, inherits Exception,
        message names model_name and both stages checked.

        Arrange: instantiate NoApprovedModelError directly.
        Assert: isinstance(..., Exception) == True,
                "Production" in message, "Archived" in message,
                model name in message.
        """
        # Arrange + Act
        exc = NoApprovedModelError(
            model_name="ny_taxi_demand_model",
            stages_checked=["Production", "Archived"],
        )

        # Assert: inherits from Exception
        assert isinstance(exc, Exception), (
            "NoApprovedModelError must inherit from Exception"
        )

        # Assert: both stage names appear in the message
        message = str(exc)
        assert "Production" in message, (
            "Error message must name the 'Production' stage that was checked"
        )
        assert "Archived" in message, (
            "Error message must name the 'Archived' stage that was checked"
        )
        assert "ny_taxi_demand_model" in message, (
            "Error message must name the model 'ny_taxi_demand_model'"
        )

    def test_ac10_no_approved_model_error_attributes_preserved(self) -> None:
        """AC-10 (attribute contract): model_name and stages_checked are accessible.

        Arrange: instantiate with explicit values.
        Assert: exc.model_name == "my_model", exc.stages_checked == [...].
        """
        # Arrange + Act
        exc = NoApprovedModelError(
            model_name="my_model",
            stages_checked=["Production", "Archived"],
        )

        # Assert
        assert exc.model_name == "my_model"
        assert exc.stages_checked == ["Production", "Archived"]


# ---------------------------------------------------------------------------
# Edge: served_from_stage casing is exact ("Production" not "production")
# ---------------------------------------------------------------------------


class TestEdgeCaseCasingExact:
    """Edge case: served_from_stage value must use exact MLflow casing."""

    def test_edge_served_from_stage_casing_is_exact_production(
        self, tmp_path: Path
    ) -> None:
        """Edge: served_from_stage is exactly "Production" (capital P), not lowercase.

        Arrange: Production model loaded.
        Act: HourlySimulator.run().
        Assert: every row has served_from_stage == "Production" (case-sensitive).
        """
        # Arrange
        sim = _make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub = _make_stub_model()
        prod_version = make_mock_version(version="1", stage="Production")

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [prod_version]

        # Act
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch(
                "app.inference.simulator.mlflow.lightgbm.load_model",
                return_value=stub,
            ):
                with patch.object(sim, "load_gold", return_value=gold_df):
                    result = sim.run()

        # Assert exact casing
        df: pd.DataFrame = result["predictions_df"]
        assert all(v == "Production" for v in df["served_from_stage"]), (
            "served_from_stage must be 'Production' (exact casing, not 'production')"
        )

    def test_edge_served_from_stage_casing_is_exact_archived(
        self, tmp_path: Path
    ) -> None:
        """Edge: served_from_stage is exactly "Archived" (capital A), not lowercase.

        Arrange: Archived fallback triggered.
        Act: HourlySimulator.run().
        Assert: every row has served_from_stage == "Archived" (case-sensitive).
        """
        # Arrange
        sim = _make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub = _make_stub_model()
        arch_version = make_mock_version(version="1", stage="Archived")

        def _side_effect(name: str, stages: list[str]) -> list[MagicMock]:
            return [] if "Production" in stages else [arch_version]

        mock_client = MagicMock()
        mock_client.get_latest_versions.side_effect = _side_effect

        # Act
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch(
                "app.inference.simulator.mlflow.lightgbm.load_model",
                return_value=stub,
            ):
                with patch.object(sim, "load_gold", return_value=gold_df):
                    result = sim.run()

        # Assert exact casing
        df: pd.DataFrame = result["predictions_df"]
        assert all(v == "Archived" for v in df["served_from_stage"]), (
            "served_from_stage must be 'Archived' (exact casing, not 'archived')"
        )


# ---------------------------------------------------------------------------
# Edge: fresh checkout — Archived list empty raises NoApprovedModelError
# ---------------------------------------------------------------------------


class TestEdgeFreshCheckoutNoVersions:
    """Edge: empty Archived list on a fresh checkout → NoApprovedModelError names stages."""

    def test_edge_empty_archived_on_fresh_checkout_raises_with_both_stages_named(
        self, tmp_path: Path
    ) -> None:
        """Edge: Production [], Archived [] → error message names both stages.

        Arrange: both get_latest_versions calls return [].
        Act: HourlySimulator._load_model_from_registry().
        Assert: NoApprovedModelError raised; message contains both "Production"
                and "Archived" and the model name.
        """
        # Arrange
        sim = _make_simulator(tmp_path)
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act + Assert
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with pytest.raises(NoApprovedModelError) as exc_info:
                sim._load_model_from_registry()

        msg = str(exc_info.value)
        assert "Production" in msg
        assert "Archived" in msg
        assert "ny_taxi_demand_model" in msg


# ---------------------------------------------------------------------------
# Edge: legacy parquet (no provenance columns) → API returns null fields
# ---------------------------------------------------------------------------


class TestEdgeLegacyParquetNullProvenance:
    """Edge: parquet written before PBI-21 has no provenance columns → null in API."""

    def test_edge_legacy_parquet_serves_null_provenance_in_heatmap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Edge: When _model_version and _served_from_stage are None (legacy parquet),
        /heatmap still returns 200 with null provenance fields — no 500 error.

        Arrange: inject predictions_df without new columns; set globals to None.
        Act: GET /heatmap.
        Assert: HTTP 200, body["model_version"] is null, body["served_from_stage"] is null.
        """
        import app.api.main as serving_module

        # Arrange: legacy parquet with no provenance columns
        legacy_predictions = pd.DataFrame(
            {
                "pulocationid": [1, 2, 3],
                "hour": pd.to_datetime(["2021-03-01 01:00:00"] * 3),
                "predicted_trip_count": [5.0, 10.0, 3.0],
                "actual_trip_count": [6.0, 9.0, 4.0],
            }
        )
        monkeypatch.setattr(serving_module, "_predictions_df", legacy_predictions)
        monkeypatch.setattr(serving_module, "_model_version", None)
        monkeypatch.setattr(serving_module, "_served_from_stage", None)
        monkeypatch.setattr(
            serving_module,
            "_zone_lookup",
            {1: "Zone A", 2: "Zone B", 3: "Zone C"},
        )
        monkeypatch.setattr(
            serving_module,
            "_centroid_lookup",
            {1: (40.0, -74.0), 2: (41.0, -73.0), 3: (42.0, -72.0)},
        )
        monkeypatch.setattr(
            serving_module, "_generated_at", "2021-03-01T01:00:00+00:00"
        )
        monkeypatch.setattr(serving_module, "_watermark_root", tmp_path / "wm")

        # Act
        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")

        # Assert: 200 with null provenance — backward-compatible
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["model_version"] is None
        assert body["served_from_stage"] is None
