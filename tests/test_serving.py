"""Tests for the FastAPI heatmap serving endpoint (PBI 3)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def sample_predictions() -> pd.DataFrame:
    """Create a small predictions DataFrame for testing."""
    return pd.DataFrame(
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
    )


@pytest.fixture()
def zone_lookup() -> dict[int, str]:
    """Create a small zone ID → name mapping."""
    return {1: "Newark Airport", 2: "Jamaica Bay", 3: "Allerton/Pelham Gardens"}


@pytest.fixture()
def client(
    sample_predictions: pd.DataFrame, zone_lookup: dict[int, str]
) -> TestClient:
    """Create a test client with mocked predictions and zone lookup."""
    import src.serving.app as serving_module

    serving_module._predictions_df = sample_predictions
    serving_module._zone_lookup = zone_lookup
    serving_module._generated_at = "2021-03-01T12:00:00+00:00"

    return TestClient(serving_module.app)


class TestHealthCheck:
    """Tests for the root health check endpoint."""

    def test_health_check_returns_ok(self, client: TestClient) -> None:
        """GET / returns 200 with health_check OK."""
        resp = client.get("/")
        assert resp.status_code == 200
        body: dict[str, object] = resp.json()
        assert body["health_check"] == "OK"


class TestHeatmapEndpoint:
    """Tests for GET /heatmap."""

    def test_heatmap_default_returns_latest_hour(self, client: TestClient) -> None:
        """GET /heatmap without params returns the latest available hour."""
        resp = client.get("/heatmap")
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["hour"] == "2021-03-01 01:00:00"
        assert body["zone_count"] == 3
        assert len(body["predictions"]) == 3

    def test_heatmap_contains_required_fields(self, client: TestClient) -> None:
        """Response contains generated_at, hour, zone_count, predictions."""
        resp = client.get("/heatmap")
        body: dict[str, Any] = resp.json()
        assert "generated_at" in body
        assert "hour" in body
        assert "zone_count" in body
        assert "predictions" in body

    def test_heatmap_prediction_schema(self, client: TestClient) -> None:
        """Each prediction has pulocationid, zone_name, predicted_trip_count."""
        resp = client.get("/heatmap")
        body: dict[str, Any] = resp.json()
        for pred in body["predictions"]:
            assert "pulocationid" in pred
            assert "zone_name" in pred
            assert "predicted_trip_count" in pred
            assert isinstance(pred["pulocationid"], int)
            assert isinstance(pred["zone_name"], str)
            assert isinstance(pred["predicted_trip_count"], float)

    def test_heatmap_sorted_descending(self, client: TestClient) -> None:
        """Predictions are sorted by predicted_trip_count descending."""
        resp = client.get("/heatmap")
        body: dict[str, Any] = resp.json()
        counts = [p["predicted_trip_count"] for p in body["predictions"]]
        assert counts == sorted(counts, reverse=True)

    def test_heatmap_filter_by_hour(self, client: TestClient) -> None:
        """GET /heatmap?hour=... filters to the requested hour."""
        resp = client.get("/heatmap", params={"hour": "2021-03-01T00:00:00"})
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["hour"] == "2021-03-01 00:00:00"
        assert body["zone_count"] == 3

    def test_heatmap_invalid_hour_returns_404(self, client: TestClient) -> None:
        """GET /heatmap?hour=<nonexistent> returns 404."""
        resp = client.get("/heatmap", params={"hour": "2099-01-01T00:00:00"})
        assert resp.status_code == 404

    def test_heatmap_bad_hour_format_returns_400(self, client: TestClient) -> None:
        """GET /heatmap?hour=<garbage> returns 400."""
        resp = client.get("/heatmap", params={"hour": "not-a-date"})
        assert resp.status_code == 400

    def test_heatmap_zone_names_resolved(self, client: TestClient) -> None:
        """Zone names are resolved from the lookup."""
        resp = client.get("/heatmap")
        body: dict[str, Any] = resp.json()
        names = {p["pulocationid"]: p["zone_name"] for p in body["predictions"]}
        assert names[1] == "Newark Airport"
        assert names[2] == "Jamaica Bay"
        assert names[3] == "Allerton/Pelham Gardens"


class TestHeatmapNoData:
    """Tests for GET /heatmap when predictions are not loaded."""

    def test_heatmap_returns_503_when_no_predictions(self) -> None:
        """GET /heatmap returns 503 if predictions are not loaded."""
        import src.serving.app as serving_module

        serving_module._predictions_df = None
        serving_module._zone_lookup = {}
        serving_module._generated_at = ""

        client = TestClient(serving_module.app)
        resp = client.get("/heatmap")
        assert resp.status_code == 503
