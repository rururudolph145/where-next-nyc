"""Tests for the /health simulated-position fields (PBI 25b FIX, AC #4, #5, #6).

After the PBI 25b FIX, the simulation position is the incremental pipeline's
DATA WATERMARK, not a clock. /health therefore reports:
- ``simulated`` — driven by the DEMO_MODE env var (AC #4).
- ``simulated_now`` (DEMO_MODE=1 only) — the current watermark position, or the
  configured ``ingest.seed_start`` when the watermark file is absent (AC #5).
- ``wall_clock_now`` (DEMO_MODE=1 only) — genuine real time (AC #5).
- ``data_publication_lag_days`` — lag of predictions vs the watermark, naive
  date arithmetic (AC #6).

All timestamps on the simulated timeline are tz-NAIVE NYC wall-clock values.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Naive NYC wall-clock seed (matches config.yml convention: ISO, no 'Z').
_SEED_START = "2021-01-18T00:00:00"

# A watermark position later than the seed (the pipeline has advanced).
_WATERMARK_POSITION = "2021-02-05T12:30:00"

_FAKE_PREDICTIONS = pd.DataFrame(
    {
        "pulocationid": [1],
        "hour": pd.to_datetime(["2021-02-05 12:00:00"]),
        "predicted_trip_count": [10.0],
        "actual_trip_count": [9.0],
    }
)

_FAKE_CONFIG: dict[str, Any] = {
    "ingest": {"seed_start": _SEED_START},
    "serving": {
        "predictions_path": "fake/predictions.parquet",
        "zone_shapefile_path": "fake/taxi_zones.shp",
    },
    "mlflow": {"model_name": "ny_taxi_demand_model"},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patched_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Patch _predictions_df, _config, and an isolated watermark root.

    Yields the (empty) watermark root so individual tests can write a control
    file to exercise the watermark-present branch.
    """
    import app.api.main as serving_module

    watermark_root = tmp_path / "watermarks" / "refresh_15min"
    monkeypatch.setattr(serving_module, "_predictions_df", _FAKE_PREDICTIONS)
    monkeypatch.setattr(serving_module, "_config", _FAKE_CONFIG)
    monkeypatch.setattr(serving_module, "_watermark_root", watermark_root)
    yield watermark_root


def _write_watermark(root: Path, position: str) -> None:
    """Write a watermark control file at *root* recording *position*."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".current_watermark").write_text(
        json.dumps({"position": position}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# TestHealthSimulatedFields
# ---------------------------------------------------------------------------


class TestHealthSimulatedFields:
    """AC #4, #5, #6 — /health simulated-position payload tests."""

    def test_health_reports_simulated_false_when_demo_mode_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC #4: DEMO_MODE unset → simulated=False, no simulated_now/wall_clock_now.

        Arrange: DEMO_MODE removed from env; MlflowClient mocked.
        Act: GET /health.
        Assert: simulated is False, simulated_now absent, wall_clock_now absent.
        """
        import app.api.main as serving_module

        # Arrange
        monkeypatch.delenv("DEMO_MODE", raising=False)
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.MlflowClient", return_value=mock_client):
            resp = TestClient(serving_module.app).get("/health")

        # Assert
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["simulated"] is False
        assert "simulated_now" not in body
        assert "wall_clock_now" not in body

    def test_health_simulated_now_falls_back_to_seed_when_no_watermark(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC #5: DEMO_MODE=1, no watermark file → simulated_now == seed_start.

        Arrange: DEMO_MODE=1; watermark root empty; MlflowClient mocked.
        Act: GET /health.
        Assert: simulated True; simulated_now == seed_start; wall_clock_now is
                tz-aware ISO-8601 and differs from the (2021) seed by > 1 year.
        """
        import app.api.main as serving_module

        # Arrange
        monkeypatch.setenv("DEMO_MODE", "1")
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.MlflowClient", return_value=mock_client):
            resp = TestClient(serving_module.app).get("/health")

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["simulated"] is True
        assert body["simulated_now"] == datetime.fromisoformat(_SEED_START).isoformat()

        wall_now = datetime.fromisoformat(body["wall_clock_now"])
        assert wall_now.tzinfo is not None  # genuine real time, tz-aware

        sim_dt = datetime.fromisoformat(body["simulated_now"])
        assert sim_dt.tzinfo is None  # simulated timeline is naive
        diff_years = (wall_now.replace(tzinfo=None) - sim_dt).total_seconds() / (
            365.25 * 24 * 3600
        )
        assert diff_years > 1.0, (
            f"Expected > 1 year gap between simulated_now ({sim_dt}) "
            f"and wall_clock_now ({wall_now}), got {diff_years:.2f} years"
        )

    def test_health_simulated_now_reflects_watermark_position(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_state: Path,
    ) -> None:
        """AC #5: DEMO_MODE=1 with a watermark file → simulated_now == watermark.

        Arrange: DEMO_MODE=1; watermark control file written at a position later
                 than the seed; MlflowClient mocked.
        Act: GET /health.
        Assert: simulated_now equals the watermark position (not the seed).
        """
        import app.api.main as serving_module

        # Arrange
        monkeypatch.setenv("DEMO_MODE", "1")
        _write_watermark(_patched_state, _WATERMARK_POSITION)
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.MlflowClient", return_value=mock_client):
            resp = TestClient(serving_module.app).get("/health")

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        expected = datetime.fromisoformat(_WATERMARK_POSITION).isoformat()
        assert body["simulated_now"] == expected

    def test_health_lag_computed_against_watermark(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_state: Path,
    ) -> None:
        """AC #6: data_publication_lag_days uses the watermark, naive arithmetic.

        Arrange: watermark at 2021-02-05; last prediction bucket at 2021-02-05
                 → lag 0. MlflowClient mocked.
        Act: GET /health.
        Assert: data_publication_lag_days == 0 (watermark date == bucket date).
        """
        import app.api.main as serving_module

        # Arrange
        monkeypatch.delenv("DEMO_MODE", raising=False)
        _write_watermark(_patched_state, _WATERMARK_POSITION)
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.MlflowClient", return_value=mock_client):
            resp = TestClient(serving_module.app).get("/health")

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["data_publication_lag_days"] == 0

    def test_health_degrades_gracefully_on_malformed_watermark(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_state: Path,
    ) -> None:
        """AC #6 (edge): malformed watermark → /health returns 200, lag None.

        Arrange: write a corrupt watermark control file; DEMO_MODE=1.
        Act: GET /health.
        Assert: 200; data_publication_lag_days is None; simulated_now is None;
                /health never raises 5xx.
        """
        import app.api.main as serving_module

        # Arrange
        monkeypatch.setenv("DEMO_MODE", "1")
        _patched_state.mkdir(parents=True, exist_ok=True)
        (_patched_state / ".current_watermark").write_text(
            "not valid json", encoding="utf-8"
        )
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.MlflowClient", return_value=mock_client):
            resp = TestClient(serving_module.app).get("/health")

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["data_publication_lag_days"] is None
        assert body["simulated_now"] is None
