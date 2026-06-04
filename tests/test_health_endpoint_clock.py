"""Tests for the /health endpoint clock transparency extension (PBI 25b AC #4, #5, #6).

Verifies that:
- DEMO_MODE unset  → simulated=False, no simulated_now/wall_clock_now fields (AC #4)
- DEMO_MODE=1      → simulated=True, both clock fields present as ISO-8601 strings (AC #5)
- Two calls ~2s apart under DEMO_MODE=1 show simulated_now advancing ~15 min (AC #6)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.clock import SimulatedClock, WallClock, set_active_clock
from orchestration.dags._clock_factory import _SpeedupClock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Known busy Monday used by make_clock() as the demo start time.
_T0 = datetime(2024, 6, 3, 0, 0, tzinfo=timezone.utc)

_FAKE_PREDICTIONS = pd.DataFrame(
    {
        "pulocationid": [1],
        "hour": pd.to_datetime(["2024-06-03 00:00:00"]),
        "predicted_trip_count": [10.0],
        "actual_trip_count": [9.0],
    }
)

_FAKE_CONFIG: dict[str, Any] = {
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
def _reset_active_clock() -> Any:
    """Reset the process-global active clock after each test.

    Prevents leakage between tests that call set_active_clock().
    """
    yield
    set_active_clock(WallClock())


@pytest.fixture()
def _predictions_patched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch _predictions_df and MlflowClient for all health endpoint tests."""
    import app.api.main as serving_module

    monkeypatch.setattr(serving_module, "_predictions_df", _FAKE_PREDICTIONS)


# ---------------------------------------------------------------------------
# TestHealthClockFields
# ---------------------------------------------------------------------------


class TestHealthClockFields:
    """AC #4, #5, #6 — /health clock transparency payload tests."""

    def test_health_reports_simulated_false_when_demo_mode_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _predictions_patched: None,
    ) -> None:
        """AC #4: DEMO_MODE unset → simulated=False, no simulated_now or wall_clock_now.

        Arrange: _predictions_df patched; DEMO_MODE removed from env; MlflowClient mocked.
        Act: GET /health.
        Assert: simulated is False, simulated_now absent, wall_clock_now absent.
        """
        import app.api.main as serving_module

        # Arrange
        monkeypatch.delenv("DEMO_MODE", raising=False)
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
            with patch("app.api.main.MlflowClient", return_value=mock_client):
                resp = TestClient(serving_module.app).get("/health")

        # Assert
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["simulated"] is False
        assert "simulated_now" not in body
        assert "wall_clock_now" not in body

    def test_health_reports_simulated_true_with_both_clocks_when_demo_mode_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _predictions_patched: None,
    ) -> None:
        """AC #5: DEMO_MODE=1 → simulated=True, both clock fields present and different.

        Arrange: DEMO_MODE=1; SimulatedClock(start=T0) registered via set_active_clock;
                 _predictions_df and MlflowClient mocked.
        Act: GET /health.
        Assert: simulated is True, simulated_now == T0.isoformat(), wall_clock_now is
                parseable ISO-8601 and differs from simulated_now by > 1 year (T0 is 2024,
                wall clock is 2026).
        """
        import app.api.main as serving_module

        # Arrange
        monkeypatch.setenv("DEMO_MODE", "1")
        sim_clock = SimulatedClock(start=_T0)
        set_active_clock(sim_clock)
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act
        with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
            with patch("app.api.main.MlflowClient", return_value=mock_client):
                resp = TestClient(serving_module.app).get("/health")

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["simulated"] is True

        # simulated_now equals the SimulatedClock start
        assert body["simulated_now"] == _T0.isoformat()

        # wall_clock_now is parseable ISO-8601
        wall_now = datetime.fromisoformat(body["wall_clock_now"])
        assert wall_now.tzinfo is not None  # must be tz-aware

        # The two times are visibly different: T0 is 2024, wall clock is ~2026
        sim_dt = datetime.fromisoformat(body["simulated_now"])
        diff_years = (wall_now - sim_dt).total_seconds() / (365.25 * 24 * 3600)
        assert diff_years > 1.0, (
            f"Expected > 1 year gap between simulated_now ({sim_dt}) "
            f"and wall_clock_now ({wall_now}), got {diff_years:.2f} years"
        )

    def test_health_simulated_now_advances_between_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _predictions_patched: None,
    ) -> None:
        """AC #6: simulated_now advances ~15 min between two calls ~2s apart (DEMO_MODE=1).

        Arrange: DEMO_MODE=1; _SpeedupClock(start=T0, speedup=450) registered.
                 Two GET /health calls separated by 2s of wall-clock time.
        Act: two /health calls with time.sleep(2) between them.
        Assert: delta between the two simulated_now values is ~900s (15 min) ±10%.
        """
        import app.api.main as serving_module

        # Arrange
        monkeypatch.setenv("DEMO_MODE", "1")
        speedup_clock = _SpeedupClock(start=_T0, speedup=450)
        set_active_clock(speedup_clock)
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        local_client = TestClient(serving_module.app)

        try:
            with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
                with patch("app.api.main.MlflowClient", return_value=mock_client):
                    # Act: first call
                    resp1 = local_client.get("/health")
                    assert resp1.status_code == 200
                    t1 = datetime.fromisoformat(resp1.json()["simulated_now"])

                    # Wall-clock sleep
                    time.sleep(2.0)

                    # Act: second call
                    resp2 = local_client.get("/health")
                    assert resp2.status_code == 200
                    t2 = datetime.fromisoformat(resp2.json()["simulated_now"])
        finally:
            speedup_clock.stop()

        # Assert: delta ≈ 15 min (900 s) within ±10%
        delta_seconds = (t2 - t1).total_seconds()
        expected_seconds = 2.0 * 450  # 900 s = 15 min
        tolerance = expected_seconds * 0.10
        assert abs(delta_seconds - expected_seconds) < tolerance, (
            f"Expected simulated_now delta ~{expected_seconds}s, "
            f"got {delta_seconds:.1f}s (tolerance ±{tolerance:.0f}s)"
        )
