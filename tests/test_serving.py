"""Tests for the FastAPI heatmap serving endpoint (PBI 3)."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_mock_version


@pytest.fixture()
def sample_predictions() -> pd.DataFrame:
    """Create a small predictions DataFrame for testing.

    Includes model_version and served_from_stage columns added by PBI 21.
    """
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
            "model_version": ["3", "3", "3", "3", "3", "3"],
            "served_from_stage": [
                "Production",
                "Production",
                "Production",
                "Production",
                "Production",
                "Production",
            ],
        }
    )


@pytest.fixture()
def zone_lookup() -> dict[int, str]:
    """Create a small zone ID → name mapping."""
    return {1: "Newark Airport", 2: "Jamaica Bay", 3: "Allerton/Pelham Gardens"}


@pytest.fixture()
def centroid_lookup() -> dict[int, tuple[float, float]]:
    """Create a small zone ID → (lat, lon) mapping with 3 d.p. values."""
    return {
        1: (40.693, -74.174),
        2: (40.618, -73.825),
        3: (40.866, -73.868),
    }


@pytest.fixture()
def client(
    sample_predictions: pd.DataFrame,
    zone_lookup: dict[int, str],
    centroid_lookup: dict[int, tuple[float, float]],
    tmp_path: Path,
) -> Generator[TestClient, None, None]:
    """Create a test client with mocked predictions, zone lookup, and centroid lookup.

    Restores all mutated module globals on teardown so tests are isolated.
    _config is redirected to tmp_path so any prediction-log writes land in a
    safe temp directory rather than the repo's data/ tree.
    """
    import app.api.main as serving_module

    orig_predictions = serving_module._predictions_df
    orig_zone_lookup = serving_module._zone_lookup
    orig_centroid_lookup = serving_module._centroid_lookup
    orig_generated_at = serving_module._generated_at
    orig_model_version = serving_module._model_version
    orig_served_from_stage = serving_module._served_from_stage
    orig_config = serving_module._config
    orig_watermark_root = serving_module._watermark_root

    serving_module._predictions_df = sample_predictions
    serving_module._zone_lookup = zone_lookup
    serving_module._centroid_lookup = centroid_lookup
    serving_module._generated_at = "2021-03-01T12:00:00+00:00"
    serving_module._model_version = "3"
    serving_module._served_from_stage = "Production"
    # Redirect log writes to tmp_path so no real disk I/O hits data/ tree.
    serving_module._config = {
        "monitoring": {"predictions_path": str(tmp_path / "monitoring_predictions")}
    }
    # Point watermark root at an empty tmp dir so read_watermark returns None
    # and the no-hour branch falls back to latest-hour (no watermark cap).
    serving_module._watermark_root = tmp_path / "watermarks" / "refresh_15min"

    yield TestClient(serving_module.app)

    serving_module._predictions_df = orig_predictions
    serving_module._zone_lookup = orig_zone_lookup
    serving_module._centroid_lookup = orig_centroid_lookup
    serving_module._generated_at = orig_generated_at
    serving_module._model_version = orig_model_version
    serving_module._served_from_stage = orig_served_from_stage
    serving_module._config = orig_config
    serving_module._watermark_root = orig_watermark_root


class TestHealthCheck:
    """Tests for the root health check endpoint."""

    def test_health_check_returns_ok(self, client: TestClient) -> None:
        """GET / returns 200 with status alive."""
        resp = client.get("/")
        assert resp.status_code == 200
        body: dict[str, object] = resp.json()
        assert body["status"] == "alive"


class TestHeatmapEndpoint:
    """Tests for GET /heatmap."""

    def test_heatmap_default_returns_latest_hour(self, client: TestClient) -> None:
        """GET /heatmap without params returns the latest available hour."""
        resp = client.get("/heatmap")
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["hour"] == "2021-03-01T01:00:00"
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
        assert body["hour"] == "2021-03-01T00:00:00"
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

    def test_heatmap_returns_503_when_no_predictions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /heatmap returns 503 if predictions are not loaded."""
        import app.api.main as serving_module

        monkeypatch.setattr(serving_module, "_predictions_df", None)
        monkeypatch.setattr(serving_module, "_zone_lookup", {})
        monkeypatch.setattr(serving_module, "_centroid_lookup", {})
        monkeypatch.setattr(serving_module, "_generated_at", "")

        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")
        assert resp.status_code == 503


class TestHeatmapEnrichedFields:
    """Tests for enriched prediction fields: latitude, longitude, bucket_time (AC #1–#8)."""

    def test_get_heatmap_prediction_schema_includes_lat_lon_bucket_time(
        self, client: TestClient
    ) -> None:
        """Every prediction object must contain latitude, longitude, and bucket_time.

        AC #1: all six required fields present on each prediction.
        """
        resp = client.get("/heatmap")
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        for pred in body["predictions"]:
            assert "pulocationid" in pred
            assert "zone_name" in pred
            assert "predicted_trip_count" in pred
            assert "latitude" in pred
            assert "longitude" in pred
            assert "bucket_time" in pred
            assert isinstance(pred["pulocationid"], int)
            assert isinstance(pred["zone_name"], str)
            assert isinstance(pred["predicted_trip_count"], float)

    def test_get_heatmap_lat_lon_rounded_to_3dp(self, client: TestClient) -> None:
        """latitude and longitude must be rounded to exactly 3 decimal places.

        AC #3: values match the centroid_lookup fixture which already holds
        3 d.p. values; the assertion verifies no extra precision is added.
        """
        resp = client.get("/heatmap")
        body: dict[str, Any] = resp.json()
        for pred in body["predictions"]:
            if pred["latitude"] is not None:
                assert pred["latitude"] == round(pred["latitude"], 3)
            if pred["longitude"] is not None:
                assert pred["longitude"] == round(pred["longitude"], 3)

    def test_get_heatmap_bucket_time_equals_hour_iso_formatted(
        self, client: TestClient
    ) -> None:
        """bucket_time must equal the envelope ``hour`` formatted as ISO-8601.

        AC #4: for the default (latest) hour, every prediction's bucket_time
        must equal the envelope hour string as an ISO timestamp.
        """
        resp = client.get("/heatmap")
        body: dict[str, Any] = resp.json()
        envelope_hour = body["hour"]
        expected_bucket_time = pd.Timestamp(envelope_hour).isoformat()
        for pred in body["predictions"]:
            assert pred["bucket_time"] == expected_bucket_time

    def test_get_heatmap_missing_centroid_zone_returns_null_lat_lon(
        self,
        client: TestClient,
        centroid_lookup: dict[int, tuple[float, float]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A zone absent from _centroid_lookup must have null lat/lon.

        AC #7: zone 99 has no centroid entry → latitude and longitude are None;
        zone_name falls back to "Unknown".
        """
        import app.api.main as serving_module

        # Override with a prediction set that includes zone 99 (no centroid).
        predictions_with_unknown = pd.DataFrame(
            {
                "pulocationid": [1, 99],
                "hour": pd.to_datetime(["2021-03-01 01:00:00", "2021-03-01 01:00:00"]),
                "predicted_trip_count": [12.0, 3.0],
                "actual_trip_count": [13.0, 2.0],
            }
        )
        monkeypatch.setattr(serving_module, "_predictions_df", predictions_with_unknown)
        # centroid_lookup has no entry for zone 99 — reuse fixture as-is
        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()

        preds_by_zone = {p["pulocationid"]: p for p in body["predictions"]}
        assert preds_by_zone[99]["latitude"] is None
        assert preds_by_zone[99]["longitude"] is None
        assert preds_by_zone[99]["zone_name"] == "Unknown"

    def test_get_heatmap_enriched_with_hour_filter(self, client: TestClient) -> None:
        """?hour=<iso> must still return enriched prediction objects.

        AC #8: explicit hour parameter returns predictions with all six fields.
        """
        resp = client.get("/heatmap", params={"hour": "2021-03-01T00:00:00"})
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        for pred in body["predictions"]:
            assert "latitude" in pred
            assert "longitude" in pred
            assert "bucket_time" in pred

    def test_get_heatmap_zone_count_equals_prediction_count(
        self, client: TestClient
    ) -> None:
        """zone_count must equal len(predictions).

        AC #5: all zones returned; no extra or missing entries.
        """
        resp = client.get("/heatmap")
        body: dict[str, Any] = resp.json()
        assert body["zone_count"] == len(body["predictions"])

    def test_get_heatmap_envelope_fields_unchanged(self, client: TestClient) -> None:
        """generated_at, hour, zone_count must still be present and correct.

        AC #6: envelope shape unchanged after enrichment.
        """
        resp = client.get("/heatmap")
        body: dict[str, Any] = resp.json()
        assert body["generated_at"] == "2021-03-01T12:00:00+00:00"
        assert body["hour"] == "2021-03-01T01:00:00"
        assert body["zone_count"] == 3

    def test_get_heatmap_sorted_descending_with_enrichment(
        self, client: TestClient
    ) -> None:
        """Predictions must remain sorted descending by predicted_trip_count.

        AC #6: sort order preserved after enrichment.
        """
        resp = client.get("/heatmap")
        body: dict[str, Any] = resp.json()
        counts = [p["predicted_trip_count"] for p in body["predictions"]]
        assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# Story-level acceptance tests — one test per acceptance criterion
# ---------------------------------------------------------------------------


class TestHeatmapAcceptance:
    """Acceptance tests for the /heatmap enrichment story.

    Each test maps directly to one numbered acceptance criterion (AC #1–#11)
    from the approved user story. These tests exercise the endpoint as an API
    consumer would: via ``TestClient`` with realistic-shape fixtures. They
    are intentionally higher-level than ``TestHeatmapEnrichedFields`` and
    include user-visible error-body assertions that the unit-style tests omit.
    """

    # ------------------------------------------------------------------
    # AC #1 — all six required fields present on every prediction object
    # ------------------------------------------------------------------

    def test_ac1_every_prediction_has_all_six_required_fields(
        self, client: TestClient
    ) -> None:
        """AC #1: pulocationid, zone_name, predicted_trip_count, latitude,
        longitude, bucket_time must all be present on every prediction.

        Arrange: client fixture has 3 zones for 2 hours.
        Act: GET /heatmap (default = latest hour, 3 zones).
        Assert: each of the 3 prediction objects contains exactly the six
        fields with their documented types.
        """
        resp = client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        predictions: list[dict[str, Any]] = body["predictions"]
        assert len(predictions) == 3, "Expected 3 zones in fixture"
        for pred in predictions:
            assert isinstance(pred["pulocationid"], int)
            assert isinstance(pred["zone_name"], str)
            assert isinstance(pred["predicted_trip_count"], float)
            assert isinstance(pred["latitude"], float)
            assert isinstance(pred["longitude"], float)
            # bucket_time must be a non-empty string (ISO-8601 parsability
            # is checked in AC #4)
            assert isinstance(pred["bucket_time"], str)
            assert pred["bucket_time"] != ""

    # ------------------------------------------------------------------
    # AC #2 — latitude / longitude come from the shared centroid helper
    # ------------------------------------------------------------------

    def test_ac2_lat_lon_values_match_centroid_fixture(
        self,
        client: TestClient,
        centroid_lookup: dict[int, tuple[float, float]],
    ) -> None:
        """AC #2: lat/lon must equal the values from the shared centroid lookup.

        The ``centroid_lookup`` fixture is injected into ``_centroid_lookup``
        in ``app.api.main`` before the request, so any value that does not
        come from that dict would be incorrect.

        Arrange: centroid_lookup maps zone 1→(40.693, -74.174), etc.
        Act: GET /heatmap.
        Assert: each prediction's latitude/longitude exactly matches the
        fixture centroid for that zone.
        """
        resp = client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        for pred in body["predictions"]:
            zone_id: int = pred["pulocationid"]
            expected_lat, expected_lon = centroid_lookup[zone_id]
            assert pred["latitude"] == expected_lat, (
                f"zone {zone_id}: expected lat {expected_lat}, got {pred['latitude']}"
            )
            assert pred["longitude"] == expected_lon, (
                f"zone {zone_id}: expected lon {expected_lon}, got {pred['longitude']}"
            )

    # ------------------------------------------------------------------
    # AC #3 — lat/lon rounded to exactly 3 decimal places
    # ------------------------------------------------------------------

    def test_ac3_lat_lon_rounded_to_3_decimal_places(self, client: TestClient) -> None:
        """AC #3: latitude and longitude must have at most 3 decimal places.

        Verifies this at the JSON level — any value that would require a
        4th decimal place when serialised is a violation visible to the consumer.

        Arrange: centroid_lookup fixture already contains 3-d.p. values.
        Act: GET /heatmap.
        Assert: for each prediction, round(value, 3) == value (no extra precision).
        """
        resp = client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        for pred in body["predictions"]:
            lat = pred["latitude"]
            lon = pred["longitude"]
            if lat is not None:
                assert round(lat, 3) == lat, (
                    f"latitude {lat} has more than 3 decimal places"
                )
            if lon is not None:
                assert round(lon, 3) == lon, (
                    f"longitude {lon} has more than 3 decimal places"
                )

    # ------------------------------------------------------------------
    # AC #4 — bucket_time equals envelope hour, ISO-8601 formatted
    # ------------------------------------------------------------------

    def test_ac4_bucket_time_equals_envelope_hour_iso8601(
        self, client: TestClient
    ) -> None:
        """AC #4: every prediction's bucket_time must be literally equal to the
        envelope ``hour`` field (both are ISO-8601 T-separated strings).

        Arrange: default request targets the latest hour (2021-03-01T01:00:00).
        Act: GET /heatmap.
        Assert: bucket_time == body["hour"] (strict string equality, not just
        timestamp equality) AND the string is parseable as ISO-8601.
        """
        resp = client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        envelope_hour: str = body["hour"]
        # Sanity: envelope hour must itself be parseable ISO-8601
        pd.Timestamp(envelope_hour)
        for pred in body["predictions"]:
            bucket_time: str = pred["bucket_time"]
            # Strict literal equality — both strings are produced by .isoformat()
            assert bucket_time == envelope_hour, (
                f"bucket_time {bucket_time!r} != envelope hour {envelope_hour!r}"
            )

    # ------------------------------------------------------------------
    # AC #5 — all zones in parquet returned; zone_count == len(predictions)
    # ------------------------------------------------------------------

    def test_ac5_zone_count_equals_number_of_predictions(
        self, client: TestClient
    ) -> None:
        """AC #5: zone_count in the envelope must equal len(predictions).
        No zones may be silently dropped or duplicated.

        Arrange: fixture has 3 zones for the latest hour.
        Act: GET /heatmap.
        Assert: zone_count == 3 == len(predictions); zone IDs are unique.
        """
        resp = client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        predictions: list[dict[str, Any]] = body["predictions"]
        assert body["zone_count"] == len(predictions), (
            "zone_count envelope field does not match actual prediction count"
        )
        zone_ids = [p["pulocationid"] for p in predictions]
        assert len(zone_ids) == len(set(zone_ids)), "Duplicate zone IDs in response"

    # ------------------------------------------------------------------
    # AC #6 — envelope fields unchanged; predictions sorted desc
    # ------------------------------------------------------------------

    def test_ac6_envelope_fields_present_and_predictions_sorted_desc(
        self, client: TestClient
    ) -> None:
        """AC #6: generated_at / hour / zone_count must be present and unchanged;
        predictions must be sorted descending by predicted_trip_count.

        Arrange: client fixture sets _generated_at = '2021-03-01T12:00:00+00:00'.
        Act: GET /heatmap.
        Assert: all three envelope fields match expected values AND trip counts
        are in non-ascending order.
        """
        resp = client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        # Envelope fields
        assert body["generated_at"] == "2021-03-01T12:00:00+00:00"
        assert body["hour"] == "2021-03-01T01:00:00"
        assert body["zone_count"] == 3
        # Sort order
        counts: list[float] = [p["predicted_trip_count"] for p in body["predictions"]]
        assert counts == sorted(counts, reverse=True), (
            "predictions are not sorted descending by predicted_trip_count"
        )

    # ------------------------------------------------------------------
    # AC #7 — missing centroid → null lat/lon; zone_name fallback "Unknown"
    # ------------------------------------------------------------------

    def test_ac7_missing_centroid_yields_null_coords_and_unknown_name(
        self,
        centroid_lookup: dict[int, tuple[float, float]],
        zone_lookup: dict[int, str],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """AC #7: a zone absent from _centroid_lookup must have latitude=null,
        longitude=null, and zone_name="Unknown" in the response.

        Arrange: inject predictions that include zone 99, which has no entry
        in the centroid_lookup or zone_lookup fixtures.
        Act: GET /heatmap.
        Assert: zone 99's prediction has null coordinates and Unknown name;
        zone 1 (which is in the lookup) still has valid coordinates.
        """
        import app.api.main as serving_module

        predictions_with_unknown = pd.DataFrame(
            {
                "pulocationid": [1, 99],
                "hour": pd.to_datetime(["2021-03-01 02:00:00", "2021-03-01 02:00:00"]),
                "predicted_trip_count": [15.0, 3.5],
                "actual_trip_count": [14.0, 2.0],
            }
        )
        monkeypatch.setattr(serving_module, "_predictions_df", predictions_with_unknown)
        monkeypatch.setattr(serving_module, "_zone_lookup", zone_lookup)
        monkeypatch.setattr(serving_module, "_centroid_lookup", centroid_lookup)
        monkeypatch.setattr(
            serving_module, "_generated_at", "2021-03-01T12:00:00+00:00"
        )
        monkeypatch.setattr(serving_module, "_watermark_root", tmp_path / "wm")

        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        preds_by_zone: dict[int, dict[str, Any]] = {
            p["pulocationid"]: p for p in body["predictions"]
        }
        # Zone 99: no centroid, no name
        assert preds_by_zone[99]["latitude"] is None
        assert preds_by_zone[99]["longitude"] is None
        assert preds_by_zone[99]["zone_name"] == "Unknown"
        # Zone 1: valid centroid retained
        assert preds_by_zone[1]["latitude"] == centroid_lookup[1][0]
        assert preds_by_zone[1]["longitude"] == centroid_lookup[1][1]

    # ------------------------------------------------------------------
    # AC #8 — ?hour=<iso> returns enriched objects
    # ------------------------------------------------------------------

    def test_ac8_hour_filter_returns_enriched_predictions(
        self, client: TestClient
    ) -> None:
        """AC #8: an explicit ?hour= query must return fully-enriched predictions
        (all six fields present) for that hour.

        Arrange: fixture has data for 2021-03-01 00:00:00 and 01:00:00.
        Act: GET /heatmap?hour=2021-03-01T00:00:00.
        Assert: 200, correct hour, 3 predictions each with latitude, longitude,
        bucket_time all populated.
        """
        resp = client.get("/heatmap", params={"hour": "2021-03-01T00:00:00"})

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["hour"] == "2021-03-01T00:00:00"
        assert len(body["predictions"]) == 3
        for pred in body["predictions"]:
            assert "latitude" in pred and pred["latitude"] is not None
            assert "longitude" in pred and pred["longitude"] is not None
            assert "bucket_time" in pred
            assert pred["bucket_time"] == body["hour"]

    # ------------------------------------------------------------------
    # AC #9 — 503 when predictions not loaded
    # ------------------------------------------------------------------

    def test_ac9_returns_503_with_detail_when_predictions_not_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #9: GET /heatmap must return 503 when no predictions are available.
        The response body must contain a 'detail' field explaining the issue.

        Arrange: set _predictions_df = None (simulates a cold start before
        the pipeline has run).
        Act: GET /heatmap.
        Assert: status 503, body contains 'detail' key (FastAPI error shape).
        """
        import app.api.main as serving_module

        monkeypatch.setattr(serving_module, "_predictions_df", None)
        monkeypatch.setattr(serving_module, "_zone_lookup", {})
        monkeypatch.setattr(serving_module, "_centroid_lookup", {})
        monkeypatch.setattr(serving_module, "_generated_at", "")

        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")

        assert resp.status_code == 503
        body: dict[str, Any] = resp.json()
        assert "detail" in body, "503 response must include a 'detail' field"
        assert body["detail"]  # non-empty

    # ------------------------------------------------------------------
    # AC #10 — 404 for unknown hour
    # ------------------------------------------------------------------

    def test_ac10_returns_404_with_detail_for_unknown_hour(
        self, client: TestClient
    ) -> None:
        """AC #10: requesting a valid-format but non-existent hour must return 404.
        The response body must contain a 'detail' field.

        Arrange: fixture has hours for 2021-03-01 only; request 2099-01-01.
        Act: GET /heatmap?hour=2099-01-01T00:00:00.
        Assert: status 404, body has 'detail'.
        """
        resp = client.get("/heatmap", params={"hour": "2099-01-01T00:00:00"})

        assert resp.status_code == 404
        body: dict[str, Any] = resp.json()
        assert "detail" in body, "404 response must include a 'detail' field"
        assert body["detail"]

    # ------------------------------------------------------------------
    # AC #11 — 400 for malformed hour
    # ------------------------------------------------------------------

    def test_ac11_returns_400_with_detail_for_malformed_hour(
        self, client: TestClient
    ) -> None:
        """AC #11: a syntactically invalid hour string must return 400.
        The response body must contain a 'detail' field.

        Arrange: pass the string 'not-a-date' as the hour parameter.
        Act: GET /heatmap?hour=not-a-date.
        Assert: status 400, body has 'detail'.
        """
        resp = client.get("/heatmap", params={"hour": "not-a-date"})

        assert resp.status_code == 400
        body: dict[str, Any] = resp.json()
        assert "detail" in body, "400 response must include a 'detail' field"
        assert body["detail"]


# ---------------------------------------------------------------------------
# I3 — Lifespan integration: verify build_centroid_lookup wiring
# ---------------------------------------------------------------------------


class TestLifespanCentroidWiring:
    """Verify that lifespan() calls build_centroid_lookup and stores the result.

    All file I/O is patched so the test exercises only the wiring inside
    ``lifespan()`` without touching the filesystem.
    """

    def test_lifespan_populates_centroid_lookup_via_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """lifespan must call build_centroid_lookup once and store its return
        value in ``_centroid_lookup``.

        Arrange:
        - Patch ``app.api.main.build_centroid_lookup`` to return a known dict.
        - Patch ``app.api.main._load_predictions`` to return a trivial DataFrame.
        - Patch ``app.api.main._load_zone_lookup`` to return a trivial dict.
        - Patch ``app.core.config.load_config`` so lifespan gets a valid config.
        Act: enter the FastAPI app via ``TestClient`` (triggers lifespan).
        Assert: the mock was called exactly once AND ``_centroid_lookup``
        matches the dict the mock returned.
        """
        import app.api.main as serving_module

        expected_centroids: dict[int, tuple[float, float]] = {
            1: (40.693, -74.174),
            2: (40.618, -73.825),
        }
        fake_predictions = pd.DataFrame(
            {
                "pulocationid": [1],
                "hour": pd.to_datetime(["2021-03-01 00:00:00"]),
                "predicted_trip_count": [5.0],
                "actual_trip_count": [4.0],
            }
        )

        mock_build_centroid = MagicMock(return_value=expected_centroids)
        mock_load_predictions = MagicMock(return_value=fake_predictions)
        mock_load_zone_lookup = MagicMock(return_value={1: "Newark Airport"})
        fake_config = {
            "serving": {
                "predictions_path": "fake/predictions.parquet",
                "zone_shapefile_path": "fake/taxi_zones.shp",
            }
        }

        monkeypatch.setattr(
            serving_module, "build_centroid_lookup", mock_build_centroid
        )
        monkeypatch.setattr(serving_module, "_load_predictions", mock_load_predictions)
        monkeypatch.setattr(serving_module, "_load_zone_lookup", mock_load_zone_lookup)

        with patch("app.api.main.load_config", return_value=fake_config):
            with TestClient(serving_module.app):
                # Inside the context manager the lifespan has completed.
                mock_build_centroid.assert_called_once_with("fake/taxi_zones.shp")
                assert serving_module._centroid_lookup == expected_centroids


# ---------------------------------------------------------------------------
# PBI 27 — GET /health endpoint tests
# ---------------------------------------------------------------------------

_FAKE_CONFIG = {
    # PBI 25b FIX: /health derives the simulation position from the watermark,
    # falling back to ingest.seed_start when no watermark file exists.
    "ingest": {"seed_start": "2021-01-18T00:00:00"},
    "serving": {
        "predictions_path": "fake/predictions.parquet",
        "zone_shapefile_path": "fake/taxi_zones.shp",
    },
    "mlflow": {"model_name": "ny_taxi_demand_model"},
}


class TestHealthEndpoint:
    """Tests for GET /health (PBI 27)."""

    @pytest.fixture(autouse=True)
    def _health_state(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Provide _config (with ingest.seed_start) and an isolated watermark root.

        PBI 25b FIX: /health derives the simulation position from the watermark,
        falling back to ingest.seed_start. These tests build a TestClient without
        entering its lifespan, so they set the module globals directly.
        """
        import app.api.main as serving_module

        monkeypatch.setattr(serving_module, "_config", _FAKE_CONFIG)
        monkeypatch.setattr(serving_module, "_watermark_root", tmp_path / "watermarks")

    def test_get_health_happy_path_returns_ok_with_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /health returns 200 ok with model_version when parquet loaded and Production version exists.

        Arrange: non-empty _predictions_df; MlflowClient returns one Production version; DEMO_MODE unset.
        Act: GET /health.
        Assert: 200, status=="ok", simulated==False (PBI 25b: dynamic), model_version is a non-empty string,
                data_publication_lag_days >= 0.
        """
        import app.api.main as serving_module

        monkeypatch.delenv("DEMO_MODE", raising=False)
        predictions = pd.DataFrame(
            {
                "pulocationid": [1],
                "hour": pd.to_datetime(["2021-03-01 00:00:00"]),
                "predicted_trip_count": [10.0],
                "actual_trip_count": [9.0],
            }
        )
        monkeypatch.setattr(serving_module, "_predictions_df", predictions)

        mock_version = make_mock_version(version="3")
        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = [mock_version]

        with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
            with patch("app.api.main.MlflowClient", return_value=mock_client_instance):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/health")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["status"] == "ok"
        assert (
            body["simulated"] is False
        )  # PBI 25b: dynamic, False when DEMO_MODE unset
        assert isinstance(body["model_version"], str)
        assert body["model_version"] != ""
        assert isinstance(body["data_publication_lag_days"], int)
        assert body["data_publication_lag_days"] >= 0

    def test_get_health_degraded_when_no_production_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /health returns 200 degraded with model_version=null when no Production version exists.

        Arrange: non-empty _predictions_df; MlflowClient returns empty version list.
        Act: GET /health.
        Assert: 200, status=="degraded", model_version is null.
        """
        import app.api.main as serving_module

        monkeypatch.delenv("DEMO_MODE", raising=False)
        predictions = pd.DataFrame(
            {
                "pulocationid": [1],
                "hour": pd.to_datetime(["2021-03-01 00:00:00"]),
                "predicted_trip_count": [10.0],
                "actual_trip_count": [9.0],
            }
        )
        monkeypatch.setattr(serving_module, "_predictions_df", predictions)

        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = []

        with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
            with patch("app.api.main.MlflowClient", return_value=mock_client_instance):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["model_version"] is None
        assert (
            body["simulated"] is False
        )  # PBI 25b: dynamic, False when DEMO_MODE unset

    def test_get_health_returns_503_when_predictions_not_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /health returns 503 when _predictions_df is None.

        Arrange: _predictions_df = None.
        Act: GET /health.
        Assert: 503 with detail field.
        """
        import app.api.main as serving_module

        monkeypatch.setattr(serving_module, "_predictions_df", None)

        local_client = TestClient(serving_module.app)
        resp = local_client.get("/health")

        assert resp.status_code == 503
        body = resp.json()
        assert "detail" in body
        assert body["detail"]

    def test_get_health_degraded_when_mlflow_client_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /health returns 200 degraded when MlflowClient raises (e.g. missing mlflow.db).

        Arrange: non-empty _predictions_df; MlflowClient constructor raises Exception.
        Act: GET /health.
        Assert: 200, status=="degraded", model_version is null.
        """
        import app.api.main as serving_module

        predictions = pd.DataFrame(
            {
                "pulocationid": [1],
                "hour": pd.to_datetime(["2021-03-01 00:00:00"]),
                "predicted_trip_count": [10.0],
                "actual_trip_count": [9.0],
            }
        )
        monkeypatch.setattr(serving_module, "_predictions_df", predictions)

        with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
            with patch(
                "app.api.main.MlflowClient", side_effect=Exception("db missing")
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["model_version"] is None

    def test_get_health_lag_is_null_when_predictions_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /health returns null lag and null bucket when predictions DataFrame is empty.

        Arrange: _predictions_df is an empty DataFrame (0 rows).
        Act: GET /health.
        Assert: data_publication_lag_days is null, last_prediction_bucket is null.
        """
        import app.api.main as serving_module

        empty_df = pd.DataFrame(
            columns=[
                "pulocationid",
                "hour",
                "predicted_trip_count",
                "actual_trip_count",
            ]
        )
        monkeypatch.setattr(serving_module, "_predictions_df", empty_df)

        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = []

        with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
            with patch("app.api.main.MlflowClient", return_value=mock_client_instance):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["data_publication_lag_days"] is None
        assert body["last_prediction_bucket"] is None

    def test_get_health_lag_is_zero_when_bucket_in_future(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /health returns data_publication_lag_days==0 when max(hour) is in the future.

        Arrange: predictions with an hour far in the future (year 2099).
        Act: GET /health.
        Assert: data_publication_lag_days == 0, never negative.
        """
        import app.api.main as serving_module

        predictions = pd.DataFrame(
            {
                "pulocationid": [1],
                "hour": pd.to_datetime(["2099-12-31 23:00:00"]),
                "predicted_trip_count": [5.0],
                "actual_trip_count": [4.0],
            }
        )
        monkeypatch.setattr(serving_module, "_predictions_df", predictions)

        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = []

        with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
            with patch("app.api.main.MlflowClient", return_value=mock_client_instance):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["data_publication_lag_days"] == 0


# ---------------------------------------------------------------------------
# PBI 27 — GET /model-info endpoint tests
# ---------------------------------------------------------------------------


class TestModelInfoEndpoint:
    """Tests for GET /model-info (PBI 27)."""

    def test_get_model_info_happy_path_returns_all_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /model-info returns 200 with all fields populated when Production version exists.

        Arrange: MlflowClient returns one Production version with both tags set.
        Act: GET /model-info.
        Assert: 200, status=="ok", model_test_wape and wape_lift_pct are floats,
                training_date is ISO-8601 parseable, stage=="Production".
        """
        import app.api.main as serving_module

        mock_version = make_mock_version(
            version="3",
            stage="Production",
            creation_timestamp_ms=1_710_058_921_000,
            tags={"model_test_wape": "0.1823", "wape_lift_pct": "12.4"},
        )
        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = [mock_version]

        with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
            with patch("app.api.main.MlflowClient", return_value=mock_client_instance):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/model-info")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["status"] == "ok"
        assert body["model_version"] == "3"
        assert body["stage"] == "Production"
        assert isinstance(body["model_test_wape"], float)
        assert isinstance(body["wape_lift_pct"], float)
        # training_date must be ISO-8601 parseable
        pd.Timestamp(body["training_date"])

    def test_get_model_info_no_production_model_returns_null_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /model-info returns 200 no_production_model with all null fields.

        Arrange: MlflowClient returns empty version list.
        Act: GET /model-info.
        Assert: 200, status=="no_production_model", all other fields null.
        """
        import app.api.main as serving_module

        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = []

        with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
            with patch("app.api.main.MlflowClient", return_value=mock_client_instance):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/model-info")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "no_production_model"
        assert body["model_version"] is None
        assert body["stage"] is None
        assert body["training_date"] is None
        assert body["model_test_wape"] is None
        assert body["wape_lift_pct"] is None

    def test_get_model_info_returns_null_wape_when_tag_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /model-info returns null for model_test_wape when tag is absent.

        Arrange: Production version exists but has no model_test_wape tag.
        Act: GET /model-info.
        Assert: 200, model_test_wape is null, no exception raised.
        """
        import app.api.main as serving_module

        mock_version = make_mock_version(
            version="2",
            stage="Production",
            tags={},  # no tags
        )
        mock_client_instance = MagicMock()
        mock_client_instance.get_latest_versions.return_value = [mock_version]

        with patch("app.api.main.load_config", return_value=_FAKE_CONFIG):
            with patch("app.api.main.MlflowClient", return_value=mock_client_instance):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/model-info")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_test_wape"] is None
        assert body["wape_lift_pct"] is None


# ---------------------------------------------------------------------------
# PBI 27 — /heatmap simulated flag test
# ---------------------------------------------------------------------------


class TestHeatmapSimulatedFlag:
    """Tests for the simulated flag added to /heatmap (PBI 27 AC #9, PBI 25b update)."""

    def test_get_heatmap_envelope_contains_simulated_flag(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /heatmap envelope includes 'simulated' bool at the top level.

        PBI 25b update: simulated is dynamic (DEMO_MODE env var).
        DEMO_MODE unset → simulated=False.

        Arrange: standard client fixture (non-empty predictions); DEMO_MODE unset.
        Act: GET /heatmap.
        Assert: body["simulated"] is False.
        """
        monkeypatch.delenv("DEMO_MODE", raising=False)
        resp = client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert isinstance(body["simulated"], bool)
        assert (
            body["simulated"] is False
        )  # PBI 25b: dynamic, False when DEMO_MODE unset


# ---------------------------------------------------------------------------
# PBI 21 — model provenance fields on /heatmap
# ---------------------------------------------------------------------------


class TestHeatmapModelProvenance:
    """Tests for model_version and served_from_stage envelope fields (PBI 21 AC #8, #12)."""

    def test_get_heatmap_envelope_contains_model_provenance_fields(
        self, client: TestClient
    ) -> None:
        """GET /heatmap returns string model_version and served_from_stage in envelope.

        AC #8, #12: predictions parquet with PBI-21 columns loaded; both fields
        appear as non-null strings at the top level of the JSON envelope.

        Arrange: client fixture sets _model_version="3", _served_from_stage="Production".
        Act: GET /heatmap.
        Assert: 200, body["model_version"] == "3", body["served_from_stage"] == "Production".
        """
        resp = client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert "model_version" in body, "model_version missing from /heatmap envelope"
        assert "served_from_stage" in body, (
            "served_from_stage missing from /heatmap envelope"
        )
        assert isinstance(body["model_version"], str)
        assert isinstance(body["served_from_stage"], str)
        assert body["model_version"] == "3"
        assert body["served_from_stage"] == "Production"

    def test_get_heatmap_provenance_null_when_legacy_parquet(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """GET /heatmap returns null provenance fields for pre-PBI-21 parquets.

        Legacy parquets lack model_version/served_from_stage columns; the API
        must set both globals to None and surface null in the envelope.

        Arrange: _model_version=None, _served_from_stage=None (legacy state).
        Act: GET /heatmap.
        Assert: 200, body["model_version"] is null, body["served_from_stage"] is null.
        """
        import app.api.main as serving_module

        predictions = pd.DataFrame(
            {
                "pulocationid": [1],
                "hour": pd.to_datetime(["2021-03-01 01:00:00"]),
                "predicted_trip_count": [5.0],
                "actual_trip_count": [4.0],
                # No model_version or served_from_stage columns — legacy parquet
            }
        )
        monkeypatch.setattr(serving_module, "_predictions_df", predictions)
        monkeypatch.setattr(serving_module, "_zone_lookup", {1: "Test Zone"})
        monkeypatch.setattr(serving_module, "_centroid_lookup", {})
        monkeypatch.setattr(
            serving_module, "_generated_at", "2021-03-01T12:00:00+00:00"
        )
        monkeypatch.setattr(serving_module, "_model_version", None)
        monkeypatch.setattr(serving_module, "_served_from_stage", None)
        monkeypatch.setattr(serving_module, "_watermark_root", tmp_path / "wm")

        local_client = TestClient(serving_module.app)
        resp = local_client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["model_version"] is None
        assert body["served_from_stage"] is None


# ---------------------------------------------------------------------------
# PBI 22 — prediction log hook tests
# ---------------------------------------------------------------------------


class TestPredictionLogHook:
    """Tests for the log_predictions hook wired into GET /heatmap (PBI 22)."""

    def test_heatmap_hook_called_on_valid_response(self, client: TestClient) -> None:
        """log_predictions is called once per successful /heatmap request.

        Arrange: patch app.api.main.prediction_log.log_predictions as MagicMock.
        Act: GET /heatmap.
        Assert: mock called once; first positional arg contains "predictions" key.

        Covers AC 6.
        """
        import app.api.main as serving_module

        mock_log = MagicMock(return_value=None)
        with patch.object(serving_module.prediction_log, "log_predictions", mock_log):
            resp = client.get("/heatmap")

        assert resp.status_code == 200
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        response_arg: dict[str, Any] = call_args[0][0]
        assert "predictions" in response_arg

    @pytest.mark.parametrize("exc_cls", [OSError, RuntimeError, ValueError])
    def test_heatmap_returns_200_when_log_raises(
        self, client: TestClient, exc_cls: type[Exception]
    ) -> None:
        """A log_predictions failure must not degrade the serving path.

        Arrange: patch log_predictions to raise exc_cls ("boom").
        Act: GET /heatmap.
        Assert: HTTP 200 and response body still contains "predictions".

        Covers AC 7 + resolved decision 2.  Parametrized over OSError,
        RuntimeError, and ValueError to match the production except Exception.
        """
        import app.api.main as serving_module

        mock_log = MagicMock(side_effect=exc_cls("boom"))
        with patch.object(serving_module.prediction_log, "log_predictions", mock_log):
            resp = client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert "predictions" in body


class TestHeatmapWatermarkFilter:
    """GET /heatmap caps the default hour to the watermark when one exists."""

    @pytest.fixture(autouse=True)
    def _restore_globals(self) -> Generator[None, None, None]:
        """Snapshot and restore all module globals mutated by this class."""
        import app.api.main as serving_module

        orig_predictions = serving_module._predictions_df
        orig_zone_lookup = serving_module._zone_lookup
        orig_centroid_lookup = serving_module._centroid_lookup
        orig_generated_at = serving_module._generated_at
        orig_model_version = serving_module._model_version
        orig_served_from_stage = serving_module._served_from_stage
        orig_config = serving_module._config
        orig_watermark_root = serving_module._watermark_root
        yield
        serving_module._predictions_df = orig_predictions
        serving_module._zone_lookup = orig_zone_lookup
        serving_module._centroid_lookup = orig_centroid_lookup
        serving_module._generated_at = orig_generated_at
        serving_module._model_version = orig_model_version
        serving_module._served_from_stage = orig_served_from_stage
        serving_module._config = orig_config
        serving_module._watermark_root = orig_watermark_root

    def _client_with_watermark(
        self,
        sample_predictions: pd.DataFrame,
        zone_lookup: dict[int, str],
        centroid_lookup: dict[int, tuple[float, float]],
        tmp_path: Path,
        watermark: datetime | None,
    ) -> TestClient:
        """Build a TestClient with module globals pointed at tmp_path.

        Mirrors the `client` fixture but lets the test control the watermark.
        Caller is responsible for restoring globals — kept minimal via a
        fresh TestClient per test and monkeypatch in the tests below.
        """
        import app.api.main as serving_module
        from app.core.watermark import write_watermark

        wm_root = tmp_path / "watermarks" / "refresh_15min"
        serving_module._predictions_df = sample_predictions
        serving_module._zone_lookup = zone_lookup
        serving_module._centroid_lookup = centroid_lookup
        serving_module._generated_at = "2021-03-01T12:00:00+00:00"
        serving_module._model_version = "3"
        serving_module._served_from_stage = "Production"
        serving_module._config = {
            "monitoring": {"predictions_path": str(tmp_path / "mon")},
            "ingest": {"seed_start": "2021-01-18T00:00:00"},
        }
        serving_module._watermark_root = wm_root
        if watermark is not None:
            write_watermark(wm_root, watermark)
        return TestClient(serving_module.app)

    def test_heatmap_default_capped_at_watermark(
        self,
        sample_predictions,
        zone_lookup,
        centroid_lookup,
        tmp_path,
        monkeypatch,
    ) -> None:
        """Watermark between the two sample hours withholds the later tick."""
        # Arrange — watermark at 00:30 (between 00:00 and 01:00)
        client = self._client_with_watermark(
            sample_predictions,
            zone_lookup,
            centroid_lookup,
            tmp_path,
            datetime(2021, 3, 1, 0, 30, 0),
        )

        # Act
        resp = client.get("/heatmap")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["hour"] == "2021-03-01T00:00:00"

    def test_heatmap_watermark_before_all_predictions_returns_503(
        self,
        sample_predictions,
        zone_lookup,
        centroid_lookup,
        tmp_path,
    ) -> None:
        """Watermark earlier than every prediction hour → 503."""
        # Arrange
        client = self._client_with_watermark(
            sample_predictions,
            zone_lookup,
            centroid_lookup,
            tmp_path,
            datetime(2021, 1, 1, 0, 0, 0),
        )

        # Act
        resp = client.get("/heatmap")

        # Assert
        assert resp.status_code == 503

    def test_heatmap_no_watermark_returns_latest(
        self,
        sample_predictions,
        zone_lookup,
        centroid_lookup,
        tmp_path,
    ) -> None:
        """No watermark file → unchanged latest-hour behavior."""
        # Arrange — watermark=None → no control file written
        client = self._client_with_watermark(
            sample_predictions,
            zone_lookup,
            centroid_lookup,
            tmp_path,
            None,
        )

        # Act
        resp = client.get("/heatmap")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["hour"] == "2021-03-01T01:00:00"

    def test_heatmap_explicit_hour_bypasses_watermark(
        self,
        sample_predictions,
        zone_lookup,
        centroid_lookup,
        tmp_path,
    ) -> None:
        """An explicit ?hour= is served even if it exceeds the watermark."""
        # Arrange — watermark at 00:30 would withhold 01:00 by default
        client = self._client_with_watermark(
            sample_predictions,
            zone_lookup,
            centroid_lookup,
            tmp_path,
            datetime(2021, 3, 1, 0, 30, 0),
        )

        # Act
        resp = client.get("/heatmap", params={"hour": "2021-03-01T01:00:00"})

        # Assert
        assert resp.status_code == 200
        assert resp.json()["hour"] == "2021-03-01T01:00:00"


# ---------------------------------------------------------------------------
# Step 6 — lifespan wires _watermark_root from config["serving"]["watermark_path"]
# ---------------------------------------------------------------------------


class TestLifespanWatermarkPathWiring:
    """Step 6 — serving.watermark_path is honoured by lifespan()."""

    def test_lifespan_sets_watermark_root_from_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """lifespan() must set _watermark_root to Path(config["serving"]["watermark_path"]).

        Arrange:
        - Provide a fake_config with a custom watermark_path.
        - Patch _load_predictions and _load_zone_lookup to avoid filesystem I/O.
        Act: enter the app via TestClient (triggers lifespan).
        Assert: _watermark_root equals the configured path.
        """
        import app.api.main as serving_module

        custom_wm_path = "custom/watermarks/refresh"
        fake_config = {
            "serving": {
                "watermark_path": custom_wm_path,
                "predictions_path": "fake/predictions.parquet",
                "zone_shapefile_path": "fake/taxi_zones.shp",
            },
            "mlflow": {"model_name": "ny_taxi_demand_model"},
        }

        fake_predictions = pd.DataFrame(
            {
                "pulocationid": [1],
                "hour": pd.to_datetime(["2021-03-01 00:00:00"]),
                "predicted_trip_count": [5.0],
                "actual_trip_count": [4.0],
            }
        )

        monkeypatch.setattr(
            serving_module,
            "_load_predictions",
            MagicMock(return_value=fake_predictions),
        )
        monkeypatch.setattr(
            serving_module, "_load_zone_lookup", MagicMock(return_value={1: "Zone A"})
        )
        monkeypatch.setattr(
            serving_module, "build_centroid_lookup", MagicMock(return_value={})
        )

        with patch("app.api.main.load_config", return_value=fake_config):
            with TestClient(serving_module.app):
                # Assert — _watermark_root reflects the configured path
                assert serving_module._watermark_root == Path(custom_wm_path)
