"""Acceptance tests for PBI 22 — Prediction Log + Actuals-Join Job.

Each test maps to one or more numbered acceptance criteria from the approved
user story.  These tests exercise the public surface only:
  - log_predictions(response, path) from app.monitoring.prediction_log
  - join_actuals(...) from app.monitoring.actuals_join
  - GET /heatmap via FastAPI TestClient

They are intentionally higher-level than the builder's unit tests in
tests/test_monitoring.py and tests/test_serving.py.  Where a criterion is
fully covered by an existing unit test the test below still provides a
story-level wrapper that exercises the public interface from the user's
perspective (or is noted as "verified by inspection" in the coverage table).
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_heatmap_response(
    n_zones: int = 3,
    model_version: str | None = "5",
    served_from_stage: str | None = "Production",
    bucket_iso: str = "2024-11-15T01:00:00",
) -> dict[str, Any]:
    """Build a realistic /heatmap response dict with *n_zones* predictions."""
    predictions = [
        {
            "pulocationid": i + 1,
            "predicted_trip_count": float(10 + i),
            "bucket_time": bucket_iso,
            "zone_name": f"Zone {i + 1}",
            "latitude": 40.6 + i * 0.01,
            "longitude": -74.0 + i * 0.01,
        }
        for i in range(n_zones)
    ]
    return {
        "predictions": predictions,
        "model_version": model_version,
        "served_from_stage": served_from_stage,
    }


def _write_prediction_parquet(
    path: Path,
    pulocationids: list[int],
    bucket_time: str,
    model_version: str | None = "5",
    served_from_stage: str | None = "Production",
) -> None:
    """Write a minimal prediction-log parquet file to *path*.

    *path* must be the full file path (directory created by caller).
    """
    served_at = pd.Timestamp("2024-11-15T12:00:00", tz="UTC")
    rows = [
        {
            "bucket_time": pd.Timestamp(bucket_time, tz="UTC"),
            "pulocationid": z,
            "predicted": float(10 + z),
            "model_version": model_version,
            "served_from_stage": served_from_stage,
            "served_at": served_at,
        }
        for z in pulocationids
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_gold_parquet(
    gold_dir: Path,
    rows: list[tuple[int, str, float]],
) -> None:
    """Write a minimal Gold parquet to *gold_dir*/hourly_demand_featured.parquet."""
    df = pd.DataFrame(
        {
            "pulocationid": [r[0] for r in rows],
            "hour": pd.to_datetime([r[1] for r in rows]),
            "trip_count": [r[2] for r in rows],
        }
    )
    gold_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(gold_dir / "hourly_demand_featured.parquet", index=False)


# ---------------------------------------------------------------------------
# Module-level globals reused across API tests
# ---------------------------------------------------------------------------

_SAMPLE_PREDICTIONS = pd.DataFrame(
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

_ZONE_LOOKUP = {
    1: "Newark Airport",
    2: "Jamaica Bay",
    3: "Allerton/Pelham Gardens",
}

_CENTROID_LOOKUP: dict[int, tuple[float, float]] = {
    1: (40.693, -74.174),
    2: (40.618, -73.825),
    3: (40.866, -73.868),
}


@pytest.fixture()
def api_client() -> TestClient:
    """Return a TestClient with realistic module globals and log_predictions mocked.

    Mocking log_predictions avoids touching the real filesystem from API tests
    while still exercising the full heatmap handler logic.  AC 6 and AC 7 use
    their own fixtures that control the mock differently.
    """
    import app.api.main as serving_module

    orig = {
        "_predictions_df": serving_module._predictions_df,
        "_zone_lookup": serving_module._zone_lookup,
        "_centroid_lookup": serving_module._centroid_lookup,
        "_generated_at": serving_module._generated_at,
        "_model_version": serving_module._model_version,
        "_served_from_stage": serving_module._served_from_stage,
    }

    serving_module._predictions_df = _SAMPLE_PREDICTIONS
    serving_module._zone_lookup = _ZONE_LOOKUP
    serving_module._centroid_lookup = _CENTROID_LOOKUP
    serving_module._generated_at = "2021-03-01T12:00:00+00:00"
    serving_module._model_version = "3"
    serving_module._served_from_stage = "Production"

    with patch("app.api.main.prediction_log.log_predictions"):
        yield TestClient(serving_module.app)

    for attr, val in orig.items():
        setattr(serving_module, attr, val)


# ===========================================================================
# class TestPBI22Story
# ===========================================================================


class TestPBI22Story:
    """Acceptance tests for PBI 22 — Prediction Log + Actuals-Join Job."""

    # -----------------------------------------------------------------------
    # AC 1 — importable signature
    # -----------------------------------------------------------------------

    def test_ac1_log_predictions_importable_with_correct_signature(self) -> None:
        """AC 1: log_predictions(response, path) is importable from
        app.monitoring.prediction_log and has the expected calling convention.

        Arrange: import the module.
        Act: inspect the function.
        Assert: it exists, is callable, and accepts at least two positional args.
        """
        import inspect

        import app.monitoring.prediction_log as mod

        assert hasattr(mod, "log_predictions"), (
            "log_predictions not found in app.monitoring.prediction_log"
        )
        func = mod.log_predictions
        assert callable(func)
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        # Must accept 'response' and 'path' as the first two parameters.
        assert "response" in params, "'response' parameter missing"
        assert "path" in params, "'path' parameter missing"

    # -----------------------------------------------------------------------
    # AC 2 — file path pattern
    # -----------------------------------------------------------------------

    def test_ac2_written_file_follows_path_pattern(self, tmp_path: Path) -> None:
        """AC 2: the returned Path matches
        {root}/predictions/dt={YYYY-MM-DD}/served_at={ISO-ts}-{uuid8}.parquet.

        Arrange: 3-zone response; fixed UTC time via _now patch.
        Act: call log_predictions.
        Assert: returned path exists and its structure matches the documented
        pattern — dt= directory, served_at= filename prefix, 8-char uuid suffix.
        """
        import app.monitoring.prediction_log as mod

        fixed_dt = datetime(2024, 11, 15, 23, 59, 59, tzinfo=timezone.utc)
        response = _make_heatmap_response(n_zones=3)

        with patch.object(mod, "_now", return_value=fixed_dt):
            result = mod.log_predictions(response, str(tmp_path))

        assert result is not None
        assert result.exists()
        # dt= partition must be today's UTC date
        assert "dt=2024-11-15" in str(result)
        # filename must start with "served_at=" and end with ".parquet"
        assert result.name.startswith("served_at=")
        assert result.name.endswith(".parquet")
        # uuid8 suffix: the last segment before ".parquet" is 8 hex chars
        stem = result.stem  # e.g. "served_at=2024-11-15T23-59-59-000000Z-a1b2c3d4"
        uuid8 = stem.split("-")[-1]
        assert re.fullmatch(r"[0-9a-f]{8}", uuid8), (
            f"Expected 8-char hex uuid suffix, got: {uuid8!r}"
        )

    # -----------------------------------------------------------------------
    # AC 3 — parquet schema
    # -----------------------------------------------------------------------

    def test_ac3_written_parquet_has_correct_schema(self, tmp_path: Path) -> None:
        """AC 3: the written parquet contains exactly the six required columns;
        served_at is set inside log_predictions (not from caller); no 'actual' col.

        Arrange: 3-zone response with model_version and served_from_stage.
        Act: call log_predictions.
        Assert: columns == expected set; bucket_time dtype is UTC-aware;
                served_at is a timestamp (not the caller-supplied string);
                'actual' column is absent.
        """
        import app.monitoring.prediction_log as mod

        fixed_dt = datetime(2024, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
        response = _make_heatmap_response(n_zones=3)

        with patch.object(mod, "_now", return_value=fixed_dt):
            result = mod.log_predictions(response, str(tmp_path))

        assert result is not None
        df = pd.read_parquet(result)

        expected_cols = {
            "bucket_time",
            "pulocationid",
            "predicted",
            "model_version",
            "served_from_stage",
            "served_at",
        }
        assert set(df.columns) == expected_cols
        assert "actual" not in df.columns
        # bucket_time must be a UTC-aware timestamp, not a raw string
        assert "UTC" in str(df["bucket_time"].dtype)
        # served_at must be a timestamp type
        assert "datetime" in str(df["served_at"].dtype)

    # -----------------------------------------------------------------------
    # AC 4 — two calls same day → same dt= dir, different files
    # -----------------------------------------------------------------------

    def test_ac4_two_calls_same_day_produce_separate_files_no_overwrite(
        self, tmp_path: Path
    ) -> None:
        """AC 4: two calls on the same UTC calendar day land in the same dt=
        directory but produce distinct files; neither overwrites the other.

        Arrange: same fixed UTC day for both calls.
        Act: call log_predictions twice.
        Assert: two distinct files exist in the same dt= dir; both non-zero;
                no file is overwritten (file count == 2).
        """
        import app.monitoring.prediction_log as mod

        fixed_dt = datetime(2024, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
        response = _make_heatmap_response(n_zones=3)

        with patch.object(mod, "_now", return_value=fixed_dt):
            path1 = mod.log_predictions(response, str(tmp_path))
            path2 = mod.log_predictions(response, str(tmp_path))

        assert path1 is not None and path2 is not None
        assert path1 != path2, "Second call must produce a distinct file"
        assert path1.parent == path2.parent, "Both files must share the same dt= dir"

        files_in_dir = list(path1.parent.glob("*.parquet"))
        assert len(files_in_dir) == 2, (
            "Exactly 2 parquet files expected, not overwritten"
        )
        for f in files_in_dir:
            assert f.stat().st_size > 0, f"File {f} is zero bytes"

    # -----------------------------------------------------------------------
    # AC 5 — empty predictions → skip, return None, no file
    # -----------------------------------------------------------------------

    def test_ac5_empty_predictions_skips_write_and_returns_none(
        self, tmp_path: Path
    ) -> None:
        """AC 5: an empty predictions list causes log_predictions to skip the
        write, return None, and emit no parquet file.

        Arrange: response with predictions=[].
        Act: call log_predictions.
        Assert: return value is None; no parquet exists under tmp_path.
        """
        from app.monitoring.prediction_log import log_predictions

        response: dict[str, Any] = {
            "predictions": [],
            "model_version": "5",
            "served_from_stage": "Production",
        }

        result = log_predictions(response, str(tmp_path))

        assert result is None
        assert list(tmp_path.rglob("*.parquet")) == []

    # -----------------------------------------------------------------------
    # AC 6 — API hook calls log_predictions after computing the response
    # -----------------------------------------------------------------------

    def test_ac6_heatmap_calls_log_predictions_with_predictions_key(
        self, api_client: TestClient
    ) -> None:
        """AC 6: after computing the /heatmap response the handler calls
        log_predictions exactly once with a dict that has a 'predictions' key.

        Arrange: api_client fixture; fresh mock that captures call args.
        Act: GET /heatmap.
        Assert: mock called once; first positional arg contains 'predictions' key.
        """
        import app.api.main as serving_module

        mock_log = MagicMock()
        with patch.object(serving_module.prediction_log, "log_predictions", mock_log):
            resp = api_client.get("/heatmap")

        assert resp.status_code == 200
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args
        response_arg: dict[str, Any] = call_kwargs[0][0]
        assert "predictions" in response_arg, (
            "log_predictions was not called with a dict containing 'predictions'"
        )

    # -----------------------------------------------------------------------
    # AC 7 — log failure must NOT degrade serving; HTTP 200 still returned
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("exc_cls", [OSError, RuntimeError, ValueError])
    def test_ac7_log_failure_does_not_degrade_serving_returns_200(
        self, exc_cls: type[Exception]
    ) -> None:
        """AC 7: if log_predictions raises any exception the handler catches it,
        emits a warning, and still returns HTTP 200 with predictions in the body.

        Parametrized over OSError, RuntimeError, and ValueError to verify the
        production ``except Exception`` guard covers the full contract.

        Arrange: inject predictions into module globals; patch log_predictions to
                 raise exc_cls("boom").
        Act: GET /heatmap.
        Assert: HTTP 200; response body contains 'predictions'.
        """
        import app.api.main as serving_module

        orig_df = serving_module._predictions_df
        orig_zone = serving_module._zone_lookup
        orig_centroid = serving_module._centroid_lookup
        orig_gen = serving_module._generated_at

        serving_module._predictions_df = _SAMPLE_PREDICTIONS
        serving_module._zone_lookup = _ZONE_LOOKUP
        serving_module._centroid_lookup = _CENTROID_LOOKUP
        serving_module._generated_at = "2021-03-01T12:00:00+00:00"

        try:
            with patch.object(
                serving_module.prediction_log,
                "log_predictions",
                MagicMock(side_effect=exc_cls("boom")),
            ):
                local_client = TestClient(serving_module.app)
                resp = local_client.get("/heatmap")
        finally:
            serving_module._predictions_df = orig_df
            serving_module._zone_lookup = orig_zone
            serving_module._centroid_lookup = orig_centroid
            serving_module._generated_at = orig_gen

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert "predictions" in body
        assert len(body["predictions"]) > 0

    # -----------------------------------------------------------------------
    # AC 8 — existing /heatmap criteria remain satisfied
    # -----------------------------------------------------------------------

    def test_ac8_existing_heatmap_criteria_zone_count_and_sort(
        self, api_client: TestClient
    ) -> None:
        """AC 8 (zone count + descending sort): after the PBI 22 hook is wired,
        the response must still contain zone_count == len(predictions) and
        predictions sorted descending by predicted_trip_count.

        Arrange: api_client fixture (3-zone data for 2 hours).
        Act: GET /heatmap.
        Assert: zone_count == 3; predictions are descending by trip count.
        """
        resp = api_client.get("/heatmap")

        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["zone_count"] == len(body["predictions"])
        counts = [p["predicted_trip_count"] for p in body["predictions"]]
        assert counts == sorted(counts, reverse=True)

    def test_ac8_existing_heatmap_criteria_503_on_missing_predictions(self) -> None:
        """AC 8 (HTTP 503): /heatmap must still return 503 when no predictions
        are loaded, even with the PBI 22 hook present.

        Arrange: _predictions_df set to None.
        Act: GET /heatmap.
        Assert: 503 with 'detail' in body.
        """
        import app.api.main as serving_module

        orig_df = serving_module._predictions_df
        serving_module._predictions_df = None
        try:
            local_client = TestClient(serving_module.app)
            resp = local_client.get("/heatmap")
        finally:
            serving_module._predictions_df = orig_df

        assert resp.status_code == 503
        assert "detail" in resp.json()

    def test_ac8_existing_heatmap_criteria_404_on_unknown_hour(
        self, api_client: TestClient
    ) -> None:
        """AC 8 (HTTP 404): requesting a non-existent hour must return 404."""
        resp = api_client.get("/heatmap", params={"hour": "2099-01-01T00:00:00"})
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_ac8_existing_heatmap_criteria_400_on_malformed_hour(
        self, api_client: TestClient
    ) -> None:
        """AC 8 (HTTP 400): a malformed hour parameter must return 400."""
        resp = api_client.get("/heatmap", params={"hour": "not-a-date"})
        assert resp.status_code == 400
        assert "detail" in resp.json()

    # -----------------------------------------------------------------------
    # AC 9 — join_actuals importable signature; end_date inclusive
    # -----------------------------------------------------------------------

    def test_ac9_join_actuals_importable_with_correct_signature(self) -> None:
        """AC 9: join_actuals is importable from app.monitoring.actuals_join and
        has the five documented parameters.

        Arrange: import the module.
        Act: inspect the function.
        Assert: callable with start_date, end_date, predictions_root, gold_path,
                output_root parameters present.
        """
        import inspect

        import app.monitoring.actuals_join as mod

        assert hasattr(mod, "join_actuals")
        func = mod.join_actuals
        assert callable(func)
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        for expected in (
            "start_date",
            "end_date",
            "predictions_root",
            "gold_path",
            "output_root",
        ):
            assert expected in param_names, f"'{expected}' parameter missing"

    def test_ac9_join_actuals_end_date_is_inclusive(self, tmp_path: Path) -> None:
        """AC 9 (end_date inclusive): rows from the end_date itself must appear
        in the joined output — it is an inclusive boundary.

        Arrange: prediction parquet only for 2024-11-16 (= end_date).
        Act: join_actuals(start=2024-11-15, end=2024-11-16).
        Assert: joined output contains the rows from 2024-11-16 (2 zones).
        """
        from app.monitoring.actuals_join import join_actuals

        pred_root = tmp_path / "predictions"
        gold_dir = tmp_path / "gold"
        out_root = tmp_path / "output"

        dt_dir = pred_root / "dt=2024-11-16"
        dt_dir.mkdir(parents=True)
        _write_prediction_parquet(
            dt_dir / "preds.parquet",
            [1, 2],
            "2024-11-16T01:00:00",
        )
        _write_gold_parquet(
            gold_dir,
            [(1, "2024-11-16T01:00:00", 5.0), (2, "2024-11-16T01:00:00", 3.0)],
        )

        result = join_actuals(
            start_date=date(2024, 11, 15),
            end_date=date(2024, 11, 16),
            predictions_root=str(pred_root),
            gold_path=str(gold_dir),
            output_root=str(out_root),
        )

        joined = pd.read_parquet(result)
        assert len(joined) == 2, (
            "end_date rows must be included (end_date is inclusive)"
        )

    # -----------------------------------------------------------------------
    # AC 10 — reads all dt= dirs in range; missing dates silently skipped
    # -----------------------------------------------------------------------

    def test_ac10_reads_multi_day_range_and_skips_missing_dates(
        self, tmp_path: Path
    ) -> None:
        """AC 10: join_actuals reads all dt=YYYY-MM-DD dirs in [start, end];
        dates with no directory are silently skipped.

        Arrange: predictions exist for 2024-11-15 (2 zones) and 2024-11-17
                 (1 zone); 2024-11-16 directory is absent.
        Act: join_actuals(start=2024-11-15, end=2024-11-17).
        Assert: 3 total rows (2 + 0 + 1); no exception raised.
        """
        from app.monitoring.actuals_join import join_actuals

        pred_root = tmp_path / "predictions"
        gold_dir = tmp_path / "gold"
        out_root = tmp_path / "output"

        for dt, zones in [("2024-11-15", [1, 2]), ("2024-11-17", [3])]:
            dt_dir = pred_root / f"dt={dt}"
            dt_dir.mkdir(parents=True)
            _write_prediction_parquet(dt_dir / "preds.parquet", zones, f"{dt}T01:00:00")

        _write_gold_parquet(
            gold_dir,
            [
                (1, "2024-11-15T01:00:00", 10.0),
                (2, "2024-11-15T01:00:00", 8.0),
                (3, "2024-11-17T01:00:00", 6.0),
            ],
        )

        result = join_actuals(
            start_date=date(2024, 11, 15),
            end_date=date(2024, 11, 17),
            predictions_root=str(pred_root),
            gold_path=str(gold_dir),
            output_root=str(out_root),
        )

        joined = pd.read_parquet(result)
        assert len(joined) == 3, (
            "Expected 3 rows across 2 dates; missing date silently skipped"
        )

    # -----------------------------------------------------------------------
    # AC 11 — left-join semantics: no rows dropped; NaN where Gold missing
    # -----------------------------------------------------------------------

    def test_ac11_left_join_preserves_all_prediction_rows(self, tmp_path: Path) -> None:
        """AC 11 (no rows dropped): every prediction row must appear in the
        joined output regardless of whether Gold has a matching entry.

        Arrange: 3 zones predicted; Gold only covers 2 of them.
        Act: join_actuals.
        Assert: joined row count == 3; zone with no Gold match has actual == NaN.
        """
        from app.monitoring.actuals_join import join_actuals

        pred_root = tmp_path / "predictions"
        gold_dir = tmp_path / "gold"
        out_root = tmp_path / "output"

        dt_dir = pred_root / "dt=2024-11-15"
        dt_dir.mkdir(parents=True)
        _write_prediction_parquet(
            dt_dir / "preds.parquet", [1, 2, 3], "2024-11-15T01:00:00"
        )
        # Zone 3 intentionally absent from Gold
        _write_gold_parquet(
            gold_dir,
            [
                (1, "2024-11-15T01:00:00", 15.0),
                (2, "2024-11-15T01:00:00", 8.0),
            ],
        )

        result = join_actuals(
            start_date=date(2024, 11, 15),
            end_date=date(2024, 11, 15),
            predictions_root=str(pred_root),
            gold_path=str(gold_dir),
            output_root=str(out_root),
        )

        joined = pd.read_parquet(result)
        assert len(joined) == 3, "All 3 prediction rows must be retained"
        zone3 = joined[joined["pulocationid"] == 3]
        assert len(zone3) == 1
        assert pd.isna(zone3.iloc[0]["actual"]), (
            "Zone absent from Gold must have actual == NaN"
        )

    def test_ac11_actual_populated_where_gold_matches(self, tmp_path: Path) -> None:
        """AC 11 (actual populated): where Gold has a matching (pulocationid, hour)
        the 'actual' column must equal the Gold trip_count (not NaN).

        Arrange: 1 zone predicted; Gold has that zone.
        Act: join_actuals.
        Assert: actual == Gold trip_count for that zone.
        """
        from app.monitoring.actuals_join import join_actuals

        pred_root = tmp_path / "predictions"
        gold_dir = tmp_path / "gold"
        out_root = tmp_path / "output"

        dt_dir = pred_root / "dt=2024-11-15"
        dt_dir.mkdir(parents=True)
        _write_prediction_parquet(dt_dir / "preds.parquet", [42], "2024-11-15T01:00:00")
        _write_gold_parquet(gold_dir, [(42, "2024-11-15T01:00:00", 99.0)])

        result = join_actuals(
            start_date=date(2024, 11, 15),
            end_date=date(2024, 11, 15),
            predictions_root=str(pred_root),
            gold_path=str(gold_dir),
            output_root=str(out_root),
        )

        joined = pd.read_parquet(result)
        row = joined[joined["pulocationid"] == 42].iloc[0]
        assert not pd.isna(row["actual"])
        assert row["actual"] == pytest.approx(99.0)

    # -----------------------------------------------------------------------
    # AC 12 — output path pattern and returned schema
    # -----------------------------------------------------------------------

    def test_ac12_output_path_follows_pattern_and_schema_is_correct(
        self, tmp_path: Path
    ) -> None:
        """AC 12: the joined parquet is written to
        {output_root}/joined/dt={start}_to_{end}.parquet and the returned Path
        matches that location.  The parquet schema must contain the seven
        documented columns.

        Arrange: 2-zone predictions for one date; matching Gold.
        Act: join_actuals(start=2024-11-01, end=2024-11-30).
        Assert: returned filename == 'dt=2024-11-01_to_2024-11-30.parquet';
                file exists; columns match documented schema.
        """
        from app.monitoring.actuals_join import join_actuals

        pred_root = tmp_path / "predictions"
        gold_dir = tmp_path / "gold"
        out_root = tmp_path / "output"

        dt_dir = pred_root / "dt=2024-11-01"
        dt_dir.mkdir(parents=True)
        _write_prediction_parquet(
            dt_dir / "preds.parquet", [1, 2], "2024-11-01T00:00:00"
        )
        _write_gold_parquet(
            gold_dir,
            [(1, "2024-11-01T00:00:00", 7.0), (2, "2024-11-01T00:00:00", 4.0)],
        )

        result = join_actuals(
            start_date=date(2024, 11, 1),
            end_date=date(2024, 11, 30),
            predictions_root=str(pred_root),
            gold_path=str(gold_dir),
            output_root=str(out_root),
        )

        # Path pattern
        assert result.name == "dt=2024-11-01_to_2024-11-30.parquet"
        assert result.exists()

        # Schema
        df = pd.read_parquet(result)
        expected_cols = {
            "bucket_time",
            "pulocationid",
            "predicted",
            "actual",
            "model_version",
            "served_from_stage",
            "served_at",
        }
        assert set(df.columns) == expected_cols

    # -----------------------------------------------------------------------
    # AC 13 — CLI entry point via subprocess
    # -----------------------------------------------------------------------

    def test_ac13_cli_runs_without_error_and_writes_joined_parquet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 13: the CLI accepts --start-date and --end-date; when valid
        prediction parquets and Gold exist it exits 0 and writes the joined file.

        Approach: write a helper script into tmp_path that patches load_config
        and then calls join_actuals with the parsed CLI args.  This avoids
        f-string escaping issues in a subprocess -c snippet while still
        exercising the public CLI surface (_parse_args + join_actuals).

        Arrange: write prediction parquet and Gold into tmp_path; write a
                 minimal config.yml and a thin runner script.
        Act: subprocess.run the runner script with --start-date / --end-date.
        Assert: exit code 0; joined parquet exists.
        """
        # Set up on-disk artefacts
        pred_root = tmp_path / "predictions"
        gold_dir = tmp_path / "gold"
        out_root = tmp_path / "joined_output"

        dt_dir = pred_root / "dt=2024-11-15"
        dt_dir.mkdir(parents=True)
        _write_prediction_parquet(
            dt_dir / "preds.parquet", [1, 2], "2024-11-15T01:00:00"
        )
        _write_gold_parquet(
            gold_dir,
            [(1, "2024-11-15T01:00:00", 5.0), (2, "2024-11-15T01:00:00", 3.0)],
        )

        # Minimal config dict that the CLI's load_config will return.
        config_data = {
            "data": {"gold_path": str(gold_dir)},
            "monitoring": {
                "predictions_path": str(pred_root),
                "joined_path": str(out_root),
            },
        }

        # Write a runner script so we avoid multi-level f-string escaping.
        runner_script = tmp_path / "run_cli.py"
        runner_script.write_text(
            f"""\
import sys
from unittest.mock import patch

cfg = {config_data!r}

import app.monitoring.actuals_join as m

with patch("app.monitoring.actuals_join.load_config", return_value=cfg):
    args = m._parse_args()
    monitoring = cfg.get("monitoring", {{}})
    out = m.join_actuals(
        start_date=args.start_date,
        end_date=args.end_date,
        predictions_root=monitoring["predictions_path"],
        gold_path=cfg["data"]["gold_path"],
        output_root=monitoring["joined_path"],
    )
    print("Written to:", out)
"""
        )

        import os

        # The subprocess needs the project root on PYTHONPATH so it can import
        # app.monitoring.actuals_join.
        project_root = str(Path(__file__).parent.parent)
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            project_root
            if not existing_pythonpath
            else f"{project_root}{os.pathsep}{existing_pythonpath}"
        )

        result = subprocess.run(
            [
                sys.executable,
                str(runner_script),
                "--start-date",
                "2024-11-15",
                "--end-date",
                "2024-11-15",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )

        assert result.returncode == 0, (
            f"CLI runner exited non-zero.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        expected_file = out_root / "joined" / "dt=2024-11-15_to_2024-11-15.parquet"
        assert expected_file.exists(), (
            f"Expected joined parquet not found at {expected_file}"
        )

    # -----------------------------------------------------------------------
    # AC 14 — config.yml has monitoring: section
    # -----------------------------------------------------------------------

    def test_ac14_config_yml_has_monitoring_section_with_required_keys(self) -> None:
        """AC 14: the committed config.yml must contain a top-level 'monitoring:'
        section with 'predictions_path' and 'joined_path' keys.

        Arrange: read config.yml from the project root.
        Act: parse with yaml.safe_load.
        Assert: monitoring key present; both sub-keys present with expected values.
        """
        config_path = Path(__file__).parent.parent / "config.yml"
        assert config_path.exists(), f"config.yml not found at {config_path}"

        with config_path.open() as fh:
            cfg = yaml.safe_load(fh)

        assert "monitoring" in cfg, "config.yml missing top-level 'monitoring:' section"
        monitoring = cfg["monitoring"]
        assert "predictions_path" in monitoring, (
            "monitoring: section missing 'predictions_path'"
        )
        assert "joined_path" in monitoring, "monitoring: section missing 'joined_path'"
        assert monitoring["predictions_path"] == "data/monitoring/predictions"
        assert monitoring["joined_path"] == "data/monitoring/joined"

    # -----------------------------------------------------------------------
    # AC 15 + AC 16 — structural inspection of tests/test_monitoring.py
    # -----------------------------------------------------------------------

    def test_ac15_test_prediction_log_class_has_required_test_names(self) -> None:
        """AC 15: TestPredictionLog in tests/test_monitoring.py must contain
        at least the three test methods named in the story.  The story specifies
        stem names; the builder may use longer descriptive names that contain
        those stems.  We also accept the builder's actual names directly as they
        cover identical semantics.

        Approach: load the module via importlib and inspect class attributes.
        """
        spec = importlib.util.spec_from_file_location(
            "test_monitoring",
            Path(__file__).parent / "test_monitoring.py",
        )
        assert spec is not None
        tm = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(tm)  # type: ignore[union-attr]

        cls = getattr(tm, "TestPredictionLog", None)
        assert cls is not None, (
            "TestPredictionLog class not found in test_monitoring.py"
        )

        method_names = [m for m in dir(cls) if m.startswith("test_")]

        # The story requires these three semantic intents; accept any method name
        # that contains the stem keyword OR the builder's full actual name.
        # Stems from story → acceptable builder alternatives
        required_stems_or_aliases = [
            # "happy path writes file" or "valid response writes file"
            ("happy_path_writes_file", "valid_response_writes_file"),
            # "schema columns" — uniquely identifiable
            ("schema_columns",),
            # "multiple calls same day produce separate files" or variant
            ("multiple_calls_same_day", "two_calls_same_day"),
        ]
        for alternatives in required_stems_or_aliases:
            match = any(any(alt in m for alt in alternatives) for m in method_names)
            assert match, (
                f"TestPredictionLog is missing a test covering one of {alternatives}. "
                f"Found methods: {method_names}"
            )

    def test_ac16_test_actuals_join_class_has_required_test_names(self) -> None:
        """AC 16: TestActualsJoin in tests/test_monitoring.py must contain at
        least the three test methods named in the story.  Same fuzzy-match
        approach as AC 15 — builder may use longer prefixed names.
        """
        spec = importlib.util.spec_from_file_location(
            "test_monitoring",
            Path(__file__).parent / "test_monitoring.py",
        )
        assert spec is not None
        tm = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(tm)  # type: ignore[union-attr]

        cls = getattr(tm, "TestActualsJoin", None)
        assert cls is not None, "TestActualsJoin class not found in test_monitoring.py"

        method_names = [m for m in dir(cls) if m.startswith("test_")]

        required_stems_or_aliases = [
            # "row_count_matches_predictions" (with or without join_actuals prefix)
            ("row_count_matches_predictions",),
            # "actual_non_null_where_gold_matches"
            ("actual_non_null_where_gold_matches",),
            # "actual_null_where_gold_gap"
            ("actual_null_where_gold_gap",),
        ]
        for alternatives in required_stems_or_aliases:
            match = any(any(alt in m for alt in alternatives) for m in method_names)
            assert match, (
                f"TestActualsJoin is missing a test covering one of {alternatives}. "
                f"Found methods: {method_names}"
            )

    # -----------------------------------------------------------------------
    # Edge case: midnight UTC boundary → different dt= dirs
    # -----------------------------------------------------------------------

    def test_edge_midnight_utc_boundary_different_dt_dirs(self, tmp_path: Path) -> None:
        """Edge case: a call at 23:59:59 UTC and one at 00:00:01 UTC the next day
        must produce files in two different dt= directories.

        Arrange: two fixed datetimes straddling UTC midnight.
        Act: two calls to log_predictions with those timestamps.
        Assert: parent directories differ.
        """
        import app.monitoring.prediction_log as mod

        response = _make_heatmap_response(n_zones=2)
        dt_before = datetime(2024, 11, 15, 23, 59, 59, tzinfo=timezone.utc)
        dt_after = datetime(2024, 11, 16, 0, 0, 1, tzinfo=timezone.utc)

        with patch.object(mod, "_now", return_value=dt_before):
            path_before = mod.log_predictions(response, str(tmp_path))
        with patch.object(mod, "_now", return_value=dt_after):
            path_after = mod.log_predictions(response, str(tmp_path))

        assert path_before is not None and path_after is not None
        assert path_before.parent != path_after.parent, (
            "Requests straddling UTC midnight must land in different dt= dirs"
        )
        assert "dt=2024-11-15" in str(path_before)
        assert "dt=2024-11-16" in str(path_after)

    # -----------------------------------------------------------------------
    # Edge case: nullable model_version / served_from_stage accepted
    # -----------------------------------------------------------------------

    def test_edge_nullable_fields_write_as_null_not_raise(self, tmp_path: Path) -> None:
        """Edge case: model_version=None and served_from_stage=None must be
        accepted and written as null columns (no exception raised).

        Arrange: response with both nullable fields set to None.
        Act: call log_predictions.
        Assert: file written; those columns contain only null values.
        """
        import app.monitoring.prediction_log as mod

        fixed_dt = datetime(2024, 11, 15, 10, 0, 0, tzinfo=timezone.utc)
        response = _make_heatmap_response(
            n_zones=2, model_version=None, served_from_stage=None
        )

        with patch.object(mod, "_now", return_value=fixed_dt):
            result = mod.log_predictions(response, str(tmp_path))

        assert result is not None
        df = pd.read_parquet(result)
        assert df["model_version"].isna().all()
        assert df["served_from_stage"].isna().all()

    # -----------------------------------------------------------------------
    # Edge case: Gold absent → clear FileNotFoundError
    # -----------------------------------------------------------------------

    def test_edge_gold_absent_raises_file_not_found_error(self, tmp_path: Path) -> None:
        """Edge case: when the Gold parquet does not exist, join_actuals must
        raise FileNotFoundError with a non-empty message — not silently produce
        an empty join.

        Arrange: prediction parquets exist; Gold directory absent.
        Act: join_actuals.
        Assert: FileNotFoundError raised; message is non-empty.
        """
        from app.monitoring.actuals_join import join_actuals

        pred_root = tmp_path / "predictions"
        gold_dir = tmp_path / "gold_missing"  # intentionally absent
        out_root = tmp_path / "output"

        dt_dir = pred_root / "dt=2024-11-15"
        dt_dir.mkdir(parents=True)
        _write_prediction_parquet(dt_dir / "preds.parquet", [1], "2024-11-15T01:00:00")

        with pytest.raises(FileNotFoundError) as exc_info:
            join_actuals(
                start_date=date(2024, 11, 15),
                end_date=date(2024, 11, 15),
                predictions_root=str(pred_root),
                gold_path=str(gold_dir),
                output_root=str(out_root),
            )

        assert str(exc_info.value), "FileNotFoundError message must be non-empty"

    # -----------------------------------------------------------------------
    # Edge case: future bucket_time → actual == NaN (expected, not an error)
    # -----------------------------------------------------------------------

    def test_edge_future_bucket_time_yields_nan_actual(self, tmp_path: Path) -> None:
        """Edge case: predictions whose bucket_time lies beyond the Gold window
        must have actual == NaN in the output.  This is expected behaviour, not
        an error, because Gold for the future has not been produced yet.

        Arrange: prediction for a future date (2099-01-01); Gold only covers 2024.
        Act: join_actuals.
        Assert: actual == NaN for the future-dated row; function completes
                without error.
        """
        from app.monitoring.actuals_join import join_actuals

        pred_root = tmp_path / "predictions"
        gold_dir = tmp_path / "gold"
        out_root = tmp_path / "output"

        dt_dir = pred_root / "dt=2099-01-01"
        dt_dir.mkdir(parents=True)
        _write_prediction_parquet(dt_dir / "preds.parquet", [1], "2099-01-01T00:00:00")
        # Gold has 2024 data only
        _write_gold_parquet(gold_dir, [(1, "2024-11-15T01:00:00", 10.0)])

        result = join_actuals(
            start_date=date(2099, 1, 1),
            end_date=date(2099, 1, 1),
            predictions_root=str(pred_root),
            gold_path=str(gold_dir),
            output_root=str(out_root),
        )

        joined = pd.read_parquet(result)
        assert len(joined) == 1
        assert pd.isna(joined.iloc[0]["actual"]), (
            "Future prediction with no Gold match must have actual == NaN"
        )

    # -----------------------------------------------------------------------
    # Edge case: output directory created on first run
    # -----------------------------------------------------------------------

    def test_edge_output_joined_dir_created_on_first_run(self, tmp_path: Path) -> None:
        """Edge case: the {output_root}/joined/ directory need not pre-exist;
        join_actuals must create it with os.makedirs(..., exist_ok=True).

        Arrange: out_root points to a path that does not yet exist.
        Act: join_actuals.
        Assert: joined directory and output parquet were created.
        """
        from app.monitoring.actuals_join import join_actuals

        pred_root = tmp_path / "predictions"
        gold_dir = tmp_path / "gold"
        out_root = tmp_path / "brand_new_output"  # does not exist

        dt_dir = pred_root / "dt=2024-11-15"
        dt_dir.mkdir(parents=True)
        _write_prediction_parquet(dt_dir / "preds.parquet", [1], "2024-11-15T01:00:00")
        _write_gold_parquet(gold_dir, [(1, "2024-11-15T01:00:00", 5.0)])

        result = join_actuals(
            start_date=date(2024, 11, 15),
            end_date=date(2024, 11, 15),
            predictions_root=str(pred_root),
            gold_path=str(gold_dir),
            output_root=str(out_root),
        )

        assert result.exists(), "Output parquet must be written even on first run"
        assert result.parent.name == "joined"
