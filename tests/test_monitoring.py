"""Tests for app/monitoring/prediction_log.py and app/monitoring/actuals_join.py."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

import app.monitoring.prediction_log as prediction_log_module
from app.monitoring.actuals_join import join_actuals
from app.monitoring.prediction_log import log_predictions


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_BUCKET_ISO = "2024-11-15T01:00:00"
_FIXED_DT = datetime(2024, 11, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_response(
    zones: int = 3,
    model_version: str | None = "5",
    served_from_stage: str | None = "Production",
) -> dict[str, Any]:
    """Build a minimal /heatmap response dict with *zones* prediction entries."""
    predictions = [
        {
            "pulocationid": i + 1,
            "predicted_trip_count": float(10 + i),
            "bucket_time": _BUCKET_ISO,
            "zone_name": f"Zone {i + 1}",
        }
        for i in range(zones)
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
) -> None:
    """Write a minimal prediction parquet to *path* (directory must exist)."""
    served_at = pd.Timestamp("2024-11-15T12:00:00", tz="UTC")
    rows = [
        {
            "bucket_time": pd.Timestamp(bucket_time, tz="UTC"),
            "pulocationid": z,
            "predicted": float(10 + z),
            "model_version": model_version,
            "served_from_stage": "Production",
            "served_at": served_at,
        }
        for z in pulocationids
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# TestPredictionLog
# ---------------------------------------------------------------------------


class TestPredictionLog:
    """Unit tests for app.monitoring.prediction_log.log_predictions."""

    def test_log_predictions_valid_response_writes_file(self, tmp_path: Path) -> None:
        """Arrange 3-zone response; act; assert returned Path exists and is readable.

        Covers AC 1, 2.
        """
        # Arrange
        response = _make_response(zones=3)

        # Act
        with patch.object(prediction_log_module, "_now", return_value=_FIXED_DT):
            result = log_predictions(response, str(tmp_path))

        # Assert
        assert result is not None
        assert result.exists()
        df = pd.read_parquet(result)
        assert len(df) == 3

    def test_log_predictions_valid_response_schema_columns(
        self, tmp_path: Path
    ) -> None:
        """Written parquet has exactly the six required columns.

        Covers AC 3.
        """
        # Arrange
        response = _make_response(zones=3)
        expected_cols = {
            "bucket_time",
            "pulocationid",
            "predicted",
            "model_version",
            "served_from_stage",
            "served_at",
        }

        # Act
        with patch.object(prediction_log_module, "_now", return_value=_FIXED_DT):
            result = log_predictions(response, str(tmp_path))

        # Assert
        assert result is not None
        df = pd.read_parquet(result)
        assert set(df.columns) == expected_cols

    def test_log_predictions_two_calls_same_day_separate_files(
        self, tmp_path: Path
    ) -> None:
        """Two calls on the same UTC day produce two distinct non-zero files.

        Covers AC 4.
        """
        # Arrange
        response = _make_response(zones=3)

        # Act
        with patch.object(prediction_log_module, "_now", return_value=_FIXED_DT):
            path1 = log_predictions(response, str(tmp_path))
            path2 = log_predictions(response, str(tmp_path))

        # Assert
        assert path1 is not None and path2 is not None
        assert path1 != path2
        assert path1.exists() and path2.exists()
        assert path1.stat().st_size > 0
        assert path2.stat().st_size > 0

    def test_log_predictions_empty_predictions_returns_none(
        self, tmp_path: Path
    ) -> None:
        """Empty predictions list returns None and writes no file.

        Covers AC 5 + resolved decision 1.
        """
        # Arrange
        response: dict[str, Any] = {
            "predictions": [],
            "model_version": "5",
            "served_from_stage": "Production",
        }

        # Act
        result = log_predictions(response, str(tmp_path))

        # Assert
        assert result is None
        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) == 0

    def test_log_predictions_null_model_version_writes_null(
        self, tmp_path: Path
    ) -> None:
        """model_version=None and served_from_stage=None write null columns.

        Covers nullable fields edge case.
        """
        # Arrange
        response = _make_response(zones=2, model_version=None, served_from_stage=None)

        # Act
        with patch.object(prediction_log_module, "_now", return_value=_FIXED_DT):
            result = log_predictions(response, str(tmp_path))

        # Assert
        assert result is not None
        df = pd.read_parquet(result)
        assert df["model_version"].isna().all()
        assert df["served_from_stage"].isna().all()

    def test_log_predictions_bucket_time_is_utc_timestamp(self, tmp_path: Path) -> None:
        """bucket_time dtype in written parquet is datetime64[us, UTC].

        Covers ISO string parsing edge case.
        """
        # Arrange
        response = _make_response(zones=2)

        # Act
        with patch.object(prediction_log_module, "_now", return_value=_FIXED_DT):
            result = log_predictions(response, str(tmp_path))

        # Assert
        assert result is not None
        df = pd.read_parquet(result)
        # The dtype string for tz-aware UTC timestamps includes "UTC"
        assert "UTC" in str(df["bucket_time"].dtype)

    def test_log_predictions_path_pattern_contains_dt_partition(
        self, tmp_path: Path
    ) -> None:
        """Returned path contains dt=YYYY-MM-DD partition directory.

        Covers AC 2 (path pattern).
        """
        # Arrange
        response = _make_response(zones=1)

        # Act
        with patch.object(prediction_log_module, "_now", return_value=_FIXED_DT):
            result = log_predictions(response, str(tmp_path))

        # Assert
        assert result is not None
        assert "dt=2024-11-15" in str(result)
        assert result.name.startswith("served_at=")


# ---------------------------------------------------------------------------
# TestActualsJoin
# ---------------------------------------------------------------------------


class TestActualsJoin:
    """Unit tests for app.monitoring.actuals_join.join_actuals."""

    def _make_gold_parquet(
        self, path: Path, rows: list[tuple[int, str, float]]
    ) -> None:
        """Write a minimal Gold parquet with (pulocationid, hour, trip_count) rows."""
        df = pd.DataFrame(
            {
                "pulocationid": [r[0] for r in rows],
                "hour": pd.to_datetime([r[1] for r in rows]),
                "trip_count": [r[2] for r in rows],
                # Gold may have other feature columns; add a dummy to verify they drop.
                "trip_count_lag_1": [0.0] * len(rows),
            }
        )
        path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path / "hourly_demand_featured.parquet", index=False)

    def _make_pred_dir(
        self, root: Path, dt: str, zones: list[int], bucket_time: str
    ) -> None:
        """Write a prediction parquet into {root}/dt={dt}/."""
        dt_dir = root / f"dt={dt}"
        dt_dir.mkdir(parents=True, exist_ok=True)
        _write_prediction_parquet(dt_dir / "served_at=xxx.parquet", zones, bucket_time)

    def test_join_actuals_row_count_matches_predictions(self, tmp_path: Path) -> None:
        """Joined output has the same number of rows as total prediction rows.

        Covers AC 10, 11, 12.
        """
        # Arrange
        pred_root = tmp_path / "predictions"
        gold_root = tmp_path / "gold"
        out_root = tmp_path / "joined_output"

        bucket = "2024-11-15T01:00:00"
        zones = [1, 2]
        self._make_pred_dir(pred_root, "2024-11-15", zones, bucket)
        self._make_gold_parquet(
            gold_root,
            [(1, "2024-11-15T01:00:00", 15.0), (2, "2024-11-15T01:00:00", 8.0)],
        )

        # Act
        result = join_actuals(
            start_date=date(2024, 11, 15),
            end_date=date(2024, 11, 15),
            predictions_root=str(pred_root),
            gold_path=str(gold_root),
            output_root=str(out_root),
        )

        # Assert
        joined = pd.read_parquet(result)
        assert len(joined) == len(zones)

    def test_join_actuals_actual_non_null_where_gold_matches(
        self, tmp_path: Path
    ) -> None:
        """actual is non-null for rows where Gold has a matching (pulocationid, hour).

        Covers AC 11.
        """
        # Arrange
        pred_root = tmp_path / "predictions"
        gold_root = tmp_path / "gold"
        out_root = tmp_path / "joined_output"

        bucket = "2024-11-15T01:00:00"
        self._make_pred_dir(pred_root, "2024-11-15", [1], bucket)
        self._make_gold_parquet(gold_root, [(1, "2024-11-15T01:00:00", 42.0)])

        # Act
        result = join_actuals(
            start_date=date(2024, 11, 15),
            end_date=date(2024, 11, 15),
            predictions_root=str(pred_root),
            gold_path=str(gold_root),
            output_root=str(out_root),
        )

        # Assert
        joined = pd.read_parquet(result)
        row = joined[joined["pulocationid"] == 1].iloc[0]
        assert not pd.isna(row["actual"])
        assert row["actual"] == pytest.approx(42.0)

    def test_join_actuals_actual_null_where_gold_gap(self, tmp_path: Path) -> None:
        """actual is NaN for rows where Gold has no matching zone.

        Covers AC 11 (left-join semantics).
        """
        # Arrange
        pred_root = tmp_path / "predictions"
        gold_root = tmp_path / "gold"
        out_root = tmp_path / "joined_output"

        bucket = "2024-11-15T01:00:00"
        # Zone 2 is in predictions but NOT in Gold.
        self._make_pred_dir(pred_root, "2024-11-15", [1, 2], bucket)
        self._make_gold_parquet(gold_root, [(1, "2024-11-15T01:00:00", 10.0)])

        # Act
        result = join_actuals(
            start_date=date(2024, 11, 15),
            end_date=date(2024, 11, 15),
            predictions_root=str(pred_root),
            gold_path=str(gold_root),
            output_root=str(out_root),
        )

        # Assert
        joined = pd.read_parquet(result)
        zone2 = joined[joined["pulocationid"] == 2]
        assert len(zone2) == 1
        assert pd.isna(zone2.iloc[0]["actual"])

    def test_join_actuals_missing_gold_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        """FileNotFoundError is raised with a non-empty message when Gold is absent.

        Covers edge case from story.
        """
        # Arrange
        pred_root = tmp_path / "predictions"
        gold_root = tmp_path / "gold_missing"
        out_root = tmp_path / "joined_output"

        self._make_pred_dir(pred_root, "2024-11-15", [1], "2024-11-15T01:00:00")

        # Act / Assert
        with pytest.raises(FileNotFoundError, match=r".+"):
            join_actuals(
                start_date=date(2024, 11, 15),
                end_date=date(2024, 11, 15),
                predictions_root=str(pred_root),
                gold_path=str(gold_root),
                output_root=str(out_root),
            )

    def test_join_actuals_missing_date_dir_silently_skipped(
        self, tmp_path: Path
    ) -> None:
        """A date with no prediction directory is silently skipped.

        Covers AC 10 + edge case for absent dates.
        """
        # Arrange
        pred_root = tmp_path / "predictions"
        gold_root = tmp_path / "gold"
        out_root = tmp_path / "joined_output"

        bucket = "2024-11-15T01:00:00"
        # Only 2024-11-15 has files; 2024-11-16 directory is absent.
        self._make_pred_dir(pred_root, "2024-11-15", [1, 2], bucket)
        self._make_gold_parquet(
            gold_root,
            [(1, "2024-11-15T01:00:00", 5.0), (2, "2024-11-15T01:00:00", 3.0)],
        )

        # Act — range includes a day with no files
        result = join_actuals(
            start_date=date(2024, 11, 15),
            end_date=date(2024, 11, 16),
            predictions_root=str(pred_root),
            gold_path=str(gold_root),
            output_root=str(out_root),
        )

        # Assert — only rows from 2024-11-15 present, no exception raised
        joined = pd.read_parquet(result)
        assert len(joined) == 2

    def test_join_actuals_output_path_matches_pattern(self, tmp_path: Path) -> None:
        """Returned Path filename equals dt={start}_to_{end}.parquet.

        Covers AC 12 + resolved decision 3.
        """
        # Arrange
        pred_root = tmp_path / "predictions"
        gold_root = tmp_path / "gold"
        out_root = tmp_path / "joined_output"

        self._make_pred_dir(pred_root, "2024-11-01", [1], "2024-11-01T00:00:00")
        self._make_gold_parquet(gold_root, [(1, "2024-11-01T00:00:00", 7.0)])

        # Act
        result = join_actuals(
            start_date=date(2024, 11, 1),
            end_date=date(2024, 11, 30),
            predictions_root=str(pred_root),
            gold_path=str(gold_root),
            output_root=str(out_root),
        )

        # Assert
        assert result.name == "dt=2024-11-01_to_2024-11-30.parquet"
        assert result.exists()
