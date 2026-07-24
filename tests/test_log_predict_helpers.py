"""Tests for orchestration_experiment/dags/exp04_log.py.

PBI EXP-04 (predict + log) — the pure log helper the DAG delegates to.
Inserts a ``log_predict`` task after ``predict``.

The real logic in ``run_log_predict`` is the parquet → heatmap reshape:
reading the predictions parquet and building the response dict that
``log_predictions`` expects. The reshape is tested against real parquet files
(``tmp_path``); ``log_predictions`` is mocked at the import boundary so tests
run without touching the filesystem for the monitoring side.

AC coverage:
  - run_log_predict reads parquet and reshapes to heatmap response with correct keys
  - run_log_predict calls log_predictions once with the response and monitoring_root
  - run_log_predict returns None and skips log_predictions when parquet is missing
  - run_log_predict returns None and skips log_predictions when parquet is empty
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

from log_predict_helpers import run_log_predict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MONITORING_ROOT = "/opt/airflow/data_exp/monitoring/predictions"

_HOUR = datetime(2021, 1, 18, 1, 0, 0)


def _write_predictions_parquet(path: Path, n_rows: int = 3) -> Path:
    """Write a minimal predictions parquet with the Predictor output schema."""
    rows = [
        {
            "pulocationid": i + 1,
            "hour": _HOUR,
            "predicted_trip_count": float(i * 2),
            "actual_trip_count": float(i * 2 + 1),
        }
        for i in range(n_rows)
    ]
    df = pd.DataFrame(rows)
    df["pulocationid"] = df["pulocationid"].astype("int64")
    df["predicted_trip_count"] = df["predicted_trip_count"].astype("float64")
    df["actual_trip_count"] = df["actual_trip_count"].astype("float64")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _write_empty_predictions_parquet(path: Path) -> Path:
    """Write a 0-row parquet with the correct schema."""
    df = pd.DataFrame(
        columns=["pulocationid", "hour", "predicted_trip_count", "actual_trip_count"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# TestRunLogPredict
# ---------------------------------------------------------------------------


class TestRunLogPredict:
    """run_log_predict reshapes predictions parquet into heatmap response."""

    def test_reshapes_parquet_into_heatmap_response(self, tmp_path: Path) -> None:
        """AC: log_predictions called once; response has correct shape and keys."""
        # Arrange
        parquet_path = tmp_path / "predictions" / "demand_predictions.parquet"
        _write_predictions_parquet(parquet_path, n_rows=3)

        with patch(
            "log_predict_helpers.log_predictions"
        ) as mock_log:
            mock_log.return_value = (
                Path(MONITORING_ROOT) / "dt=2021-01-18" / "f.parquet"
            )

            # Act
            run_log_predict(str(parquet_path), monitoring_root=MONITORING_ROOT)

        # Assert — called once
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args

        response: dict[str, Any] = call_kwargs[0][0]
        logged_path: str = call_kwargs[1]["path"]

        # response has the envelope keys
        assert "predictions" in response
        assert "model_version" in response
        assert "served_from_stage" in response

        # each prediction item has the required keys
        predictions = response["predictions"]
        assert len(predictions) == 3
        for item in predictions:
            assert "pulocationid" in item
            assert "predicted_trip_count" in item
            assert "bucket_time" in item
            # bucket_time is an ISO string (log_predictions parses it as Timestamp)
            _ = datetime.fromisoformat(item["bucket_time"])

        # path kwarg is the monitoring root
        assert logged_path == MONITORING_ROOT

    def test_missing_file_skips_without_calling_log(self, tmp_path: Path) -> None:
        """AC: missing parquet → returns None, log_predictions NOT called."""
        # Arrange
        parquet_path = tmp_path / "does_not_exist.parquet"

        with patch(
            "log_predict_helpers.log_predictions"
        ) as mock_log:
            # Act
            result = run_log_predict(str(parquet_path), monitoring_root=MONITORING_ROOT)

        # Assert
        assert result is None
        mock_log.assert_not_called()

    def test_empty_parquet_skips_without_calling_log(self, tmp_path: Path) -> None:
        """AC: 0-row parquet → returns None, log_predictions NOT called (covers EXP-03b empty-gold)."""
        # Arrange
        parquet_path = tmp_path / "predictions" / "demand_predictions.parquet"
        _write_empty_predictions_parquet(parquet_path)

        with patch(
            "log_predict_helpers.log_predictions"
        ) as mock_log:
            # Act
            result = run_log_predict(str(parquet_path), monitoring_root=MONITORING_ROOT)

        # Assert
        assert result is None
        mock_log.assert_not_called()
