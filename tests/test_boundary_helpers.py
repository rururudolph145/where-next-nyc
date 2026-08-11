"""Tests for orchestration_experiment/dags/exp05_boundary.py.

PBI EXP-05 (Idempotency Boundary) — the pure Airflow-free boundary predicate.

``is_dataset_exhausted`` answers whether the next window to process starts at or
past the fixed 2021 dataset maximum. The DAG wrapper raises
``AirflowSkipException`` when this returns True, short-circuiting the pipeline
so the watermark never advances past the dataset end.

AC coverage:
  - returns False when window_start is strictly before dataset_max
  - returns True when window_start equals dataset_max (inclusive boundary)
  - returns True when window_start is past dataset_max

Step 5 (hardcoded-values audit):
  - config["ingest"]["dataset_max"] parses to seconds=57 (transcription guard)
  - _check_no_new_data reads dataset_max from config (not module-level constant)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from boundary_helpers import is_dataset_exhausted  # type: ignore[import-not-found]

# Module-level constant shared across all three tests (EXP idiom: avoid repeated
# datetime literals in AAA blocks).
DATASET_MAX = datetime(2021, 3, 31, 23, 59, 57)


class TestIsDatasetExhausted:
    """Boundary predicate covers before / at / after the dataset maximum."""

    def test_before_boundary_returns_false(self) -> None:
        """A window starting before dataset_max still has data to process."""
        # Arrange
        window_start = datetime(2021, 3, 31, 23, 45, 0)

        # Act
        result = is_dataset_exhausted(window_start, dataset_max=DATASET_MAX)

        # Assert
        assert result is False

    def test_at_boundary_returns_true(self) -> None:
        """A window starting exactly at dataset_max has no remaining data."""
        # Arrange
        window_start = DATASET_MAX

        # Act
        result = is_dataset_exhausted(window_start, dataset_max=DATASET_MAX)

        # Assert
        assert result is True

    def test_after_boundary_returns_true(self) -> None:
        """A window starting past dataset_max is fully beyond the dataset."""
        # Arrange
        window_start = datetime(2021, 4, 1, 0, 0, 0)

        # Act
        result = is_dataset_exhausted(window_start, dataset_max=DATASET_MAX)

        # Assert
        assert result is True


class TestDatasetMaxConfigParsing:
    """Step 5 — dataset_max is config-derived; seconds=57 transcription guard."""

    def test_config_dataset_max_seconds_is_57_not_59(self) -> None:
        """config.yml ingest.dataset_max parses to second=57 (not 59).

        seconds=57 is intentional — deriving from end_year_month yields
        23:59:59 (2 s too late) which caused two overrun DAG runs.
        """
        # Arrange
        from app.core.config import load_config

        config = load_config()

        # Act
        dataset_max = datetime.fromisoformat(config.ingest.dataset_max)

        # Assert
        assert dataset_max.second == 57, (
            f"Expected second=57, got {dataset_max.second} — "
            "transcription error: check config.yml ingest.dataset_max"
        )
        assert dataset_max == datetime(2022, 3, 31, 23, 59, 57)

    def test_check_no_new_data_reads_dataset_max_from_config(self) -> None:
        """_check_no_new_data skips when window_start >= config dataset_max.

        This test patches load_config to inject a custom dataset_max and
        verifies the callable reads it (rather than using the module-level
        DATASET_MAX constant). Requires Airflow to be importable — skips in
        the host env.
        """
        pytest.importorskip("airflow", reason="Airflow not installed in host env")

        from airflow.exceptions import AirflowSkipException  # type: ignore[import-not-found]

        from orchestration.dags.the_15_minute_refresh_inference_dag import (
            _check_no_new_data,
        )

        # Arrange — custom dataset_max set to 2021-01-20 so a Jan-21 window triggers skip
        custom_max = datetime(2021, 1, 20, 0, 0, 0)
        fake_config = {"ingest": {"dataset_max": custom_max.isoformat()}}

        window = {
            "window_start": custom_max.isoformat(),  # equals dataset_max → exhausted
            "window_end": datetime(2021, 1, 20, 0, 15, 0).isoformat(),
        }
        ti_mock = MagicMock()
        ti_mock.xcom_pull.return_value = window

        # Act / Assert — AirflowSkipException raised because config dataset_max is respected
        with patch(
            "orchestration.dags.the_15_minute_refresh_inference_dag.load_config",
            return_value=fake_config,
        ):
            with pytest.raises(AirflowSkipException):
                _check_no_new_data(ti=ti_mock)
