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
"""

from __future__ import annotations

from datetime import datetime

from boundary_helpers import is_dataset_exhausted

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
