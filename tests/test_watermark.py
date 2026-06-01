"""Tests for app/core/watermark.py — DAG idempotency primitive.

PBI 25b: covers the watermark contract used by refresh_15min_dag.
A watermark is an empty marker file at:

    {root}/{YYYY-MM-DD}/{bucket:02d}.done

If the file exists, the (date, bucket) work unit has been completed and
must not be re-run. The DAG's check_watermark task uses is_complete()
to short-circuit; write_done uses mark_complete() to record success.

AC coverage:
  AC #9   — DAG short-circuits at check_watermark when .done file exists
  AC #10  — Successful run creates watermark; re-trigger is a no-op
  Edge   — data/watermarks/ absent on first run (mark_complete creates parents)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.core.watermark import bucket_path, is_complete, mark_complete

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

D0 = date(2024, 6, 3)  # the known busy Monday used by the demo


# ---------------------------------------------------------------------------
# TestBucketPath
# ---------------------------------------------------------------------------


class TestBucketPath:
    """bucket_path is a pure-function path builder; no I/O."""

    def test_returns_root_slash_date_slash_bucket_done(self, tmp_path: Path) -> None:
        """bucket_path encodes (root, date, bucket) as the .done file path.

        AC #10.
        """
        # Arrange
        root = tmp_path / "watermarks" / "refresh_15min"

        # Act
        result = bucket_path(root=root, day=D0, bucket=0)

        # Assert
        assert result == root / "2024-06-03" / "00.done"

    def test_bucket_is_zero_padded_to_two_digits(self, tmp_path: Path) -> None:
        """Bucket index 7 must serialize as '07', not '7'.

        Zero-pad keeps lexicographic order equal to chronological order so
        directory listings are intuitive to a human reviewer.
        """
        # Arrange
        root = tmp_path

        # Act
        result = bucket_path(root=root, day=D0, bucket=7)

        # Assert
        assert result.name == "07.done"

    def test_bucket_95_is_last_slot_of_day(self, tmp_path: Path) -> None:
        """Bucket 95 corresponds to 23:45 UTC — the final quarter-hour.

        Edge — boundary value.
        """
        # Arrange / Act
        result = bucket_path(root=tmp_path, day=D0, bucket=95)

        # Assert
        assert result.name == "95.done"

    @pytest.mark.parametrize("bad_bucket", [-1, 96, 100])
    def test_invalid_bucket_raises_valueerror(
        self, tmp_path: Path, bad_bucket: int
    ) -> None:
        """Bucket index must be in [0, 95]; reject anything outside.

        Edge — invalid input rejected at the API boundary.
        """
        # Arrange / Act / Assert
        with pytest.raises(ValueError):
            bucket_path(root=tmp_path, day=D0, bucket=bad_bucket)


# ---------------------------------------------------------------------------
# TestIsComplete
# ---------------------------------------------------------------------------


class TestIsComplete:
    """is_complete checks for the .done file's existence; no side effects."""

    def test_returns_false_when_done_file_absent(self, tmp_path: Path) -> None:
        """Fresh root with no .done files must report incomplete.

        AC #9 — happy-path: clean state means work must run.
        """
        # Arrange
        root = tmp_path

        # Act
        result = is_complete(root=root, day=D0, bucket=0)

        # Assert
        assert result is False

    def test_returns_true_when_done_file_present(self, tmp_path: Path) -> None:
        """After mark_complete, is_complete must return True for that bucket.

        AC #9.
        """
        # Arrange
        root = tmp_path
        mark_complete(root=root, day=D0, bucket=12)

        # Act
        result = is_complete(root=root, day=D0, bucket=12)

        # Assert
        assert result is True

    def test_is_per_bucket_not_per_day(self, tmp_path: Path) -> None:
        """A .done file for bucket 12 must not affect bucket 13.

        Edge — sibling-bucket independence.
        """
        # Arrange
        root = tmp_path
        mark_complete(root=root, day=D0, bucket=12)

        # Act / Assert
        assert is_complete(root=root, day=D0, bucket=12) is True
        assert is_complete(root=root, day=D0, bucket=13) is False


# ---------------------------------------------------------------------------
# TestMarkComplete
# ---------------------------------------------------------------------------


class TestMarkComplete:
    """mark_complete writes an empty .done file; idempotent."""

    def test_creates_done_file_at_correct_path(self, tmp_path: Path) -> None:
        """mark_complete must create the file at bucket_path's output.

        AC #10.
        """
        # Arrange
        root = tmp_path
        expected = bucket_path(root=root, day=D0, bucket=5)

        # Act
        mark_complete(root=root, day=D0, bucket=5)

        # Assert
        assert expected.is_file()

    def test_creates_parent_directories_on_first_run(self, tmp_path: Path) -> None:
        """mark_complete must create the {date}/ subdir if it doesn't exist.

        Edge — data/watermarks/ absent on first run (story edge case).
        """
        # Arrange
        root = tmp_path / "does" / "not" / "exist" / "yet"
        assert not root.exists()

        # Act
        mark_complete(root=root, day=D0, bucket=0)

        # Assert
        assert root.is_dir()
        assert (root / "2024-06-03" / "00.done").is_file()

    def test_is_idempotent(self, tmp_path: Path) -> None:
        """Calling mark_complete twice on the same bucket must not raise.

        AC #10 — re-triggering the same (date, bucket) is a no-op.
        """
        # Arrange
        root = tmp_path

        # Act
        mark_complete(root=root, day=D0, bucket=42)
        mark_complete(root=root, day=D0, bucket=42)  # second call

        # Assert
        assert is_complete(root=root, day=D0, bucket=42) is True
