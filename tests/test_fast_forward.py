"""Tests for scripts/fast_forward.py — range builder, gap-jump decision, tick body."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from app.core.config import load_config


# ---------------------------------------------------------------------------
# Step 4 — tick body: advances watermark exactly one tick and calls run_* helpers
# ---------------------------------------------------------------------------


class TestAdvanceOneTick:
    """Tick body advances the watermark by slice_duration and invokes pipeline helpers."""

    def test_tick_advances_watermark_by_slice_duration(self, tmp_path: Path) -> None:
        """advance_one_tick() writes watermark = position + slice_duration."""
        from scripts.fast_forward import advance_one_tick

        # Arrange: position well inside the 2022 range
        config = load_config()
        wm_root = tmp_path / "watermarks"
        wm_root.mkdir()
        position = datetime(2022, 1, 15, 0, 0, 0)
        duration = pd.Timedelta(config.ingest.slice_duration).to_pytimedelta()
        expected_end = position + duration

        with (
            patch("scripts.fast_forward.run_ingest") as mock_ingest,
            patch("scripts.fast_forward.run_validate") as mock_validate,
            patch("scripts.fast_forward.run_feature") as mock_feature,
            patch("scripts.fast_forward.run_predict") as mock_predict,
        ):
            mock_ingest.return_value = "bronze.parquet"
            mock_validate.return_value = "bronze.parquet"
            mock_feature.return_value = "bronze.parquet"
            mock_predict.return_value = "predictions.parquet"

            # Act
            advance_one_tick(position, config=config, watermark_root=wm_root)

        # Assert: watermark now equals end of first tick
        from app.core.watermark import read_watermark

        written = read_watermark(wm_root)
        assert written == expected_end

    def test_tick_calls_pipeline_helpers_in_order(self, tmp_path: Path) -> None:
        """advance_one_tick() calls ingest → validate → feature → predict in order."""
        from scripts.fast_forward import advance_one_tick

        config = load_config()
        wm_root = tmp_path / "watermarks"
        wm_root.mkdir()
        position = datetime(2022, 1, 15, 0, 0, 0)

        call_order: list[str] = []

        def record(name: str) -> MagicMock:
            m = MagicMock()

            def _side_effect(*a: object, **kw: object) -> str:
                call_order.append(name)
                return "bronze.parquet"

            m.side_effect = _side_effect
            return m

        with (
            patch("scripts.fast_forward.run_ingest", record("ingest")),
            patch("scripts.fast_forward.run_validate", record("validate")),
            patch("scripts.fast_forward.run_feature", record("feature")),
            patch("scripts.fast_forward.run_predict", record("predict")),
        ):
            advance_one_tick(position, config=config, watermark_root=wm_root)

        # Assert: helpers called exactly once, in pipeline order
        assert call_order == ["ingest", "validate", "feature", "predict"]


# ---------------------------------------------------------------------------
# Step 5 — contiguous month ranges (pure helper)
# ---------------------------------------------------------------------------


class TestContiguousMonthRanges:
    """_contiguous_month_ranges groups YYYY_MM dirs into calendar-adjacent runs."""

    def test_current_disk_layout_produces_two_ranges(self) -> None:
        """Pruned disk: [2021_01..03] and [2022_01..03] → two ranges, one gap."""
        from scripts.fast_forward import _contiguous_month_ranges

        # Arrange
        dirs = ["2021_01", "2021_02", "2021_03", "2022_01", "2022_02", "2022_03"]

        # Act
        ranges = _contiguous_month_ranges(dirs)

        # Assert
        assert ranges == [
            ["2021-01", "2021-02", "2021-03"],
            ["2022-01", "2022-02", "2022-03"],
        ]

    def test_noise_dirs_do_not_break_grouping(self) -> None:
        """Noise dirs adjacent to each other group together; non-adjacent form gaps.

        2008-12 and 2009-01 are calendar-adjacent → same group.
        The gap between 2009-01 and 2021-01 produces a separate group.
        Algorithm is disk-scan-driven, not hardcoded."""
        from scripts.fast_forward import _contiguous_month_ranges

        # Arrange: re-introduce noise dirs to prove robustness
        dirs = [
            "2008_12",
            "2009_01",
            "2021_01",
            "2021_02",
            "2021_03",
            "2021_04",
            "2021_05",
            "2022_01",
            "2022_02",
            "2022_03",
            "2022_04",
        ]

        # Act
        ranges = _contiguous_month_ranges(dirs)

        # Assert: 2008-12 and 2009-01 are adjacent (Dec→Jan) → same group
        assert ranges == [
            ["2008-12", "2009-01"],
            ["2021-01", "2021-02", "2021-03", "2021-04", "2021-05"],
            ["2022-01", "2022-02", "2022-03", "2022-04"],
        ]


# ---------------------------------------------------------------------------
# Step 6 — gap-jump decision (pure helper)
# ---------------------------------------------------------------------------


class TestNextAction:
    """_next_action returns RUN, JUMP, or STOP for a given position."""

    def _current_disk_ranges(self) -> list[list[str]]:
        return [
            ["2021-01", "2021-02", "2021-03"],
            ["2022-01", "2022-02", "2022-03"],
        ]

    def test_position_inside_first_range_returns_run(self) -> None:
        """Mid-Q1-2021 position → RUN (data exists in first range)."""
        from scripts.fast_forward import _next_action

        pos = datetime(2021, 2, 15, 0, 0, 0)
        until = datetime(2022, 3, 31, 23, 59, 57)
        action, target = _next_action(pos, self._current_disk_ranges(), until=until)
        assert action == "RUN"
        assert target is None

    def test_position_in_gap_returns_jump_to_next_range_start(self) -> None:
        """Position in gap (Apr 2021 — no raw data) → JUMP to 2022-01-01."""
        from scripts.fast_forward import _next_action

        # April 2021 is in the gap between range-1 (end 2021-04-01) and range-2 (start 2022-01-01)
        pos = datetime(2021, 4, 15, 0, 0, 0)
        until = datetime(2022, 3, 31, 23, 59, 57)
        action, target = _next_action(pos, self._current_disk_ranges(), until=until)
        assert action == "JUMP"
        assert target == datetime(2022, 1, 1, 0, 0, 0)

    def test_position_at_or_past_until_returns_stop(self) -> None:
        """Position >= until → STOP."""
        from scripts.fast_forward import _next_action

        pos = datetime(2022, 3, 31, 23, 59, 57)
        until = datetime(2022, 3, 31, 23, 59, 57)
        action, target = _next_action(pos, self._current_disk_ranges(), until=until)
        assert action == "STOP"
        assert target is None
