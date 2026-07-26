"""Assertion: the `.done` sentinel machinery is no longer part of `app.core.watermark`.

Post-promotion (design doc §6.2), the sentinel helpers (`bucket_path`,
`is_complete`, `mark_complete`) and the `_BUCKETS_PER_DAY`/`_validate_bucket`
support code are all removed. This test locks that deletion — if a future
refactor accidentally re-exports them, this test catches it.
"""

from __future__ import annotations

import pytest


class TestSentinelHelpersRemoved:
    """The four .done-sentinel symbols are not importable from app.core.watermark."""

    def test_bucket_path_is_gone(self) -> None:
        """bucket_path is not part of the public API."""
        # Arrange / Act / Assert
        with pytest.raises(ImportError):
            from app.core.watermark import bucket_path  # type: ignore[attr-defined]  # noqa: F401

    def test_is_complete_is_gone(self) -> None:
        """is_complete is not part of the public API."""
        # Arrange / Act / Assert
        with pytest.raises(ImportError):
            from app.core.watermark import is_complete  # type: ignore[attr-defined]  # noqa: F401

    def test_mark_complete_is_gone(self) -> None:
        """mark_complete is not part of the public API."""
        # Arrange / Act / Assert
        with pytest.raises(ImportError):
            from app.core.watermark import mark_complete  # type: ignore[attr-defined]  # noqa: F401
