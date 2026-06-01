"""Watermark primitive for DAG idempotency.

A watermark is an empty marker file recording that a ``(date, bucket)`` work
unit has been completed. ``refresh_15min_dag`` uses this to make re-running
the same bucket a no-op:

    {root}/{YYYY-MM-DD}/{bucket:02d}.done

There are 96 buckets per UTC day (one per quarter-hour); bucket 0 is
00:00–00:15 UTC and bucket 95 is 23:45–24:00 UTC.

This module lives under ``app/core/`` so the dependency-direction invariant
(``app/`` never imports from ``orchestration/``) holds: the DAG depends on
the watermark, not the other way around.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

_BUCKETS_PER_DAY = 96


def _validate_bucket(bucket: int) -> None:
    if not 0 <= bucket < _BUCKETS_PER_DAY:
        raise ValueError(f"bucket must be in [0, {_BUCKETS_PER_DAY - 1}]; got {bucket}")


def bucket_path(*, root: Path, day: date, bucket: int) -> Path:
    """Return the ``.done`` file path for ``(day, bucket)`` under ``root``.

    Pure function — no I/O.

    Args:
        root: Base directory holding all watermarks (typically
            ``data/watermarks/refresh_15min``).
        day: UTC date the bucket belongs to.
        bucket: Quarter-hour index in ``[0, 95]``.

    Raises:
        ValueError: if ``bucket`` is outside ``[0, 95]``.
    """
    _validate_bucket(bucket)
    return root / day.isoformat() / f"{bucket:02d}.done"


def is_complete(*, root: Path, day: date, bucket: int) -> bool:
    """Return True iff the ``.done`` file for ``(day, bucket)`` exists.

    Read-only; no side effects.

    Raises:
        ValueError: if ``bucket`` is outside ``[0, 95]``.
    """
    return bucket_path(root=root, day=day, bucket=bucket).is_file()


def mark_complete(*, root: Path, day: date, bucket: int) -> None:
    """Create the ``.done`` file for ``(day, bucket)``; idempotent.

    Creates ``root`` and the ``{day}/`` subdirectory if either is missing.
    Re-calling on an already-marked bucket is a no-op.

    Raises:
        ValueError: if ``bucket`` is outside ``[0, 95]``.
    """
    path = bucket_path(root=root, day=day, bucket=bucket)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
