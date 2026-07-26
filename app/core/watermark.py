"""Data-watermark primitive for incremental DAG runs.

The watermark is a single JSON control file recording how far the pipeline
has advanced through the fixed 2021 raw timeline:

    {root}/.current_watermark  ->  {"position": "<naive-iso>"}

There are two functions for reading/writing that state and one pure helper
(`next_window`) for computing the next ingest window from the current
position.

This module lives under ``app/core/`` so the dependency-direction invariant
(``app/`` never imports from ``orchestration/``) holds: the DAG depends on
the watermark, not the other way around.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Control-file name holding the incremental-pipeline position. Lives under
# ``data/watermarks/refresh_15min/``.
_WATERMARK_FILE = ".current_watermark"


def _watermark_file(root: Path) -> Path:
    """Return the control-file path holding the data watermark under *root*."""
    return root / _WATERMARK_FILE


def read_watermark(root: Path) -> datetime | None:
    """Return the current data watermark, or ``None`` if not yet set.

    Args:
        root: Watermark base directory (typically
            ``data/watermarks/refresh_15min``).

    Returns:
        The stored position as a NAIVE ``datetime``, or ``None`` when the
        control file is absent (first run — caller should fall back to the
        configured ``ingest.seed_start``).

    Raises:
        ValueError: if the control file exists but is malformed (invalid
            JSON, missing ``position`` key, or unparseable timestamp).
    """
    path = _watermark_file(root)
    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        position = payload["position"]
        dt = datetime.fromisoformat(position)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed watermark control file at {path}: {exc}") from exc

    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def write_watermark(root: Path, dt: datetime) -> None:
    """Atomically persist the data watermark *dt* under *root*.

    Writes to a ``.tmp`` sibling then ``os.replace`` to make the update
    atomic — a crash mid-write never leaves a half-written control file, and
    a failed run leaves the previous watermark untouched (the next run then
    reprocesses the same slice).

    Args:
        root: Watermark base directory.
        dt: The NAIVE position to record. A tz-aware value is coerced to
            naive defensively (the contract is naive throughout).
    """
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)

    root.mkdir(parents=True, exist_ok=True)
    final = _watermark_file(root)
    tmp = final.with_name(_WATERMARK_FILE + ".tmp")
    tmp.write_text(json.dumps({"position": dt.isoformat()}), encoding="utf-8")
    os.replace(tmp, final)


def next_window(
    position: datetime,
    duration: timedelta,
) -> tuple[datetime, datetime]:
    """Return the ingest window ``(position, position + duration)``. Pure.

    Args:
        position: The current watermark position (naive datetime).
        duration: The window length.

    Returns:
        A ``(start, end)`` tuple. ``start == position``; ``end == position + duration``.
    """
    return position, position + duration
