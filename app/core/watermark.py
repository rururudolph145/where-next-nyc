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

import json
import os
from datetime import date, datetime
from pathlib import Path

_BUCKETS_PER_DAY = 96

# Control-file name holding the incremental-pipeline position. Lives under
# ``data/watermarks/refresh_15min/``. Distinct from the per-(day, bucket)
# ``.done`` markers above: this records the *data* watermark — how far the
# refresh_15min_dag has advanced through the fixed 2021 raw timeline.
_WATERMARK_FILE = ".current_watermark"


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


# ---------------------------------------------------------------------------
# Data watermark (incremental-pipeline position)
# ---------------------------------------------------------------------------
#
# The refresh_15min_dag walks the FIXED 2021 raw timeline one slice at a
# time. The "simulation position" is a data watermark — a single JSON
# control file — not a clock. Each run reads the watermark, ingests the
# next slice, and advances it.
#
# Timezone contract: ``tpep_pickup_datetime`` in the raw/bronze data is
# ``datetime64[us]`` and tz-NAIVE; the timestamps ARE the canonical NYC
# wall-clock timeline. The watermark is therefore stored and returned as a
# NAIVE datetime (ISO-8601 with no ``Z`` / no offset). This keeps the
# window filter a naive-vs-naive comparison and avoids the 5h offset / the
# ``TypeError: Cannot compare tz-naive and tz-aware`` that tz-aware values
# would introduce.


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
        # Defensive: the contract is naive. Strip any tzinfo rather than
        # propagate a tz-aware value into the naive-vs-naive window filter.
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
