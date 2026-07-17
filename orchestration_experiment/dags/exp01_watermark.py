"""Pure watermark primitive for the EXP-01 "now-signal" DAG.

Airflow-free by design: the experiment container has Airflow but the host test
env does not, so the DAG's logic lives here where pytest can import it. The DAG
file (``exp01_watermark_now_signal_dag.py``) is a thin wrapper that injects the
concrete path / seed / slice and calls these functions.

On-disk format and semantics deliberately mirror ``app/core/watermark`` so a
later refactor to the shared helper is a drop-in with zero data migration:
  * JSON ``{"position": "<naive-iso>"}``.
  * tz-NAIVE throughout (raw ``tpep_pickup_datetime`` is naive NYC wall-clock).
  * atomic write (``.tmp`` + ``os.replace``) — a crash mid-write never corrupts
    the file, and a failed run leaves the previous watermark intact.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path


def read_position(path: Path, *, seed: datetime) -> datetime:
    """Return the stored simulation position, or *seed* on the first run.

    Args:
        path: The watermark control file.
        seed: Fallback position when the control file does not yet exist.

    Returns:
        The current position as a NAIVE datetime.

    Raises:
        ValueError: if the control file exists but is malformed.
    """
    if not path.is_file():
        return seed

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dt = datetime.fromisoformat(payload["position"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed watermark control file at {path}: {exc}") from exc

    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def next_window(position: datetime, *, slice_: timedelta) -> tuple[datetime, datetime]:
    """Return the ingest window ``(position, position + slice_)``. Pure."""
    return position, position + slice_


def write_position(path: Path, position: datetime) -> None:
    """Atomically persist *position* as ``{"position": "<naive-iso>"}``.

    Writes to a ``.tmp`` sibling then ``os.replace`` for atomicity. A tz-aware
    value is coerced to naive defensively (the contract is naive throughout).
    """
    if position.tzinfo is not None:
        position = position.replace(tzinfo=None)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"position": position.isoformat()}), encoding="utf-8")
    os.replace(tmp, path)
