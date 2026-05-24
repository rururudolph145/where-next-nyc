"""Stateful 15-minute bucket clock for the inference simulation.

The clock persists the current simulated bucket timestamp to a JSON file
on disk. Each DAG run advances the clock by one bucket. This lets a
real-time scheduler (Airflow firing every minute) drive a simulated
15-minute cadence over historical data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_STATE_DIR = os.environ.get("INFERENCE_STATE_DIR", "/opt/airflow/state")
STATE_FILENAME = "bucket_clock.json"
BUCKET_FREQ = pd.Timedelta("15min")


@dataclass(frozen=True)
class ClockState:
    """Current and next simulated bucket timestamps."""

    current: pd.Timestamp
    next: pd.Timestamp


def _state_path(state_dir: str = DEFAULT_STATE_DIR) -> Path:
    return Path(state_dir) / STATE_FILENAME


def initialise(start: pd.Timestamp, state_dir: str = DEFAULT_STATE_DIR) -> ClockState:
    """Initialise the clock at ``start`` if no state file exists yet.

    Idempotent: if the file already exists, the existing state is
    returned unchanged.
    """
    path = _state_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return read(state_dir)
    state = ClockState(current=start, next=start + BUCKET_FREQ)
    _write(state, path)
    return state


def read(state_dir: str = DEFAULT_STATE_DIR) -> ClockState:
    """Return the current clock state.

    Raises FileNotFoundError if the clock has not been initialised.
    """
    path = _state_path(state_dir)
    raw = json.loads(path.read_text())
    return ClockState(
        current=pd.Timestamp(raw["current"]),
        next=pd.Timestamp(raw["next"]),
    )


def advance(state_dir: str = DEFAULT_STATE_DIR) -> ClockState:
    """Advance the clock by one 15-min bucket and persist."""
    state = read(state_dir)
    new = ClockState(current=state.next, next=state.next + BUCKET_FREQ)
    _write(new, _state_path(state_dir))
    return new


def _write(state: ClockState, path: Path) -> None:
    path.write_text(
        json.dumps(
            {"current": state.current.isoformat(), "next": state.next.isoformat()},
            indent=2,
        )
    )
