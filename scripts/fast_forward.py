"""fast_forward.py — advance the inference watermark to a target date on the host.

Replays the DAG's per-tick pipeline body (ingest → validate → feature → predict)
using the same ``orchestration/lib`` helpers, so the script and the scheduled DAG
stay byte-faithful. Gap months (no raw data) are jumped by writing the watermark
forward without running ingest, avoiding empty-bronze validation failures.

Usage::

    uv run python scripts/fast_forward.py --until 2022-01-31
    uv run python scripts/fast_forward.py --until 2022-03-31

The ``--until`` date is inclusive: the loop stops when the watermark reaches or
exceeds that timestamp. Fast-forward assumes raw data is the six pruned month dirs
(2021_01..03, 2022_01..03); the gap-jump is computed from a runtime scan, so
re-adding raw months changes the ranges automatically.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.core.config import AppConfig, load_config
from app.core.watermark import next_window, read_watermark, write_watermark
from feature_helpers import build_feature_config, run_feature  # type: ignore[import-not-found]
from ingest_helpers import build_ingest_config, run_ingest  # type: ignore[import-not-found]
from predict_helpers import build_predict_config, run_predict  # type: ignore[import-not-found]
from validate_helpers import build_validate_config, run_validate  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers — unit-tested in isolation (no I/O, no Airflow)
# ---------------------------------------------------------------------------


def _contiguous_month_ranges(dir_names: list[str]) -> list[list[str]]:
    """Group YYYY_MM directory names into calendar-adjacent runs.

    Args:
        dir_names: Unsorted list of directory names like ``"2021_01"``.

    Returns:
        List of groups, each group a list of ``"YYYY-MM"`` strings in
        ascending order. Directories that are not calendar-adjacent to any
        neighbour form single-element groups.
    """

    def _to_ym(name: str) -> tuple[int, int]:
        y, m = name.split("_")
        return int(y), int(m)

    def _next_ym(y: int, m: int) -> tuple[int, int]:
        if m == 12:
            return y + 1, 1
        return y, m + 1

    def _fmt(y: int, m: int) -> str:
        return f"{y:04d}-{m:02d}"

    parsed = sorted(_to_ym(d) for d in dir_names)
    if not parsed:
        return []

    groups: list[list[str]] = [[_fmt(*parsed[0])]]
    for prev, curr in zip(parsed, parsed[1:]):
        if curr == _next_ym(*prev):
            groups[-1].append(_fmt(*curr))
        else:
            groups.append([_fmt(*curr)])
    return groups


def _next_action(
    pos: datetime,
    ranges: list[list[str]],
    *,
    until: datetime,
) -> tuple[str, datetime | None]:
    """Decide what to do next given the current watermark position.

    Args:
        pos: Current watermark position (naive datetime).
        ranges: Contiguous-month range list from ``_contiguous_month_ranges``.
        until: Target stop timestamp (inclusive).

    Returns:
        ``("RUN", None)`` when pos is inside a range (data exists).
        ``("JUMP", start_of_next_range)`` when pos is in a gap.
        ``("STOP", None)`` when pos >= until or no next range exists.
    """
    if pos >= until:
        return "STOP", None

    def _range_start(r: list[str]) -> datetime:
        y, m = r[0].split("-")
        return datetime(int(y), int(m), 1)

    def _range_end_exclusive(r: list[str]) -> datetime:
        y, m = r[-1].split("-")
        # First day of the month after the last month in the range
        if int(m) == 12:
            return datetime(int(y) + 1, 1, 1)
        return datetime(int(y), int(m) + 1, 1)

    for i, rng in enumerate(ranges):
        start = _range_start(rng)
        end = _range_end_exclusive(rng)
        if start <= pos < end:
            return "RUN", None
        if pos < start:
            # pos is in a gap before this range
            return "JUMP", start

    return "STOP", None


# ---------------------------------------------------------------------------
# Tick body — mirrors the DAG's per-tick pipeline body on the host
# ---------------------------------------------------------------------------


def advance_one_tick(
    position: datetime,
    *,
    config: AppConfig,
    watermark_root: Path,
    predict: bool = True,
) -> datetime:
    """Run one pipeline tick starting at *position* and advance the watermark.

    Uses host-relative paths (no container prefix) via the config values
    directly — the isolation seam is still present (build_*_config overrides),
    but the paths are the same as in config.yml on the host.

    Args:
        position: Current watermark position (start of the tick window).
        config: Loaded AppConfig.
        watermark_root: Host path to the watermark directory.
        predict: When False, skip the predict step. Use this for initial data
            population runs before a model has been trained.

    Returns:
        The new watermark position (end of the tick window).
    """
    duration = pd.Timedelta(config.ingest.slice_duration).to_pytimedelta()
    seed_start = datetime.fromisoformat(config.ingest.seed_start)
    _, window_end = next_window(position, duration)

    # Build host-relative override configs (identity paths — no container prefix)
    ingest_cfg = build_ingest_config(
        config,
        raw_root=config.data.raw_path,
        bronze_root=config.data.bronze_path,
    )
    validate_cfg = build_validate_config(config, silver_root=config.data.silver_path)
    feature_cfg = build_feature_config(
        config,
        silver_root=config.data.silver_path,
        gold_root=config.data.gold_path,
        shapefile_path=config.feature.zone_shapefile_path,
        model_root=config.model.store_path,
    )

    bronze_path = run_ingest(window_end, seed_start=seed_start, config=ingest_cfg)
    run_validate(bronze_path, config=validate_cfg)
    run_feature(bronze_path, config=feature_cfg)

    if predict:
        predict_cfg = build_predict_config(
            config,
            gold_root=config.data.gold_path,
            model_root=config.model.store_path,
            predictions_path=config.inference.output_path,
        )
        run_predict(config=predict_cfg, since=position)

    write_watermark(watermark_root, window_end)
    return window_end


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main(until: datetime, *, predict: bool = True) -> None:
    """Advance the inference watermark to *until*, gap-jumping dead windows.

    Scans ``config.data.raw_path`` at runtime to discover available month dirs
    and computes contiguous ranges. Ticks within a range run the full pipeline;
    positions in a gap jump directly to the next range start.

    Args:
        until: Target stop timestamp (inclusive).
        predict: When False, skip the predict step on every tick. Use this for
            initial data population runs before a model has been trained.
    """
    config = load_config()
    watermark_root = Path(config.serving.watermark_path)
    seed_start = datetime.fromisoformat(config.ingest.seed_start)

    position = read_watermark(watermark_root) or seed_start

    # Discover month ranges from disk
    raw_path = Path(config.data.raw_path)
    dir_names = [d.name for d in sorted(raw_path.iterdir()) if d.is_dir()]
    ranges = _contiguous_month_ranges(dir_names)

    logger.info("fast_forward: starting at %s, target %s", position, until)
    logger.info("fast_forward: discovered ranges: %s", ranges)
    if not predict:
        logger.info("fast_forward: predict=False — skipping predict step on every tick")

    while True:
        action, jump_target = _next_action(position, ranges, until=until)

        if action == "STOP":
            logger.info("fast_forward: reached target %s — done", until)
            break

        if action == "JUMP":
            assert jump_target is not None
            logger.info("fast_forward: gap-jump %s → %s", position, jump_target)
            write_watermark(watermark_root, jump_target)
            position = jump_target
            continue

        # RUN: one pipeline tick
        logger.info("fast_forward: tick from %s", position)
        position = advance_one_tick(
            position, config=config, watermark_root=watermark_root, predict=predict
        )
        logger.info("fast_forward: watermark now %s", position)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Advance inference watermark by replaying the DAG's tick body."
    )
    parser.add_argument(
        "--until",
        required=True,
        help="Target date (inclusive), e.g. 2022-01-31",
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Skip the predict step (use when no model exists yet)",
    )
    args = parser.parse_args()

    until_dt = datetime.fromisoformat(args.until)
    main(until=until_dt, predict=not args.no_predict)
