"""Actuals-join job: enriches prediction logs with Gold trip-count actuals.

Reads all prediction log parquets for a given date range, left-joins them
against the Gold ``hourly_demand_featured.parquet`` on
``(pulocationid, bucket_time == hour)``, and writes one joined parquet file
per CLI invocation.

Usage (CLI)::

    uv run python -m app.monitoring.actuals_join \\
        --start-date 2024-11-01 \\
        --end-date   2024-11-30

Design note — deferred wiring (see PBI-MON-01)
-----------------------------------------------
The 15-minute inference DAG intentionally writes ``actual_trip_count = NaN``
in the prediction history.  At prediction time the DAG feeds an
inference-only Gold slice (no ``trip_count`` label) to the predictor, so the
actual is structurally absent even though Gold already contains the value for
the previous bucket.

This job is the intended backfill: it reads Gold directly and left-joins on
``(pulocationid, bucket_time == hour)`` to fill those NaNs for all buckets
where ``bucket_time <= Gold max hour``.

It is **not** wired into the 15-minute DAG by design — actuals backfill is a
monitoring/evaluation concern that runs on a separate (e.g. daily) cadence.
Wiring it into the inference DAG would couple model serving with monitoring
and do a full Gold read every 15 minutes unnecessarily.

TODO (PBI-MON-01): implement a daily ``actuals_backfill_dag`` that calls
``join_actuals`` for the prior calendar day and writes results to
``data/monitoring/joined/``.  This output is also the input for the closed
training-loop retraining trigger (deferred mode-3 work).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.core.config import load_config


def join_actuals(
    start_date: date,
    end_date: date,
    predictions_root: str,
    gold_path: str,
    output_root: str,
) -> Path:
    """Join prediction logs against Gold actuals for a date range.

    Reads all parquet files under ``{predictions_root}/dt=YYYY-MM-DD/`` for
    every calendar date in ``[start_date, end_date]`` (inclusive), left-joins
    them against the Gold ``hourly_demand_featured.parquet`` on
    ``(pulocationid, bucket_time == hour)``, and writes the result to
    ``{output_root}/joined/dt={start}_to_{end}.parquet``.

    Dates with no prediction files are silently skipped.

    Args:
        start_date: First date of the prediction window to include.
        end_date: Last date of the prediction window to include (inclusive).
        predictions_root: Root directory of prediction logs
            (e.g. ``"data/monitoring/predictions"``).
        gold_path: Directory containing ``hourly_demand_featured.parquet``
            (e.g. ``"data/gold/yellow_tripdata"``).
        output_root: Root directory for joined output files
            (e.g. ``"data/monitoring/joined"``).

    Returns:
        The ``Path`` of the written joined parquet file.

    Raises:
        FileNotFoundError: If the Gold parquet file does not exist.
        SystemExit: If no prediction files are found in the date range.
    """
    # Collect all prediction parquet files in the date range.
    frames: list[pd.DataFrame] = []
    current = start_date
    while current <= end_date:
        dt_dir = os.path.join(predictions_root, f"dt={current.isoformat()}")
        pattern = os.path.join(dt_dir, "*.parquet")
        matched = glob.glob(pattern)
        for fp in sorted(matched):
            frames.append(pd.read_parquet(fp))
        current += timedelta(days=1)

    if not frames:
        print(
            f"No prediction files found between {start_date} and {end_date}. "
            "Ensure the simulator has run and prediction logs exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    predictions_df = pd.concat(frames, ignore_index=True)

    # Load Gold parquet; raise clearly if absent.
    gold_file = os.path.join(gold_path, "hourly_demand_featured.parquet")
    if not os.path.exists(gold_file):
        raise FileNotFoundError(
            f"Gold parquet not found: {gold_file}. "
            "Run the feature-engineering pipeline first."
        )
    gold_df = pd.read_parquet(gold_file)

    # Align Gold timestamps to UTC so the join key matches bucket_time.
    gold_df["hour"] = pd.to_datetime(gold_df["hour"]).dt.tz_localize("UTC")

    # Left-join: keep all prediction rows; attach trip_count as actual.
    joined = predictions_df.merge(
        gold_df[["pulocationid", "hour", "trip_count"]],
        left_on=["pulocationid", "bucket_time"],
        right_on=["pulocationid", "hour"],
        how="left",
    )
    joined = joined.rename(columns={"trip_count": "actual"})
    # Drop the redundant 'hour' column added by the merge.
    if "hour" in joined.columns:
        joined = joined.drop(columns=["hour"])

    # Ensure column order matches the spec.
    col_order = [
        "bucket_time",
        "pulocationid",
        "predicted",
        "actual",
        "model_version",
        "served_from_stage",
        "served_at",
    ]
    joined = joined[[c for c in col_order if c in joined.columns]]

    start_str = start_date.isoformat()
    end_str = end_date.isoformat()
    out_dir = os.path.join(output_root, "joined")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"dt={start_str}_to_{end_str}.parquet")
    joined.to_parquet(out_path, index=False)

    return Path(out_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the actuals-join job.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with ``start_date`` and ``end_date`` as ``date`` objects.
    """
    parser = argparse.ArgumentParser(
        description="Join prediction logs against Gold actuals."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="First date of the prediction window (inclusive).",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Last date of the prediction window (inclusive).",
    )
    ns = parser.parse_args(argv)
    ns.start_date = date.fromisoformat(ns.start_date)
    ns.end_date = date.fromisoformat(ns.end_date)
    return ns


if __name__ == "__main__":
    args = _parse_args()
    config = load_config()
    monitoring = config.get("monitoring", {})
    out = join_actuals(
        start_date=args.start_date,
        end_date=args.end_date,
        predictions_root=monitoring.get(
            "predictions_path", "data/monitoring/predictions"
        ),
        gold_path=config["data"]["gold_path"],
        output_root=monitoring.get("joined_path", "data/monitoring/joined"),
    )
    print(f"Joined output written to: {out}")
