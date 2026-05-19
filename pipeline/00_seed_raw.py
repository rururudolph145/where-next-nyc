"""Step 0 — Seed raw layer: split Bronze monthly parquets into daily files.

Reads monthly parquet files from data/bronze/ and splits them by day into
data/raw/yellow_tripdata/YYYY_MM/YYYY_MM_DD.parquet.

This is a one-time setup step that transforms the bulk Socrata download into
the raw folder structure that simulates real-world trip event arrival.

Usage:
    uv run python pipeline/00_seed_raw.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

RAW_BASE = "data/raw/yellow_tripdata"
PICKUP_COL = "tpep_pickup_datetime"


def split_month_to_daily(parquet_path: str, raw_base: str) -> int:
    """Split a monthly Bronze parquet into daily raw files.

    Args:
        parquet_path: Path to the monthly Bronze parquet file.
        raw_base: Base directory for raw output (e.g. ``data/raw/yellow_tripdata``).

    Returns:
        Number of daily files written.
    """
    df = pd.read_parquet(parquet_path)

    if PICKUP_COL not in df.columns:
        logging.warning(
            "Column %s not found in %s, skipping.", PICKUP_COL, parquet_path
        )
        return 0

    df[PICKUP_COL] = pd.to_datetime(df[PICKUP_COL], errors="coerce")
    df = df.dropna(subset=[PICKUP_COL])

    df["_date"] = df[PICKUP_COL].dt.date
    files_written = 0

    for day, group in df.groupby("_date"):
        year = day.year  # type: ignore[union-attr]
        month = day.month  # type: ignore[union-attr]
        day_num = day.day  # type: ignore[union-attr]

        month_tag = f"{year}_{month:02d}"
        day_tag = f"{year}_{month:02d}_{day_num:02d}"

        out_dir = os.path.join(raw_base, month_tag)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{day_tag}.parquet")

        group.drop(columns=["_date"]).to_parquet(out_path, index=False)
        logging.info(
            "  %s → %s (%s rows)",
            day_tag,
            out_path,
            f"{len(group):,}",
        )
        files_written += 1

    return files_written


def run_seed_raw() -> None:
    """Scan all Bronze monthly parquets and split into daily raw files."""
    config: dict[str, Any] = load_config()
    bronze_base: str = config["data"]["bronze_path"]

    if not os.path.isdir(bronze_base):
        logging.error("Bronze path does not exist: %s", bronze_base)
        sys.exit(1)

    logging.info("Scanning Bronze path: %s", bronze_base)
    logging.info("Output raw path: %s", RAW_BASE)

    total_files = 0

    for month_dir in sorted(os.listdir(bronze_base)):
        month_path = os.path.join(bronze_base, month_dir)
        if not os.path.isdir(month_path):
            continue

        for fname in sorted(os.listdir(month_path)):
            if not fname.endswith(".parquet"):
                continue

            parquet_path = os.path.join(month_path, fname)
            logging.info("Processing %s", parquet_path)
            files_written = split_month_to_daily(parquet_path, RAW_BASE)
            total_files += files_written

    logging.info("Done. %d daily files written to %s", total_files, RAW_BASE)


if __name__ == "__main__":
    run_seed_raw()
