"""Re-export ``split_month_to_daily`` for importability.

The canonical implementation lives in ``pipeline/00_seed_raw.py`` which
cannot be imported directly because its filename starts with a digit.
This thin wrapper re-implements the function to keep it importable.
"""

import logging
import os

import pandas as pd

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
