"""Step 0b — Seed March 2021 raw data.

Downloads the ``yellow_tripdata_2021-03.parquet`` file from the NYC TLC
CloudFront distribution, saves it to the Bronze layer, then splits it
into daily files under the raw layer using the existing
:func:`pipeline.00_seed_raw.split_month_to_daily` helper.

Usage:
    uv run python pipeline/00_seed_march.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import seed_raw_helper
from app.core.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

TLC_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-03.parquet"
)
MONTH_TAG = "2021_03"


def download_march(bronze_base: str) -> str:
    """Download March 2021 yellow taxi parquet into the Bronze layer.

    Args:
        bronze_base: Root Bronze directory (e.g. ``data/bronze/yellow_tripdata``).

    Returns:
        Path to the downloaded parquet file.
    """
    out_dir = os.path.join(bronze_base, MONTH_TAG)
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{MONTH_TAG}.parquet")

    if os.path.exists(dest):
        logging.info("Already downloaded: %s", dest)
        return dest

    logging.info("Downloading %s → %s", TLC_URL, dest)
    urlretrieve(TLC_URL, dest)  # noqa: S310
    logging.info("Download complete: %s", dest)
    return dest


def run_seed_march() -> None:
    """Download March 2021 data and split into daily raw files."""
    config: dict[str, Any] = load_config()
    bronze_base: str = os.path.dirname(config["data"]["bronze_path"])

    parquet_path = download_march(bronze_base)

    raw_base = config["data"]["raw_path"]
    logging.info("Splitting %s into daily files under %s", parquet_path, raw_base)
    n_files = seed_raw_helper.split_month_to_daily(parquet_path, raw_base)
    logging.info("Done. %d daily files written.", n_files)


if __name__ == "__main__":
    run_seed_march()
