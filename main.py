import os
import logging
from typing import Any

import yaml
import pandas as pd

from src.steps.ingest import Ingestion
from src.steps.clean import DataCleaner

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def _iter_months(start: str, end: str) -> list[tuple[int, int]]:
    """Return a list of (year, month) tuples from *start* to *end* inclusive.

    Args:
        start: Year-month string in ``YYYY-MM`` format (e.g. ``"2022-01"``).
        end: Year-month string in ``YYYY-MM`` format (e.g. ``"2022-04"``).

    Returns:
        List of ``(year, month)`` tuples covering the range.
    """
    s_year, s_month = (int(x) for x in start.split("-"))
    e_year, e_month = (int(x) for x in end.split("-"))
    months: list[tuple[int, int]] = []
    year, month = s_year, s_month
    while (year, month) <= (e_year, e_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def run_pipeline() -> None:
    """Run the ingest → clean pipeline month by month.

    For each month in the configured ``ingest.start_year_month`` to
    ``ingest.end_year_month`` range:
    1. Fetch raw trip data from the NYC Open Data API → Bronze layer.
    2. Clean the raw data and promote it → Silver layer.
    """
    with open("config.yml") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    ingestor = Ingestion()
    cleaner = DataCleaner(config)

    start_ym: str = config["ingest"]["start_year_month"]
    end_ym: str = config["ingest"]["end_year_month"]

    months = _iter_months(start_ym, end_ym)
    logging.info("Pipeline will process %d month(s): %s → %s", len(months), start_ym, end_ym)

    for i, (year, month) in enumerate(months, 1):
        tag = f"{year}_{month:02d}"
        logging.info("--- [%d/%d] Ingesting %s ---", i, len(months), tag)

        # ── Ingest: fetch one month → Bronze ────────────────────────────
        bronze_path = ingestor.fetch_month(year, month)

        if not os.path.exists(bronze_path):
            logging.warning("No data returned for %s, skipping clean step.", tag)
            continue

        # ── Clean: Bronze → Silver ──────────────────────────────────────
        df = pd.read_parquet(bronze_path)
        cleaned = cleaner.clean(df)

        silver_dir = "data/silver"
        os.makedirs(silver_dir, exist_ok=True)
        silver_path = f"{silver_dir}/yellow_tripdata_cleaned_{tag}.parquet"
        cleaned.to_parquet(silver_path, index=False)
        logging.info("Silver → %s  (%s rows)", silver_path, f"{len(cleaned):,}")

    logging.info("Pipeline complete. Processed %d month(s).", len(months))


if __name__ == "__main__":
    run_pipeline()
