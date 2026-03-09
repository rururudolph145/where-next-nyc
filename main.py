import os
import logging
import yaml
import pandas as pd
from datetime import date, timedelta

from steps.ingest import Ingestion, START_DATE, END_DATE
from steps.clean import DataCleaner

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

# Stop early after this many dates so the pipeline can be tested quickly.
# Set to None (or a very large number) to run the full date range.
MAX_DATES: int = 2


def run_pipeline() -> None:
    """Run the ingest → clean pipeline day by day.

    For each calendar day between START_DATE and END_DATE:
    1. Fetch raw trip data from the NYC Open Data API → Bronze layer.
    2. Clean the raw data and promote it → Silver layer.

    The loop exits as soon as *MAX_DATES* days have been processed or the
    END_DATE is reached, whichever comes first.
    """
    with open("config.yml") as f:
        config: dict[str, object] = yaml.safe_load(f)

    ingestor = Ingestion()
    cleaner = DataCleaner(config)  # type: ignore[arg-type]

    current = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)
    dates_fetched = 0

    while current <= end:
        date_str = current.isoformat()
        logging.info("--- [%d] Ingesting %s ---", dates_fetched + 1, date_str)

        # ── Ingest: fetch one day → Bronze ────────────────────────────────
        bronze_path = ingestor.fetch_day(date_str)

        if not os.path.exists(bronze_path):
            logging.warning("No data returned for %s, skipping clean step.", date_str)
            current += timedelta(days=1)
            continue

        # ── Clean: Bronze → Silver ─────────────────────────────────────────
        df = pd.read_parquet(bronze_path)
        cleaned = cleaner.clean(df)

        silver_dir = "data/silver"
        os.makedirs(silver_dir, exist_ok=True)
        silver_path = f"{silver_dir}/yellow_tripdata_cleaned_{date_str}.parquet"
        cleaned.to_parquet(silver_path, index=False)
        logging.info("Silver → %s  (%s rows)", silver_path, f"{len(cleaned):,}")

        dates_fetched += 1
        current += timedelta(days=1)

        if dates_fetched >= MAX_DATES:
            logging.info(
                "Stopped after %d date(s) (MAX_DATES fast-test limit).", MAX_DATES
            )
            break

    logging.info("Pipeline complete. Processed %d date(s).", dates_fetched)


if __name__ == "__main__":
    run_pipeline()
