"""Step 1 — Ingestion: consolidate raw daily parquets → Bronze layer."""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_config
from app.ingestion.ingest import Ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def run_ingestion() -> None:
    """Consolidate raw daily parquets into a single Bronze parquet file."""
    config: dict[str, Any] = load_config()
    logging.info(
        "Ingestion: %s → %s",
        config["ingest"]["start_year_month"],
        config["ingest"]["end_year_month"],
    )

    bronze_path = Ingestion(config).run()
    logging.info("Ingestion complete → %s", bronze_path)


if __name__ == "__main__":
    run_ingestion()
