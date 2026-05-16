"""Step 2 — Cleaning: Bronze → Silver (validate, clean, save)."""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.clean import DataCleaner
from src.shared.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def run_clean() -> None:
    """Clean the Bronze parquet and produce a Silver parquet."""
    config: dict[str, Any] = load_config()
    logging.info("Cleaning bronze → silver")

    cleaned = DataCleaner(config).run()
    logging.info("Cleaning complete. Silver rows: %s", f"{len(cleaned):,}")


if __name__ == "__main__":
    run_clean()
