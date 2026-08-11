"""bulk_advance.py — one-shot ingest→clean→feature→set watermark.

Use this to jump the watermark to any target date without running the
15-min tick loop. Correct for bulk backfill; fast_forward is for live
cadence only.

Usage:
    uv run python scripts/bulk_advance.py --until 2022-01-08
    uv run python scripts/bulk_advance.py --until 2022-03-31
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from app.core.config import load_config
from app.core.watermark import write_watermark
from app.ingestion.clean import DataCleaner
from app.ingestion.feature import FeatureEngineer
from app.ingestion.ingest import Ingestion

logger = logging.getLogger(__name__)


def main(until: datetime) -> None:
    cfg = load_config()
    seed_start = datetime.fromisoformat(cfg.ingest.seed_start)
    watermark_root = Path(cfg.serving.watermark_path)

    logger.info("bulk_advance: ingest [%s, %s)", seed_start, until)
    Ingestion(config=cfg).run(window_start=seed_start, window_end=until)

    logger.info("bulk_advance: clean")
    DataCleaner(config=cfg).run()

    logger.info("bulk_advance: feature")
    FeatureEngineer(config=cfg).run()

    write_watermark(watermark_root, until)
    logger.info("bulk_advance: watermark set to %s — done", until)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="One-shot bulk ingest→clean→feature + set watermark."
    )
    parser.add_argument(
        "--until",
        required=True,
        help="Target watermark date, e.g. 2022-01-08",
    )
    args = parser.parse_args()
    main(until=datetime.fromisoformat(args.until))
