"""Step 3 — Feature engineering: Silver → Gold layer."""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_config
from app.ingestion.feature import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def run_feature() -> None:
    """Run feature engineering on the full Silver dataset to produce Gold layer.

    Builds hourly demand panel, adds temporal/cyclical/lag features and
    zone coordinates, then saves to the configured Gold path.
    """
    config: dict[str, Any] = load_config()
    feature_eng = FeatureEngineer(config)

    logging.info("Running feature engineering: Silver → Gold")
    gold_df = feature_eng.run()
    logging.info("Feature engineering complete. Gold rows: %s", f"{len(gold_df):,}")


if __name__ == "__main__":
    run_feature()
