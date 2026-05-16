import logging
from typing import Any

import yaml

from src.ingestion.ingest import Ingestion
from src.ingestion.clean import DataCleaner
from src.ingestion.feature import FeatureEngineer
from src.training.train import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def run_pipeline() -> None:
    """Run the ingest → clean → feature engineering → training pipeline.

    1. Consolidate raw daily parquets → single Bronze file.
    2. Clean Bronze → Silver.
    3. Feature engineering on Silver → Gold.
    4. Train demand prediction model, log to MLflow, save artifact.
    """
    with open("config.yml") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    # ── Step 1: Ingest – raw → Bronze ────────────────────────────────
    logging.info("Step 1: Ingestion (raw → bronze)")
    Ingestion(config).run()

    # ── Step 2: Clean – Bronze → Silver ──────────────────────────────
    logging.info("Step 2: Cleaning (bronze → silver)")
    DataCleaner(config).run()

    # ── Step 3: Feature Engineering – Silver → Gold ──────────────────
    logging.info("Step 3: Feature engineering (silver → gold)")
    FeatureEngineer(config).run()

    # ── Step 4: Training – Gold → Model ──────────────────────────────
    logging.info("Step 4: Training")
    trainer = Trainer(config)
    result = trainer.run()
    logging.info(
        "Training complete. Val MAE=%.2f, Test MAE=%.2f. Model → %s",
        result["val_metrics"]["mae"],
        result["test_metrics"]["mae"],
        result["model_path"],
    )

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
